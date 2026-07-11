//! KIF (柿木) 形式棋譜パーサ．
//!
//! cshogi の `KIF.py` (pure Python) と同一の観測可能挙動を基本とし
//! (golden fixture parity を `tests/kifu_parity.rs` で実証)，以下は
//! 安全側に拡張・変更している:
//!
//! - **不成**: cshogi は正規表現に一致せず指し手を黙って読み飛ばす
//!   (以降の盤面文脈が壊れる) が，本実装は不成として正しく解釈する
//! - **同** の後の全角空白は省略可 (cshogi は「同　」のみ)
//! - **BOD (局面図)**: cshogi は黙って無視し初期局面が平手になってしまう
//!   が，本実装は明示的にエラーにする (誤った学習データの混入防止)
//! - **変化：** 行以降は読まない (cshogi は本譜に変化の指し手を連結して
//!   しまう)．「まで」行があれば cshogi も読まないため観測差は出ない
//!
//! cshogi 互換で維持している quirk (意図的):
//!
//! - 初期局面は **手合割ヘッダのみ**が決める
//! - endgame / win は **「まで」行のみ**から決まる (投了行等は無視)
//! - 投了等の終局手に消費時間が付くと `times` に追記される
//!   (`times.len() == moves.len() + 1` になり得る)
//! - 合法でない指し手も `moves` には追加される (盤面には反映されない)
//! - 認識できない行は無視 (KIF は lenient)
//!
//! 入力は UTF-8 文字列 (`&str`)．Shift_JIS (.kif) のデコードは呼び出し側
//! (Python 層) の責務．

use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::moves::Move;
use crate::sfen;
use crate::types::{Piece, PieceType, Square};

use super::record::{
    encode_drop, encode_move, GameRecord, KifuParseError, WIN_BLACK, WIN_DRAW, WIN_WHITE,
};

/// 手合割 → 初期局面 SFEN (cshogi `HANDYCAP_SFENS` 互換)．
fn handicap_sfen(name: &str) -> Option<&'static str> {
    Some(match name {
        "平手" => sfen::HIRATE_SFEN,
        "香落ち" => "lnsgkgsn1/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "右香落ち" => "1nsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "角落ち" => "lnsgkgsnl/1r7/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "飛車落ち" => "lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "飛香落ち" => "lnsgkgsn1/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "二枚落ち" => "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "三枚落ち" => "lnsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "四枚落ち" => "1nsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "五枚落ち" => "2sgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "左五枚落ち" => "1nsgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "六枚落ち" => "2sgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "八枚落ち" => "3gkg3/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "十枚落ち" => "4k4/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        _ => return None,
    })
}

/// 駒漢字 (正規化後の 1 文字) → 駒種．
fn piece_type_from_kanji(c: char) -> Option<PieceType> {
    Some(match c {
        '歩' => PieceType::Pawn,
        '香' => PieceType::Lance,
        '桂' => PieceType::Knight,
        '銀' => PieceType::Silver,
        '角' => PieceType::Bishop,
        '飛' => PieceType::Rook,
        '金' => PieceType::Gold,
        '玉' => PieceType::King,
        'と' => PieceType::ProPawn,
        '杏' => PieceType::ProLance,
        '圭' => PieceType::ProKnight,
        '全' => PieceType::ProSilver,
        '馬' => PieceType::Horse,
        '龍' => PieceType::Dragon,
        _ => return None,
    })
}

/// 全角数字 (１-９) → 1-9．
fn zenkaku_digit(c: char) -> Option<u8> {
    match c {
        '１'..='９' => Some((c as u32 - '１' as u32) as u8 + 1),
        _ => None,
    }
}

/// 漢数字 (一-九) → 1-9．
fn kanji_digit(c: char) -> Option<u8> {
    "一二三四五六七八九"
        .chars()
        .position(|k| k == c)
        .map(|i| i as u8 + 1)
}

/// 指し手の正規化 (cshogi 互換): 王→玉，竜→龍，成銀→全，成桂→圭，成香→杏．
fn normalize_move_line(line: &str) -> String {
    line.replace('王', "玉")
        .replace('竜', "龍")
        .replace("成銀", "全")
        .replace("成桂", "圭")
        .replace("成香", "杏")
}

/// 終局を表す語 (指し手欄に現れるもの)．
///
/// cshogi の `MOVE_RE` に含まれる 8 語のみ (「入玉勝ち」は cshogi の
/// 正規表現に含まれず無視されるため，互換のため含めない)．
const END_WORDS: [&str; 8] = [
    "中断",
    "投了",
    "持将棋",
    "千日手",
    "詰み",
    "切れ負け",
    "反則勝ち",
    "反則負け",
];

/// 指し手行のパース結果．
enum MoveLine {
    /// 通常の指し手
    Move {
        /// None = 「同」(直前の指し手と同じ移動先)
        to: Option<Square>,
        piece: PieceType,
        /// 駒打ち (「打」表記，または移動元 (00))
        drop: bool,
        promote: bool,
        from: Option<Square>,
        time: Option<i32>,
    },
    /// 終局手 (投了・中断など)
    End { time: Option<i32> },
}

/// `( 0:16/00:00:16)` 形式の消費時間をパースし，秒とパース後位置を返す．
///
/// cshogi 互換: `/` の前半のみを `:` で分割し `Σ t_i * 60^i` (逆順)．
fn parse_time(chars: &[char], mut i: usize) -> (Option<i32>, usize) {
    let start = i;
    // \s* (Unicode 空白: 全角空白含む — Python re の \s 互換)
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    if i >= chars.len() || chars[i] != '(' {
        return (None, start);
    }
    i += 1;
    while i < chars.len() && chars[i] == ' ' {
        i += 1;
    }
    let mut spent = String::new();
    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == ':') {
        spent.push(chars[i]);
        i += 1;
    }
    if spent.is_empty() || i >= chars.len() || chars[i] != '/' {
        return (None, start);
    }
    i += 1;
    let mut total_seen = false;
    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == ':') {
        total_seen = true;
        i += 1;
    }
    if !total_seen || i >= chars.len() || chars[i] != ')' {
        return (None, start);
    }
    i += 1;
    let mut sec: i64 = 0;
    for (k, part) in spent.split(':').rev().enumerate() {
        let v: i64 = part.parse().unwrap_or(0);
        sec += v * 60i64.pow(k as u32);
    }
    (Some(sec.clamp(i32::MIN as i64, i32::MAX as i64) as i32), i)
}

/// 指し手行 (`   1 ７六歩(77)   ( 0:01/00:00:01)` 等) をパースする．
///
/// 指し手行でなければ `None`．
fn parse_move_line(line: &str) -> Option<MoveLine> {
    let normalized = normalize_move_line(line);
    let chars: Vec<char> = normalized.chars().collect();
    let mut i = 0;
    // \A *[0-9]+
    while i < chars.len() && chars[i] == ' ' {
        i += 1;
    }
    let num_start = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i == num_start {
        return None;
    }
    // \s+
    let ws_start = i;
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    if i == ws_start {
        return None;
    }

    // 終局語
    let rest: String = chars[i..].iter().collect();
    for w in END_WORDS {
        if rest.starts_with(w) {
            let after = i + w.chars().count();
            let (time, _) = parse_time(&chars, after);
            return Some(MoveLine::End { time });
        }
    }

    // 移動先: 「同」または 全角数字+漢数字
    let to: Option<Square> = if chars.get(i) == Some(&'同') {
        i += 1;
        // cshogi は「同　」(全角空白必須) だが省略も許容する
        if chars.get(i) == Some(&'　') {
            i += 1;
        }
        None
    } else {
        let file = zenkaku_digit(*chars.get(i)?)?;
        let rank = kanji_digit(*chars.get(i + 1)?)?;
        i += 2;
        Some(Square::new(file - 1, rank - 1))
    };

    // 駒
    let piece = piece_type_from_kanji(*chars.get(i)?)?;
    i += 1;

    // 打 | [不]成? ( <from> )
    let mut promote = false;
    if chars.get(i) == Some(&'打') {
        i += 1;
    } else {
        if chars.get(i) == Some(&'不') && chars.get(i + 1) == Some(&'成') {
            // cshogi は不成を読み飛ばすが，本実装は不成として解釈する
            i += 2;
        } else if chars.get(i) == Some(&'成') {
            promote = true;
            i += 1;
        }
        if chars.get(i) != Some(&'(') {
            return None;
        }
        let f = chars.get(i + 1)?.to_digit(10)? as u8;
        let r = chars.get(i + 2)?.to_digit(10)? as u8;
        if chars.get(i + 3) != Some(&')') {
            return None;
        }
        i += 4;
        if f == 0 && r == 0 {
            // 移動元 (00) は駒打ち扱い (cshogi 互換)
        } else if (1..=9).contains(&f) && (1..=9).contains(&r) {
            let (time, _) = parse_time(&chars, i);
            return Some(MoveLine::Move {
                to,
                piece,
                drop: false,
                promote,
                from: Some(Square::new(f - 1, r - 1)),
                time,
            });
        } else {
            return None;
        }
    }

    let (time, _) = parse_time(&chars, i);
    Some(MoveLine::Move {
        to,
        piece,
        drop: true,
        promote,
        from: None,
        time,
    })
}

/// 「まで」行 (`まで7手で先手の勝ち` 等) をパースする．
///
/// 戻り値: `Some((win, endgame))`．行が「まで」行でなければ `None`．
#[allow(clippy::type_complexity)]
fn parse_result_line(line: &str) -> Option<(Option<u8>, &'static str)> {
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    // 　* (全角空白のみ — cshogi RESULT_RE 互換)
    while i < chars.len() && chars[i] == '　' {
        i += 1;
    }
    let rest: String = chars[i..].iter().collect();
    let rest = rest.strip_prefix("まで")?;
    let rest = rest.strip_prefix('、').unwrap_or(rest);
    let digits_end = rest
        .char_indices()
        .find(|(_, c)| !c.is_ascii_digit())
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    if digits_end == 0 {
        return None;
    }
    let rest = rest[digits_end..].strip_prefix("手で")?;

    for (side_c, is_black) in [('先', true), ('下', true), ('後', false), ('上', false)] {
        if let Some(r) = rest.strip_prefix(side_c) {
            let r = r.strip_prefix("手の")?;
            // 長い語から順に照合 (勝ち は 入玉勝ち/反則勝ち の接尾)
            return if r.starts_with("入玉勝ち") {
                Some((Some(if is_black { WIN_BLACK } else { WIN_WHITE }), "%KACHI"))
            } else if r.starts_with("反則勝ち") {
                Some((
                    Some(if is_black { WIN_BLACK } else { WIN_WHITE }),
                    if is_black {
                        "%+ILLEGAL_ACTION"
                    } else {
                        "%-ILLEGAL_ACTION"
                    },
                ))
            } else if r.starts_with("反則負け") {
                Some((
                    Some(if is_black { WIN_WHITE } else { WIN_BLACK }),
                    "%ILLEGAL_MOVE",
                ))
            } else if r.starts_with("勝ち") {
                Some((Some(if is_black { WIN_BLACK } else { WIN_WHITE }), "%TORYO"))
            } else {
                None
            };
        }
    }
    if rest.starts_with("中断") {
        Some((None, "%CHUDAN"))
    } else if rest.starts_with("千日手") || rest.starts_with("持将棋") {
        // cshogi 互換: 持将棋も %SENNICHITE / DRAW になる
        Some((Some(WIN_DRAW), "%SENNICHITE"))
    } else {
        None
    }
}

/// KIF 形式棋譜 (単一対局) をパースする．
pub fn parse_kif_str(content: &str) -> Result<GameRecord, KifuParseError> {
    let mut record = GameRecord {
        sfen: sfen::HIRATE_SFEN.to_string(),
        ..GameRecord::default()
    };

    let mut board = Board::new();
    // コメント (指し手 index → テキスト)．最後に moves と同長へ整列する
    let mut comments: Vec<(usize, String)> = Vec::new();
    let mut header_comments: Vec<String> = Vec::new();
    // 「同」の解決に使う直前に盤へ反映した指し手の移動先 (cshogi は
    // board.peek() — 合法で push された手のみが対象)
    let mut last_pushed_to: Option<Square> = None;

    for (i, raw_line) in content.split('\n').enumerate() {
        let line_no = i + 1;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('*') {
            // 指し手コメント (** は * 1 個分を剥がす — cshogi 互換)
            let text = rest.strip_prefix('*').unwrap_or(rest);
            if record.moves.is_empty() {
                header_comments.push(text.to_string());
            } else {
                let idx = record.moves.len() - 1;
                match comments.last_mut() {
                    Some((last_idx, buf)) if *last_idx == idx => {
                        buf.push('\n');
                        buf.push_str(text);
                    }
                    _ => comments.push((idx, text.to_string())),
                }
            }
            continue;
        }

        if let Some((key, value)) = line.split_once('：') {
            let value = value.trim_end_matches('　');
            match key {
                "先手" | "下手" => {
                    record.names[0] = Some(value.to_string());
                }
                "後手" | "上手" => {
                    record.names[1] = Some(value.to_string());
                }
                // cshogi は持駒ヘッダを読まない (バグ由来の no-op) — 互換
                "先手の持駒" | "下手の持駒" | "後手の持駒" | "上手の持駒" => {}
                "手合割" => {
                    let sfen_str = handicap_sfen(value).ok_or_else(|| {
                        KifuParseError::new(line_no, format!("未対応の手合割です: {value}"))
                    })?;
                    record.sfen = sfen_str.to_string();
                    board.set_sfen(sfen_str).expect("handicap SFEN must parse");
                }
                "変化" => {
                    // 変化以降は読まない (cshogi は「まで」行が無いと本譜に
                    // 連結してしまうが，本実装は安全側で打ち切る)
                    break;
                }
                _ => {
                    record.var_info.push((key.to_string(), value.to_string()));
                }
            }
            continue;
        }

        // BOD (局面図) は未対応: 黙って平手扱いにすると誤データを生むため
        // 明示エラー (cshogi は黙って無視する)
        if line.starts_with('|') || line.starts_with("+---") {
            return Err(KifuParseError::new(
                line_no,
                "BOD (局面図) からの初期局面指定は未対応です".to_string(),
            ));
        }

        if let Some(parsed) = parse_move_line(line) {
            match parsed {
                MoveLine::End { time } => {
                    // cshogi 互換: 終局手は endgame に影響せず，時間のみ
                    // times に追記される
                    if let Some(t) = time {
                        record.times.push(t);
                    }
                }
                MoveLine::Move {
                    to,
                    piece,
                    drop,
                    promote,
                    from,
                    time,
                } => {
                    let to_sq = match to {
                        Some(sq) => sq,
                        None => last_pushed_to.ok_or_else(|| {
                            KifuParseError::new(
                                line_no,
                                "「同」の対象となる直前の指し手がありません".to_string(),
                            )
                        })?,
                    };
                    let (encoded, mv) = if drop {
                        let encoded = encode_drop(to_sq, piece).ok_or_else(|| {
                            KifuParseError::new(line_no, format!("打てない駒です: {line}"))
                        })?;
                        (encoded, Move::new_drop(to_sq, piece))
                    } else {
                        let from_sq = from.ok_or_else(|| {
                            KifuParseError::new(line_no, format!("移動元がありません: {line}"))
                        })?;
                        let from_piece = Piece::from_raw_u8(board.piece_at(from_sq));
                        let pt_from = from_piece.piece_type().ok_or_else(|| {
                            KifuParseError::new(line_no, format!("移動元に駒がありません: {line}"))
                        })?;
                        let cap_piece = Piece::from_raw_u8(board.piece_at(to_sq));
                        let encoded =
                            encode_move(from_sq, to_sq, promote, pt_from, cap_piece.piece_type());
                        let mv = Move::new_move(
                            from_sq,
                            to_sq,
                            promote,
                            cap_piece.raw_u8(),
                            pt_from as u8,
                        );
                        (encoded, mv)
                    };
                    record.moves.push(encoded);
                    // cshogi 互換: 合法手のみ盤面に反映する
                    let m16 = mv.to_move16();
                    let is_legal = generate_legal_moves(&mut board)
                        .iter()
                        .any(|lm| lm.to_move16() == m16);
                    if is_legal {
                        board.do_move(mv);
                        last_pushed_to = Some(to_sq);
                    }
                    if let Some(t) = time {
                        record.times.push(t);
                    }
                }
            }
            continue;
        }

        if let Some((win, endgame)) = parse_result_line(line) {
            record.win = win;
            record.endgame = Some(endgame.to_string());
            // 終局以降の行 (変化など) は読まない (cshogi 互換)
            break;
        }

        // その他の行 (ヘッダの区切り・# コメント・& しおり等) は無視
    }

    record.scores = vec![0; record.moves.len()];
    record.comments = vec![String::new(); record.moves.len()];
    for (idx, text) in comments {
        record.comments[idx] = text;
    }
    record.header_comment = header_comments.join("\n");

    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC: &str = "手合割：平手\n\
                         先手：A\n\
                         後手：B\n\
                            1 ７六歩(77)   ( 0:01/00:00:01)\n\
                            2 ３四歩(33)   ( 0:02/00:00:03)\n\
                         まで2手で先手の勝ち\n";

    #[test]
    fn test_basic_moves_and_result() {
        let rec = parse_kif_str(BASIC).unwrap();
        assert_eq!(rec.moves, vec![0x00011e3b, 0x00010a15]);
        assert_eq!(rec.times, vec![1, 2]);
        assert_eq!(rec.win, Some(WIN_BLACK));
        assert_eq!(rec.endgame.as_deref(), Some("%TORYO"));
        assert_eq!(rec.names[0].as_deref(), Some("A"));
        assert_eq!(rec.sfen, sfen::HIRATE_SFEN);
    }

    #[test]
    fn test_funari_is_parsed() {
        // cshogi は不成を読み飛ばすが本実装は解釈する (安全側拡張)
        let kif = "手合割：平手\n\
                   先手：A\n\
                   後手：B\n\
                      1 ７六歩(77)\n\
                      2 ３四歩(33)\n\
                      3 ２二角不成(88)\n";
        let rec = parse_kif_str(kif).unwrap();
        assert_eq!(rec.moves.len(), 3);
        // 2二角不成: from=8八(70) to=2二(10) 角(5) 角捕獲(5) 不成
        let m = rec.moves[2];
        assert_eq!(m & 0x7f, 10);
        assert_eq!((m >> 7) & 0x7f, 70);
        assert_eq!((m >> 14) & 1, 0);
        assert_eq!((m >> 16) & 0xf, 5);
        assert_eq!((m >> 20) & 0xf, 5);
    }

    #[test]
    fn test_dou_without_zenkaku_space() {
        // 「同」直後に全角空白が無い表記も受理する (lenient 拡張)
        let kif = "手合割：平手\n\
                      1 ７六歩(77)\n\
                   2 ３四歩(33)\n\
                   3 ２二角成(88)\n\
                   4 同銀(31)\n";
        let rec = parse_kif_str(kif).unwrap();
        assert_eq!(rec.moves.len(), 4);
        assert_eq!(rec.moves[3] & 0x7f, 10); // to = 2二
    }

    #[test]
    fn test_bod_is_error() {
        let kif = "後手の持駒：なし\n\
                   +---------------------------+\n\
                   |v香v桂v銀v金v玉v金v銀v桂v香|一\n";
        assert!(parse_kif_str(kif).is_err());
    }

    #[test]
    fn test_unknown_handicap_is_error() {
        assert!(parse_kif_str("手合割：その他\n").is_err());
    }

    #[test]
    fn test_lenient_unknown_lines() {
        let kif = "# ---- generated ----\n\
                   手数----指手---------消費時間--\n\
                   &しおり\n\
                   まだ指し手なし(これも無視される行)\n";
        let rec = parse_kif_str(kif).unwrap();
        assert!(rec.moves.is_empty());
        assert!(rec.endgame.is_none());
        assert!(rec.win.is_none());
    }

    #[test]
    fn test_variation_header_stops_parsing() {
        let kif = "手合割：平手\n\
                      1 ７六歩(77)\n\
                   変化：1手\n\
                   1 ２六歩(27)\n";
        let rec = parse_kif_str(kif).unwrap();
        assert_eq!(rec.moves.len(), 1);
    }
}
