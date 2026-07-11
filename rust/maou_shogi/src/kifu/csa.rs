//! CSA 標準棋譜ファイル形式 (V2.2) パーサ．
//!
//! cshogi の CSA パーサ (C++ `parser.h`) と同一の観測可能挙動を基本とし
//! (golden fixture parity を `tests/kifu_parity.rs` で実証)，cshogi が
//! 未対応 (segfault) の範囲は V2.2 仕様に準拠して拡張している:
//!
//! - `P+` / `P-` による個別駒配置・持駒 (`00` 枡)・`AL` (残り全駒)
//! - 指し手行のカンマ区切り後続文 (`+7776FU,T12`)
//!
//! cshogi 互換で維持している quirk (意図的):
//!
//! - `'** <score>` コメントは**直前の指し手**の score/comment を上書きする
//! - 終局 (`%` 行) 後の `T` 行は `times` に追記される (`times.len() ==
//!   moves.len() + 1` になり得る)
//! - `'summary:illegal move:...` (floodgate) が endgame を上書きする
//! - 認識できない行はエラー (CSA は strict)

use crate::board::Board;
use crate::moves::Move;
use crate::sfen;
use crate::types::{Color, Piece, PieceType, Square, HAND_KINDS};

use super::record::{
    encode_drop, encode_move, GameRecord, KifuParseError, WIN_BLACK, WIN_DRAW, WIN_WHITE,
};

/// CSA 駒コード → 駒種．
fn piece_type_from_csa(code: &str) -> Option<PieceType> {
    Some(match code {
        "FU" => PieceType::Pawn,
        "KY" => PieceType::Lance,
        "KE" => PieceType::Knight,
        "GI" => PieceType::Silver,
        "KI" => PieceType::Gold,
        "KA" => PieceType::Bishop,
        "HI" => PieceType::Rook,
        "OU" => PieceType::King,
        "TO" => PieceType::ProPawn,
        "NY" => PieceType::ProLance,
        "NK" => PieceType::ProKnight,
        "NG" => PieceType::ProSilver,
        "UM" => PieceType::Horse,
        "RY" => PieceType::Dragon,
        _ => return None,
    })
}

/// "+FU" / "-FU" / " * " → 駒 (色付き)．
fn piece_from_csa(token: &str) -> Option<Piece> {
    if token == " * " || token == " *" {
        return Some(Piece::EMPTY);
    }
    let color = match token.as_bytes().first()? {
        b'+' => Color::Black,
        b'-' => Color::White,
        _ => return None,
    };
    let pt = piece_type_from_csa(token.get(1..3)?)?;
    Some(Piece::new(color, pt))
}

/// 数字文字列の先頭部分を i32 として解釈する (C++ `stoi` 互換: 後続の
/// 非数字は無視)．
fn parse_leading_i32(s: &str) -> Option<i32> {
    let s = s.trim_start();
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if c == '-' || c == '+' {
            if i != 0 {
                break;
            }
        } else if !c.is_ascii_digit() {
            break;
        }
        end = i + c.len_utf8();
    }
    s[..end].parse().ok()
}

/// 数値文字列の先頭部分を f32 として解釈する (C++ `stof` 互換)．
fn parse_leading_f32(s: &str) -> Option<f32> {
    let s = s.trim_start();
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if matches!(c, '0'..='9' | '-' | '+' | '.' | 'e' | 'E') {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    while end > 0 && s[..end].parse::<f32>().is_err() {
        end -= 1;
    }
    if end == 0 {
        None
    } else {
        s[..end].parse().ok()
    }
}

/// 初期局面ブロック (P 行と手番行) を SFEN に変換する．
fn parse_position(lines: &[String], line_no: usize) -> Result<String, KifuParseError> {
    let err = |msg: String| KifuParseError::new(line_no, msg);

    let mut squares = [Piece::EMPTY; 81];
    let mut hand = [[0u8; HAND_KINDS]; 2];
    let mut turn = Color::Black;

    for line in lines {
        let bytes = line.as_bytes();
        if bytes[0] != b'P' {
            // 手番行 (+ / -)
            turn = match bytes[0] {
                b'+' => Color::Black,
                b'-' => Color::White,
                _ => {
                    return Err(err(format!("不正な局面行: {line}")));
                }
            };
            continue;
        }
        match bytes.get(1) {
            // PI: 平手初形から指定駒を取り除く (駒落ち)
            Some(b'I') => {
                let hirate = sfen::parse_sfen(sfen::HIRATE_SFEN).expect("HIRATE_SFEN must parse");
                squares = hirate.squares;
                let mut idx = 2;
                while idx + 4 <= line.len() {
                    let file = bytes[idx].wrapping_sub(b'0');
                    let rank = bytes[idx + 1].wrapping_sub(b'0');
                    let code = &line[idx + 2..idx + 4];
                    idx += 4;
                    let pt = piece_type_from_csa(code)
                        .ok_or_else(|| err(format!("不正な駒コード: {code}")))?;
                    if !(1..=9).contains(&file) || !(1..=9).contains(&rank) {
                        return Err(err(format!("不正な駒落ち枡: {line}")));
                    }
                    let sq = Square::new(file - 1, rank - 1);
                    let cur = squares[sq.index()];
                    if cur.piece_type() != Some(pt) {
                        return Err(err(format!("駒落ち指定の駒が盤上と一致しません: {line}")));
                    }
                    squares[sq.index()] = Piece::EMPTY;
                }
            }
            // P1..P9: 1 段 9 枡の一括指定
            Some(r @ b'1'..=b'9') => {
                let rank = r - b'1';
                for i in 0..9u8 {
                    let start = 2 + 3 * (i as usize);
                    let token = if start + 3 <= line.len() {
                        &line[start..start + 3]
                    } else if start + 2 <= line.len() {
                        // 行末の " * " は末尾空白が落ちていても許容する
                        &line[start..start + 2]
                    } else if start >= line.len() {
                        " * "
                    } else {
                        return Err(err(format!("P{}行が短すぎます", rank + 1)));
                    };
                    let piece = piece_from_csa(token)
                        .ok_or_else(|| err(format!("不正な駒トークン: {token:?}")))?;
                    // 行内は 9筋→1筋 の順 (col = 8-i)
                    let sq = Square::new(8 - i, rank);
                    squares[sq.index()] = piece;
                }
            }
            // P+ / P-: 個別配置 (00 は持駒，AL は残り全駒)
            Some(c @ (b'+' | b'-')) => {
                let color = if *c == b'+' {
                    Color::Black
                } else {
                    Color::White
                };
                let mut idx = 2;
                while idx + 4 <= line.len() {
                    let file = bytes[idx].wrapping_sub(b'0');
                    let rank = bytes[idx + 1].wrapping_sub(b'0');
                    let code = &line[idx + 2..idx + 4];
                    idx += 4;
                    if code == "AL" {
                        if file != 0 || rank != 0 {
                            return Err(err("AL は持駒 (00) にのみ指定できます".to_string()));
                        }
                        distribute_remaining(&squares, &mut hand, color);
                        continue;
                    }
                    let pt = piece_type_from_csa(code)
                        .ok_or_else(|| err(format!("不正な駒コード: {code}")))?;
                    if file == 0 && rank == 0 {
                        let hi = pt
                            .hand_index()
                            .ok_or_else(|| err(format!("持駒にできない駒です: {code}")))?;
                        hand[color.index()][hi] += 1;
                    } else if (1..=9).contains(&file) && (1..=9).contains(&rank) {
                        let sq = Square::new(file - 1, rank - 1);
                        squares[sq.index()] = Piece::new(color, pt);
                    } else {
                        return Err(err(format!("不正な配置枡: {line}")));
                    }
                }
            }
            _ => {
                return Err(err(format!("不正な局面行: {line}")));
            }
        }
    }

    Ok(sfen::to_sfen(&squares, turn, &hand, 1))
}

/// AL: 盤上と両者の持駒に無い残り全駒を `color` の持駒に加える．
fn distribute_remaining(squares: &[Piece; 81], hand: &mut [[u8; HAND_KINDS]; 2], color: Color) {
    let mut used = [0u8; HAND_KINDS];
    for piece in squares.iter() {
        if let Some(pt) = piece.piece_type() {
            if let Some(hi) = pt.captured_to_hand().hand_index() {
                used[hi] += 1;
            }
        }
    }
    for hi in 0..HAND_KINDS {
        let total = PieceType::MAX_HAND_COUNT[hi];
        let taken = used[hi] + hand[0][hi] + hand[1][hi];
        if total > taken {
            hand[color.index()][hi] += total - taken;
        }
    }
}

/// CSA 指し手 6 文字 ("7776FU" 等) を盤面文脈で解決する．
///
/// 戻り値は (cshogi 互換 32-bit int, 盤面更新用の [`Move`])．
fn parse_csa_move(board: &Board, s: &str, line_no: usize) -> Result<(u32, Move), KifuParseError> {
    let err = |msg: String| KifuParseError::new(line_no, msg);
    let bytes = s.as_bytes();
    if bytes.len() < 6 {
        return Err(err(format!("指し手が短すぎます: {s}")));
    }
    let from_file = bytes[0].wrapping_sub(b'0');
    let from_rank = bytes[1].wrapping_sub(b'0');
    let to_file = bytes[2].wrapping_sub(b'0');
    let to_rank = bytes[3].wrapping_sub(b'0');
    let code = &s[4..6];
    let pt_code =
        piece_type_from_csa(code).ok_or_else(|| err(format!("不正な駒コード: {code}")))?;
    if !(1..=9).contains(&to_file) || !(1..=9).contains(&to_rank) {
        return Err(err(format!("不正な移動先: {s}")));
    }
    let to = Square::new(to_file - 1, to_rank - 1);
    let turn = board.turn();

    if from_file == 0 && from_rank == 0 {
        // 駒打ち
        let hi = pt_code
            .hand_index()
            .filter(|_| pt_code.drop_move_index().is_some())
            .ok_or_else(|| err(format!("打てない駒です: {s}")))?;
        let (black_hand, white_hand) = board.pieces_in_hand();
        let count = match turn {
            Color::Black => black_hand[hi],
            Color::White => white_hand[hi],
        };
        if count == 0 {
            return Err(err(format!("持駒がありません: {s}")));
        }
        if !Piece::from_raw_u8(board.piece_at(to)).is_empty() {
            return Err(err(format!("移動先に駒があります: {s}")));
        }
        let encoded =
            encode_drop(to, pt_code).ok_or_else(|| err(format!("打てない駒です: {s}")))?;
        Ok((encoded, Move::new_drop(to, pt_code)))
    } else {
        if !(1..=9).contains(&from_file) || !(1..=9).contains(&from_rank) {
            return Err(err(format!("不正な移動元: {s}")));
        }
        let from = Square::new(from_file - 1, from_rank - 1);
        let piece = Piece::from_raw_u8(board.piece_at(from));
        let pt_from = piece
            .piece_type()
            .ok_or_else(|| err(format!("移動元に駒がありません: {s}")))?;
        if piece.color() != Some(turn) {
            return Err(err(format!("移動元が手番の駒ではありません: {s}")));
        }
        let promote = if pt_from == pt_code {
            false
        } else if pt_from.promoted() == Some(pt_code) {
            true
        } else {
            return Err(err(format!("駒コードが移動元の駒と一致しません: {s}")));
        };
        let cap_piece = Piece::from_raw_u8(board.piece_at(to));
        let captured_pt = match cap_piece.piece_type() {
            Some(pt) => {
                if cap_piece.color() == Some(turn) {
                    return Err(err(format!("自駒の枡には動けません: {s}")));
                }
                Some(pt)
            }
            None => None,
        };
        let encoded = encode_move(from, to, promote, pt_from, captured_pt);
        let mv = Move::new_move(from, to, promote, cap_piece.raw_u8(), pt_from as u8);
        Ok((encoded, mv))
    }
}

/// CSA 棋譜 (単一対局) をパースする．
///
/// 複数対局を含む文字列は [`parse_csa_multi`] を使うこと ("/" 行は
/// cshogi 互換でエラーになる)．
pub fn parse_csa_str(content: &str) -> Result<GameRecord, KifuParseError> {
    let mut record = GameRecord {
        names: [Some(String::new()), Some(String::new())],
        win: Some(WIN_DRAW),
        ..GameRecord::default()
    };

    let mut board = Board::new();
    let mut pos_initialized = false;
    let mut position_lines: Vec<String> = Vec::new();
    let mut current_turn_seen = false;
    // 敗者 (未確定は None)
    let mut lose_color: Option<Color> = None;

    for (i, raw_line) in content.split('\n').enumerate() {
        let line_no = i + 1;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        let err = |msg: String| KifuParseError::new(line_no, msg);

        if line.is_empty() {
            // skip
        } else if line.starts_with('\'') {
            if let Some(rest) = line.strip_prefix("'black_rate:") {
                if let Some((_, v)) = rest.split_once(':') {
                    if let Some(f) = parse_leading_f32(v) {
                        record.ratings[0] = f;
                    }
                }
            } else if let Some(rest) = line.strip_prefix("'white_rate:") {
                if let Some((_, v)) = rest.split_once(':') {
                    if let Some(f) = parse_leading_f32(v) {
                        record.ratings[1] = f;
                    }
                }
            } else if let Some(rest) = line.strip_prefix("'** ") {
                // 評価値コメント: 直前の指し手に付与 (cshogi 互換で上書き)
                if let Some(last) = record.moves.len().checked_sub(1) {
                    let score_str = rest.split_once(' ').map(|(s, _)| s).unwrap_or(rest);
                    if let Some(score) = parse_leading_i32(score_str) {
                        record.scores[last] = score;
                    }
                    record.comments[last] = rest.to_string();
                } else {
                    record.header_comment.push_str(line);
                    record.header_comment.push('\n');
                }
            } else {
                record.header_comment.push_str(line);
                record.header_comment.push('\n');
                // floodgate: 'summary:<reason>:...
                if let Some(rest) = line.strip_prefix("'summary:") {
                    if let Some((reason, _)) = rest.split_once(':') {
                        match reason {
                            "illegal move" => {
                                record.endgame = Some("%ILLEGAL_MOVE".to_string());
                                lose_color = Some(board.turn());
                            }
                            "max_moves" => {
                                record.endgame = Some("%JISHOGI".to_string());
                            }
                            "abnormal" => {
                                record.endgame = Some("%ERROR".to_string());
                            }
                            _ => {}
                        }
                    }
                }
            }
        } else if line.starts_with('V') {
            record.version = line.to_string();
        } else if let Some(rest) = line.strip_prefix('N') {
            match rest.as_bytes().first() {
                Some(b'+') => record.names[0] = Some(rest[1..].to_string()),
                Some(b'-') => record.names[1] = Some(rest[1..].to_string()),
                _ => {}
            }
        } else if let Some(rest) = line.strip_prefix('$') {
            let (k, v) = rest.split_once(':').unwrap_or((rest, ""));
            record.var_info.push((k.to_string(), v.to_string()));
        } else if line.starts_with('P') {
            position_lines.push(line.to_string());
        } else if line.starts_with('+') || line.starts_with('-') {
            if line.len() == 1 {
                current_turn_seen = true;
                position_lines.push(line.to_string());
            } else {
                if !pos_initialized {
                    return Err(err("局面定義より前に指し手があります".to_string()));
                }
                if line.len() < 7 {
                    return Err(err(format!("不正な指し手行: {line}")));
                }
                let (encoded, mv) = parse_csa_move(&board, &line[1..7], line_no)?;
                record.moves.push(encoded);
                board.do_move(mv);
                record.times.push(0);
                record.scores.push(0);
                record.comments.push(String::new());
                // カンマ区切りの後続文 (V2.2)．cshogi 互換の ,'コメント に
                // 加え，,T<秒> も解釈する (それ以外は無視)
                let mut rest = &line[7..];
                loop {
                    if let Some(comment) = rest.strip_prefix(",'") {
                        let last = record.comments.len() - 1;
                        record.comments[last] = comment.to_string();
                        break;
                    } else if let Some(t) = rest.strip_prefix(",T") {
                        if let Some(sec) = parse_leading_i32(t) {
                            let last = record.times.len() - 1;
                            record.times[last] = sec;
                        }
                        rest = t.find(',').map(|p| &t[p..]).unwrap_or("");
                    } else {
                        break;
                    }
                }
            }
        } else if let Some(rest) = line.strip_prefix('T') {
            let sec =
                parse_leading_i32(rest).ok_or_else(|| err(format!("不正な消費時間行: {line}")))?;
            if record.endgame.is_none() {
                if let Some(last) = record.times.len().checked_sub(1) {
                    record.times[last] = sec;
                }
            } else {
                record.times.push(sec);
            }
        } else if line.starts_with('%') {
            if !pos_initialized {
                return Err(err("局面定義より前に特殊手があります".to_string()));
            }
            // カンマ以降は無視 (cshogi 互換)
            let stmt = line.split_once(',').map(|(s, _)| s).unwrap_or(line);
            match stmt {
                "%TORYO" | "%TIME_UP" | "%ILLEGAL_MOVE" => {
                    lose_color = Some(board.turn());
                }
                "%+ILLEGAL_ACTION" => lose_color = Some(Color::Black),
                "%-ILLEGAL_ACTION" => lose_color = Some(Color::White),
                "%KACHI" => lose_color = Some(board.turn().opponent()),
                _ => {}
            }
            record.endgame = Some(stmt.to_string());
        } else if line == "/" {
            return Err(err(
                "複数対局はサポートしていません (parse_csa_multi を使用)".to_string(),
            ));
        } else {
            return Err(err(format!("不正な行: {line}")));
        }

        if !pos_initialized && current_turn_seen {
            pos_initialized = true;
            record.sfen = parse_position(&position_lines, line_no)?;
            board.set_sfen(&record.sfen).map_err(|e| {
                KifuParseError::new(line_no, format!("初期局面の SFEN 変換に失敗: {e}"))
            })?;
        }
    }

    record.win = Some(match lose_color {
        Some(Color::Black) => WIN_WHITE,
        Some(Color::White) => WIN_BLACK,
        None => WIN_DRAW,
    });

    Ok(record)
}

/// CSA 棋譜 (複数対局可) をパースする．
///
/// cshogi (Python 層) 互換で `"\n/\n"` を区切りとして分割する．
pub fn parse_csa_multi(content: &str) -> Result<Vec<GameRecord>, KifuParseError> {
    content.split("\n/\n").map(parse_csa_str).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hirate_pi() {
        let rec = parse_csa_str("V2.2\nPI\n+\n%CHUDAN\n").unwrap();
        assert_eq!(rec.sfen, sfen::HIRATE_SFEN);
        assert_eq!(rec.endgame.as_deref(), Some("%CHUDAN"));
        assert_eq!(rec.win, Some(WIN_DRAW));
        assert_eq!(rec.version, "V2.2");
    }

    #[test]
    fn test_p_plus_individual_placement() {
        // cshogi は P+/P- で segfault するため spec 準拠の期待値でテスト．
        // 5一玉(後手)・5九玉(先手)・2八飛(先手)・先手持駒 金1
        let content = "V2.2\n\
                       P-51OU\n\
                       P+59OU28HI\n\
                       P+00KI\n\
                       +\n\
                       %CHUDAN\n";
        let rec = parse_csa_str(content).unwrap();
        assert_eq!(rec.sfen, "4k4/9/9/9/9/9/9/7R1/4K4 b G 1");
    }

    #[test]
    fn test_p_minus_al_remaining() {
        // AL: 盤上 (玉 2・飛 1) と先手持駒 (金 1) を除く全てが後手持駒
        let content = "V2.2\n\
                       P-51OU\n\
                       P+59OU28HI\n\
                       P+00KI\n\
                       P-00AL\n\
                       +\n\
                       %CHUDAN\n";
        let rec = parse_csa_str(content).unwrap();
        // 後手持駒: 飛1 角2 金3 銀4 桂4 香4 歩18
        assert_eq!(rec.sfen, "4k4/9/9/9/9/9/9/7R1/4K4 b Gr2b3g4s4n4l18p 1");
    }

    #[test]
    fn test_comma_statement_extension() {
        // V2.2 のカンマ区切り: ,T は cshogi 非対応 (無視される) だが
        // 本実装は解釈する
        let rec = parse_csa_str("V2.2\nPI\n+\n+7776FU,T12\n%TORYO\n").unwrap();
        assert_eq!(rec.times, vec![12]);
        assert_eq!(rec.moves, vec![0x00011e3b]);
        // 1 手後は後手番 → 後手負け → 先手勝ち
        assert_eq!(rec.win, Some(WIN_BLACK));
    }

    #[test]
    fn test_inline_comment_with_commas() {
        // ,'コメント はカンマを含めて行末まで (cshogi 互換)
        let rec = parse_csa_str("V2.2\nPI\n+\n+7776FU,'foo,bar\n%CHUDAN\n").unwrap();
        assert_eq!(rec.comments, vec!["foo,bar".to_string()]);
    }

    #[test]
    fn test_invalid_line_is_error() {
        let e = parse_csa_str("V2.2\nPI\n+\nGARBAGE\n").unwrap_err();
        assert_eq!(e.line_no, 4);
    }

    #[test]
    fn test_move_before_position_is_error() {
        assert!(parse_csa_str("V2.2\n+7776FU\n").is_err());
    }

    #[test]
    fn test_illegal_csa_move_is_error() {
        // 移動元に駒が無い
        assert!(parse_csa_str("V2.2\nPI\n+\n+5555FU\n").is_err());
        // 駒コード不一致
        assert!(parse_csa_str("V2.2\nPI\n+\n+7776KI\n").is_err());
        // 持駒なしで打つ
        assert!(parse_csa_str("V2.2\nPI\n+\n+0055KA\n").is_err());
    }
}
