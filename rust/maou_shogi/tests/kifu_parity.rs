//! CSA/KIF パーサの cshogi (oracle) parity テスト．
//!
//! fixtures/kifu/ の golden は cshogi 0.9.7 の CSA.Parser / KIF.Parser の
//! 出力から scratchpad/gen_kifu_golden.py で生成した (形式はスクリプトの
//! docstring 参照)．本テストは maou_shogi::kifu の出力が oracle と一致する
//! ことを検証する (CLAUDE.md TRIPWIRE: cshogi 由来実装の置換前 parity)．

use maou_shogi::kifu::{parse_csa_multi, parse_kif_str, GameRecord};

/// golden ファイル 1 レコード分の期待値．
#[derive(Debug, Default)]
struct Golden {
    sfen: String,
    version: Option<String>,
    endgame: Option<String>,
    win: Option<u8>,
    nameb: Option<String>,
    namew: Option<String>,
    ratingb: Option<f32>,
    ratingw: Option<f32>,
    var: Vec<(String, String)>,
    moves: Vec<u32>,
    times: Vec<i32>,
    scores: Vec<i32>,
    has_scores: bool,
    comments: Vec<(usize, String)>,
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_goldens(text: &str) -> Vec<Golden> {
    let mut records = Vec::new();
    let mut cur: Option<Golden> = None;
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if line == "record" {
            cur = Some(Golden::default());
            continue;
        }
        if line == "end" {
            records.push(cur.take().expect("end without record"));
            continue;
        }
        let g = cur.as_mut().expect("field outside record");
        let (tag, rest) = line.split_once('\t').unwrap_or((line, ""));
        match tag {
            "sfen" => g.sfen = rest.to_string(),
            "version" => g.version = Some(unescape(rest)),
            "endgame" => g.endgame = Some(rest.to_string()),
            "win" => g.win = Some(rest.parse().unwrap()),
            "nameb" => g.nameb = Some(unescape(rest)),
            "namew" => g.namew = Some(unescape(rest)),
            "ratingb" => g.ratingb = Some(rest.parse().unwrap()),
            "ratingw" => g.ratingw = Some(rest.parse().unwrap()),
            "var" => {
                let (k, v) = rest.split_once('\t').expect("var needs k/v");
                g.var.push((unescape(k), unescape(v)));
            }
            "m" => g.moves.push(rest.parse().unwrap()),
            "t" => g.times.push(rest.parse().unwrap()),
            "s" => {
                g.has_scores = true;
                g.scores.push(rest.parse().unwrap());
            }
            "c" => {
                let (idx, text) = rest.split_once('\t').expect("c needs idx");
                g.comments.push((idx.parse().unwrap(), unescape(text)));
            }
            other => panic!("unknown golden tag: {other}"),
        }
    }
    records
}

/// GameRecord と golden 1 レコードの完全一致を検証する．
fn assert_record_matches(name: &str, rec: &GameRecord, g: &Golden) {
    assert_eq!(rec.sfen, g.sfen, "{name}: sfen mismatch");
    if let Some(v) = &g.version {
        assert_eq!(&rec.version, v, "{name}: version mismatch");
    } else {
        assert!(rec.version.is_empty(), "{name}: unexpected version");
    }
    assert_eq!(
        rec.endgame.as_deref(),
        g.endgame.as_deref(),
        "{name}: endgame mismatch"
    );
    assert_eq!(rec.win, g.win, "{name}: win mismatch");
    assert_eq!(rec.names[0], g.nameb, "{name}: black name mismatch");
    assert_eq!(rec.names[1], g.namew, "{name}: white name mismatch");
    if let Some(r) = g.ratingb {
        assert_eq!(rec.ratings[0], r, "{name}: black rating mismatch");
    }
    if let Some(r) = g.ratingw {
        assert_eq!(rec.ratings[1], r, "{name}: white rating mismatch");
    }
    assert_eq!(rec.var_info, g.var, "{name}: var_info mismatch");
    assert_eq!(rec.moves, g.moves, "{name}: moves mismatch");
    assert_eq!(rec.times, g.times, "{name}: times mismatch");
    if g.has_scores {
        assert_eq!(rec.scores, g.scores, "{name}: scores mismatch");
    } else {
        // KIF: cshogi に scores は無い．本実装は moves と同長の 0 列
        assert_eq!(
            rec.scores,
            vec![0; rec.moves.len()],
            "{name}: KIF scores must be zero-filled"
        );
    }
    // comments: golden の (idx, text) が一致し，それ以外は空であること
    assert_eq!(
        rec.comments.len(),
        rec.moves.len(),
        "{name}: comments must align with moves"
    );
    let mut expected = vec![String::new(); rec.moves.len()];
    for (idx, text) in &g.comments {
        expected[*idx] = text.clone();
    }
    assert_eq!(rec.comments, expected, "{name}: comments mismatch");
}

macro_rules! fixture {
    ($name:literal, $ext:literal) => {
        (
            $name,
            include_str!(concat!("fixtures/kifu/", $name, $ext)),
            include_str!(concat!("fixtures/kifu/", $name, ".golden.txt")),
        )
    };
}

#[test]
fn test_csa_parity() {
    let cases = [
        fixture!("test_data_1", ".csa"),
        fixture!("test_data_2", ".csa"),
        fixture!("test_data_3", ".csa"),
        fixture!("test_data_no_moves", ".csa"),
        fixture!("csa_handicap_pi", ".csa"),
        fixture!("csa_inline_comment", ".csa"),
        fixture!("csa_illegal_move", ".csa"),
        fixture!("csa_rows_kachi", ".csa"),
        fixture!("csa_multi_game", ".csa"),
    ];
    for (name, content, golden_text) in cases {
        let goldens = parse_goldens(golden_text);
        let records =
            parse_csa_multi(content).unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
        assert_eq!(
            records.len(),
            goldens.len(),
            "{name}: record count mismatch"
        );
        for (i, (rec, g)) in records.iter().zip(goldens.iter()).enumerate() {
            assert_record_matches(&format!("{name}[{i}]"), rec, g);
        }
    }
}

#[test]
fn test_kif_parity() {
    let cases = [
        fixture!("kif_gen_test_data_1", ".kifu"),
        fixture!("kif_gen_test_data_2", ".kifu"),
        fixture!("kif_gen_test_data_3", ".kifu"),
        fixture!("kif_edge_basic", ".kifu"),
        fixture!("kif_edge_normalize", ".kifu"),
        fixture!("kif_edge_handicap", ".kifu"),
        fixture!("kif_edge_sennichite", ".kifu"),
        fixture!("kif_edge_jishogi", ".kifu"),
        fixture!("kif_edge_illegal_win_w", ".kifu"),
        fixture!("kif_edge_illegal_lose_b", ".kifu"),
        fixture!("kif_edge_nyugyoku", ".kifu"),
        fixture!("kif_edge_no_result", ".kifu"),
    ];
    for (name, content, golden_text) in cases {
        let goldens = parse_goldens(golden_text);
        assert_eq!(goldens.len(), 1, "{name}: KIF golden must be single");
        let record = parse_kif_str(content).unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
        assert_record_matches(name, &record, &goldens[0]);
    }
}

/// CSA 棋譜の指し手を実際に盤面に適用して最終局面まで進められること
/// (エンコーディングと盤面遷移の整合性) を検証する．
#[test]
fn test_csa_moves_replayable_on_board() {
    use maou_shogi::board::Board;

    let content = include_str!("fixtures/kifu/test_data_1.csa");
    let record = &parse_csa_multi(content).unwrap()[0];
    let mut board = Board::new();
    board.set_sfen(&record.sfen).unwrap();
    for (i, &m) in record.moves.iter().enumerate() {
        // cshogi 互換 int の下位 16bit から手を復元して適用
        let mv = board
            .move_from_move16((m & 0xffff) as u16)
            .unwrap_or_else(|| panic!("move {i} ({m:#x}) not resolvable"));
        board.do_move(mv);
    }
}
