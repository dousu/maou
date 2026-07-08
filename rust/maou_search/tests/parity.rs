//! Python 正実装との parity テスト．
//!
//! fixture は `scratchpad` の生成スクリプト (worklog 参照) で
//! Python 実装 (`feature.py` / `label.py`) から生成した golden data．
//! - `label_table_golden.txt`: 全 (from, to, promo) 盤上手 + 全駒打ちのラベル網羅表
//! - `positions_golden.txt`: 実局面での盤面/持ち駒エンコードと合法手ラベル

use std::collections::HashMap;

use maou_search::feature::{encode_board, encode_hand, BOARD_FEATURE_LEN, HAND_FEATURE_LEN};
use maou_search::label::{move_label, try_move_label, MOVE_LABELS_NUM};
use maou_shogi::board::Board;
use maou_shogi::movegen::generate_legal_moves;
use maou_shogi::moves::Move;
use maou_shogi::types::Color;

const LABEL_TABLE: &str = include_str!("fixtures/label_table_golden.txt");
const POSITIONS: &str = include_str!("fixtures/positions_golden.txt");

/// 盤上手・駒打ちの全域でラベルが Python 実装と一致すること．
///
/// fixture は 16-bit cshogi 互換エンコード (to | from << 7 | promo << 14，
/// 駒打ちは from = 81 + drop_move_index) の生の move 値で Python 側と共有する．
#[test]
fn test_label_table_exhaustive_parity() {
    let mut checked = 0usize;
    for line in LABEL_TABLE.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("labels_num") => {
                let n: usize = it.next().unwrap().parse().unwrap();
                assert_eq!(MOVE_LABELS_NUM, n, "ラベル総数が Python と一致すること");
            }
            Some("m") => {
                let from: u32 = it.next().unwrap().parse().unwrap();
                let promo: u32 = it.next().unwrap().parse().unwrap();
                for (to, tok) in it.enumerate() {
                    let expected: i32 = tok.parse().unwrap();
                    let raw = to as u32 | (from << 7) | (promo << 14);
                    let got = try_move_label(Color::Black, Move::from_raw_u32(raw));
                    assert_eq!(
                        got.map(i32::from).unwrap_or(-1),
                        expected,
                        "from={from} to={to} promo={promo}"
                    );
                    checked += 1;
                }
            }
            Some("d") => {
                let dmi: u32 = it.next().unwrap().parse().unwrap();
                for (to, tok) in it.enumerate() {
                    let expected: i32 = tok.parse().unwrap();
                    let raw = to as u32 | ((81 + dmi) << 7);
                    let got = try_move_label(Color::Black, Move::from_raw_u32(raw));
                    assert_eq!(
                        got.map(i32::from).unwrap_or(-1),
                        expected,
                        "drop dmi={dmi} to={to}"
                    );
                    checked += 1;
                }
            }
            _ => panic!("不明な fixture 行: {line}"),
        }
    }
    // 81*81*2 (盤上手) + 7*81 (駒打ち)
    assert_eq!(checked, 81 * 81 * 2 + 7 * 81);
}

/// 実局面での特徴量エンコードと合法手ラベルが Python 実装と一致すること．
/// 後手番局面 (回転 + ID swap + 座標正規化) を含む．
#[test]
fn test_positions_golden_parity() {
    let mut lines = POSITIONS.lines().peekable();
    let mut positions = 0usize;
    while let Some(line) = lines.next() {
        let sfen = line.strip_prefix("sfen ").expect("sfen 行");
        let board_line = lines.next().unwrap().strip_prefix("board ").unwrap();
        let hand_line = lines.next().unwrap().strip_prefix("hand ").unwrap();
        let moves_line = lines.next().unwrap().strip_prefix("moves ").unwrap();
        let n_moves: usize = moves_line.parse().unwrap();

        let expected_board: Vec<i32> = board_line
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let expected_hand: Vec<f32> = hand_line
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let mut expected_labels: HashMap<String, u16> = HashMap::new();
        for _ in 0..n_moves {
            let mv_line = lines.next().unwrap();
            let (usi, label) = mv_line.split_once(' ').unwrap();
            expected_labels.insert(usi.to_string(), label.parse().unwrap());
        }
        assert_eq!(lines.next(), Some("end"));

        let mut board = Board::empty();
        board.set_sfen(sfen).expect("golden の SFEN は正当");

        let mut got_board = [0i32; BOARD_FEATURE_LEN];
        encode_board(&board, &mut got_board);
        assert_eq!(
            got_board.as_slice(),
            expected_board.as_slice(),
            "board: {sfen}"
        );

        let mut got_hand = [0f32; HAND_FEATURE_LEN];
        encode_hand(&board, &mut got_hand);
        assert_eq!(
            got_hand.as_slice(),
            expected_hand.as_slice(),
            "hand: {sfen}"
        );

        let legal = generate_legal_moves(&mut board);
        assert_eq!(legal.len(), n_moves, "合法手数: {sfen}");
        let turn = board.turn();
        for m in legal {
            let usi = m.to_usi();
            let expected = expected_labels
                .get(&usi)
                .unwrap_or_else(|| panic!("Python 側に無い合法手 {usi}: {sfen}"));
            assert_eq!(move_label(turn, m), *expected, "label {usi}: {sfen}");
        }
        positions += 1;
    }
    assert!(positions >= 10, "fixture に十分な局面数があること");
}
