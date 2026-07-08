//! NN 入力特徴量エンコード (board ID 9×9 + 持ち駒 14)．
//!
//! Python 実装 `src/maou/app/pre_process/feature.py` の Rust 移植．
//! 現行 NN の入力は embedding ベースの駒 ID 盤面 `board` (9,9) int32 と
//! 持ち駒枚数 `hand` (14,) float32 (生の枚数，正規化なし)．
//!
//! # 正規化規則 (Python と同一)
//!
//! - 手番視点に正規化する: 後手番なら盤面を 180 度回転し，駒 ID の先手/後手を
//!   入れ替える．正規化後は手番側の駒が常に 1-14，相手側が 15-28
//! - `board` のメモリ配置は row-major `[段][筋]` (段 0 = 一段，筋 0 = 1筋)．
//!   maou_shogi の square (col-major: `sq = 筋*9 + 段`) から転置して書き込む
//! - `hand` は手番側 7 種 (歩香桂銀金角飛) が先頭，相手側 7 種が後続
//!
//! 移植の正しさは Python 正実装から生成した golden fixture
//! (`tests/fixtures/positions_golden.txt`) との一致で検証している．

use maou_shogi::board::Board;
use maou_shogi::types::Color;

/// board 特徴量の要素数 (9×9)．
pub const BOARD_FEATURE_LEN: usize = 81;
/// hand 特徴量の要素数 (手番側 7 + 相手側 7)．
pub const HAND_FEATURE_LEN: usize = 14;

/// cshogi 互換駒 ID (0-30) → PieceId (0-28) 変換表．
/// PieceId は 金(5) が 角(6)・飛(7) より前に来る点が cshogi と異なる．
const PIECE_ID_TABLE: [u8; 31] = [
    0, // EMPTY
    1, 2, 3, 4, // BPAWN..BSILVER → FU..GI
    6, 7, // BBISHOP → KA, BROOK → HI
    5, // BGOLD → KI
    8, // BKING → OU
    9, 10, 11, 12, 13, 14, // BPROM_* → TO..RYU
    0, 0, // 15, 16: cshogi 未使用ギャップ
    15, 16, 17, 18, // WPAWN..WSILVER
    20, 21, // WBISHOP → KA+14, WROOK → HI+14
    19, // WGOLD → KI+14
    22, // WKING
    23, 24, 25, 26, 27, 28, // WPROM_*
];

/// 先手 (1-14) と後手 (15-28) の駒 ID を入れ替える．EMPTY (0) は不変．
#[inline]
fn swap_piece_id(id: u8) -> u8 {
    match id {
        1..=14 => id + 14,
        15..=28 => id - 14,
        _ => id,
    }
}

/// 盤面を手番視点の駒 ID 配列 (row-major `[段][筋]`) にエンコードする．
///
/// NN の `board` 入力 (int32) に合わせて i32 で書き込む．
pub fn encode_board(board: &Board, out: &mut [i32; BOARD_FEATURE_LEN]) {
    let pieces = board.pieces();
    match board.turn() {
        Color::Black => {
            for (sq, &p) in pieces.iter().enumerate() {
                let id = PIECE_ID_TABLE[p as usize];
                let (col, row) = (sq / 9, sq % 9);
                out[row * 9 + col] = i32::from(id);
            }
        }
        Color::White => {
            for (sq, &p) in pieces.iter().enumerate() {
                let id = swap_piece_id(PIECE_ID_TABLE[p as usize]);
                let (col, row) = (sq / 9, sq % 9);
                out[(8 - row) * 9 + (8 - col)] = i32::from(id);
            }
        }
    }
}

/// 持ち駒を手番側先頭の枚数配列にエンコードする．
///
/// NN の `hand` 入力 (float32) に合わせて f32 (生の枚数) で書き込む．
pub fn encode_hand(board: &Board, out: &mut [f32; HAND_FEATURE_LEN]) {
    let (black, white) = board.pieces_in_hand();
    let (own, opp) = match board.turn() {
        Color::Black => (black, white),
        Color::White => (white, black),
    };
    for (o, &c) in out[..7].iter_mut().zip(own.iter()) {
        *o = f32::from(c);
    }
    for (o, &c) in out[7..].iter_mut().zip(opp.iter()) {
        *o = f32::from(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_piece_id_roundtrip() {
        for id in 0..=28u8 {
            let swapped = swap_piece_id(id);
            assert_eq!(swap_piece_id(swapped), id);
            if id == 0 {
                assert_eq!(swapped, 0);
            } else {
                assert_ne!(swapped, id);
            }
        }
    }

    #[test]
    fn test_encode_startpos_black_perspective() {
        let mut board = Board::empty();
        board
            .set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .expect("正当な SFEN");
        let mut out = [0i32; BOARD_FEATURE_LEN];
        encode_board(&board, &mut out);
        // 一段目 (row 0) は後手の香桂銀金王金銀桂香 (9筋→1筋の SFEN 表記を
        // [段][筋] で読むと筋 0 = 1筋が末尾の l)
        assert_eq!(&out[0..9], &[16, 17, 18, 19, 22, 19, 18, 17, 16]);
        // 七段目 (row 6) は先手の歩が 9 枚
        assert_eq!(&out[54..63], &[1; 9]);
        let mut hand = [0f32; HAND_FEATURE_LEN];
        encode_hand(&board, &mut hand);
        assert_eq!(hand, [0f32; HAND_FEATURE_LEN]);
    }
}
