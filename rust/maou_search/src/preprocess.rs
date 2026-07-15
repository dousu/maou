//! HCPE 前処理の一括エンコード (hash / moveLabel / resultValue / 特徴量 / 合法手マスク)．
//!
//! Python 実装の per-position ループの hot path を一括移植したもの:
//!
//! - `src/maou/app/pre_process/hcpe_transform.py` の `_process_single_array`
//!   (N 局面の set_hcp → hash / move label / result value)
//! - `src/maou/app/utility/stage2_data_generation.py` のチャンクループ
//!   (unique 局面の特徴量 + 合法手ラベルマスク)
//!
//! Python 側の集計 (numpy argsort/unique/bincount，DuckDB) は対象外で，
//! 本モジュールは局面単位の変換だけを担う．移植の正しさは Python 正実装から
//! 生成した golden fixture (tests/maou/app/{pre_process,utility}/resources/golden/)
//! との bit-exact 一致で検証している．

use std::fmt;

use maou_shogi::board::Board;
use maou_shogi::hcp::{self, HCP_SIZE};
use maou_shogi::movegen::generate_legal_moves;
use maou_shogi::moves::Move;
use maou_shogi::types::Color;

use crate::feature::{encode_board_ids, encode_hand_counts, BOARD_FEATURE_LEN, HAND_FEATURE_LEN};
use crate::label::{try_move_label, MOVE_LABELS_NUM};

/// 一括前処理のエラー．
#[derive(Debug)]
pub enum PreprocessError {
    /// 入力バッファ長が HCP_SIZE の倍数でない / 配列長が不一致．
    InvalidInput(String),
    /// HCP デコード失敗 (index 番目の局面)．
    InvalidHcp { index: usize, message: String },
    /// move label に変換できない指し手 (index 番目の局面)．
    IllegalMove { index: usize, move16: i16 },
}

impl fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PreprocessError::InvalidInput(msg) => {
                write!(f, "invalid input: {msg}")
            }
            PreprocessError::InvalidHcp { index, message } => {
                write!(f, "invalid HCP at index {index}: {message}")
            }
            PreprocessError::IllegalMove { index, move16 } => {
                write!(
                    f,
                    "Can not transform illegal move to move label. \
                     (index {index}, move16 {move16})"
                )
            }
        }
    }
}

impl std::error::Error for PreprocessError {}

/// gameResult (HCPE 規約: 0=引き分け, 1=先手勝ち, 2=後手勝ち) から
/// 手番側の教師値 (1=手番側勝ち, 0=負け, 0.5=引き分け) を計算する．
///
/// 旧実装は Python `make_result_value` の gameResult 規約取り違え
/// (`shogi.Result` 旧定義 BLACK_WIN=0/WHITE_WIN=1/DRAW=2 で解釈) を
/// bit-exact 移植していた．規約修正に伴い Python 側 (`shogi.Result` の
/// 値を HCPE 規約へ変更) と同一コミットで正しい対応に更新済み．
#[inline]
pub fn result_value(turn: Color, game_result: i8) -> f32 {
    match (turn, game_result) {
        (Color::Black, 1) | (Color::White, 2) => 1.0,
        (Color::Black, 2) | (Color::White, 1) => 0.0,
        _ => 0.5,
    }
}

/// `process_hcpes` の結果列 (zobrist hash, move label, result value)．
pub type ProcessedColumns = (Vec<u64>, Vec<u16>, Vec<f32>);

/// フラットな HCP バッファ (N×32 bytes) を局面数と検証付きで分割する．
fn split_hcps(hcps: &[u8]) -> Result<usize, PreprocessError> {
    if !hcps.len().is_multiple_of(HCP_SIZE) {
        return Err(PreprocessError::InvalidInput(format!(
            "HCP buffer length {} is not a multiple of {HCP_SIZE}",
            hcps.len()
        )));
    }
    Ok(hcps.len() / HCP_SIZE)
}

/// index 番目の HCP から盤面を復元する．
fn board_at(hcps: &[u8], index: usize) -> Result<Board, PreprocessError> {
    let mut buf = [0u8; HCP_SIZE];
    buf.copy_from_slice(&hcps[index * HCP_SIZE..(index + 1) * HCP_SIZE]);
    hcp::from_hcp(&buf).map_err(|e| PreprocessError::InvalidHcp {
        index,
        message: e.to_string(),
    })
}

/// N 局面の (zobrist hash, move label, result value) を一括計算する．
///
/// `hcps` はフラットな N×32 バイト，`move16s`/`game_results` は N 要素．
/// Python `_process_single_array` の統合ループ相当 (集計は含まない)．
pub fn process_hcpes(
    hcps: &[u8],
    move16s: &[i16],
    game_results: &[i8],
) -> Result<ProcessedColumns, PreprocessError> {
    let n = split_hcps(hcps)?;
    if move16s.len() != n || game_results.len() != n {
        return Err(PreprocessError::InvalidInput(format!(
            "length mismatch: hcp {n}, move16 {}, game_result {}",
            move16s.len(),
            game_results.len()
        )));
    }
    let mut hashes = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let board = board_at(hcps, i)?;
        let turn = board.turn();
        let m = Move::from_raw_u32(move16s[i] as u16 as u32);
        let label = try_move_label(turn, m).ok_or(PreprocessError::IllegalMove {
            index: i,
            move16: move16s[i],
        })?;
        hashes.push(board.hash());
        labels.push(label);
        results.push(result_value(turn, game_results[i]));
    }
    Ok((hashes, labels, results))
}

/// N 局面の zobrist hash を一括計算する．
///
/// stage2 データ生成の重複排除 (unique HCP 収集) 用．
pub fn hcp_hashes(hcps: &[u8]) -> Result<Vec<u64>, PreprocessError> {
    let n = split_hcps(hcps)?;
    let mut hashes = Vec::with_capacity(n);
    for i in 0..n {
        hashes.push(board_at(hcps, i)?.hash());
    }
    Ok(hashes)
}

/// N 局面の NN 入力特徴量 (boardIdPositions N×81 u8, piecesInHand N×14 u8) を
/// 一括エンコードする．
///
/// Python `Board.get_normalized_board_id_positions` /
/// `get_normalized_pieces_in_hand` 相当
/// (手番視点正規化込み)．
pub fn encode_hcp_features(hcps: &[u8]) -> Result<(Vec<u8>, Vec<u8>), PreprocessError> {
    let n = split_hcps(hcps)?;
    let mut boards = vec![0u8; n * BOARD_FEATURE_LEN];
    let mut hands = vec![0u8; n * HAND_FEATURE_LEN];
    for i in 0..n {
        let board = board_at(hcps, i)?;
        let out: &mut [u8; BOARD_FEATURE_LEN] = (&mut boards
            [i * BOARD_FEATURE_LEN..(i + 1) * BOARD_FEATURE_LEN])
            .try_into()
            .expect("chunk length is BOARD_FEATURE_LEN");
        encode_board_ids(&board, out);
        let out: &mut [u8; HAND_FEATURE_LEN] = (&mut hands
            [i * HAND_FEATURE_LEN..(i + 1) * HAND_FEATURE_LEN])
            .try_into()
            .expect("chunk length is HAND_FEATURE_LEN");
        encode_hand_counts(&board, out);
    }
    Ok((boards, hands))
}

/// N 局面の合法手ラベルマスク (N×1496 u8, 1=合法) を一括生成する．
///
/// ラベルは手番視点に正規化する．Python 正実装 (stage2_data_generation.py) は
/// 後手番のとき正規化済み盤面を SFEN 経由で再構築してから合法手を列挙するが，
/// 「回転盤面の合法手を先手視点でラベル化」と「元盤面の合法手を後手視点で
/// ラベル化」は回転対称性により同値なので，再構築せず後者で実装している
/// (golden fixture の後手番局面で同値性を検証済み)．
pub fn legal_move_masks(hcps: &[u8]) -> Result<Vec<u8>, PreprocessError> {
    let n = split_hcps(hcps)?;
    let mut masks = vec![0u8; n * MOVE_LABELS_NUM];
    for i in 0..n {
        let mut board = board_at(hcps, i)?;
        let turn = board.turn();
        let mask = &mut masks[i * MOVE_LABELS_NUM..(i + 1) * MOVE_LABELS_NUM];
        for m in generate_legal_moves(&mut board) {
            let label = try_move_label(turn, m).ok_or(PreprocessError::IllegalMove {
                index: i,
                move16: 0,
            })?;
            mask[label as usize] = 1;
        }
    }
    Ok(masks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::move_label;

    /// 平手初期局面の HCP を作る．
    fn startpos_hcp() -> Vec<u8> {
        let board = Board::new();
        hcp::to_hcp(&board)
            .expect("startpos は HCP 化できる")
            .to_vec()
    }

    #[test]
    fn test_result_value_hcpe_convention() {
        // gameResult は HCPE 規約 (0=draw, 1=black win, 2=white win)．
        // 手番側視点の教師値 (1=勝ち, 0=負け, 0.5=引き分け) を返す．
        assert_eq!(result_value(Color::Black, 0), 0.5);
        assert_eq!(result_value(Color::Black, 1), 1.0);
        assert_eq!(result_value(Color::Black, 2), 0.0);
        assert_eq!(result_value(Color::White, 0), 0.5);
        assert_eq!(result_value(Color::White, 1), 0.0);
        assert_eq!(result_value(Color::White, 2), 1.0);
    }

    #[test]
    fn test_process_hcpes_startpos() {
        let board = Board::new();
        let hcps = startpos_hcp();
        // 7g7f (先手番)
        let m = board.move_from_usi("7g7f").expect("合法手");
        let m16 = maou_shogi::moves::move16(m);
        let (hashes, labels, results) =
            process_hcpes(&hcps, &[m16 as i16], &[1]).expect("有効な入力");
        assert_eq!(hashes, vec![board.hash()]);
        assert_eq!(labels, vec![move_label(Color::Black, m)]);
        // gameResult=1 (先手勝ち) × 先手番 → 手番側勝ち = 1.0
        assert_eq!(results, vec![1.0]);
    }

    #[test]
    fn test_process_hcpes_length_mismatch() {
        let hcps = startpos_hcp();
        assert!(process_hcpes(&hcps, &[0, 0], &[1, 1]).is_err());
        assert!(process_hcpes(&hcps[..31], &[0], &[1]).is_err());
    }

    #[test]
    fn test_encode_hcp_features_matches_scalar_encoders() {
        // 後手番局面 (先手が 7g7f を指した直後) で正規化込みの一致を確認する
        let mut board = Board::new();
        let m = board.move_from_usi("7g7f").expect("合法手");
        board.do_move(m);
        let hcp_bytes = hcp::to_hcp(&board).expect("HCP 化できる");

        let (boards, hands) = encode_hcp_features(&hcp_bytes).expect("有効な入力");

        let mut expected_board = [0u8; BOARD_FEATURE_LEN];
        encode_board_ids(&board, &mut expected_board);
        let mut expected_hand = [0u8; HAND_FEATURE_LEN];
        encode_hand_counts(&board, &mut expected_hand);
        assert_eq!(boards, expected_board.to_vec());
        assert_eq!(hands, expected_hand.to_vec());
    }

    #[test]
    fn test_legal_move_masks_white_turn_rotation_equivalence() {
        // 後手番: 「元盤面の合法手を後手視点でラベル化」が正規化済み盤面の
        // 合法手数と一致し，マスクの立つ数 = 合法手のユニークラベル数になる
        let mut board = Board::new();
        let m = board.move_from_usi("7g7f").expect("合法手");
        board.do_move(m);
        assert_eq!(board.turn(), Color::White);
        let hcp_bytes = hcp::to_hcp(&board).expect("HCP 化できる");

        let masks = legal_move_masks(&hcp_bytes).expect("有効な入力");
        let legal = generate_legal_moves(&mut board);
        let count = masks.iter().filter(|&&x| x == 1).count();
        // 初期局面付近では合法手のラベルは全て異なる
        assert_eq!(count, legal.len());
        for m in legal {
            let label = try_move_label(Color::White, m).expect("合法手はラベル化できる");
            assert_eq!(masks[label as usize], 1);
        }
    }
}
