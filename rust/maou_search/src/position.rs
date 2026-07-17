//! 局面セットアップ (基準局面 SFEN + USI 指し手列 → root 局面 + 対局履歴)．
//!
//! USI の `position sfen <sfen> moves ...` 規約の単一実装．PyO3 バインディング
//! (maou_rust) と USI エンジン (maou_usi) の両方がこれを使う．

use maou_shogi::board::{Board, SfenError};
use maou_shogi::movegen::generate_legal_moves;

use crate::repetition::HistoryEntry;

/// [`build_board_and_history`] のエラー．
#[derive(Debug)]
pub enum PositionSetupError {
    /// SFEN が不正．
    Sfen(SfenError),
    /// 非合法または不正な指し手 (USI 表記)．
    IllegalMove(String),
}

impl std::fmt::Display for PositionSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionSetupError::Sfen(e) => write!(f, "不正な SFEN: {e:?}"),
            PositionSetupError::IllegalMove(usi) => {
                write!(f, "非合法または不正な指し手: {usi}")
            }
        }
    }
}

impl std::error::Error for PositionSetupError {}

/// 基準局面 SFEN + USI 指し手列から root 局面と対局履歴 (千日手判定用) を構築する．
///
/// 履歴は root 自身を含まず古い順
/// ([`crate::search::Searcher::search_with_history`] の規約)．各指し手は
/// その局面の合法手列挙と USI 表記一致で検証される (非合法手は
/// [`PositionSetupError::IllegalMove`])．
pub fn build_board_and_history(
    sfen: &str,
    moves: &[String],
) -> Result<(Board, Vec<HistoryEntry>), PositionSetupError> {
    let mut board = Board::empty();
    board.set_sfen(sfen).map_err(PositionSetupError::Sfen)?;
    let mut history: Vec<HistoryEntry> = Vec::with_capacity(moves.len());
    for usi in moves {
        let mut probe = board.clone();
        let mv = generate_legal_moves(&mut probe)
            .into_iter()
            .find(|m| m.to_usi() == *usi)
            .ok_or_else(|| PositionSetupError::IllegalMove(usi.clone()))?;
        history.push(HistoryEntry::from_board(&board));
        board.do_move(mv);
    }
    Ok((board, history))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 平手初期局面 (maou_shogi::sfen::HIRATE_SFEN は private のため試験用に複製)．
    const HIRATE_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

    #[test]
    fn test_build_board_and_history_startpos() {
        let (board, history) = build_board_and_history(HIRATE_SFEN, &[]).unwrap();
        assert!(history.is_empty());
        // 平手初期局面の合法手は 30 手
        let mut probe = board.clone();
        assert_eq!(generate_legal_moves(&mut probe).len(), 30);
    }

    #[test]
    fn test_build_board_and_history_moves() {
        let moves: Vec<String> = ["7g7f", "3c3d"].iter().map(|s| s.to_string()).collect();
        let (_, history) = build_board_and_history(HIRATE_SFEN, &moves).unwrap();
        // 履歴は root 自身を含まない = 指し手数と同数
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_build_board_and_history_illegal_move() {
        let moves: Vec<String> = vec!["7g7e".to_string()];
        let err = build_board_and_history(HIRATE_SFEN, &moves)
            .err()
            .expect("非合法手はエラーになること");
        match err {
            PositionSetupError::IllegalMove(usi) => assert_eq!(usi, "7g7e"),
            other => panic!("IllegalMove を期待: {other:?}"),
        }
    }

    #[test]
    fn test_build_board_and_history_bad_sfen() {
        assert!(matches!(
            build_board_and_history("not-a-sfen", &[]),
            Err(PositionSetupError::Sfen(_))
        ));
    }
}
