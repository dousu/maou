use crate::board::Board;
use crate::movegen;
use crate::moves::Move;
use crate::types::{Color, Piece};

/// 履歴情報．
#[derive(Debug, Clone)]
struct StateInfo {
    /// この局面のZobrist hash．
    hash: u64,
    /// 直前の手で取った駒．
    captured: Piece,
    /// 直前の手．
    last_move: Move,
    /// この局面で手番側が王手をかけているか．
    gives_check: bool,
}

/// 局面(Board + 履歴管理)．
///
/// 連続王手の千日手判定のために局面履歴を管理する．
#[derive(Clone)]
pub struct Position {
    pub board: Board,
    history: Vec<StateInfo>,
}

impl Position {
    /// 平手初期局面で生成する．
    pub fn new() -> Position {
        Position {
            board: Board::new(),
            history: Vec::new(),
        }
    }

    /// SFEN文字列から生成する．
    pub fn from_sfen(sfen: &str) -> Result<Position, String> {
        let mut board = Board::new();
        board.set_sfen(sfen)?;
        Ok(Position {
            board,
            history: Vec::new(),
        })
    }

    /// 手を実行する．
    pub fn do_move(&mut self, m: Move) {
        let _us = self.board.turn;

        let captured = self.board.do_move(m);

        // 相手の玉に王手をかけているか(do_move後は手番が交代済み)
        let gives_check = self.board.is_in_check(self.board.turn);

        self.history.push(StateInfo {
            hash: self.board.hash,
            captured,
            last_move: m,
            gives_check,
        });
    }

    /// 手を取り消す．
    pub fn undo_move(&mut self) {
        let state = self.history.pop().expect("no move to undo");
        self.board.undo_move(state.last_move, state.captured);
    }

    /// 合法手を生成する．
    ///
    /// 基本的な合法手に加えて，連続王手の千日手になる手を除外する．
    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = movegen::generate_legal_moves(&self.board);

        // 連続王手の千日手チェック
        moves.retain(|&m| !self.is_perpetual_check_move(m));

        moves
    }

    /// この手を指すと連続王手の千日手が成立するかを判定する．
    ///
    /// 条件:
    /// - 手を指した後の局面hashが履歴に3回以上出現している
    /// - かつ，その間の自分の全ての手が王手だった
    fn is_perpetual_check_move(&self, m: Move) -> bool {
        // 手を仮に指してhashを計算
        let mut board_copy = self.board.clone();
        let _captured = board_copy.do_move(m);
        let new_hash = board_copy.hash;

        // この手で王手をかけるかチェック
        let gives_check = board_copy.is_in_check(board_copy.turn);
        if !gives_check {
            // この手が王手でなければ連続王手千日手にはならない
            return false;
        }

        // 同一局面の出現回数をカウント
        // 連続王手かどうかも同時にチェック
        let _us = self.board.turn;
        let mut repetition_count = 0u32;
        let _all_checks = true;

        // 履歴を逆順に見ていく
        for (_i, state) in self.history.iter().enumerate().rev() {
            if state.hash == new_hash {
                repetition_count += 1;
            }

            // 自分の手番の履歴をチェック
            // historyのインデックスのパリティで手番を判定
            // 最新のhistoryは相手の手なので，1つ前が自分の手
            // ...ただし，do_moveはまだ行っていないので，
            // history[len-1]は直前の相手の手の結果

            // 同一局面に遡る間で，自分の手が全て王手かチェック
        }

        // 現在の手を含めて4回目の同一局面(= 履歴に3回)
        if repetition_count >= 3 {
            // 連続王手チェック: 全ての過去の出現から現在までの自分の手が王手か
            // 簡易実装: 直近の同一局面出現間隔で全ての手が王手だったか
            return self.check_perpetual_check_detail(new_hash, gives_check);
        }

        false
    }

    /// 連続王手千日手の詳細判定．
    fn check_perpetual_check_detail(&self, target_hash: u64, _current_gives_check: bool) -> bool {
        // 履歴を逆順に辿り，同一局面間の全ての自分の手が王手かチェック
        let history_len = self.history.len();
        let mut count = 0u32;

        // 自分の手番のインデックス: history_lenからの偶数間隔
        // (最新のhistoryエントリは直前の手の結果)
        for i in (0..history_len).rev() {
            if self.history[i].hash == target_hash {
                count += 1;
            }

            // 自分の手番の状態(gives_check)をチェック
            // gives_checkは「この手を指した結果，相手玉に王手がかかっているか」
            // 自分の手かどうかはインデックスのパリティで判断
            // history[i]のgives_checkが自分の手の結果なら，
            // そのgives_checkがfalseの区間があれば連続王手ではない

            // 簡易判定: 同一局面間の全てのgives_checkがtrue
            if count >= 3 {
                // 最後の同一局面から現在まで，全てのgives_checkがtrue
                let mut all_check = true;
                let mut found = 0u32;
                for j in (0..history_len).rev() {
                    if self.history[j].hash == target_hash {
                        found += 1;
                        if found >= 3 {
                            break;
                        }
                    }
                    if !self.history[j].gives_check {
                        all_check = false;
                        break;
                    }
                }
                return all_check;
            }
        }

        false
    }

    /// SFEN文字列を返す．
    pub fn sfen(&self) -> String {
        self.board.sfen()
    }

    /// 現在の手番を返す．
    pub fn turn(&self) -> Color {
        self.board.turn
    }

    /// Zobrist hashを返す．
    pub fn hash(&self) -> u64 {
        self.board.hash
    }
}

impl Default for Position {
    fn default() -> Self {
        Position::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_do_undo() {
        let mut pos = Position::new();
        let original_sfen = pos.sfen();
        let original_hash = pos.hash();

        let moves = pos.legal_moves();
        assert!(!moves.is_empty());

        let m = moves[0];
        pos.do_move(m);
        assert_ne!(pos.sfen(), original_sfen);

        pos.undo_move();
        assert_eq!(pos.sfen(), original_sfen);
        assert_eq!(pos.hash(), original_hash);
    }
}
