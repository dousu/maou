use crate::board::{Board, SfenError};
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
    pub(crate) board: Board,
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
    pub fn from_sfen(sfen: &str) -> Result<Position, SfenError> {
        let mut board = Board::empty();
        board.set_sfen(sfen)?;
        Ok(Position {
            board,
            history: Vec::new(),
        })
    }

    /// 手を実行する．
    ///
    /// # 前提条件
    /// - 手番は先後交互に呼び出すこと．連続して同じ色が手を指した場合，
    ///   連続王手の千日手判定(`is_perpetual_check_move`)のparity計算が不正になる．
    pub fn do_move(&mut self, m: Move) {
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
    ///
    /// 履歴が空の場合は `None` を返す．
    pub fn undo_move(&mut self) -> Option<Move> {
        let state = self.history.pop()?;
        self.board.undo_move(state.last_move, state.captured);
        Some(state.last_move)
    }

    /// 合法手を生成する．
    ///
    /// 基本的な合法手に加えて，連続王手の千日手になる手を除外する．
    pub fn legal_moves(&mut self) -> Vec<Move> {
        let mut moves = movegen::generate_legal_moves(&mut self.board);

        // 連続王手の千日手チェック
        // retain with &mut selfが使えないため，手動でフィルタリング
        let mut filtered = Vec::with_capacity(moves.len());
        for m in moves.drain(..) {
            if !self.is_perpetual_check_move(m) {
                filtered.push(m);
            }
        }

        filtered
    }

    /// この手を指すと連続王手の千日手が成立するかを判定する．
    ///
    /// 条件:
    /// - この手が王手である
    /// - 手を指した後の局面hashが履歴に3回以上出現(計4回目)
    /// - 最初の出現から現在まで，自分の全ての手が王手だった
    fn is_perpetual_check_move(&mut self, m: Move) -> bool {
        // 手を仮に指してhashを計算(do_move/undo_moveで元に戻す)
        let captured = self.board.do_move(m);
        let new_hash = self.board.hash;

        // この手で王手をかけるかチェック
        let gives_check = self.board.is_in_check(self.board.turn);

        // 元に戻す
        self.board.undo_move(m, captured);

        if !gives_check {
            return false;
        }

        // 同一局面hashの出現回数と最初の出現位置を単一パスで取得
        //
        // 先手の手は偶数インデックス(0,2,4,...)，後手は奇数インデックス(通常)．
        // ただし from_sfen で後手番開始した場合は history_len % 2 と turn が
        // 一致しないため，turn.index() を直接使用する．
        let history_len = self.history.len();
        let my_parity = self.board.turn.index();

        let mut count = 0usize;
        let mut first_idx = None;
        for (i, s) in self.history.iter().enumerate() {
            if s.hash == new_hash {
                count += 1;
                if first_idx.is_none() {
                    first_idx = Some(i);
                }
            }
        }

        if count < 3 {
            return false;
        }

        // 最初の出現から現在まで，自分の全ての手が王手かチェック
        let first_idx = first_idx.unwrap();
        for i in first_idx..history_len {
            if i % 2 == my_parity && !self.history[i].gives_check {
                return false;
            }
        }

        true
    }

    /// 詰み(王手されていて合法手がない状態)かどうかを判定する．
    pub fn is_checkmate(&mut self) -> bool {
        self.board.is_in_check(self.board.turn) && self.legal_moves().is_empty()
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

    /// USI文字列から合法手を検索するヘルパー．
    fn find_move(pos: &mut Position, usi: &str) -> Option<Move> {
        pos.legal_moves().into_iter().find(|m| m.to_usi() == usi)
    }

    #[test]
    fn test_perpetual_check() {
        // 連続王手の千日手テスト
        // 白玉1a, 黒龍3b, 黒玉9i
        // 龍が3b↔3aを往復して王手し続ける
        //
        // サイクル:
        //   Black: 3b→3a (王手) → White: 1a→1b → Black: 3a→3b (王手) → White: 1b→1a
        // 4サイクル目で連続王手の千日手が成立
        let mut pos = Position::from_sfen("8k/6+R2/9/9/9/9/9/9/K8 b - 1").unwrap();

        let initial_hash = pos.hash();

        // 1サイクル目: 全ての手が合法
        assert!(find_move(&mut pos, "3b3a").is_some(), "cycle 1: 3b3a should be legal");
        let m = find_move(&mut pos, "3b3a").unwrap(); pos.do_move(m);
        let m = find_move(&mut pos, "1a1b").unwrap(); pos.do_move(m);
        let m = find_move(&mut pos, "3a3b").unwrap(); pos.do_move(m);
        let m = find_move(&mut pos, "1b1a").unwrap(); pos.do_move(m);

        // 1サイクル後: 元の局面に戻る
        assert_eq!(pos.hash(), initial_hash, "hash should return to initial after 1 cycle");

        // 2サイクル目
        let m = find_move(&mut pos, "3b3a").unwrap(); pos.do_move(m);
        let m = find_move(&mut pos, "1a1b").unwrap(); pos.do_move(m);
        let m = find_move(&mut pos, "3a3b").unwrap(); pos.do_move(m);
        let m = find_move(&mut pos, "1b1a").unwrap(); pos.do_move(m);

        // 3サイクル目
        let m = find_move(&mut pos, "3b3a").unwrap(); pos.do_move(m);
        let m = find_move(&mut pos, "1a1b").unwrap(); pos.do_move(m);
        let m = find_move(&mut pos, "3a3b").unwrap(); pos.do_move(m);
        let m = find_move(&mut pos, "1b1a").unwrap(); pos.do_move(m);

        // 4サイクル目: 連続王手の千日手が成立 → 3b3aが合法手から除外される
        assert!(
            find_move(&mut pos, "3b3a").is_none(),
            "cycle 4: 3b3a should be excluded (perpetual check)"
        );

        // 他の手はまだ合法
        let moves = pos.legal_moves();
        let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
        assert!(
            !moves.is_empty(),
            "should still have other legal moves"
        );
        assert!(
            !usi_moves.contains(&"3b3a".to_string()),
            "3b3a must not be in legal moves"
        );

        // 龍の他の移動手(王手でない手)は合法
        assert!(
            find_move(&mut pos, "3b3c").is_some(),
            "non-check move 3b3c should be legal"
        );
    }

    #[test]
    fn test_no_perpetual_check_if_broken() {
        // 連続王手を途中で中断した場合は千日手にならない
        // 2サイクル後に別の手を指し，その後同じパターンに戻っても
        // 連続王手の千日手にはならない
        let mut pos = Position::from_sfen("8k/6+R2/9/9/9/9/9/9/K8 b - 1").unwrap();

        // 2サイクル実行
        for _ in 0..2 {
            let m = find_move(&mut pos, "3b3a").unwrap(); pos.do_move(m);
            let m = find_move(&mut pos, "1a1b").unwrap(); pos.do_move(m);
            let m = find_move(&mut pos, "3a3b").unwrap(); pos.do_move(m);
            let m = find_move(&mut pos, "1b1a").unwrap(); pos.do_move(m);
        }

        // 中断: 龍を別のマスに動かす(王手でない手)
        let m = find_move(&mut pos, "3b3c").unwrap(); pos.do_move(m);
        // 白は適当に動く
        let m = find_move(&mut pos, "1a1b").unwrap(); pos.do_move(m);
        // 龍を3bに戻す
        let m = find_move(&mut pos, "3c3b").unwrap(); pos.do_move(m);
        // 白も戻る
        let m = find_move(&mut pos, "1b1a").unwrap(); pos.do_move(m);

        // 再度サイクルを開始 → 連続王手が途切れているので千日手にならない
        // (中断前の2回 + 中断後1回 = 3回だが，間に王手でない手がある)
        assert!(
            find_move(&mut pos, "3b3a").is_some(),
            "3b3a should still be legal after breaking perpetual check"
        );
    }
}
