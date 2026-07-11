//! 探索パス上の祖先局面を保持する **flat な探索パス スタック**．
//!
//! 探索パス上の祖先局面 (千日手検出用の `path_depths` / path dominance 用の `dom_path`) を
//! 連続配列で保持する．`FxHashMap` で持つと bucket が確保領域全体に散在し，毎ノードの
//! insert/remove/get が散在 DRAM アクセスとなって探索がメモリ帯域律速になりやすい．
//!
//! ここでは探索パスを連続配列で持ち，線形走査で祖先一致を調べる．パス長は高々探索深さ
//! (数十手程度) と小さく，連続配列の逆順走査は L1/L2 に収まり予測可能で memory-bound に
//! ならない．
//!
//! ## 探索不変性
//! - 探索 DFS は append/exit が厳密に LIFO (search 関数の入口で push・出口で pop)．
//! - 探索パス上で同一キーは高々 1 回しか現れない (再出現は千日手として検出され do_move しない)．
//!
//! → 逆順線形走査が返す祖先は HashMap が返すものと完全一致するため，置換は探索結果を変えない．

use super::search_result::Hand;

/// 千日手検出用の flat パス スタック．
///
/// `board.hash` (全局面 hash, 手番・持駒込) をキーに，パス上の祖先 ply を保持する．
/// メソッド名/シグネチャは `FxHashMap` 互換 (insert/remove/get/clear)．
pub(super) struct PathStack {
    /// (full board hash, ply) を push 順に保持．
    entries: Vec<(u64, u32)>,
}

impl PathStack {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::with_capacity(128),
        }
    }

    /// solve ごとにパスを空にする．
    #[inline]
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    /// パス入口で祖先 (hash, ply) を push する．
    ///
    /// パス上で同一 hash は高々 1 回なので上書きは発生しない (単純 push)．
    #[inline]
    pub(super) fn insert(&mut self, hash: u64, ply: u32) {
        self.entries.push((hash, ply));
    }

    /// パス出口で祖先を取り除く．
    ///
    /// LIFO 規律により対象は通常末尾だが，`rposition` で末尾側の一致を取り除くことで
    /// 「そのキーのエントリを 1 つ消す」意味論を保つ．
    #[inline]
    pub(super) fn remove(&mut self, hash: &u64) {
        if let Some(pos) = self.entries.iter().rposition(|&(h, _)| h == *hash) {
            self.entries.remove(pos);
        }
    }

    /// パス上に同一 hash の祖先があればその ply を返す．
    ///
    /// 逆順走査で最も新しい (= 最も深い) 祖先を返す (パス上一意なので結果は一意)．
    #[inline]
    pub(super) fn get(&self, hash: &u64) -> Option<&u32> {
        self.entries
            .iter()
            .rev()
            .find(|&&(h, _)| h == *hash)
            .map(|(_, p)| p)
    }
}

/// path dominance 用の flat パス スタック．
///
/// `position_key` (盤面のみ hash) をキーに，パス上の祖先 (攻め方持駒, ply) を保持する．
/// 子局面が同一 board_key かつ攻め方持駒が祖先以下 (= 劣位) なら反復として刈る．
pub(super) struct DomPathStack {
    /// (position_key, attacker_hand, ply) を push 順に保持．
    entries: Vec<(u64, Hand, u32)>,
}

impl DomPathStack {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::with_capacity(128),
        }
    }

    #[inline]
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    /// パス入口で祖先 (position_key, attacker_hand, ply) を push する．
    #[inline]
    pub(super) fn push(&mut self, pos: u64, hand: Hand, ply: u32) {
        self.entries.push((pos, hand, ply));
    }

    /// パス出口で当該 position_key の最も新しい祖先を取り除く．
    #[inline]
    pub(super) fn pop(&mut self, pos: u64) {
        if let Some(idx) = self.entries.iter().rposition(|&(p, _, _)| p == pos) {
            self.entries.remove(idx);
        }
    }

    /// 子局面 `(pos, child_hand)` を劣位とする祖先 (同一 board_key かつ攻め方持駒が child 以上)
    /// があればその ply を返す．
    ///
    /// 逆順走査で `pos` 一致かつ `hand_gte(ancestor_hand, child_hand)` の最初の祖先を返す．
    #[inline]
    pub(super) fn find_dominator(&self, pos: u64, child_hand: &Hand) -> Option<u32> {
        self.entries
            .iter()
            .rev()
            .find(|&&(p, h, _)| p == pos && super::hand_gte(&h, child_hand))
            .map(|&(_, _, d)| d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pathstack_insert_get_remove_lifo() {
        let mut s = PathStack::new();
        s.insert(0xAA, 1);
        s.insert(0xBB, 3);
        s.insert(0xCC, 5);
        assert_eq!(s.get(&0xBB), Some(&3));
        assert_eq!(s.get(&0xCC), Some(&5));
        assert_eq!(s.get(&0xDD), None);
        // LIFO: 末尾 pop．
        s.remove(&0xCC);
        assert_eq!(s.get(&0xCC), None);
        assert_eq!(s.get(&0xAA), Some(&1));
    }

    #[test]
    fn pathstack_get_returns_deepest() {
        // 逆順走査は最も新しい (深い) 祖先を返す (パス上一意前提だが順序保証を確認)．
        let mut s = PathStack::new();
        s.insert(0x10, 2);
        s.insert(0x10, 8); // 仮に重複が起きても新しい方を返す
        assert_eq!(s.get(&0x10), Some(&8));
    }

    fn hand(pawns: u8, rooks: u8) -> Hand {
        let mut h = [0u8; crate::types::HAND_KINDS];
        h[0] = pawns;
        h[6] = rooks;
        h
    }

    #[test]
    fn dompath_find_dominator_superset() {
        let mut s = DomPathStack::new();
        let pos_a: u64 = 0xAAAA_0000_1111;
        let pos_b: u64 = 0xBBBB_0000_2222;
        s.push(pos_a, hand(2, 1), 4);
        // 子 hand(1,0) は祖先 hand(2,1) 以下 (劣位) → dominator あり (ply 4)．
        assert_eq!(s.find_dominator(pos_a, &hand(1, 0)), Some(4));
        // 子 hand(3,0) は祖先より歩が多い → 劣位でない → None．
        assert_eq!(s.find_dominator(pos_a, &hand(3, 0)), None);
        // 別 position → None．
        assert_eq!(s.find_dominator(pos_b, &hand(1, 0)), None);
    }

    #[test]
    fn dompath_pop_removes_latest_for_pos() {
        let mut s = DomPathStack::new();
        s.push(0x1, hand(1, 0), 2);
        s.push(0x1, hand(5, 0), 6); // 同 position の深い祖先
        s.pop(0x1); // 深い方 (ply 6) を取り除く
        assert_eq!(s.find_dominator(0x1, &hand(0, 0)), Some(2));
    }
}
