//! KH `Node` の path 保持 (`ContainsInPath` / VisitHistory) 相当の **flat な探索パス スタック**．
//!
//! mid_v3 / mid_v4 は探索パス上の祖先局面を，従来 `FxHashMap` (`v3_path` 千日手 / `v4_dom_path`
//! path dominance) で保持していた．HashMap は bucket が確保領域全体に散在するため，毎ノードの
//! insert/remove/get が **散在 DRAM アクセス**となり (V4PROF `cl_other` pathcheck)，探索が
//! メモリ帯域律速になる主因の一つだった (host 負荷で 2.3× 振れる; KH は安定)．
//!
//! KH/YaneuraOu は探索パスを **連続配列**で持ち，`Node::ContainsInPath` のように線形走査して
//! 祖先一致を調べる (`mid_v4` の `contains_in_path_v4` = `v4_stack` の線形走査と同型)．パス長は
//! 高々探索深さ (39te で ~55) と小さく，連続配列の逆順走査は L1/L2 に収まり予測可能で
//! **memory-bound でない**．本モジュールはこの flat 構造を提供し，HashMap を置換する．
//!
//! ## 探索不変性 (bit-identical)
//! - 探索 DFS は append/exit が厳密に LIFO (search 関数の入口で push・出口で pop)．
//! - 探索パス上で同一キーは高々 1 回しか現れない (再出現は千日手として検出され do_move しない)．
//! → 逆順線形走査が返す祖先は HashMap が返すものと完全一致するため，置換は探索結果を変えない
//!   (canonical mid_v3 18,539 / 39te node 不変で検証)．

use super::search_result::Hand;

/// 千日手検出用の flat パス スタック (旧 `v3_path: FxHashMap<u64, u32>` の drop-in 置換)．
///
/// `board.hash` (全局面 hash, 手番・持駒込) をキーに，パス上の祖先 ply を保持する．
/// メソッド名/シグネチャは `FxHashMap` 互換 (insert/remove/get/clear) で，呼び出し側を
/// 変更せずに差し替えられる．
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

    /// solve ごとにパスを空にする (`FxHashMap::clear` 相当)．
    #[inline]
    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    /// パス入口で祖先 (hash, ply) を push する (`FxHashMap::insert` 相当)．
    ///
    /// パス上で同一 hash は高々 1 回なので上書きは発生しない (単純 push)．
    #[inline]
    pub(super) fn insert(&mut self, hash: u64, ply: u32) {
        self.entries.push((hash, ply));
    }

    /// パス出口で祖先を取り除く (`FxHashMap::remove` 相当)．
    ///
    /// LIFO 規律により対象は通常末尾だが，`rposition` で末尾側の一致を取り除くことで
    /// HashMap と完全に同じ「そのキーのエントリを 1 つ消す」意味論を保つ．
    #[inline]
    pub(super) fn remove(&mut self, hash: &u64) {
        if let Some(pos) = self.entries.iter().rposition(|&(h, _)| h == *hash) {
            self.entries.remove(pos);
        }
    }

    /// パス上に同一 hash の祖先があればその ply を返す (`FxHashMap::get` 相当)．
    ///
    /// 逆順走査で最も新しい (= 最も深い) 祖先を返す (パス上一意なので結果は HashMap と同一)．
    #[inline]
    pub(super) fn get(&self, hash: &u64) -> Option<&u32> {
        self.entries
            .iter()
            .rev()
            .find(|&&(h, _)| h == *hash)
            .map(|(_, p)| p)
    }
}

/// path dominance 用の flat パス スタック (旧 `v4_dom_path: FxHashMap<u64, Vec<(Hand, u32)>>`)．
///
/// `position_key` (盤面のみ hash) をキーに，パス上の祖先 (攻め方持駒, ply) を保持する．
/// 子局面が同一 board_key かつ攻め方持駒が祖先以下 (= 劣位) なら反復として刈る
/// (KH `IsRepetitionOrInferiorAfter`)．
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

    /// パス出口で当該 position_key の最も新しい祖先を取り除く
    /// (旧 `get_mut(&pk).pop()` + 空なら `remove(&pk)` と同一意味論)．
    #[inline]
    pub(super) fn pop(&mut self, pos: u64) {
        if let Some(idx) = self.entries.iter().rposition(|&(p, _, _)| p == pos) {
            self.entries.remove(idx);
        }
    }

    /// 子局面 `(pos, child_hand)` を劣位とする祖先 (同一 board_key かつ攻め方持駒が child 以上)
    /// があればその ply を返す (旧 `get(&pos).iter().rev().find(hand_gte(h, child))`)．
    ///
    /// 逆順走査で `pos` 一致かつ `hand_gte(ancestor_hand, child_hand)` の最初の祖先を返す
    /// (HashMap が per-key Vec を逆順走査するのと同一順序・同一結果)．
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
        s.insert(0x10, 8); // 仮に重複が起きても HashMap 上書きと同じく新しい方
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
