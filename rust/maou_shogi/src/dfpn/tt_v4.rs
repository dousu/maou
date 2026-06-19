//! KH `tt::RegularTable` + `tt::Query` + `tt::RepetitionTable` の忠実移植
//! (`regular_table.hpp` / `ttquery.hpp` / `repetition_table.hpp`)．
//!
//! mid_v4 (KH verbatim 再現) の置換表本体．[`super::ttentry::Entry`] (len-aware/cross-hand) を
//! 循環配列で管理し，cluster を走査して複数 entry 横断で pn/dn を合成する (KH `Query::LookUp`)．
//!
//! ## single-thread 適応 (並列化は対象外)
//! - KH `Query` は entry への生ポインタ (`initial_entry_pointer_`, `cached_entry_`) を保持するが，
//!   Rust borrow checker と相性が悪いため **cluster 先頭 index を保持し table 参照を都度渡す**形に
//!   変える (意味論は同一)．
//! - thread ノイズ (`tt_noise`) は **main thread (=single-thread) では載らない** (ttquery.hpp:42
//!   「main thread にはノイズを載せない」) ため除外する．atomic/lock も plain 化．

use super::mate_len::MateLen;
use super::search_result::{BitSet64, Depth, Hand, PnDn, SearchAmount, SearchResult};
use super::ttentry::{Entry, NULL_HAND};
use rustc_hash::FxHashMap;

/// KH `ApplyDeltaHand` (hands.hpp:83)．`target + (diff_dst - diff_src)` を駒種別に
/// `[0, MAX_HAND_COUNT]` へ clamp する (cross-hand 親持駒の補正)．
#[inline]
fn apply_delta_hand(target: Hand, diff_src: Hand, diff_dst: Hand) -> Hand {
    let mut res = [0u8; crate::types::HAND_KINDS];
    for i in 0..crate::types::HAND_KINDS {
        let v = target[i] as i32 + diff_dst[i] as i32 - diff_src[i] as i32;
        res[i] = v.clamp(0, crate::types::PieceType::MAX_HAND_COUNT[i] as i32) as u8;
    }
    res
}

/// KH `tt::RegularTable` (regular_table.hpp:133)．循環配列でエントリを管理する通常テーブル．
pub(super) struct RegularTable {
    entries: Vec<Entry>,
}

impl RegularTable {
    /// 要素数 `num_entries` (最低 1) でテーブルを確保し全 null 初期化する (KH `Resize`)．
    pub(super) fn new(num_entries: usize) -> Self {
        let n = num_entries.max(1);
        RegularTable {
            entries: vec![Entry::null(); n],
        }
    }
    /// 全エントリを null 化する (KH `Clear`)．
    pub(super) fn clear(&mut self) {
        for e in &mut self.entries {
            *e = Entry::null();
        }
    }
    /// 保存可能要素数 (KH `Capacity`)．
    pub(super) fn capacity(&self) -> usize {
        self.entries.len()
    }
    /// `board_key` の cluster 先頭 index (KH `PointerOf`; Stockfish 方式で mod を回避)．
    #[inline]
    pub(super) fn pointer_of(&self, board_key: u64) -> usize {
        let hash_low = board_key & 0xffff_ffff;
        ((hash_low as u128 * self.entries.len() as u128) >> 32) as usize
    }

    /// KH `CalculateHashRate` (regular_table.hpp:198)．`kHashfullCalcEntries`=10000 個を
    /// stride 334 でサンプルし使用率を見積もる (全走査を避ける近似)．
    pub(super) fn calculate_hash_rate(&self) -> f64 {
        const SAMPLES: usize = 10_000;
        let n = self.entries.len();
        if n == 0 {
            return 1.0;
        }
        let samples = SAMPLES.min(n);
        let mut used = 0usize;
        let mut idx = 1usize % n;
        for _ in 0..samples {
            if !self.entries[idx].is_null() {
                used += 1;
            }
            idx += 334;
            if idx >= n {
                idx -= n;
            }
        }
        used as f64 / samples as f64
    }

    /// KH `CollectGarbage` (regular_table.hpp:226)．`kGcSamplingEntries`=20000 個の amount を
    /// サンプルし，下位 `ratio` 分位を閾値に，それ以下の amount の entry を null 化して間引く．
    /// max_amount が飽和に近ければ全 amount を半減 (CutAmount)．最後に compaction で穴を詰める．
    pub(super) fn collect_garbage(&mut self, ratio: f64) {
        const SAMPLES: usize = 20_000;
        let n = self.entries.len();
        if n == 0 {
            return;
        }
        let mut amounts: Vec<SearchAmount> = Vec::with_capacity(SAMPLES.min(n));
        let mut idx = 0usize;
        let mut scanned = 0usize;
        while amounts.len() < SAMPLES && scanned < n {
            if !self.entries[idx].is_null() {
                amounts.push(self.entries[idx].amount());
            }
            idx += 334;
            if idx >= n {
                idx -= n;
            }
            scanned += 1;
        }
        if amounts.is_empty() {
            return;
        }
        // 下位 ratio 分位 (KH: nth_element pivot)．
        let pivot = ((amounts.len() as f64 * ratio) as usize)
            .max(1)
            .min(amounts.len() - 1);
        amounts.sort_unstable();
        let amount_threshold = amounts[pivot];
        let max_amount = *amounts.last().unwrap();
        let should_cut = max_amount > SearchAmount::MAX / 8;
        for e in &mut self.entries {
            if e.is_null() {
                continue;
            }
            if e.amount() <= amount_threshold {
                e.set_null();
            } else if should_cut {
                e.cut_amount();
            }
        }
        self.compact_entries();
    }

    /// KH `CompactEntries` (regular_table.hpp:344)．null 穴を詰めて各 entry を `PointerOf` から
    /// なるべく手前へ再配置する (open addressing の probe chain 整合; 先頭周辺は wrap で目を瞑る)．
    fn compact_entries(&mut self) {
        let n = self.entries.len();
        if n == 0 {
            return;
        }
        for i in 0..n {
            if self.entries[i].is_null() {
                continue;
            }
            let start = self.pointer_of(self.entries[i].board_key());
            let mut j = start;
            while j != i {
                if self.entries[j].is_null() {
                    self.entries[j] = self.entries[i];
                    self.entries[i].set_null();
                    break;
                }
                j += 1;
                if j == n {
                    j = 0;
                }
            }
        }
    }
}

/// KH `tt::RepetitionTable` (repetition_table.hpp)．経路 (path_key) 依存の千日手結果を記録する．
/// 開アドレス法 + 世代 GC を持つ KH に対し，意味論を保った `FxHashMap` で移植する
/// (path_key ごとに 1 つの (depth, len)．Insert の更新規則は KH:100-130 と同一)．
pub(super) struct RepetitionTable {
    map: FxHashMap<u64, (Depth, MateLen)>,
}

impl RepetitionTable {
    pub(super) fn new() -> Self {
        RepetitionTable {
            map: FxHashMap::default(),
        }
    }
    /// 前回探索結果を消す (KH `NewSearch`/`Clear`)．
    pub(super) fn clear(&mut self) {
        self.map.clear();
    }
    /// KH `Insert` (repetition_table.hpp:100)．
    pub(super) fn insert(&mut self, path_key: u64, depth: Depth, len: MateLen) {
        match self.map.get_mut(&path_key) {
            None => {
                self.map.insert(path_key, (depth, len));
            }
            Some(slot) => {
                if slot.1 != len {
                    // len が変われば上書き
                    *slot = (depth, len);
                } else if slot.0 <= depth {
                    // 同 len なら深い (= 千日手が浅くまで及ばない) 方を採る
                    slot.0 = depth;
                }
            }
        }
    }
    /// KH `Contains` (repetition_table.hpp:138)．
    /// 記録された table_len が探索 len 以上なら，その千日手はこの探索にも適用される．
    pub(super) fn contains(&self, path_key: u64, len: MateLen) -> Option<(Depth, MateLen)> {
        match self.map.get(&path_key) {
            Some(&(depth, table_len)) if table_len >= len => Some((depth, table_len)),
            _ => None,
        }
    }
}

/// LookUp/SetResult の文脈 (KH `Query` のメンバから生ポインタを除いたもの)．
/// 1 局面 (board_key, hand, depth, path_key) に対応し，cluster 先頭 index を cache する．
#[derive(Clone, Copy)]
pub(super) struct TtContext {
    start_idx: usize,
    path_key: u64,
    board_key: u64,
    hand: Hand,
    depth: Depth,
}

impl TtContext {
    #[inline]
    pub(super) fn board_key_hand(&self) -> (u64, Hand) {
        (self.board_key, self.hand)
    }
}

/// KH `tt::TranspositionTable` (regular + repetition を束ねる)．
pub(super) struct TranspositionTable {
    regular: RegularTable,
    rep: RepetitionTable,
    gc_count: u64,
}

impl TranspositionTable {
    pub(super) fn new(num_entries: usize) -> Self {
        TranspositionTable {
            regular: RegularTable::new(num_entries),
            rep: RepetitionTable::new(),
            gc_count: 0,
        }
    }
    /// 新規探索 (KH `NewSearch`): repetition table のみ消去 (通常表は GC に任せるが本移植では明示)．
    pub(super) fn new_search(&mut self) {
        self.rep.clear();
    }
    /// 通常表を全消去 (KH `Clear`)．
    pub(super) fn clear(&mut self) {
        self.regular.clear();
        self.rep.clear();
    }
    pub(super) fn capacity(&self) -> usize {
        self.regular.capacity()
    }

    /// これまでに実行した GC 回数 (診断用)．
    pub(super) fn gc_count(&self) -> u64 {
        self.gc_count
    }

    /// hashfull が `kExecuteGcHashRate`(=0.5) 以上なら GC を実行する (KH komoring_heights.cpp:157)．
    /// 通常テーブルのみで判断する (千日手テーブルは mid_v4 では無制限 `FxHashMap`)．
    /// テーブル満杯による `look_up` の O(cap) probe 退化を防ぐ．戻り値 = GC を実行したか．
    pub(super) fn maybe_collect_garbage(&mut self) -> bool {
        if self.regular.calculate_hash_rate() >= 0.5 {
            self.regular.collect_garbage(0.5);
            self.gc_count += 1;
            true
        } else {
            false
        }
    }

    /// KH `Query::LookUpParent` (ttquery.hpp:195) + `UpdateParentCandidate` (ttentry.hpp:412)．
    /// cluster の board_key 一致 entry を全走査し，優等/劣等局面から pn/dn 境界を合成する (KH 忠実)．
    /// 返す**親**は exact-hand entry を優先し，無い場合のみ cross-hand 推論 (`ApplyDeltaHand`) に
    /// fall back する．**KH は dominant bound の親で常に上書きするが，maou ではそれが false DAG 辺を
    /// 生み退行する** (29te 11,143→13,092)．exact 優先 + cross-hand fallback が正 (29te 11,286 /
    /// 39te 14.5M→13.4M / 62s→58s; いずれも STRICT sound)．pn/dn 境界は KH どおり cross-hand 合成のまま．
    pub(super) fn look_up_parent(
        &self,
        board_key: u64,
        hand: Hand,
    ) -> Option<(u64, Hand, PnDn, PnDn)> {
        use super::ttentry::hand_is_equal_or_superior;
        let mut pn: PnDn = 1;
        let mut dn: PnDn = 1;
        let mut exact_parent: Option<(u64, Hand)> = None;
        let mut cross_bk: u64 = 0;
        let mut cross_hand: Hand = NULL_HAND;
        let cap = self.regular.entries.len();
        let mut idx = self.regular.pointer_of(board_key);
        for _ in 0..cap {
            let e = &self.regular.entries[idx];
            if e.is_null() {
                break;
            }
            if e.is_for(board_key) {
                let entry_hand = e.get_hand();
                let eph = e.get_parent_hand();
                if entry_hand == hand && eph != NULL_HAND {
                    exact_parent = Some((e.parent_board_key(), eph));
                }
                let is_inferior = hand_is_equal_or_superior(entry_hand, hand);
                let is_superior = hand_is_equal_or_superior(hand, entry_hand);
                if is_inferior && e.pn() > pn {
                    pn = e.pn();
                    if eph != NULL_HAND && (cross_hand == NULL_HAND || pn > dn) {
                        cross_bk = e.parent_board_key();
                        cross_hand = apply_delta_hand(eph, entry_hand, hand);
                    }
                }
                if is_superior && e.dn() > dn {
                    dn = e.dn();
                    if eph != NULL_HAND && (cross_hand == NULL_HAND || dn > pn) {
                        cross_bk = e.parent_board_key();
                        cross_hand = apply_delta_hand(eph, entry_hand, hand);
                    }
                }
            }
            idx += 1;
            if idx == cap {
                idx = 0;
            }
        }
        if let Some((pbk, ph)) = exact_parent {
            Some((pbk, ph, pn, dn))
        } else if cross_hand == NULL_HAND {
            None
        } else {
            Some((cross_bk, cross_hand, pn, dn))
        }
    }

    /// 1 局面の Query 文脈を構築する (KH `BuildQuery`)．
    pub(super) fn build_query(
        &self,
        path_key: u64,
        board_key: u64,
        hand: Hand,
        depth: Depth,
    ) -> TtContext {
        TtContext {
            start_idx: self.regular.pointer_of(board_key),
            path_key,
            board_key,
            hand,
            depth,
        }
    }

    /// `ctx` の cluster 先頭 cache line を投機的に prefetch する (memory latency hiding)．
    ///
    /// TT は ~8M entry × ~72B = ~576MB と L3 を遥かに超え，`look_up` の cluster 先頭アクセスは
    /// ほぼ DRAM miss (~100ns) になる (= tt_lookup phase が memory-bound な理由)．child loop では
    /// `build_query` 直後・`look_up` 前に dom_path / v3_path の HashMap lookup が入るため，ここで
    /// prefetch を発行すると DRAM fetch がそれらと重なり latency を一部隠せる．**純粋な hint で
    /// 探索結果に影響しない** (search-invariant)．x86_64 以外では no-op (per-arch wheel ゆえ可搬)．
    #[inline]
    pub(super) fn prefetch(&self, ctx: &TtContext) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // start_idx は pointer_of で [0, len) に収まる (build_query が算出)．
            let ptr = self.regular.entries.as_ptr().add(ctx.start_idx) as *const i8;
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ctx;
    }

    /// **置換表の読み出し** (KH `Query::LookUp`, ttquery.hpp:117)．
    /// cluster を走査し，一致/優等/劣等局面から pn/dn を合成して `SearchResult` を返す．
    /// エントリが無ければ `eval`(InitialPnDn) を seed として MakeFirstVisit する (遅延評価)．
    pub(super) fn look_up<F: FnOnce() -> (PnDn, PnDn)>(
        &mut self,
        ctx: &TtContext,
        len: MateLen,
        does_have_old_child: &mut bool,
        eval: F,
    ) -> SearchResult {
        let mut pn: PnDn = 1;
        let mut dn: PnDn = 1;
        let mut amount: SearchAmount = 1;
        let mut found_exact = false;
        let mut sum_mask = BitSet64::full();

        let cap = self.regular.entries.len();
        let mut idx = ctx.start_idx;
        for _ in 0..cap {
            if self.regular.entries[idx].is_null() {
                break; // cluster 終端 (KH `!itr->IsNull()`)
            }
            if self.regular.entries[idx].is_for(ctx.board_key) {
                let matched = self.regular.entries[idx].look_up(
                    ctx.hand,
                    ctx.depth,
                    len,
                    &mut pn,
                    &mut dn,
                    does_have_old_child,
                );
                if matched {
                    let e = &self.regular.entries[idx];
                    amount = amount.max(e.amount());
                    if pn == 0 {
                        return SearchResult::make_final(
                            true,
                            e.get_hand(),
                            e.proven_len(),
                            amount,
                        );
                    } else if dn == 0 {
                        return SearchResult::make_final(
                            false,
                            e.get_hand(),
                            e.disproven_len(),
                            amount,
                        );
                    } else if e.get_hand() == ctx.hand {
                        // 一致局面 (exact)
                        if e.is_possible_repetition() {
                            if let Some((depth, table_len)) = self.rep.contains(ctx.path_key, len) {
                                return SearchResult::make_repetition(
                                    ctx.hand, table_len, amount, depth,
                                );
                            }
                        }
                        found_exact = true;
                        sum_mask = e.sum_mask();
                    }
                }
            }
            idx += 1;
            if idx == cap {
                idx = 0;
            }
        }

        // single-thread: ノイズなし (KH main thread)．
        if found_exact {
            return SearchResult::make_unknown(pn, dn, len, amount, sum_mask);
        }
        let (init_pn, init_dn) = eval();
        pn = pn.max(init_pn);
        dn = dn.max(init_dn);
        SearchResult::make_first_visit(pn, dn, len, amount)
    }

    /// **置換表の書き込み** (KH `Query::SetResult`, ttquery.hpp:248)．
    pub(super) fn set_result(
        &mut self,
        ctx: &TtContext,
        result: SearchResult,
        parent: (u64, Hand),
    ) {
        if result.pn() == 0 {
            self.set_final(ctx, true, result);
        } else if result.dn() == 0 {
            if result.is_repetition() {
                self.set_repetition(ctx, result);
            } else {
                self.set_final(ctx, false, result);
            }
        } else {
            self.set_unknown(ctx, result, parent);
        }
    }

    /// `board_key`+`hand` のエントリを探し，無ければ null slot を Init して返す (KH `FindOrCreate`)．
    fn find_or_create(&mut self, board_key: u64, hand: Hand) -> usize {
        let start = self.regular.pointer_of(board_key);
        let cap = self.regular.entries.len();
        let mut idx = start;
        for _ in 0..cap {
            if self.regular.entries[idx].is_null() {
                self.regular.entries[idx].init(board_key, hand);
                return idx;
            }
            if self.regular.entries[idx].is_for_hand(board_key, hand) {
                return idx;
            }
            idx += 1;
            if idx == cap {
                idx = 0;
            }
        }
        // テーブル満杯 (GC 未移植時の fallback; KH は GC で回避)．先頭を上書き．
        self.regular.entries[start].init(board_key, hand);
        start
    }

    /// 詰み/不詰の書き込み (KH `SetFinal<kIsProven>`)．**proof/disproof hand 下に格納** (cross-hand)．
    fn set_final(&mut self, ctx: &TtContext, proven: bool, result: SearchResult) {
        let hand = result.hand();
        let idx = self.find_or_create(ctx.board_key, hand);
        let len = result.len();
        let amount = result.amount();
        if proven {
            self.regular.entries[idx].update_proven(len, amount);
        } else {
            self.regular.entries[idx].update_disproven(len, amount);
        }
    }

    /// 千日手の書き込み (KH `SetRepetition`)．rep flag を立て path_key を rep table へ記録．
    fn set_repetition(&mut self, ctx: &TtContext, result: SearchResult) {
        let idx = self.find_or_create(ctx.board_key, ctx.hand);
        self.regular.entries[idx].set_possible_repetition();
        self.rep
            .insert(ctx.path_key, result.repetition_start(), result.len());
    }

    /// 探索中 (unknown) の書き込み (KH `SetUnknown`)．現局面 hand 下に格納．
    fn set_unknown(&mut self, ctx: &TtContext, result: SearchResult, parent: (u64, Hand)) {
        let idx = self.find_or_create(ctx.board_key, ctx.hand);
        self.regular.entries[idx].update_unknown(
            ctx.depth,
            result.pn(),
            result.dn(),
            result.amount(),
            result.sum_mask(),
            parent.0,
            parent.1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::ttentry::NULL_HAND;
    use super::*;
    use crate::types::HAND_KINDS;

    fn hand(pawns: u8) -> Hand {
        let mut h = [0u8; HAND_KINDS];
        h[0] = pawns;
        h
    }
    fn null_parent() -> (u64, Hand) {
        (0, NULL_HAND)
    }
    const BK: u64 = 0xABCD_1234_5678_9ABC;

    #[test]
    fn unknown_roundtrip() {
        let mut tt = TranspositionTable::new(1024);
        let q = tt.build_query(0x11, BK, hand(2), 5);
        // 初回 LookUp: エントリ無し => eval seed．
        let mut old = false;
        let r0 = tt.look_up(&q, MateLen::from_len(10), &mut old, || (4, 6));
        assert_eq!((r0.pn(), r0.dn()), (4, 6));
        assert!(r0.is_first_visit());

        // 書き込み後の LookUp で pn/dn/sum_mask を復元．
        let mut sm = BitSet64::full();
        sm.reset(2);
        let stored = SearchResult::make_unknown(7, 9, MateLen::from_len(10), 3, sm);
        tt.set_result(&q, stored, null_parent());

        let mut old2 = false;
        let r1 = tt.look_up(&q, MateLen::from_len(10), &mut old2, || (1, 1));
        assert_eq!((r1.pn(), r1.dn()), (7, 9));
        assert!(!r1.is_first_visit());
        assert!(!r1.sum_mask().test(2));
    }

    #[test]
    fn proven_len_aware_roundtrip() {
        let mut tt = TranspositionTable::new(1024);
        let q = tt.build_query(0x22, BK, hand(1), 3);
        // まず探索中 (unknown pn=3,dn=5) を格納し，その後 len=8 で詰みを格納する (実探索の順序)．
        tt.set_result(
            &q,
            SearchResult::make_unknown(3, 5, MateLen::from_len(8), 1, BitSet64::full()),
            null_parent(),
        );
        tt.set_result(
            &q,
            SearchResult::make_final(true, hand(1), MateLen::from_len(8), 2),
            null_parent(),
        );

        // len>=8 で LookUp => proven (len-aware: 探索 len が proven_len 以上)．
        let mut old = false;
        let r = tt.look_up(&q, MateLen::from_len(9), &mut old, || (1, 1));
        assert!(r.is_final());
        assert_eq!(r.pn(), 0);

        // len<8 では proven にならず，格納済 unknown pn/dn (3,5) を返す (eval seed は呼ばれない)．
        let mut old2 = false;
        let r2 = tt.look_up(&q, MateLen::from_len(5), &mut old2, || (99, 99));
        assert!(!r2.is_final());
        assert_eq!((r2.pn(), r2.dn()), (3, 5));
    }

    #[test]
    fn cross_hand_superior_lookup() {
        let mut tt = TranspositionTable::new(1024);
        // hand=1 で len=8 詰みを格納．
        let q1 = tt.build_query(0x33, BK, hand(1), 3);
        tt.set_result(
            &q1,
            SearchResult::make_final(true, hand(1), MateLen::from_len(8), 1),
            null_parent(),
        );
        // 同 board・hand=3 (優等) で LookUp => 優等局面として詰み．
        let q3 = tt.build_query(0x33, BK, hand(3), 3);
        let mut old = false;
        let r = tt.look_up(&q3, MateLen::from_len(10), &mut old, || (1, 1));
        assert!(r.is_final());
        assert_eq!(r.pn(), 0); // cross-hand proven
    }

    #[test]
    fn repetition_roundtrip() {
        let mut tt = TranspositionTable::new(1024);
        let q = tt.build_query(0x44, BK, hand(2), 7);
        // 千日手結果 (dn=0, rep_start=2) を格納．
        let rep = SearchResult::make_repetition(hand(2), MateLen::from_len(0), 1, 2);
        tt.set_result(&q, rep, null_parent());
        // LookUp は rep flag + rep_table.Contains で千日手を返す (len <= 記録 len)．
        let mut old = false;
        let r = tt.look_up(&q, MateLen::from_len(0), &mut old, || (1, 1));
        assert!(r.is_final());
        assert_eq!(r.dn(), 0);
        assert!(r.is_repetition());
        assert_eq!(r.repetition_start(), 2);
    }

    #[test]
    fn distinct_boards_do_not_collide() {
        let mut tt = TranspositionTable::new(1024);
        let qa = tt.build_query(0x1, 0xAAAA, hand(1), 1);
        let qb = tt.build_query(0x2, 0xBBBB, hand(1), 1);
        tt.set_result(
            &qa,
            SearchResult::make_unknown(5, 5, MateLen::from_len(10), 1, BitSet64::full()),
            null_parent(),
        );
        let mut old = false;
        // 別 board は first visit のまま (eval seed)．
        let rb = tt.look_up(&qb, MateLen::from_len(10), &mut old, || (2, 3));
        assert!(rb.is_first_visit());
        assert_eq!((rb.pn(), rb.dn()), (2, 3));
    }
}
