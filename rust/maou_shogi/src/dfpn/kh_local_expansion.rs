//! KH `tt::LocalExpansion` の忠実移植 (`local_expansion.hpp`) — **core (δ/閾値/選択) 部分**．
//!
//! mid_v4 (KH verbatim 再現) の探索ノード展開．maou 旧 `MidLocalExpansion` (local_expansion.rs) は
//! **resort 後に `RecalcDelta` しない** stale-sum 近道を取っていた (local_expansion.rs:410-415; 速度のため
//! co-adapt)．本移植は KH どおり **毎 sort/resort 後に正確な δ を再計算**する (近道なし)．
//!
//! ## 本ファイルの範囲 (3a)
//! - 構築済み `results_/idx_/sum_mask_/move_evals` 上で動く δ/閾値/選択ロジック:
//!   `GetPn/Dn/Phi/Delta`・`FrontPnDnThresholds`・`RecalcDelta`・`ResortFront/Back/ExcludedBack`・
//!   `UpdateBestChild`．これらは movegen/TT に依存せず単体テスト可能 (= stale-sum 近道が居た核心)．
//! - 構築 (MovePicker/TT LookUp/DML 配線/1 手詰) と `GetWin/LoseResult` (proof/disproof hand) は
//!   3b (movegen/TT 依存) で接続する．
//!
//! KH 定数: `kForceSumPnDn = kInfinitePnDn/1024` (local_expansion.hpp:35)．

use super::mate_len::{MateLen, MINUS1_MATE_LEN};
use super::search_result::{
    clamp_pn_dn, compare_results, BitSet64, Depth, Hand, Ordering3, PnDn, SearchAmount,
    SearchResult, K_INFINITE_PN_DN,
};
use super::tt_v4::TtContext;
use crate::board::Board;
use crate::moves::Move;
use std::cmp::Ordering;

/// KH `detail::kForceSumPnDn` (local_expansion.hpp:35)．δ がこの値以上の子は max 集約へ落とす．
const K_FORCE_SUM_PN_DN: PnDn = K_INFINITE_PN_DN / 1024;

/// KH `kAncestorSearchThreshold` (double_count_elimination.hpp:53) = 3 * kPnDnUnit(=2)．
/// 親子間 pn/dn 差がこれを超えたら二重カウントでないとみなす (mid_v4 は unit-2 scale)．
pub(super) const K_ANCESTOR_SEARCH_THRESHOLD: PnDn = 6;

// ===== libstdc++ `std::sort` (introsort) の忠実移植 (gcc 13 `bits/stl_algo.h`) =====
// KH `std::sort(idx_.begin(), idx_.end(), comp)` の完全同点手の置換を bit 単位で再現する．
// `lt(a, b)` = a が b より前 (strict weak ordering)．`_S_threshold = 16`．

/// libstdc++ `std::__insertion_sort` の guarded linear insert (`__unguarded_linear_insert`)．
/// `last` 位置の要素を `[.., last)` の整列済列へ後ろから挿入する (`lt(val, prev)` で停止 = stable)．
fn ks_linear_insert<F: Fn(u32, u32) -> bool>(v: &mut [u32], last: usize, lt: &F) {
    let val = v[last];
    let mut l = last;
    while l > 0 && lt(val, v[l - 1]) {
        v[l] = v[l - 1];
        l -= 1;
    }
    v[l] = val;
}

/// libstdc++ `__insertion_sort(first, last)` (range [first, last))．
fn ks_insertion_sort<F: Fn(u32, u32) -> bool>(v: &mut [u32], first: usize, last: usize, lt: &F) {
    if first == last {
        return;
    }
    for i in (first + 1)..last {
        if lt(v[i], v[first]) {
            // 先頭より小: [first, i) を 1 つ後ろへずらし v[i] を先頭へ (`__move_backward`)．
            let val = v[i];
            let mut j = i;
            while j > first {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[first] = val;
        } else {
            ks_linear_insert(v, i, lt);
        }
    }
}

/// libstdc++ `__move_median_to_first(result, a, b, c)`: 3 値中央値を `result` へ swap．
fn ks_median_to_first<F: Fn(u32, u32) -> bool>(
    v: &mut [u32],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    lt: &F,
) {
    if lt(v[a], v[b]) {
        if lt(v[b], v[c]) {
            v.swap(result, b);
        } else if lt(v[a], v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if lt(v[a], v[c]) {
        v.swap(result, a);
    } else if lt(v[b], v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

/// libstdc++ `__unguarded_partition(first, last, pivot)`: pivot 値で [first, last) を分割し境界を返す．
fn ks_partition<F: Fn(u32, u32) -> bool>(
    v: &mut [u32],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    lt: &F,
) -> usize {
    loop {
        while lt(v[first], v[pivot]) {
            first += 1;
        }
        last -= 1;
        while lt(v[pivot], v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

/// libstdc++ `__unguarded_partition_pivot(first, last)`: median-of-3 を pivot に [first+1, last) を分割．
fn ks_partition_pivot<F: Fn(u32, u32) -> bool>(
    v: &mut [u32],
    first: usize,
    last: usize,
    lt: &F,
) -> usize {
    let mid = first + (last - first) / 2;
    ks_median_to_first(v, first, first + 1, mid, last - 1, lt);
    ks_partition(v, first + 1, last, first, lt)
}

/// libstdc++ heapsort fallback (`__partial_sort(first, last, last)` = make_heap + sort_heap)．
/// depth_limit 枯渇時のみ (move 数 < 64 では発火しないが忠実性のため移植)．
fn ks_heapsort<F: Fn(u32, u32) -> bool>(v: &mut [u32], first: usize, last: usize, lt: &F) {
    let n = last - first;
    if n < 2 {
        return;
    }
    // make_heap (max-heap): [first, last)．
    let sift_down = |v: &mut [u32], mut hole: usize, len: usize, lt: &F| {
        let val = v[first + hole];
        let mut child = hole;
        while child < (len - 1) / 2 {
            child = 2 * child + 2;
            if lt(v[first + child], v[first + child - 1]) {
                child -= 1;
            }
            v[first + hole] = v[first + child];
            hole = child;
        }
        if len % 2 == 0 && child == (len - 2) / 2 {
            child = 2 * child + 1;
            v[first + hole] = v[first + child];
            hole = child;
        }
        // push val up (libstdc++ __push_heap)．
        let top = 0usize;
        let mut idx = hole;
        while idx > top {
            let parent = (idx - 1) / 2;
            if lt(v[first + parent], val) {
                v[first + idx] = v[first + parent];
                idx = parent;
            } else {
                break;
            }
        }
        v[first + idx] = val;
    };
    let mut parent = n / 2;
    loop {
        parent -= 1;
        sift_down(v, parent, n, lt);
        if parent == 0 {
            break;
        }
    }
    // sort_heap: 末尾へ最大値を運ぶ．
    let mut end = n;
    while end > 1 {
        end -= 1;
        v.swap(first, first + end);
        sift_down(v, 0, end, lt);
    }
}

/// libstdc++ `__introsort_loop(first, last, depth_limit)`．
fn ks_introsort_loop<F: Fn(u32, u32) -> bool>(
    v: &mut [u32],
    first: usize,
    mut last: usize,
    mut depth: i32,
    lt: &F,
) {
    const THRESHOLD: usize = 16;
    while last - first > THRESHOLD {
        if depth == 0 {
            ks_heapsort(v, first, last, lt);
            return;
        }
        depth -= 1;
        let cut = ks_partition_pivot(v, first, last, lt);
        ks_introsort_loop(v, cut, last, depth, lt);
        last = cut;
    }
}

/// libstdc++ `std::sort(first, last, comp)` の忠実移植 (introsort + final insertion sort)．
fn kh_std_sort<F: Fn(u32, u32) -> bool>(v: &mut [u32], lt: &F) {
    const THRESHOLD: usize = 16;
    let n = v.len();
    if n < 2 {
        return;
    }
    // depth_limit = 2 * floor(log2(n))  (libstdc++ `std::__lg(n) * 2`)．
    let depth = 2 * (usize::BITS - 1 - n.leading_zeros()) as i32;
    ks_introsort_loop(v, 0, n, depth, lt);
    // __final_insertion_sort: 先頭 16 を guarded, 残りを unguarded で挿入．
    if n > THRESHOLD {
        ks_insertion_sort(v, 0, THRESHOLD, lt);
        for i in THRESHOLD..n {
            ks_linear_insert(v, i, lt);
        }
    } else {
        ks_insertion_sort(v, 0, n, lt);
    }
}

/// `V4HAND` 計装: 指定 sfen prefix のノードの final hand (proof/disproof) を dump (KH `KHHAND` と突合)．
fn dump_v4hand(board: &Board, proven: bool, hand: &[u8; crate::types::HAND_KINDS]) {
    if let Some(prefix) = super::mid_v4::v4hand_prefix() {
        let sfen = board.sfen();
        if sfen.starts_with(prefix.as_str()) {
            let kind = if proven { "proof" } else { "disproof" };
            // hand 順: 歩 香 桂 銀 金 角 飛 (hand_index)．
            eprintln!(
                "V4HAND {} P{} L{} N{} S{} G{} B{} R{} sfen={}",
                kind, hand[0], hand[1], hand[2], hand[3], hand[4], hand[5], hand[6], sfen
            );
        }
    }
}

/// `V4HAND` 計装の子内訳版: 指定 sfen prefix のノードで集約前の各子 (move + 子 hand) を dump．
/// proof/disproof hand 再帰の駒種差 (R vs G) がどの子由来かを特定するため．
fn dump_v4hand_child(board: &Board, m: Move, hand: &[u8; crate::types::HAND_KINDS]) {
    if let Some(prefix) = super::mid_v4::v4hand_prefix() {
        if board.sfen().starts_with(prefix.as_str()) {
            eprintln!(
                "V4HAND   child {} P{} L{} N{} S{} G{} B{} R{}",
                m.to_usi(),
                hand[0],
                hand[1],
                hand[2],
                hand[3],
                hand[4],
                hand[5],
                hand[6]
            );
        }
    }
}

/// KH `BranchRootEdge` (double_count_elimination.hpp:74)．二重カウントの分岐元の辺．
#[derive(Clone, Copy)]
pub(super) struct BranchRootEdge {
    /// 分岐元局面の (board_key, hand)．
    pub(super) branch_root: (u64, Hand),
    /// 分岐元の子 (合流路を遡った直前) の (board_key, hand)．
    pub(super) child: (u64, Hand),
    /// 分岐元が OR node なら true．
    pub(super) branch_root_is_or_node: bool,
}

/// KH `tt::LocalExpansion` の core (local_expansion.hpp:112)．
pub(super) struct LocalExpansion {
    or_node: bool,
    len: MateLen,
    /// 合法手 (KH `mp_`)．δ math では不使用 (BestMove と deferred 数のみ)．
    moves: Vec<Move>,
    /// 各手の MoveBriefEvaluation (KH `mp_[i].value`)．comparer の最終 tie-break．
    move_evals: Vec<i32>,
    /// 各手の TT query 文脈 (KH `queries_`)．親が子結果を TT へ書く際に使う．
    queries: Vec<TtContext>,
    /// 各手の現在の探索結果 (KH `results_`, raw index 添字)．
    results: Vec<SearchResult>,
    /// 「良さげ順」に並べた raw index スタック (KH `idx_`)．deferred 手は含まない．
    idx: Vec<u32>,
    /// idx_ 先頭の証明済 (φ=0) 手の個数 (KH `excluded_moves_`)．
    excluded_moves: usize,
    /// δ を和で計上する子の集合 (KH `sum_mask_`)．
    sum_mask: BitSet64,
    sum_delta_except_best: PnDn,
    max_delta_except_best: PnDn,
    /// 古い (浅い) 子の結果を使ったか (KH `does_have_old_child_`)．TCA 判定用．
    does_have_old_child: bool,
    /// DML next chain (raw index → 次に復活すべき手, -1=無し) (KH `delayed_move_list_.Next`)．
    dml_next: Vec<i32>,
    /// MultiPv (KH `multi_pv_`; 1 以上)．
    multi_pv: u32,
    /// 本ノードの (position_key, attacker_hand) (KH `key_hand_pair_`)．EliminateDoubleCount で
    /// 分岐元一致判定に使う．`set_key_hand_pair` で構築後に設定する (未設定は (0, 空))．
    key_hand_pair: (u64, Hand),
}

impl LocalExpansion {
    /// 構築済みパーツから core を組み立てる (KH ctor 末尾の `sort + RecalcDelta` 相当)．
    /// 3b の本構築 (movegen/TT) または単体テストから呼ぶ．`idx` は deferred を除いた raw index 列．
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_parts(
        or_node: bool,
        len: MateLen,
        moves: Vec<Move>,
        move_evals: Vec<i32>,
        queries: Vec<TtContext>,
        results: Vec<SearchResult>,
        idx: Vec<u32>,
        sum_mask: BitSet64,
        does_have_old_child: bool,
        dml_next: Vec<i32>,
        multi_pv: u32,
    ) -> Self {
        let mut e = LocalExpansion {
            or_node,
            len,
            moves,
            move_evals,
            queries,
            results,
            idx,
            excluded_moves: 0,
            sum_mask,
            sum_delta_except_best: 0,
            max_delta_except_best: 0,
            does_have_old_child,
            dml_next,
            multi_pv: multi_pv.max(1),
            key_hand_pair: (0, [0u8; crate::types::HAND_KINDS]),
        };
        // 構築時点で既に φ=0 の手があれば excluded_moves を進める (KH ctor CHILD_LOOP_END)．
        for k in 0..e.idx.len() {
            let raw = e.idx[k] as usize;
            if e.results[raw].phi(or_node) == 0 {
                if e.excluded_moves >= (e.multi_pv as usize).saturating_sub(1) {
                    break;
                }
                e.excluded_moves += 1;
            }
        }
        e.sort_idx();
        e.recalc_delta();
        e
    }

    // ---- comparer (KH MakeComparer: SearchResultComparer → 同点なら mp_.value) ----
    #[inline]
    fn compare_idx(&self, i: u32, j: u32) -> Ordering {
        match compare_results(
            self.or_node,
            self.results[i as usize],
            self.results[j as usize],
        ) {
            Ordering3::Less => Ordering::Less,
            Ordering3::Greater => Ordering::Greater,
            Ordering3::Equivalent => self.move_evals[i as usize].cmp(&self.move_evals[j as usize]),
        }
    }
    /// idx_ 全体を comparer で sort する (KH `std::sort(idx_.begin(), idx_.end(), ...)`)．
    fn sort_idx(&mut self) {
        let mut idx = std::mem::take(&mut self.idx);
        if super::mid_v4::khorder_enabled() && super::mid_v4::introsort_enabled() {
            // KH tie-break 忠実化: libstdc++ `std::sort`(introsort) を bit 単位で再現する (V4_INTROSORT 時)．
            // 入力順 (V4_KHORDER で movegen 順を KH に一致) + comparator + sort algorithm が揃い，
            // >16 子ノードの完全同点 tie の置換が KH と一致する (stable sort では movegen 順を保ち乖離)．
            kh_std_sort(&mut idx, &|a, b| self.compare_idx(a, b) == Ordering::Less);
        } else {
            // default = stable sort (movegen 順保持)．KH を std::stable_sort にした診断ビルドと tie 順が揃う．
            idx.sort_by(|&a, &b| self.compare_idx(a, b));
        }
        self.idx = idx;
    }

    // ---- accessors ----
    #[inline]
    pub(super) fn empty(&self) -> bool {
        self.idx.is_empty()
    }
    /// 現時点の最善手 (KH `BestMove`)．`!IsFinal()` 前提．
    #[inline]
    pub(super) fn best_move(&self) -> Move {
        self.moves[self.idx[self.excluded_moves] as usize]
    }
    /// 現 best の raw index (= idx[excluded_moves])．親が子 query/結果を引く際に使う．
    #[inline]
    pub(super) fn front_raw(&self) -> usize {
        self.idx[self.excluded_moves] as usize
    }
    /// sort 済 idx の (move, pn, dn, eval, amount) 列 (KH KHSEL ダンプと突合する divergence-hunting 用)．
    pub(super) fn trace_children(&self) -> Vec<(Move, PnDn, PnDn, i32, u32)> {
        self.idx
            .iter()
            .map(|&r| {
                let res = &self.results[r as usize];
                (
                    self.moves[r as usize],
                    res.pn(),
                    res.dn(),
                    self.move_evals[r as usize],
                    res.amount(),
                )
            })
            .collect()
    }
    /// raw index の TT query 文脈 (KH `queries_[i_raw]`)．
    #[inline]
    pub(super) fn query_at(&self, raw: usize) -> TtContext {
        self.queries[raw]
    }
    /// 現 best 子の seed/結果 (first-visit 時の初期値; KH `FrontResult`)．
    #[inline]
    pub(super) fn front_result(&self) -> SearchResult {
        self.results[self.idx[self.excluded_moves] as usize]
    }
    #[inline]
    pub(super) fn does_have_old_child(&self) -> bool {
        self.does_have_old_child
    }
    #[inline]
    pub(super) fn front_is_first_visit(&self) -> bool {
        self.front_result().is_first_visit()
    }
    #[inline]
    pub(super) fn front_sum_mask(&self) -> BitSet64 {
        self.front_result().sum_mask()
    }
    #[inline]
    pub(super) fn len(&self) -> MateLen {
        self.len
    }
    /// 本ノードの (board_key, hand) を設定する (KH ctor `key_hand_pair_`)．build 後に呼ぶ．
    #[inline]
    pub(super) fn set_key_hand_pair(&mut self, kh: (u64, Hand)) {
        self.key_hand_pair = kh;
    }
    /// 本ノードの (board_key, hand) (KH `GetBoardKeyHandPair`)．
    #[inline]
    pub(super) fn key_hand_pair(&self) -> (u64, Hand) {
        self.key_hand_pair
    }
    /// 現 best の δ (KH `FrontResult().Delta(or_node)`)．EliminateDoubleCount の ancestor 判定用．
    #[inline]
    pub(super) fn front_delta(&self) -> PnDn {
        self.front_result().delta(self.or_node)
    }
    /// 本ノードの現 δ (KH `GetDelta`)．EliminateDoubleCount の ShouldStopAncestorSearch 用．
    #[inline]
    pub(super) fn delta(&self) -> PnDn {
        self.get_delta()
    }
    #[inline]
    pub(super) fn is_or_node(&self) -> bool {
        self.or_node
    }

    /// KH `ResolveDoubleCountIfBranchRoot` (local_expansion.hpp:478)．本ノードが二重カウントの
    /// 分岐元 (`edge.branch_root`) なら，合流していた子を sum→max 集約へ落として δ を再計算する．
    /// 戻り値 = 本ノードが分岐元だったか (= ancestor walk を打ち切るべきか)．
    pub(super) fn resolve_double_count_if_branch_root(&mut self, edge: BranchRootEdge) -> bool {
        if edge.branch_root != self.key_hand_pair {
            return false;
        }
        // KH: sum_mask_.Reset(idx_.front())．best の sum bit を落とす (get_delta が live 反映)．
        if let Some(&front) = self.idx.first() {
            self.sum_mask.reset(front as usize);
        }
        for k in (self.excluded_moves + 1)..self.idx.len() {
            let i_raw = self.idx[k] as usize;
            if self.queries[i_raw].board_key_hand() == edge.child {
                if self.sum_mask.test(i_raw) {
                    self.sum_mask.reset(i_raw);
                    self.recalc_delta();
                }
                break;
            }
        }
        true
    }

    /// KH `ShouldStopAncestorSearch` (local_expansion.hpp:502)．分岐元と node 種別が異なれば
    /// 続行 (false)．同種別で δ が best 子より閾値超に大きければ二重カウントの影響は小さく打ち切る．
    pub(super) fn should_stop_ancestor_search(&self, branch_root_is_or_node: bool) -> bool {
        if self.or_node != branch_root_is_or_node {
            return false;
        }
        let delta_diff = self.get_delta().saturating_sub(self.front_delta());
        delta_diff > K_ANCESTOR_SEARCH_THRESHOLD
    }

    // ---- pn/dn (KH GetPn/GetDn/GetPhi/GetDelta) ----
    /// φ値 (KH `GetPhi`, local_expansion.hpp:533)．
    fn get_phi(&self) -> PnDn {
        let front_phi = if self.excluded_moves < self.idx.len() {
            self.front_result().phi(self.or_node)
        } else {
            K_INFINITE_PN_DN
        };
        if front_phi >= K_INFINITE_PN_DN && self.excluded_moves > 0 {
            return 0;
        }
        front_phi
    }
    /// δ値 (KH `GetDelta`, local_expansion.hpp:548)．**後回し手の deferred penalty 込み**．
    fn get_delta(&self) -> PnDn {
        if self.idx.is_empty() {
            return 0;
        }
        let best_raw = self.idx[self.excluded_moves] as usize;
        let best_delta = self.results[best_raw].delta(self.or_node);
        let mut sum_delta = self.sum_delta_except_best;
        let mut max_delta = self.max_delta_except_best;
        if self.sum_mask.test(best_raw) {
            sum_delta = clamp_pn_dn(sum_delta + best_delta);
        } else {
            max_delta = max_delta.max(best_delta);
        }
        // 後回し手 1 つにつき 1/8 点減点 (小数切捨, 但し 1 を下回るなら 1)．KH local_expansion.hpp:567-571．
        if self.moves.len() > self.idx.len() {
            let penalty = ((self.moves.len() - self.idx.len()) / 8).max(1) as PnDn;
            sum_delta = clamp_pn_dn(sum_delta + penalty);
        }
        let raw_delta = clamp_pn_dn(sum_delta + max_delta);
        if self.excluded_moves > 0 && raw_delta == 0 {
            return K_INFINITE_PN_DN;
        }
        raw_delta
    }
    #[inline]
    fn get_pn(&self) -> PnDn {
        if self.or_node {
            self.get_phi()
        } else {
            self.get_delta()
        }
    }
    #[inline]
    fn get_dn(&self) -> PnDn {
        if self.or_node {
            self.get_delta()
        } else {
            self.get_phi()
        }
    }

    /// 現局面の unknown 結果 (KH `GetUnknownResult`, :727)．proven/disproven は 3b (hand 必要)．
    pub(super) fn current_result_unknown(&self) -> SearchResult {
        let amount = self
            .front_result()
            .amount()
            .saturating_add(self.moves.len() as SearchAmount - 1);
        SearchResult::make_unknown(
            self.get_pn(),
            self.get_dn(),
            self.len,
            amount,
            self.sum_mask,
        )
    }
    /// 現局面が proven か (φ=0)．3b の current_result 分岐用．
    #[inline]
    pub(super) fn is_proven(&self) -> bool {
        self.get_phi() == 0
    }
    /// 現局面が disproven/repetition か (δ=0)．
    #[inline]
    pub(super) fn is_disproven(&self) -> bool {
        self.get_delta() == 0
    }

    /// 現局面の探索結果 (KH `CurrentResult`, :332)．win/lose/unknown を判定し hand/len を付ける．
    ///
    /// proof/disproof hand は **現局面の攻め方持駒** (`attacker_hand`) をそのまま使う．これは KH の
    /// `HandSet` 極小化を**意図的に省略**している (sound だが非極小)．極小化 (`before_hand`/`ProofHandSet`)
    /// は cross-hand TT が position-only board_key で有効化された途端 **mate-39 の偽証明**を生む
    /// (= 移植に bug あり; full hand は sound かつ KH 並ノード)．`board` は現局面 (do_move 前の親局面)．
    pub(super) fn current_result(&self, board: &Board, depth: Depth) -> SearchResult {
        let attacker = if self.or_node {
            board.turn
        } else {
            board.turn.opponent()
        };
        let attacker_hand = board.hand[attacker.index()];
        let use_handset = super::mid_v4::handset_enabled();
        if self.get_phi() == 0 {
            // 手番 win (KH GetWinResult): OR=詰み proven / AND=逃れ disproven．
            // 「最も良い手」は excluded に関係なく idx[0] (KH: FrontResult は使えない)．
            let front = self.results[self.idx[0] as usize];
            let mate_len = front.len().add(1);
            let amount = front
                .amount()
                .saturating_add(self.moves.len().max(1) as SearchAmount - 1);
            // KH GetWinResult の hand: OR=BeforeHand(best, child 証明駒) / AND=child 反証駒 + 駒打ち補正．
            let hand = if use_handset {
                let best_move = self.moves[self.idx[0] as usize];
                if self.or_node {
                    super::proof_hand::before_hand(board, best_move, front.hand())
                } else {
                    super::proof_hand::and_node_escape_disproof(
                        board,
                        best_move,
                        front.hand(),
                        attacker,
                    )
                }
            } else {
                attacker_hand
            };
            // KH GetWinResult AND-case (逃れ disproven): best 逃れ子が千日手なら MakeRepetition で返す
            // (local_expansion.hpp:667)．これを欠くと path 依存の千日手を position-keyed な通常 disproof
            // として TT へ書き，別 path で過剰適用してしまう (Emplace#224 で 7g8h を誤って disproof 済に)．
            if !self.or_node && front.is_repetition() && front.repetition_start() < depth {
                return SearchResult::make_repetition(
                    attacker_hand,
                    mate_len,
                    amount,
                    front.repetition_start(),
                );
            }
            dump_v4hand(board, self.or_node, &hand);
            SearchResult::make_final(self.or_node, hand, mate_len, amount)
        } else if self.get_delta() == 0 {
            // 手番 lose (KH GetLoseResult): OR=不詰 disproven / AND=詰み proven．
            // 千日手: 先頭子が repetition (rep_start < depth) なら伝播する (GHI soundness)．
            let front = self.results[self.idx[0] as usize];
            if front.is_repetition() && front.repetition_start() < depth {
                let mate_len = front.len().add(1);
                let amount = front
                    .amount()
                    .saturating_add(self.moves.len().max(1) as SearchAmount - 1);
                // KH: 千日手は n.OrHand() (full attacker hand) で返す．
                return SearchResult::make_repetition(
                    attacker_hand,
                    mate_len,
                    amount,
                    front.repetition_start(),
                );
            }
            // mate_len: OR=min(len_, 最短子 len) / AND=最長子 len，+1．amount=max 子 amount + (子数-1)．
            // KH GetLoseResult: OR は `mate_len = len_` 起点で min を取る (len_ cap; 旧実装は DEPTH_MAX_PLUS1
            // 起点で cap を欠き disproven_len が過大 → len-aware cross-hand が過剰適用していた)．
            let mut mate_len = if self.or_node {
                self.len
            } else {
                MINUS1_MATE_LEN
            };
            let mut amount: SearchAmount = 1;
            for &ir in &self.idx {
                let r = self.results[ir as usize];
                amount = amount.max(r.amount());
                if self.or_node {
                    if r.len() < mate_len {
                        mate_len = r.len();
                    }
                } else if r.len() > mate_len {
                    mate_len = r.len();
                }
            }
            amount = amount.saturating_add(self.moves.len().max(1) as SearchAmount - 1);
            // KH GetLoseResult の hand: OR=DisproofHandSet(子の before_hand 反証駒 min)+remove_if /
            // AND=ProofHandSet(子の証明駒 max)+add_if．
            let hand = if use_handset {
                if self.or_node {
                    let mut set = super::proof_hand::DisproofHandSet::new();
                    for &ir in &self.idx {
                        let r = self.results[ir as usize];
                        let cm = self.moves[ir as usize];
                        let cdh = super::proof_hand::before_hand(board, cm, r.hand());
                        set.update(&cdh);
                    }
                    set.get(board)
                } else {
                    let mut set = super::proof_hand::ProofHandSet::new();
                    for &ir in &self.idx {
                        let ch = self.results[ir as usize].hand();
                        dump_v4hand_child(board, self.moves[ir as usize], &ch);
                        set.update(&ch);
                    }
                    set.get(board)
                }
            } else {
                attacker_hand
            };
            // OR lose=disproven(false) / AND lose=proven(true)．
            dump_v4hand(board, !self.or_node, &hand);
            SearchResult::make_final(!self.or_node, hand, mate_len.add(1), amount)
        } else {
            self.current_result_unknown()
        }
    }

    // ---- 閾値 (KH GetSecondPhi/NewThdeltaForBestMove/FrontPnDnThresholds) ----
    fn get_second_phi(&self) -> PnDn {
        if self.idx.len() <= self.excluded_moves + 1 {
            return K_INFINITE_PN_DN;
        }
        self.results[self.idx[self.excluded_moves + 1] as usize].phi(self.or_node)
    }
    fn new_thdelta_for_best_move(&self, thdelta: PnDn) -> PnDn {
        let mut delta_except_best = self.sum_delta_except_best;
        if self.moves.len() > self.idx.len() {
            let penalty = ((self.moves.len() - self.idx.len()) / 8).max(1) as PnDn;
            delta_except_best = clamp_pn_dn(delta_except_best + penalty);
        }
        if self.sum_mask.test(self.idx[self.excluded_moves] as usize) {
            delta_except_best = clamp_pn_dn(delta_except_best + self.max_delta_except_best);
        }
        if thdelta >= delta_except_best {
            clamp_pn_dn(thdelta - delta_except_best)
        } else {
            0
        }
    }
    /// 子に渡す (pn, dn) 閾値 (KH `FrontPnDnThresholds`, :428)．
    pub(super) fn front_pn_dn_thresholds(&self, thpn: PnDn, thdn: PnDn) -> (PnDn, PnDn) {
        let (thphi, thdelta) = if self.or_node {
            (thpn, thdn)
        } else {
            (thdn, thpn)
        };
        let child_thphi = thphi.min(self.get_second_phi().saturating_add(1));
        let child_thdelta = self.new_thdelta_for_best_move(thdelta);
        if self.or_node {
            (child_thphi, child_thdelta)
        } else {
            (child_thdelta, child_thphi)
        }
    }

    /// V4THX: KH `FrontPnDnThresholds` の KHTHX dump と同形式の breakdown 文字列を返す (診断専用)．
    pub(super) fn thx_breakdown(&self, thpn: PnDn, thdn: PnDn) -> String {
        let (thphi, thdelta) = if self.or_node {
            (thpn, thdn)
        } else {
            (thdn, thpn)
        };
        let (cthpn, cthdn) = self.front_pn_dn_thresholds(thpn, thdn);
        let (best_pn, best_dn) = if self.excluded_moves < self.idx.len() {
            let r = &self.results[self.idx[self.excluded_moves] as usize];
            (r.pn(), r.dn())
        } else {
            (0, 0)
        };
        format!(
            "or={} th=({},{}) thphi={} thdelta={} 2ndphi={} sumd={} maxd={} excl={} nmoves={} nidx={} best_pn={} best_dn={} -> cth=({},{})",
            self.or_node as i32, thpn, thdn, thphi, thdelta, self.get_second_phi(),
            self.sum_delta_except_best, self.max_delta_except_best, self.excluded_moves,
            self.moves.len(), self.idx.len(), best_pn, best_dn, cthpn, cthdn
        )
    }

    /// δ 一時変数を全再計算する (KH `RecalcDelta`, :615)．**maou 旧近道はこれを resort 後に省いていた**．
    fn recalc_delta(&mut self) {
        self.sum_delta_except_best = 0;
        self.max_delta_except_best = 0;
        for k in (self.excluded_moves + 1)..self.idx.len() {
            let raw = self.idx[k] as usize;
            let delta_i = self.results[raw].delta(self.or_node);
            if self.sum_mask.test(raw) {
                self.sum_delta_except_best = clamp_pn_dn(self.sum_delta_except_best + delta_i);
            } else {
                self.max_delta_except_best = self.max_delta_except_best.max(delta_i);
            }
        }
    }

    // ---- resort (KH ResortFront/Back/ExcludedBack; lower_bound + rotate) ----
    /// 先頭 [excluded_moves] を [excluded_moves+1, end) の整列済列へ挿入 (KH `ResortFront`, :736)．
    fn resort_front(&mut self) {
        if self.idx.len() > self.excluded_moves + 1 {
            let front = self.idx[self.excluded_moves];
            let lo = self.excluded_moves + 1;
            let rel =
                self.idx[lo..].partition_point(|&x| self.compare_idx(x, front) == Ordering::Less);
            let itr = lo + rel;
            self.idx[self.excluded_moves..itr].rotate_left(1);
        }
    }
    /// 末尾を [excluded_moves, end-1) の整列済列へ挿入 (KH `ResortBack`, :748)．
    fn resort_back(&mut self) {
        if self.idx.len() > self.excluded_moves + 1 {
            let back = self.idx[self.idx.len() - 1];
            let lo = self.excluded_moves;
            let hi = self.idx.len() - 1;
            let rel =
                self.idx[lo..hi].partition_point(|&x| self.compare_idx(x, back) == Ordering::Less);
            let itr = lo + rel;
            self.idx[itr..].rotate_right(1);
        }
    }
    /// [0, excluded_moves] の末尾を前半整列済列へ挿入 (KH `ResortExcludedBack`, :760)．
    fn resort_excluded_back(&mut self) {
        if self.excluded_moves > 0 {
            let val = self.idx[self.excluded_moves];
            let rel = self.idx[..self.excluded_moves]
                .partition_point(|&x| self.compare_idx(x, val) == Ordering::Less);
            self.idx[rel..self.excluded_moves + 1].rotate_right(1);
        }
    }

    /// 最善子の結果を反映する (KH `UpdateBestChild`, :347)．TT 書き込み (KH の query.SetResult) は
    /// 単一所有権の都合で探索ループ側 (module 5) に委ねる．ここでは局所状態のみ更新．
    pub(super) fn update_best_child(&mut self, search_result: SearchResult) {
        let old_i_raw = self.idx[self.excluded_moves] as usize;
        self.results[old_i_raw] = search_result;
        if !search_result.is_final() && search_result.delta(self.or_node) >= K_FORCE_SUM_PN_DN {
            self.sum_mask.reset(old_i_raw);
        }

        if search_result.phi(self.or_node) == 0 {
            // 後から見つかった手の方が良いかもしれないので前半を整列し直す
            self.resort_excluded_back();
            if self.excluded_moves >= (self.multi_pv as usize).saturating_sub(1) {
                return;
            }
            self.excluded_moves += 1;
            if self.excluded_moves >= self.moves.len() {
                return;
            }
        }

        let has_dml_next = self.dml_next[old_i_raw] >= 0;
        if search_result.is_final() && has_dml_next {
            if search_result.delta(self.or_node) == 0 {
                self.resort_front();
            }
            // 後回しにした手を復活させる
            let mut curr = self.dml_next[old_i_raw];
            while curr >= 0 {
                let cur_raw = curr as usize;
                self.idx.push(cur_raw as u32);
                self.resort_back();
                if self.results[cur_raw].delta(self.or_node) > 0 {
                    break; // まだ結論の出ていない子がいた
                }
                curr = self.dml_next[cur_raw];
            }
            self.recalc_delta();
        } else {
            if search_result.phi(self.or_node) > 0 {
                // 旧 best が delta_except_best へ加わるので差分計算
                let old_is_sum_delta = self.sum_mask.test(old_i_raw);
                if old_is_sum_delta {
                    self.sum_delta_except_best = clamp_pn_dn(
                        self.sum_delta_except_best + self.results[old_i_raw].delta(self.or_node),
                    );
                } else {
                    self.max_delta_except_best = self
                        .max_delta_except_best
                        .max(self.results[old_i_raw].delta(self.or_node));
                }
                self.resort_front();
            }
            let new_i_raw = self.idx[self.excluded_moves] as usize;
            let new_delta = self.results[new_i_raw].delta(self.or_node);
            let new_is_sum_delta = self.sum_mask.test(new_i_raw);
            if new_is_sum_delta {
                self.sum_delta_except_best = self.sum_delta_except_best.saturating_sub(new_delta);
            } else if new_delta < self.max_delta_except_best {
                // new_best を抜いても max_delta_except_best は不変
            } else {
                // max_delta の再計算が必要
                self.recalc_delta();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// kh_std_sort (libstdc++ introsort 移植) が >16 要素の完全同点で stable sort と**異なる**置換を
    /// 生むことを確認する (= introsort が tie を並べ替えている証拠; no-op バグ検出)．
    #[test]
    fn kh_std_sort_reorders_ties_unstable() {
        // key = value/100 で比較 (多数が同点)．18 要素 (>16) で introsort path を通す．
        let keys: Vec<i32> = vec![
            500, 400, 600, 600, 400, 600, 200, 200, 200, 400, 600, 100, 600, 100, 600, 100, 600,
            600,
        ];
        let lt = |a: u32, b: u32| keys[a as usize] < keys[b as usize];
        let mut intro: Vec<u32> = (0..18u32).collect();
        kh_std_sort(&mut intro, &lt);
        let mut stable: Vec<u32> = (0..18u32).collect();
        stable.sort_by(|&a, &b| keys[a as usize].cmp(&keys[b as usize]));
        // sort の正しさ (key 昇順) は両者満たすこと．
        for w in intro.windows(2) {
            assert!(
                keys[w[0] as usize] <= keys[w[1] as usize],
                "introsort not sorted"
            );
        }
        // 完全同点の置換が stable と異なる (introsort=unstable) ことを確認．
        assert_ne!(
            intro, stable,
            "kh_std_sort behaved like stable sort (no tie reorder)"
        );
    }

    fn unk(pn: PnDn, dn: PnDn) -> SearchResult {
        SearchResult::make_unknown(pn, dn, MateLen::from_len(10), 1, BitSet64::full())
    }
    fn proven() -> SearchResult {
        SearchResult::make_final(true, [0; crate::types::HAND_KINDS], MateLen::from_len(3), 1)
    }

    /// n 子・全 sum・deferred/DML 無しの OR node を構築する helper．
    fn or_node(child_pn_dn: &[(PnDn, PnDn)]) -> LocalExpansion {
        let n = child_pn_dn.len();
        let moves: Vec<Move> = (0..n as u32).map(Move).collect();
        let evals: Vec<i32> = (0..n as i32).collect();
        let results: Vec<SearchResult> = child_pn_dn.iter().map(|&(p, d)| unk(p, d)).collect();
        let idx: Vec<u32> = (0..n as u32).collect();
        LocalExpansion::from_parts(
            true,
            MateLen::from_len(10),
            moves,
            evals,
            vec![],
            results,
            idx,
            BitSet64::full(),
            false,
            vec![-1; n],
            1,
        )
    }

    #[test]
    fn or_aggregate_min_pn_sum_dn() {
        // OR: pn = min child pn, dn = sum child dn．
        let e = or_node(&[(5, 3), (2, 4), (9, 1)]);
        assert_eq!(e.get_pn(), 2); // min pn
        assert_eq!(e.get_dn(), 3 + 4 + 1); // sum dn (best の dn + sum_delta_except_best)
    }

    #[test]
    fn front_thresholds_or_second_phi() {
        // best=pn2, second=pn5 => child_thphi = min(thpn, second+1) = min(100, 6) = 6．
        let e = or_node(&[(5, 3), (2, 4), (9, 1)]);
        let (cthpn, cthdn) = e.front_pn_dn_thresholds(100, 1000);
        assert_eq!(cthpn, 6); // min(100, second_phi(5)+1)
                              // child_thdelta = thdn - sum_delta_except_best(他子 dn = 3+1=4) = 1000-4 = 996．
        assert_eq!(cthdn, 996);
    }

    #[test]
    fn update_best_child_recalcs_delta_after_resort() {
        // best=pn2 を pn8 へ更新 => 別の子が best になり，sum_delta を再計算する必要がある．
        let mut e = or_node(&[(5, 3), (2, 4), (9, 1)]);
        assert_eq!(e.get_pn(), 2);
        // best (idx 先頭 = pn2 の子) を (8, 7) へ更新．
        e.update_best_child(unk(8, 7));
        // 新 best = pn5 の子．OR pn = min = 5．
        assert_eq!(e.get_pn(), 5);
        // dn = sum of all child dn (更新後): 旧 best 7, pn5 子 3, pn9 子 1 => best(3) + others(7+1)=11．
        assert_eq!(e.get_dn(), 3 + 7 + 1);
    }

    #[test]
    fn deferred_penalty_inflates_delta() {
        // moves=5 だが idx=3 (2 手 deferred) => penalty = max(2/8,1)=1 が sum_delta へ加算．
        let moves: Vec<Move> = (0..5u32).map(Move).collect();
        let evals: Vec<i32> = (0..5i32).collect();
        let results: Vec<SearchResult> = vec![unk(2, 4), unk(5, 3), unk(9, 1)];
        let idx: Vec<u32> = vec![0, 1, 2];
        let e = LocalExpansion::from_parts(
            true,
            MateLen::from_len(10),
            moves,
            evals,
            vec![],
            results,
            idx,
            BitSet64::full(),
            false,
            vec![-1; 3],
            1,
        );
        // dn = best dn(4) + others(3+1) + deferred penalty(1) = 9．
        assert_eq!(e.get_dn(), 4 + 3 + 1 + 1);
    }

    #[test]
    fn proven_child_makes_or_node_proven() {
        // OR node + multi_pv=1: 子が 1 つ proven (φ=0) になれば node 自体が proven．
        // KH: excluded_moves(0) >= multi_pv(1)-1 で return し，front=proven 子のまま get_phi=0．
        let mut e = or_node(&[(2, 4), (5, 3), (9, 1)]);
        e.update_best_child(proven()); // best (pn2) 子が詰み
        assert_eq!(e.get_pn(), 0);
        assert!(e.is_proven());
    }
}
