//! 探索ノード展開の core (δ/閾値/選択) 部分．
//!
//! 毎 sort/resort 後に正確な δ を再計算する (stale-sum 近道なし)．
//!
//! ## 本ファイルの範囲
//! - 構築済み `results/idx/sum_mask/move_evals` 上で動く δ/閾値/選択ロジック:
//!   pn/dn/φ/δ・front 閾値・δ 再計算・front/back/excluded-back resort・best child 更新．
//!   これらは movegen/TT に依存せず単体テスト可能．
//! - 構築 (movegen/TT LookUp/DML 配線/1 手詰) と win/lose result (proof/disproof hand) は
//!   movegen/TT 依存側で接続する．

use crate::board::Board;
use crate::dfpn::mate_len::{MateLen, MINUS1_MATE_LEN};
use crate::dfpn::search_result::{
    clamp_pn_dn, compare_results, BitSet64, Depth, Hand, Ordering3, PnDn, SearchAmount,
    SearchResult, K_INFINITE_PN_DN,
};
use crate::dfpn::tt::TtContext;
use crate::moves::Move;
use std::cmp::Ordering;

/// δ がこの値以上の子は max 集約へ落とす閾値 (= kInfinitePnDn / 1024)．
const K_FORCE_SUM_PN_DN: PnDn = K_INFINITE_PN_DN / 1024;

/// 親子間 pn/dn 差がこれを超えたら二重カウントでないとみなす閾値 (= 3 * pn/dn unit(2))．
pub(super) const K_ANCESTOR_SEARCH_THRESHOLD: PnDn = 6;

// ===== introsort (insertion sort + quicksort + heapsort fallback) =====
// idx を comparer で sort する際の完全同点手の置換を introsort 固有の順序で行う．
// `lt(a, b)` = a が b より前 (strict weak ordering)．小区間閾値 = 16．

/// insertion sort の guarded linear insert．
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

/// insertion sort (range [first, last))．
fn ks_insertion_sort<F: Fn(u32, u32) -> bool>(v: &mut [u32], first: usize, last: usize, lt: &F) {
    if first == last {
        return;
    }
    for i in (first + 1)..last {
        if lt(v[i], v[first]) {
            // 先頭より小: [first, i) を 1 つ後ろへずらし v[i] を先頭へ．
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

/// 3 値 (a, b, c) の中央値を `result` 位置へ swap する．
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

/// pivot 値で [first, last) を分割し境界を返す．
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

/// median-of-3 を pivot に [first+1, last) を分割する．
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

/// heapsort fallback (make_heap + sort_heap)．
/// depth_limit 枯渇時のみ (move 数 < 64 では発火しない)．
fn ks_heapsort<F: Fn(u32, u32) -> bool>(v: &mut [u32], first: usize, last: usize, lt: &F) {
    let n = last - first;
    if n < 2 {
        return;
    }
    // make_heap (max-heap) を [first, last) に構築する．
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
        // push val up (val を heap 内の正しい位置まで親方向へ持ち上げる)．
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

/// introsort 本体ループ (depth_limit 枯渇で heapsort へ切替)．
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

/// introsort + final insertion sort による sort．
fn std_sort<F: Fn(u32, u32) -> bool>(v: &mut [u32], lt: &F) {
    const THRESHOLD: usize = 16;
    let n = v.len();
    if n < 2 {
        return;
    }
    // depth_limit = 2 * floor(log2(n))．
    let depth = 2 * (usize::BITS - 1 - n.leading_zeros()) as i32;
    ks_introsort_loop(v, 0, n, depth, lt);
    // 仕上げ: 先頭 16 を guarded, 残りを unguarded insertion で整列する．
    if n > THRESHOLD {
        ks_insertion_sort(v, 0, THRESHOLD, lt);
        for i in THRESHOLD..n {
            ks_linear_insert(v, i, lt);
        }
    } else {
        ks_insertion_sort(v, 0, n, lt);
    }
}

/// `HAND` 計装: 指定 sfen prefix のノードの final hand (proof/disproof) を dump (診断専用)．
fn dump_hand(board: &Board, proven: bool, hand: &[u8; crate::types::HAND_KINDS]) {
    if let Some(prefix) = super::hand_prefix() {
        let sfen = board.sfen();
        if sfen.starts_with(prefix.as_str()) {
            let kind = if proven { "proof" } else { "disproof" };
            // hand 順: 歩 香 桂 銀 金 角 飛 (hand_index)．
            eprintln!(
                "HAND {} P{} L{} N{} S{} G{} B{} R{} sfen={}",
                kind, hand[0], hand[1], hand[2], hand[3], hand[4], hand[5], hand[6], sfen
            );
        }
    }
}

/// `HAND` 計装の子内訳版: 指定 sfen prefix のノードで集約前の各子 (move + 子 hand) を dump．
/// proof/disproof hand 再帰の駒種差 (R vs G) がどの子由来かを特定するため．
fn dump_hand_child(board: &Board, m: Move, hand: &[u8; crate::types::HAND_KINDS]) {
    if let Some(prefix) = super::hand_prefix() {
        if board.sfen().starts_with(prefix.as_str()) {
            eprintln!(
                "HAND   child {} P{} L{} N{} S{} G{} B{} R{}",
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

/// 二重カウントの分岐元の辺．
#[derive(Clone, Copy)]
pub(super) struct BranchRootEdge {
    /// 分岐元局面の (board_key, hand)．
    pub(super) branch_root: (u64, Hand),
    /// 分岐元の子 (合流路を遡った直前) の (board_key, hand)．
    pub(super) child: (u64, Hand),
    /// 分岐元が OR node なら true．
    pub(super) branch_root_is_or_node: bool,
}

/// 探索ノード展開の core．
pub(crate) struct LocalExpansion {
    or_node: bool,
    len: MateLen,
    /// 合法手．δ math では不使用 (best move と deferred 数のみ)．
    moves: Vec<Move>,
    /// 各手の brief evaluation．comparer の最終 tie-break．
    move_evals: Vec<i32>,
    /// 各手の TT query 文脈．親が子結果を TT へ書く際に使う．
    queries: Vec<TtContext>,
    /// 各手の現在の探索結果 (raw index 添字)．
    results: Vec<SearchResult>,
    /// 「良さげ順」に並べた raw index スタック．deferred 手は含まない．
    idx: Vec<u32>,
    /// idx 先頭の証明済 (φ=0) 手の個数．
    excluded_moves: usize,
    /// δ を和で計上する子の集合．
    sum_mask: BitSet64,
    sum_delta_except_best: PnDn,
    max_delta_except_best: PnDn,
    /// 古い (浅い) 子の結果を使ったか．TCA 判定用．
    does_have_old_child: bool,
    /// DML next chain (raw index → 次に復活すべき手, -1=無し)．
    dml_next: Vec<i32>,
    /// MultiPv (1 以上)．
    multi_pv: u32,
    /// 本ノードの (position_key, attacker_hand)．二重カウント除去で
    /// 分岐元一致判定に使う．`set_key_hand_pair` で構築後に設定する (未設定は (0, 空))．
    key_hand_pair: (u64, Hand),
    /// [案A] AND node の透過中合いマス集合 (build 時設定; 未設定/OR node は空)．AND-proven の
    /// mate_len 集計でこれらへの合駒 drop 子を無駄合いとして手数から除外する (child.len().sub(2))．
    chain_sqs: crate::bitboard::Bitboard,
}

impl LocalExpansion {
    /// 構築済みパーツから core を組み立てる (末尾で sort + δ 再計算)．
    /// movegen/TT 側の本構築または単体テストから呼ぶ．`idx` は deferred を除いた raw index 列．
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
            chain_sqs: crate::bitboard::Bitboard::EMPTY,
        };
        // 構築時点で既に φ=0 の手があれば excluded_moves を進める．
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

    /// 所有する 6 本の Vec を取り出して返す (node pop 時に buffer pool へ返却するため)．
    /// `self` を消費する (残りのフィールドは drop される)．順序は `from_parts` の引数順に対応:
    /// `(moves, move_evals, queries, results, idx, dml_next)`．
    #[allow(clippy::type_complexity)]
    pub(super) fn into_buffers(
        self,
    ) -> (
        Vec<Move>,
        Vec<i32>,
        Vec<TtContext>,
        Vec<SearchResult>,
        Vec<u32>,
        Vec<i32>,
    ) {
        (
            self.moves,
            self.move_evals,
            self.queries,
            self.results,
            self.idx,
            self.dml_next,
        )
    }

    // ---- comparer (SearchResult 比較 → 同点なら move eval) ----
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
    /// idx 全体を comparer で sort する．
    fn sort_idx(&mut self) {
        let mut idx = std::mem::take(&mut self.idx);
        if super::introsort_enabled() {
            // introsort で sort する．
            // >16 子ノードの完全同点 tie の置換が introsort 固有の並べ替えになる
            // (stable sort では movegen 順を保つため別の順序になる)．
            std_sort(&mut idx, &|a, b| self.compare_idx(a, b) == Ordering::Less);
        } else {
            // default = stable sort (movegen 順保持)．
            idx.sort_by(|&a, &b| self.compare_idx(a, b));
        }
        self.idx = idx;
    }

    // ---- accessors ----
    #[inline]
    pub(super) fn empty(&self) -> bool {
        self.idx.is_empty()
    }
    /// 現時点の最善手．non-final 前提．
    #[inline]
    pub(super) fn best_move(&self) -> Move {
        self.moves[self.idx[self.excluded_moves] as usize]
    }
    /// 現 best の raw index (= idx[excluded_moves])．親が子 query/結果を引く際に使う．
    #[inline]
    pub(super) fn front_raw(&self) -> usize {
        self.idx[self.excluded_moves] as usize
    }
    /// sort 済 idx の (move, pn, dn, eval, amount) 列 (診断専用)．
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
    /// raw index の TT query 文脈．
    #[inline]
    pub(super) fn query_at(&self, raw: usize) -> TtContext {
        self.queries[raw]
    }
    /// 現 best 子の seed/結果 (first-visit 時の初期値)．
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
    /// [案A] 透過中合いマス集合を設定する (build 後に呼ぶ; AND-proven mate_len 集計で使用)．
    #[inline]
    pub(super) fn set_chain_sqs(&mut self, chain_sqs: crate::bitboard::Bitboard) {
        self.chain_sqs = chain_sqs;
    }

    /// 本ノードの (board_key, hand) を設定する．build 後に呼ぶ．
    #[inline]
    pub(super) fn set_key_hand_pair(&mut self, kh: (u64, Hand)) {
        self.key_hand_pair = kh;
    }
    /// 本ノードの (board_key, hand)．
    #[inline]
    pub(super) fn key_hand_pair(&self) -> (u64, Hand) {
        self.key_hand_pair
    }
    /// 現 best の δ．二重カウント除去の ancestor 判定用．
    #[inline]
    pub(super) fn front_delta(&self) -> PnDn {
        self.front_result().delta(self.or_node)
    }
    #[inline]
    pub(super) fn is_or_node(&self) -> bool {
        self.or_node
    }

    /// 本ノードが二重カウントの分岐元 (`edge.branch_root`) なら，合流していた子を
    /// sum→max 集約へ落として δ を再計算する．
    /// 戻り値 = 本ノードが分岐元だったか (= ancestor walk を打ち切るべきか)．
    pub(super) fn resolve_double_count_if_branch_root(&mut self, edge: BranchRootEdge) -> bool {
        if edge.branch_root != self.key_hand_pair {
            return false;
        }
        // best の sum bit を落とす (get_delta が live 反映)．
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

    /// 分岐元と node 種別が異なれば続行 (false)．同種別で δ が best 子より閾値超に大きければ
    /// 二重カウントの影響は小さく打ち切る．
    pub(super) fn should_stop_ancestor_search(&self, branch_root_is_or_node: bool) -> bool {
        if self.or_node != branch_root_is_or_node {
            return false;
        }
        let delta_diff = self.get_delta().saturating_sub(self.front_delta());
        delta_diff > K_ANCESTOR_SEARCH_THRESHOLD
    }

    // ---- pn/dn ----
    /// φ値．
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
    /// δ値．**後回し手の deferred penalty 込み**．
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
        // 後回し手 1 つにつき 1/8 点減点 (小数切捨, 但し 1 を下回るなら 1)．
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

    /// 現局面の unknown 結果．proven/disproven は hand を要するため別途扱う．
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
    /// 現局面の探索結果．win/lose/unknown を判定し hand/len を付ける．
    ///
    /// `handset_enabled` が false のとき，proof/disproof hand は **現局面の攻め方持駒**
    /// (`attacker_hand`) をそのまま使う (HandSet 極小化を省略; sound だが非極小)．極小化
    /// (`before_hand`/`ProofHandSet`) は cross-hand TT が position-only board_key で有効化された
    /// 場合に偽証明を生むため，full hand を既定とする (full hand は sound)．
    /// `board` は現局面 (do_move 前の親局面)．
    pub(super) fn current_result(&self, board: &Board, depth: Depth) -> SearchResult {
        let attacker = if self.or_node {
            board.turn
        } else {
            board.turn.opponent()
        };
        let attacker_hand = board.hand[attacker.index()];
        let use_handset = super::handset_enabled();
        if self.get_phi() == 0 {
            // 手番 win: OR=詰み proven / AND=逃れ disproven．
            // 「最も良い手」は excluded に関係なく idx[0]．
            let front = self.results[self.idx[0] as usize];
            let mate_len = front.len().add(1);
            let amount = front
                .amount()
                .saturating_add(self.moves.len().max(1) as SearchAmount - 1);
            // win の hand: OR=before_hand(best, child 証明駒) / AND=child 反証駒 + 駒打ち補正．
            let hand = if use_handset {
                let best_move = self.moves[self.idx[0] as usize];
                if self.or_node {
                    // OR win = 詰み proven．full proof hand 診断時は極小化を切り full hand 格納．
                    if super::full_proof_hand_enabled() {
                        attacker_hand
                    } else {
                        crate::dfpn::proof_hand::before_hand(board, best_move, front.hand())
                    }
                } else {
                    crate::dfpn::proof_hand::and_node_escape_disproof(
                        board,
                        best_move,
                        front.hand(),
                        attacker,
                    )
                }
            } else {
                attacker_hand
            };
            // AND-case (逃れ disproven): best 逃れ子が千日手なら repetition で返す．
            // これを欠くと path 依存の千日手を position-keyed な通常 disproof として TT へ書き，
            // 別 path で過剰適用してしまう．
            if !self.or_node && front.is_repetition() && front.repetition_start() < depth {
                return SearchResult::make_repetition(
                    attacker_hand,
                    mate_len,
                    amount,
                    front.repetition_start(),
                );
            }
            dump_hand(board, self.or_node, &hand);
            SearchResult::make_final(self.or_node, hand, mate_len, amount)
        } else if self.get_delta() == 0 {
            // 手番 lose: OR=不詰 disproven / AND=詰み proven．
            // 千日手: 先頭子が repetition (rep_start < depth) なら伝播する (GHI soundness)．
            let front = self.results[self.idx[0] as usize];
            if front.is_repetition() && front.repetition_start() < depth {
                let mate_len = front.len().add(1);
                let amount = front
                    .amount()
                    .saturating_add(self.moves.len().max(1) as SearchAmount - 1);
                // 千日手は full attacker hand で返す．
                return SearchResult::make_repetition(
                    attacker_hand,
                    mate_len,
                    amount,
                    front.repetition_start(),
                );
            }
            // mate_len: OR=min(len, 最短子 len) / AND=最長子 len，+1．amount=max 子 amount + (子数-1)．
            // OR は `mate_len = len` 起点で min を取る (len cap を効かせ disproven_len の過大化を防ぎ，
            // len-aware cross-hand の過剰適用を避ける)．
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
                } else {
                    // AND-proven max-resistance．[案A] 透過中合い (chain マス) への合駒 drop 子は
                    // 無駄合いゆえ `sub(2)` (末尾 `+1` 後に `r.len()-1` = 取り返し後局面の手数 =
                    // 無駄合い-free)．非中合い子は通常の `r.len()`．len-budget credit と対で，AND の
                    // 手数が無駄合いで膨らまず find_shortest が真の最短へ収束する．
                    let cm = self.moves[ir as usize];
                    let contrib = if cm.is_drop() && self.chain_sqs.contains(cm.to_sq()) {
                        r.len().sub(2)
                    } else {
                        r.len()
                    };
                    if contrib > mate_len {
                        mate_len = contrib;
                    }
                }
            }
            amount = amount.saturating_add(self.moves.len().max(1) as SearchAmount - 1);
            // lose の hand: OR=DisproofHandSet(子の before_hand 反証駒 min)+remove_if /
            // AND=ProofHandSet(子の証明駒 max)+add_if．
            let hand = if use_handset {
                if self.or_node {
                    let mut set = crate::dfpn::proof_hand::DisproofHandSet::new();
                    for &ir in &self.idx {
                        let r = self.results[ir as usize];
                        let cm = self.moves[ir as usize];
                        let cdh = crate::dfpn::proof_hand::before_hand(board, cm, r.hand());
                        set.update(&cdh);
                    }
                    set.get(board)
                } else if super::full_proof_hand_enabled() {
                    // AND lose = 詰み proven．full proof hand 診断時は極小化を切り full hand 格納．
                    attacker_hand
                } else {
                    let mut set = crate::dfpn::proof_hand::ProofHandSet::new();
                    for &ir in &self.idx {
                        let ch = self.results[ir as usize].hand();
                        dump_hand_child(board, self.moves[ir as usize], &ch);
                        set.update(&ch);
                    }
                    set.get(board)
                }
            } else {
                attacker_hand
            };
            // OR lose=disproven(false) / AND lose=proven(true)．
            dump_hand(board, !self.or_node, &hand);
            SearchResult::make_final(!self.or_node, hand, mate_len.add(1), amount)
        } else {
            self.current_result_unknown()
        }
    }

    // ---- 閾値 ----
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
    /// 子に渡す (pn, dn) 閾値．
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

    /// 子へ渡す閾値計算の breakdown 文字列を返す (診断専用)．
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
        let mut kids = String::from(" kids:");
        for (k, &i_raw) in self.idx.iter().enumerate() {
            if k >= 16 {
                break;
            }
            let r = &self.results[i_raw as usize];
            let star = if self.sum_mask.test(i_raw as usize) {
                "*"
            } else {
                ""
            };
            kids.push_str(&format!(
                " {}:{}/{}{}",
                self.moves[i_raw as usize].to_usi(),
                r.pn(),
                r.dn(),
                star
            ));
        }
        format!(
            "or={} th=({},{}) thphi={} thdelta={} 2ndphi={} sumd={} maxd={} excl={} nmoves={} nidx={} best_pn={} best_dn={} -> cth=({},{}){}",
            self.or_node as i32, thpn, thdn, thphi, thdelta, self.get_second_phi(),
            self.sum_delta_except_best, self.max_delta_except_best, self.excluded_moves,
            self.moves.len(), self.idx.len(), best_pn, best_dn, cthpn, cthdn, kids
        )
    }

    /// δ 一時変数 (sum_delta_except_best / max_delta_except_best) を全再計算する．
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

    // ---- resort (lower_bound + rotate による 1 要素の再挿入) ----
    /// 先頭 [excluded_moves] を [excluded_moves+1, end) の整列済列へ挿入する．
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
    /// 末尾を [excluded_moves, end-1) の整列済列へ挿入する．
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
    /// [0, excluded_moves] の末尾を前半整列済列へ挿入する．
    fn resort_excluded_back(&mut self) {
        if self.excluded_moves > 0 {
            let val = self.idx[self.excluded_moves];
            let rel = self.idx[..self.excluded_moves]
                .partition_point(|&x| self.compare_idx(x, val) == Ordering::Less);
            self.idx[rel..self.excluded_moves + 1].rotate_right(1);
        }
    }

    /// 最善子の結果を反映する．TT 書き込みは単一所有権の都合で探索ループ側に委ねる．
    /// ここでは局所状態 (results/idx/sum_mask/δ 一時変数/excluded_moves) のみ更新する．
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

    /// std_sort (introsort) が >16 要素の完全同点で stable sort と**異なる**置換を
    /// 生むことを確認する (= introsort が tie を並べ替えている証拠; no-op バグ検出)．
    #[test]
    fn std_sort_reorders_ties_unstable() {
        // key = value/100 で比較 (多数が同点)．18 要素 (>16) で introsort path を通す．
        let keys: Vec<i32> = vec![
            500, 400, 600, 600, 400, 600, 200, 200, 200, 400, 600, 100, 600, 100, 600, 100, 600,
            600,
        ];
        let lt = |a: u32, b: u32| keys[a as usize] < keys[b as usize];
        let mut intro: Vec<u32> = (0..18u32).collect();
        std_sort(&mut intro, &lt);
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
            "std_sort behaved like stable sort (no tie reorder)"
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
        // excluded_moves(0) >= multi_pv(1)-1 で return し，front=proven 子のまま get_phi=0．
        let mut e = or_node(&[(2, 4), (5, 3), (9, 1)]);
        e.update_best_child(proven()); // best (pn2) 子が詰み
        assert_eq!(e.get_pn(), 0);
    }
}
