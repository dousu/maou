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

use super::mate_len::{MateLen, DEPTH_MAX_PLUS1_MATE_LEN, MINUS1_MATE_LEN};
use super::search_result::{
    clamp_pn_dn, compare_results, BitSet64, Depth, Ordering3, PnDn, SearchAmount, SearchResult,
    K_INFINITE_PN_DN,
};
use super::proof_hand::{before_hand, ProofHandSet};
use super::tt_v4::TtContext;
use crate::board::Board;
use crate::moves::Move;
use std::cmp::Ordering;

/// KH `detail::kForceSumPnDn` (local_expansion.hpp:35)．δ がこの値以上の子は max 集約へ落とす．
const K_FORCE_SUM_PN_DN: PnDn = K_INFINITE_PN_DN / 1024;

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
        match compare_results(self.or_node, self.results[i as usize], self.results[j as usize]) {
            Ordering3::Less => Ordering::Less,
            Ordering3::Greater => Ordering::Greater,
            Ordering3::Equivalent => self.move_evals[i as usize].cmp(&self.move_evals[j as usize]),
        }
    }
    /// idx_ 全体を comparer で sort する (KH `std::sort(idx_.begin(), idx_.end(), ...)`)．
    fn sort_idx(&mut self) {
        let mut idx = std::mem::take(&mut self.idx);
        idx.sort_by(|&a, &b| self.compare_idx(a, b));
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
        SearchResult::make_unknown(self.get_pn(), self.get_dn(), self.len, amount, self.sum_mask)
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
    /// **proof hand は KH `HandSet` の極小化を移植**: OR proof は子の証明駒を best_move 手前へ
    /// 逆算 (`before_hand`)，AND proof は全子の証明駒を要素 max 集約 (`ProofHandSet`)．極小証明駒は
    /// TT の cross-hand Superior/Inferior 再利用を効かせ breadth を抑える．disproof hand は当面
    /// 攻め方持駒で代用 (sound; mate 木では発火頻度が低く 29te node に効かないため後回し)．
    /// `board` は現局面 (do_move していない親局面)．
    pub(super) fn current_result(&self, board: &Board, depth: Depth) -> SearchResult {
        let attacker = if self.or_node {
            board.turn
        } else {
            board.turn.opponent()
        };
        let attacker_hand = board.hand[attacker.index()];
        if self.get_phi() == 0 {
            // 手番 win (KH GetWinResult): OR=詰み proven / AND=逃れ disproven．
            // 「最も良い手」は excluded に関係なく idx[0] (KH: FrontResult は使えない)．
            let front = self.results[self.idx[0] as usize];
            let mate_len = front.len().add(1);
            let amount = front
                .amount()
                .saturating_add(self.moves.len().max(1) as SearchAmount - 1);
            let hand = if self.or_node {
                // OR proof: 子の証明駒を best_move 手前へ逆算した極小証明駒 (KH BeforeHand)．
                before_hand(board, self.moves[self.idx[0] as usize], front.hand())
            } else {
                attacker_hand
            };
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
                return SearchResult::make_repetition(
                    attacker_hand,
                    mate_len,
                    amount,
                    front.repetition_start(),
                );
            }
            // mate_len: OR=最短子 len / AND=最長子 len，+1．amount=max 子 amount + (子数-1)．
            let mut mate_len = if self.or_node {
                DEPTH_MAX_PLUS1_MATE_LEN
            } else {
                MINUS1_MATE_LEN
            };
            let mut amount: SearchAmount = 1;
            // AND proof: 全子の証明駒を要素 max 集約 (KH HandSet{ProofHandTag})．
            let mut proof_set = ProofHandSet::new();
            for &ir in &self.idx {
                let r = self.results[ir as usize];
                amount = amount.max(r.amount());
                if self.or_node {
                    if r.len() < mate_len {
                        mate_len = r.len();
                    }
                } else {
                    if r.len() > mate_len {
                        mate_len = r.len();
                    }
                    proof_set.update(&r.hand());
                }
            }
            amount = amount.saturating_add(self.moves.len().max(1) as SearchAmount - 1);
            // OR lose=disproven(false; 攻め方持駒で代用) / AND lose=proven(true; 極小証明駒)．
            let hand = if self.or_node {
                attacker_hand
            } else {
                proof_set.get(board)
            };
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
        let (thphi, thdelta) = if self.or_node { (thpn, thdn) } else { (thdn, thpn) };
        let child_thphi = thphi.min(self.get_second_phi().saturating_add(1));
        let child_thdelta = self.new_thdelta_for_best_move(thdelta);
        if self.or_node {
            (child_thphi, child_thdelta)
        } else {
            (child_thdelta, child_thphi)
        }
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
            let rel = self.idx[lo..].partition_point(|&x| self.compare_idx(x, front) == Ordering::Less);
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
            let rel = self.idx[lo..hi].partition_point(|&x| self.compare_idx(x, back) == Ordering::Less);
            let itr = lo + rel;
            self.idx[itr..].rotate_right(1);
        }
    }
    /// [0, excluded_moves] の末尾を前半整列済列へ挿入 (KH `ResortExcludedBack`, :760)．
    fn resort_excluded_back(&mut self) {
        if self.excluded_moves > 0 {
            let val = self.idx[self.excluded_moves];
            let rel =
                self.idx[..self.excluded_moves].partition_point(|&x| self.compare_idx(x, val) == Ordering::Less);
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
                    self.sum_delta_except_best =
                        clamp_pn_dn(self.sum_delta_except_best + self.results[old_i_raw].delta(self.or_node));
                } else {
                    self.max_delta_except_best =
                        self.max_delta_except_best.max(self.results[old_i_raw].delta(self.or_node));
                }
                self.resort_front();
            }
            let new_i_raw = self.idx[self.excluded_moves] as usize;
            let new_delta = self.results[new_i_raw].delta(self.or_node);
            let new_is_sum_delta = self.sum_mask.test(new_i_raw);
            if new_is_sum_delta {
                self.sum_delta_except_best =
                    self.sum_delta_except_best.saturating_sub(new_delta);
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
