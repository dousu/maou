//! mid_v2: KH SearchImpl 風の本格的 re-design (v0.82.0 開始)．
//!
//! ## 設計目標
//!
//! 既存 mid() (3000+ 行) の代替として，KH `komoring_heights.cpp::SearchImpl`
//! のロジックを Rust に忠実移植する．既存 mid() は保持し並走．
//!
//! 解決を狙う問題: 29te 算式 gap 6300×．bottleneck は AND/OR scan coverage
//! 1-2% の **scattershot 探索**．KH の `LocalExpansion` は per-defender
//! 持続 commitment を `excluded_moves_++` + `idx_` 再 sort + sum_mask で実現．
//!
//! ## アーキテクチャ
//!
//! ```text
//! mid_v2(node, thpn, thdn, len, inc_flag) ->  SearchResult {
//!     loop {
//!         curr = local.current_result()
//!         if curr.pn >= thpn || curr.dn >= thdn: break
//!         best_move = local.best_move()
//!         (cthpn, cthdn) = local.front_thresholds(thpn, thdn)
//!         do_move
//!         child_result = if is_first_visit { local.front_result() }
//!                        else { mid_v2(child, cthpn, cthdn, len-1, inc_flag) }
//!         undo_move
//!         local.update_best_child(child_result)
//!     }
//!     curr
//! }
//! ```
//!
//! ## 段階導入
//!
//! Phase 1 (current): スケルトンと API stub．
//! Phase 2: `MidLocalExpansion` の主要 method 実装 (BestMove, UpdateBestChild,
//!          CurrentResult, FrontPnDnThresholds)．
//! Phase 3: `mid_v2()` 関数の主ループ実装．OR/AND 統合．
//! Phase 4: tsume_5 / 39te Mate(15) / 29te で検証．既存 mid() の機能
//!          (DML, HandSet, SNDA, K-M, chain) を順次移植．

#![allow(dead_code)]  // Phase 1: 全 API stub，使用は Phase 3 から

use crate::moves::Move;

/// KH `SearchResult` 相当．現局面の探索結果を表す．
#[derive(Debug, Clone, Copy)]
pub(super) struct MidSearchResult {
    /// proof number (= 攻め方の証明難度)．0 で win 確定．
    pub pn: u32,
    /// disproof number (= 守り方の反証難度)．0 で lose 確定．
    pub dn: u32,
    /// 現局面に到達してから消費した nodes 数 (TCA 用診断)．
    pub amount: u32,
    /// 詰み手数情報 (proven の場合のみ意味あり)．
    pub mate_distance: u16,
    /// 初回訪問フラグ (TCA の `inc_flag--` 判定用)．
    pub is_first_visit: bool,
}

impl MidSearchResult {
    pub(super) fn new_unknown(pn: u32, dn: u32) -> Self {
        Self {
            pn,
            dn,
            amount: 0,
            mate_distance: 0,
            is_first_visit: true,
        }
    }

    pub(super) fn new_win(mate_distance: u16) -> Self {
        Self {
            pn: 0,
            dn: u32::MAX,
            amount: 1,
            mate_distance,
            is_first_visit: false,
        }
    }

    pub(super) fn new_lose(mate_distance: u16) -> Self {
        Self {
            pn: u32::MAX,
            dn: 0,
            amount: 1,
            mate_distance,
            is_first_visit: false,
        }
    }

    pub(super) fn is_final(&self) -> bool {
        self.pn == 0 || self.dn == 0
    }

    /// `or_node = true` で attacker 視点 phi=pn，false で defender 視点 phi=dn．
    pub(super) fn phi(&self, or_node: bool) -> u32 {
        if or_node { self.pn } else { self.dn }
    }

    pub(super) fn delta(&self, or_node: bool) -> u32 {
        if or_node { self.dn } else { self.pn }
    }
}

/// KH `LocalExpansion` 相当．mid_v2 の per-call state を保持し，子展開を管理する．
///
/// 現フレームの children list を sorted index で管理し，BestMove() で
/// 次に展開する手を提供する．子の探索結果は UpdateBestChild() で内部 cache に
/// 反映し，re-sort で次回の BestMove() を更新する．
pub(super) struct MidLocalExpansion {
    /// OR ノードか AND ノードか．
    or_node: bool,
    /// 全合法手 (位置生成順)．mp_ の Rust 対応．
    moves: Vec<Move>,
    /// sorted index array (phi 昇順)．`idx_[excluded_moves_]` が現在の "best"．
    idx: Vec<u32>,
    /// proven 化済み children のカウント．BestMove() で `idx_[excluded_moves_]` を返す．
    excluded_moves: usize,
    /// 各 child の現在の (pn, dn, amount)．`UpdateBestChild` で更新．
    results: Vec<MidSearchResult>,
    /// 各 child が sum aggregation 対象か (true = sum，false = max)．BitSet64 相当．
    sum_mask: u64,
    /// キャッシュ: best 以外の delta sum．
    sum_delta_except_best: u32,
    /// キャッシュ: best 以外の delta max．
    max_delta_except_best: u32,
    /// `DoesHaveOldChild`: 子の中に old visit (= TT cache 経由) があるか．
    has_old_child: bool,
}

impl MidLocalExpansion {
    /// 新規 expansion を構築．
    ///
    /// `moves`: 全合法手．`initial_results`: 各手の初期 (pn, dn)．
    pub(super) fn new(
        or_node: bool,
        moves: Vec<Move>,
        initial_results: Vec<MidSearchResult>,
    ) -> Self {
        let n = moves.len();
        let mut idx: Vec<u32> = (0..n as u32).collect();
        // 初期 sort: phi 昇順
        idx.sort_by_key(|&i| initial_results[i as usize].phi(or_node));
        let sum_mask = if n >= 64 { u64::MAX } else { (1u64 << n).wrapping_sub(1) };
        let mut expansion = Self {
            or_node,
            moves,
            idx,
            excluded_moves: 0,
            results: initial_results,
            sum_mask,
            sum_delta_except_best: 0,
            max_delta_except_best: 0,
            has_old_child: false,
        };
        expansion.recalc_delta();
        expansion
    }

    /// 空か (= empty)．
    pub(super) fn empty(&self) -> bool {
        self.idx.is_empty() || self.excluded_moves >= self.idx.len()
    }

    /// BestMove: `idx_[excluded_moves_]` の手を返す．
    pub(super) fn best_move(&self) -> Move {
        let raw = self.idx[self.excluded_moves] as usize;
        self.moves[raw]
    }

    /// FrontResult: 現在 best とされている child の SearchResult．
    pub(super) fn front_result(&self) -> &MidSearchResult {
        let raw = self.idx[self.excluded_moves] as usize;
        &self.results[raw]
    }

    /// FrontIsFirstVisit: 現 best が初回訪問か．
    pub(super) fn front_is_first_visit(&self) -> bool {
        self.front_result().is_first_visit
    }

    /// DoesHaveOldChild: TCA トリガ判定．
    pub(super) fn does_have_old_child(&self) -> bool {
        self.has_old_child
    }

    /// CurrentResult: 現フレームの集約 (pn, dn) を計算し SearchResult として返す．
    ///
    /// or_node なら pn = min(child_pn) | sum(child_pn), dn = sum(child_dn) (要素別)．
    /// and_node なら pn = sum(child_pn), dn = min(child_dn)．
    /// (実際には phi/delta で抽象化して同じロジックで処理)．
    pub(super) fn current_result(&self) -> MidSearchResult {
        let phi = self.get_phi();
        let delta = self.get_delta();
        // pn/dn に戻す
        let (pn, dn) = if self.or_node { (phi, delta) } else { (delta, phi) };
        if pn == 0 {
            MidSearchResult::new_win(0) // mate_distance は別途計算
        } else if dn == 0 {
            MidSearchResult::new_lose(0)
        } else {
            MidSearchResult::new_unknown(pn, dn)
        }
    }

    /// 子の結果を反映．search_result は recurse から戻ってきた値．
    pub(super) fn update_best_child(&mut self, search_result: MidSearchResult) {
        let old_raw = self.idx[self.excluded_moves] as usize;
        self.results[old_raw] = search_result;

        // sum_mask 動的切替: delta が kForceSumPnDn (= 4M) 以上なら max 集約に
        const FORCE_SUM_THRESHOLD: u32 = u32::MAX / 1024;
        if !search_result.is_final() && search_result.delta(self.or_node) >= FORCE_SUM_THRESHOLD {
            self.sum_mask &= !(1u64 << old_raw);
        }

        // proven 化した child を excluded_moves に進める
        if search_result.phi(self.or_node) == 0 {
            self.excluded_moves += 1;
        }

        // re-sort: 現 best の評価が変わったので idx_ を更新
        // 完全 sort でなく，excluded_moves 以降のみ sort
        let or_node = self.or_node;
        let results = &self.results;
        if self.excluded_moves < self.idx.len() {
            self.idx[self.excluded_moves..]
                .sort_by_key(|&i| results[i as usize].phi(or_node));
        }

        self.recalc_delta();
    }

    /// 子に渡す phi/delta 閾値を計算する．
    ///
    /// KH の `FrontPnDnThresholds`:
    /// - child_thphi = min(thphi, second_phi + 1)
    /// - child_thdelta = thdelta - sum_delta_except_best - [if sum_mask[best] max_delta_except_best]
    pub(super) fn front_pn_dn_thresholds(&self, thpn: u32, thdn: u32) -> (u32, u32) {
        let (thphi, thdelta) = if self.or_node { (thpn, thdn) } else { (thdn, thpn) };
        let second_phi = self.get_second_phi();
        let child_thphi = thphi.min(second_phi.saturating_add(1));
        let child_thdelta = self.new_thdelta_for_best_move(thdelta);
        if self.or_node {
            (child_thphi, child_thdelta)
        } else {
            (child_thdelta, child_thphi)
        }
    }

    /// 2 番目の child の phi 値．selection tie-break / threshold 計算用．
    fn get_second_phi(&self) -> u32 {
        let second_idx_pos = self.excluded_moves + 1;
        if second_idx_pos >= self.idx.len() {
            return u32::MAX;
        }
        let second_raw = self.idx[second_idx_pos] as usize;
        self.results[second_raw].phi(self.or_node)
    }

    /// best 以外の child の合計 delta を考慮して，best に与える child_thdelta を計算する．
    fn new_thdelta_for_best_move(&self, thdelta: u32) -> u32 {
        let mut delta_except_best = self.sum_delta_except_best;
        if (self.sum_mask >> (self.idx[self.excluded_moves] as u64)) & 1 == 1 {
            delta_except_best = delta_except_best.saturating_add(self.max_delta_except_best);
        }
        if thdelta >= delta_except_best {
            thdelta.saturating_sub(delta_except_best)
        } else {
            0
        }
    }

    /// 現フレームの phi: OR ノードなら min(pn), AND ノードなら min(dn)．
    fn get_phi(&self) -> u32 {
        if self.empty() {
            return 0;
        }
        self.front_result().phi(self.or_node)
    }

    /// 現フレームの delta: OR ノードなら sum(dn)，AND ノードなら sum(pn)．
    fn get_delta(&self) -> u32 {
        if self.empty() {
            return 0;
        }
        let raw = self.idx[self.excluded_moves] as usize;
        let best_delta = self.results[raw].delta(self.or_node);
        let mut total = best_delta;
        if (self.sum_mask >> (raw as u64)) & 1 == 1 {
            total = total.saturating_add(self.sum_delta_except_best);
        } else {
            // best が sum_mask off の場合は best の delta は含めず max_except でカバー
            total = self.sum_delta_except_best.saturating_add(self.max_delta_except_best);
        }
        total
    }

    /// sum_delta_except_best と max_delta_except_best を再計算する．
    fn recalc_delta(&mut self) {
        self.sum_delta_except_best = 0;
        self.max_delta_except_best = 0;
        for &i_raw in self.idx.iter().skip(self.excluded_moves + 1) {
            let d = self.results[i_raw as usize].delta(self.or_node);
            if (self.sum_mask >> (i_raw as u64)) & 1 == 1 {
                self.sum_delta_except_best = self.sum_delta_except_best.saturating_add(d);
            } else {
                if d > self.max_delta_except_best {
                    self.max_delta_except_best = d;
                }
            }
        }
    }
}

/// 簡易 mid_v2 (DfPnSolver と独立して単体実行できる skeleton)．
///
/// Phase 2: TT 不使用，children を直接展開し proven 判定を行うミニ実装．
/// 主目的は MidLocalExpansion の loop 構造を verify すること．
///
/// Phase 3 で DfPnSolver の TT/board 統合版に発展させる．
pub(super) struct MiniMidContext {
    /// max_recursion depth．無限ループ防止．
    pub max_depth: u32,
    /// 探索した node 数 (mid_v2 entry 回数)．
    pub nodes_searched: u64,
    /// 上限 nodes (テスト用)．
    pub max_nodes: u64,
}

impl MiniMidContext {
    pub(super) fn new(max_depth: u32, max_nodes: u64) -> Self {
        Self {
            max_depth,
            nodes_searched: 0,
            max_nodes,
        }
    }
}

/// mid_v2 のコアループ (KH SearchImpl 移植版)．
///
/// `expansion`: 現フレームの展開状態 (子は外部から注入)．
/// `expand_child`: child の MidLocalExpansion を生成する callback．
/// `recurse`: child フレーム recursion を行う callback (返り値 = child result)．
///
/// 戻り値: 現フレームの集約 SearchResult．
pub(super) fn search_impl<E, R>(
    expansion: &mut MidLocalExpansion,
    thpn: u32,
    thdn: u32,
    _len: u32,
    inc_flag: &mut u32,
    ctx: &mut MiniMidContext,
    mut expand_child: E,
    mut recurse: R,
) -> MidSearchResult
where
    E: FnMut(Move) -> MidLocalExpansion,
    R: FnMut(&mut MidLocalExpansion, u32, u32, &mut u32) -> MidSearchResult,
{
    let orig_thpn = thpn;
    let orig_thdn = thdn;
    let orig_inc_flag = *inc_flag;
    let mut cur_thpn = thpn;
    let mut cur_thdn = thdn;

    let mut curr = expansion.current_result();
    if expansion.does_have_old_child() {
        *inc_flag = inc_flag.saturating_add(1);
    }
    if *inc_flag > 0 {
        extend_search_threshold(&mut cur_thpn, &mut cur_thdn, &curr);
    }

    ctx.nodes_searched += 1;
    while curr.pn < cur_thpn && curr.dn < cur_thdn {
        if ctx.nodes_searched >= ctx.max_nodes {
            break;
        }
        if expansion.empty() {
            break;
        }
        let best_move = expansion.best_move();
        let is_first = expansion.front_is_first_visit();
        let (child_thpn, child_thdn) = expansion.front_pn_dn_thresholds(cur_thpn, cur_thdn);

        let mut child_expansion = expand_child(best_move);
        let child_result = if is_first {
            let initial = child_expansion.current_result();
            if *inc_flag > 0 {
                *inc_flag -= 1;
            }
            if initial.pn >= child_thpn || initial.dn >= child_thdn {
                initial
            } else {
                recurse(&mut child_expansion, child_thpn, child_thdn, inc_flag)
            }
        } else {
            recurse(&mut child_expansion, child_thpn, child_thdn, inc_flag)
        };

        expansion.update_best_child(child_result);
        curr = expansion.current_result();

        // TCA で延長した threshold は orig に戻し，inc_flag > 0 なら再延長
        cur_thpn = orig_thpn;
        cur_thdn = orig_thdn;
        if *inc_flag > 0 {
            extend_search_threshold(&mut cur_thpn, &mut cur_thdn, &curr);
        } else if *inc_flag == 0 && orig_inc_flag > 0 {
            break;
        }
    }

    *inc_flag = (*inc_flag).min(orig_inc_flag);
    curr
}

/// KH `ExtendSearchThreshold` の Rust 移植．
/// curr.pn, curr.dn を見て thpn/thdn を少し拡張する．
fn extend_search_threshold(thpn: &mut u32, thdn: &mut u32, curr: &MidSearchResult) {
    // KH の実装: 現在の pn/dn より少し大きく
    const EXTEND_DENOM: u32 = 4;
    if curr.pn < u32::MAX {
        let extra = curr.pn / EXTEND_DENOM + 1;
        *thpn = thpn.saturating_add(extra).min(u32::MAX - 1);
    }
    if curr.dn < u32::MAX {
        let extra = curr.dn / EXTEND_DENOM + 1;
        *thdn = thdn.saturating_add(extra).min(u32::MAX - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::Move;

    #[test]
    fn test_mid_search_result_basics() {
        let win = MidSearchResult::new_win(15);
        assert!(win.is_final());
        assert_eq!(win.pn, 0);
        assert_eq!(win.mate_distance, 15);

        let lose = MidSearchResult::new_lose(0);
        assert!(lose.is_final());
        assert_eq!(lose.dn, 0);

        let unk = MidSearchResult::new_unknown(100, 200);
        assert!(!unk.is_final());
    }

    #[test]
    fn test_or_node_phi_delta() {
        let r = MidSearchResult::new_unknown(50, 100);
        // OR (or_node=true): phi=pn=50, delta=dn=100
        assert_eq!(r.phi(true), 50);
        assert_eq!(r.delta(true), 100);
        // AND (or_node=false): phi=dn=100, delta=pn=50
        assert_eq!(r.phi(false), 100);
        assert_eq!(r.delta(false), 50);
    }

    #[test]
    fn test_local_expansion_construct() {
        let moves = vec![Move(1), Move(2), Move(3)];
        let results = vec![
            MidSearchResult::new_unknown(100, 50),
            MidSearchResult::new_unknown(30, 80),  // best for OR (min pn)
            MidSearchResult::new_unknown(50, 60),
        ];
        let exp = MidLocalExpansion::new(true, moves.clone(), results);
        assert_eq!(exp.best_move(), Move(2));
        assert!(!exp.empty());
        // 2 番目に小さい pn は idx 2 の 50
        assert_eq!(exp.get_second_phi(), 50);
    }

    #[test]
    fn test_update_best_child_proven() {
        let moves = vec![Move(1), Move(2)];
        let results = vec![
            MidSearchResult::new_unknown(100, 50),
            MidSearchResult::new_unknown(30, 80),
        ];
        let mut exp = MidLocalExpansion::new(true, moves, results);
        assert_eq!(exp.best_move(), Move(2));
        // best (Move(2)) が proven 化した
        let proven = MidSearchResult::new_win(10);
        exp.update_best_child(proven);
        // excluded_moves が進み，次の best は元 idx 0 (Move(1))
        assert_eq!(exp.best_move(), Move(1));
    }

    /// search_impl の最小動作: 1 step で proven 子が見つかると現フレームも proven．
    #[test]
    fn test_search_impl_immediate_proof() {
        // OR ノードで child 0 が proven (pn=0, dn=INF)，他は unknown
        let moves = vec![Move(1), Move(2)];
        let results = vec![
            MidSearchResult::new_win(5),                 // 既に proven (初期から)
            MidSearchResult::new_unknown(100, 100),
        ];
        let mut exp = MidLocalExpansion::new(true, moves, results);
        // best_move は proven child (phi=0)
        assert_eq!(exp.best_move(), Move(1));
        // current_result は OR で min(pn)=0 → win
        let cur = exp.current_result();
        assert_eq!(cur.pn, 0);
    }

    /// search_impl がループを抜けるテスト (proven 到達)．
    #[test]
    fn test_search_impl_loop_completes() {
        // OR ノード，2 children．初期は両方 unknown．
        // recurse callback でいずれかが proven 化 (count > 2 で win)．
        let moves = vec![Move(1), Move(2)];
        let results = vec![
            MidSearchResult::new_unknown(50, 100),
            MidSearchResult::new_unknown(60, 100),
        ];
        let mut exp = MidLocalExpansion::new(true, moves, results);

        let mut ctx = MiniMidContext::new(10, 100);
        let mut inc_flag = 0u32;
        let result = search_impl(
            &mut exp,
            1000, 1000, 10,
            &mut inc_flag,
            &mut ctx,
            // expand_child: 子に dummy expansion を返す
            |_m| {
                MidLocalExpansion::new(false, vec![Move(99)], vec![MidSearchResult::new_unknown(1, 1)])
            },
            // recurse: 直接 win を返す (mate immediate)
            |_child_exp, _ctpn, _ctdn, _inc| {
                MidSearchResult::new_win(5)
            },
        );
        // 1 回 recurse で win が返り，OR ノードは min(pn)=0 で proven
        assert_eq!(result.pn, 0);
    }

    /// search_impl が node budget で break する．
    #[test]
    fn test_search_impl_node_limit() {
        let moves = vec![Move(1), Move(2)];
        let results = vec![
            MidSearchResult::new_unknown(50, 100),
            MidSearchResult::new_unknown(60, 100),
        ];
        let mut exp = MidLocalExpansion::new(true, moves, results);

        let mut ctx = MiniMidContext::new(10, 1); // budget = 1
        let mut inc_flag = 0u32;
        let result = search_impl(
            &mut exp,
            1000, 1000, 10,
            &mut inc_flag,
            &mut ctx,
            |_m| MidLocalExpansion::new(false, vec![Move(99)], vec![MidSearchResult::new_unknown(1, 1)]),
            |_child_exp, _ctpn, _ctdn, _inc| {
                // recurse は呼ばれるが進捗なし
                MidSearchResult::new_unknown(50, 100)
            },
        );
        // budget=1 で 1 回 entry 後 break
        assert_eq!(ctx.nodes_searched, 1);
        // proven 化していない
        assert_ne!(result.pn, 0);
    }
}
