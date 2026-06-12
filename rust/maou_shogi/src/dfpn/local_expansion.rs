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
pub struct MidSearchResult {
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
    /// KH `min_depth < depth16` 相当: TT entry が shallower ply で保存されたか．
    /// TCA `has_old_child` の精密化に使用．
    pub is_shallow: bool,
    /// KH `FinalData::repetition_start` 相当 (Phase 29)．千日手 (cycle) に依存した
    /// disproof (dn==0) の場合，cycle が始まった ply を持つ (= taint)．`REPETITION_NONE`
    /// (u32::MAX) なら非 repetition．disproof が `repetition_start < 自 ply` のとき cycle 内
    /// (path-dependent) なので上へ伝播し，`>= 自 ply` で resolve する (KH local_expansion.hpp:578)．
    pub repetition_start: u32,
}

/// `repetition_start` の sentinel: 千日手依存でない．
pub(super) const REPETITION_NONE: u32 = u32::MAX;

impl MidSearchResult {
    pub(super) fn new_unknown(pn: u32, dn: u32) -> Self {
        Self {
            pn,
            dn,
            amount: 0,
            mate_distance: 0,
            is_first_visit: true,
            is_shallow: false,
            repetition_start: REPETITION_NONE,
        }
    }

    pub(super) fn new_win(mate_distance: u16) -> Self {
        Self {
            pn: 0,
            dn: u32::MAX,
            amount: 1,
            mate_distance,
            is_first_visit: false,
            is_shallow: false,
            repetition_start: REPETITION_NONE,
        }
    }

    pub(super) fn new_lose(mate_distance: u16) -> Self {
        Self {
            pn: u32::MAX,
            dn: 0,
            amount: 1,
            mate_distance,
            is_first_visit: false,
            is_shallow: false,
            repetition_start: REPETITION_NONE,
        }
    }

    /// 千日手 (cycle) による disproof．KH `MakeRepetition` 相当 (pn=INF, dn=0 + taint)．
    /// `rep_start` は cycle が始まった ply．
    pub(super) fn new_repetition(rep_start: u32) -> Self {
        Self {
            pn: u32::MAX,
            dn: 0,
            amount: 1,
            mate_distance: 0,
            is_first_visit: false,
            is_shallow: false,
            repetition_start: rep_start,
        }
    }

    /// 千日手依存の disproof か (KH `FinalData::IsRepetition`)．
    pub(super) fn is_repetition(&self) -> bool {
        self.dn == 0 && self.repetition_start != REPETITION_NONE
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
    /// Phase 14 (v0.93.0): 現フレームの局面 full_hash．DAG correction で
    /// expansion_stack 上の祖先と一致判定．
    position_fh: u64,
    /// 全合法手 (位置生成順)．mp_ の Rust 対応．
    pub(super) moves: Vec<Move>,
    /// sorted index array (phi 昇順)．`idx_[excluded_moves_]` が現在の "best"．
    pub(super) idx: Vec<u32>,
    /// proven 化済み children のカウント．BestMove() で `idx_[excluded_moves_]` を返す．
    excluded_moves: usize,
    /// 各 child の現在の (pn, dn, amount)．`UpdateBestChild` で更新．
    pub(super) results: Vec<MidSearchResult>,
    /// 各 child が sum aggregation 対象か (true = sum，false = max)．BitSet64 相当．
    sum_mask: u64,
    /// キャッシュ: best 以外の delta sum．
    sum_delta_except_best: u32,
    /// キャッシュ: best 以外の delta max．
    max_delta_except_best: u32,
    /// `DoesHaveOldChild`: 子の中に old visit (= TT cache 経由) があるか．
    has_old_child: bool,
    /// Phase 11 (v0.90.0): DelayedMoveList chain (KH `delayed_move_list.hpp` 移植)．
    /// `dml_prev[i]` = chain 上で i の直前の move index (-1 = なし)．
    /// `dml_next[i]` = 直後 (-1 = なし)．`is_deferred[i]` = 初期 idx に含めない (=
    /// prev の final 待ち)．
    dml_prev: Vec<i32>,
    dml_next: Vec<i32>,
    /// Phase 21: deferred penalty の除数 (0 = 無効，8 = KH 準拠)．
    deferred_penalty_denom: u32,
    /// Phase 22: deferred penalty に `.max(1)` floor を適用 (KH=true)．
    deferred_penalty_floor: bool,
    /// parity compound 実験 (V3_RECALC): eval 再 sort 後に recalc_delta を行うか．
    recalc_on_resort: bool,
    /// Phase 22: 1+ε 閾値 epsilon (KH デフォルト 1; maou 試験 PN_UNIT=16)．
    threshold_epsilon: u32,
    /// Phase 23 (G4): KH `MoveBriefEvaluation` 相当の move score．
    /// sort 比較で phi が同点のとき tie-break に使う (小さいほうが優先)．
    /// 0 で初期化．solver.rs 側で `set_move_evals` を呼んで実値を投入．
    move_evals: Vec<i32>,
    /// KH coherent mode (Phase 30): `child_ordering` で KH `SearchResultComparer`
    /// の δ値 (基準 2) と amount (基準 5) tie-break を有効化する．既存 (false) は
    /// δ tie-break を省略していた (PN_UNIT=16 スケールでは move 選択を悪化させたため)．
    /// KH 同等の unit=2 スケール (param_kh_scale) と併用して初めて正しく機能する．
    kh_full_comparer: bool,
    /// Phase 31: `kh_full_comparer` を AND ノードのみに限定する．δ tie-break は OR ノードでは
    /// 王手選択を変え非最短 (Mate-31) を誘発するが，deep breadth の主因は AND (defender) fan-out．
    /// AND のみ δ を効かせれば mate 長 (OR の選択) を保ったまま defender selectivity を上げられる仮説．
    kh_full_and_only: bool,
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
        Self::new_with_fh(or_node, moves, initial_results, 0)
    }

    /// Phase 14 (v0.93.0): position_fh 付き構築．旧挙動 DML (AND drops only)．
    pub(super) fn new_with_fh(
        or_node: bool,
        moves: Vec<Move>,
        initial_results: Vec<MidSearchResult>,
        position_fh: u64,
    ) -> Self {
        Self::new_with_fh_dml(or_node, moves, initial_results, position_fh, false, true)
    }

    /// Phase 26: KH parity DML を選択可能にした構築．
    /// `kh_dml=true` で非駒打ち成/不成 deferral を OR/AND 両方で有効化．
    /// `us_is_black` は敵陣判定用 (side-to-move の色)．
    pub(super) fn new_with_fh_dml(
        or_node: bool,
        moves: Vec<Move>,
        initial_results: Vec<MidSearchResult>,
        position_fh: u64,
        kh_dml: bool,
        us_is_black: bool,
    ) -> Self {
        Self::new_with_fh_dml_chuai(
            or_node,
            moves,
            initial_results,
            position_fh,
            kh_dml,
            us_is_black,
            None,
        )
    }

    /// Phase 33m: KH cross-square「無意味な中合い」束ねを有効化できる構築版．
    /// `chuai` を渡すと build_delayed_chain_chuai 経由で合駒の breadth collapse を行う
    /// (delayed_move_list.hpp:151-156)．`None` なら従来の same-to_sq のみ chain．
    pub(super) fn new_with_fh_dml_chuai(
        or_node: bool,
        moves: Vec<Move>,
        initial_results: Vec<MidSearchResult>,
        position_fh: u64,
        kh_dml: bool,
        us_is_black: bool,
        chuai: Option<&[bool]>,
    ) -> Self {
        let mut expansion = Self::new_empty();
        expansion.moves = moves;
        expansion.results = initial_results;
        expansion.rebuild(or_node, position_fh, kh_dml, us_is_black, chuai);
        expansion
    }

    /// Pool 再利用用の空 expansion (全 Vec capacity 0)．`reset_for_fill` →
    /// moves/results 投入 → `rebuild` の順で再構築する．
    /// (既存の `empty(&self) -> bool` 述語と紛らわしいので `new_` 接頭辞)．
    pub(super) fn new_empty() -> Self {
        Self {
            or_node: false,
            position_fh: 0,
            moves: Vec::new(),
            idx: Vec::new(),
            excluded_moves: 0,
            results: Vec::new(),
            sum_mask: 0,
            sum_delta_except_best: 0,
            max_delta_except_best: 0,
            has_old_child: false,
            dml_prev: Vec::new(),
            dml_next: Vec::new(),
            deferred_penalty_denom: 8,
            deferred_penalty_floor: false,
            recalc_on_resort: false,
            threshold_epsilon: 1,
            move_evals: Vec::new(),
            kh_full_comparer: false,
            kh_full_and_only: false,
        }
    }

    /// Pool 再利用: moves/results を空にする (capacity 維持)．
    /// 呼び出し側はこの後 moves/results を投入してから `rebuild` を呼ぶ．
    pub(super) fn reset_for_fill(&mut self) {
        self.moves.clear();
        self.results.clear();
    }

    /// `moves`/`results` 投入済みの self を `new_with_fh_dml_chuai` と**同一ロジック**で
    /// 再構築する (idx/dml/sum_mask/move_evals/スカラを全て初期化)．
    /// pool 再利用時も全フィールドがコンストラクタと同値になるよう，前回の
    /// 設定値 (kh_full_comparer 等) は必ずここでリセットする．
    pub(super) fn rebuild(
        &mut self,
        or_node: bool,
        position_fh: u64,
        kh_dml: bool,
        us_is_black: bool,
        chuai: Option<&[bool]>,
    ) {
        let n = self.moves.len();

        // Phase 11 (v0.90.0): DelayedMoveList chain 構築．AND ノードの同 to_sq
        // drops を chain 化．Phase 12 (v0.91.0): prev が **non-final のとき** のみ defer．
        // KH `local_expansion.hpp:181-194` 移植．prev が TT cache から既に proven 等で
        // final なら defer しない (= 初期 idx に含める)．これで AND の curr.pn=0 が
        // chain の 1 件目だけで成立する soundness 違反を防ぐ．
        // Phase 26: kh_dml=true で非駒打ち成/不成 deferral を追加 (build_delayed_chain 参照)．
        build_delayed_chain_chuai_into(
            &self.moves, or_node, kh_dml, us_is_black, chuai,
            &mut self.dml_prev, &mut self.dml_next,
        );

        self.idx.clear();
        self.idx.reserve(n);
        for i in 0..n {
            let mut deferred = false;
            let mut cur = self.dml_prev[i];
            while cur >= 0 {
                if !self.results[cur as usize].is_final() {
                    deferred = true;
                    break;
                }
                cur = self.dml_prev[cur as usize];
            }
            if !deferred {
                self.idx.push(i as u32);
            }
        }
        // 初期 sort: KH SearchResultComparer 順 (phi → delta → ...)．
        // 構築時は move_evals 未設定 (全 0) なので tie-break は無効．
        // solver 側が set_move_evals を呼んで実 eval を投入後に再 sort される．
        {
            let results = &self.results;
            self.idx.sort_by(|&i, &j| {
                child_ordering(
                    or_node,
                    &results[i as usize], 0,
                    &results[j as usize], 0,
                )
            });
        }

        let mut sum_mask = if n >= 64 { u64::MAX } else { (1u64 << n).wrapping_sub(1) };
        // Phase 22: KH `kForceSumPnDn = kInfinitePnDn / 1024` 相当．
        // child の初期 delta が一定値以上の場合 sum_mask off (max 集約に切替)．
        // KH `local_expansion.hpp:177` を移植．
        const FORCE_MAX_DELTA: u32 = (u32::MAX / 2) / 1024;
        for (i, r) in self.results.iter().enumerate().take(64) {
            if r.delta(or_node) >= FORCE_MAX_DELTA {
                sum_mask &= !(1u64 << i);
            }
        }
        self.or_node = or_node;
        self.position_fh = position_fh;
        self.excluded_moves = 0;
        self.sum_mask = sum_mask;
        self.sum_delta_except_best = 0;
        self.max_delta_except_best = 0;
        self.has_old_child = self.results.iter().any(|r| r.is_shallow);
        self.deferred_penalty_denom = 8;
        self.deferred_penalty_floor = false;
        self.recalc_on_resort = false;
        self.threshold_epsilon = 1;
        self.move_evals.clear();
        self.move_evals.resize(n, 0i32);
        self.kh_full_comparer = false;
        self.kh_full_and_only = false;
        self.recalc_delta();
    }

    /// Phase 23 (G4): KH `MoveBriefEvaluation` の値を設定し idx を再 sort する．
    /// `evals[i]` は `moves[i]` に対応する score (小さいほうが優先)．
    /// コンストラクト直後に呼ぶ前提．サイズは `moves.len()` と一致すること．
    pub(super) fn set_move_evals(&mut self, evals: Vec<i32>) {
        debug_assert_eq!(evals.len(), self.moves.len());
        self.move_evals = evals;
        self.resort_by_evals();
    }

    /// `set_move_evals` の slice 版 (pool 再利用用)．`move_evals` の capacity を
    /// 維持したまま内容を差し替えて再 sort する．挙動は `set_move_evals` と同一．
    pub(super) fn set_move_evals_slice(&mut self, evals: &[i32]) {
        debug_assert_eq!(evals.len(), self.moves.len());
        self.move_evals.clear();
        self.move_evals.extend_from_slice(evals);
        self.resort_by_evals();
    }

    /// move_evals 投入後の idx 再 sort (tie-break が変わるため KH comparer で再 sort)．
    fn resort_by_evals(&mut self) {
        let or_node = self.or_node;
        let kh_full = self.effective_kh_full();
        let results = &self.results;
        let me_evals = &self.move_evals;
        if self.excluded_moves < self.idx.len() {
            self.idx[self.excluded_moves..].sort_by(|&i, &j| {
                child_ordering_ex(
                    or_node,
                    &results[i as usize], me_evals[i as usize],
                    &results[j as usize], me_evals[j as usize],
                    kh_full,
                )
            });
        }
        // NOTE (2026-06-12 parity hunt): KH は最終 sort 後に RecalcDelta する
        // (local_expansion.hpp:220-221) が，単発で recalc_delta() を KH に合わせると
        // **29te +42% / 39te +29% に退行**する (STRICT 健全)．stale sum の under-aggregation
        // は「loop を長く回し re-entry を減らす」方向に共適応して net 有利のため．
        // V3_RECALC=1 (compound 格子実験用 env) のときのみ KH 一致の recalc を行う．
        if self.recalc_on_resort {
            self.recalc_delta();
        }
    }

    /// Phase 21: deferred penalty 除数を設定 (0 = 無効)．
    /// 正の値の場合: penalty = deferred_count / denom (floor なし)．
    pub(super) fn set_deferred_penalty_denom(&mut self, denom: u32) {
        self.deferred_penalty_denom = denom;
    }

    /// Phase 22: 1+ε 閾値 epsilon (`second_phi + epsilon`) を設定．
    pub(super) fn set_threshold_epsilon(&mut self, eps: u32) {
        self.threshold_epsilon = eps.max(1);
    }

    /// Phase 22: deferred penalty floor (`.max(1)`) を設定．
    pub(super) fn set_deferred_penalty_floor(&mut self, floor: bool) {
        self.deferred_penalty_floor = floor;
    }

    /// parity compound 実験 (V3_RECALC): eval 再 sort 後に recalc_delta を行うか
    /// (KH RecalcDelta 一致)．default false (off; v2.8.7 NOTE の単発退行を参照)．
    pub(super) fn set_recalc_on_resort(&mut self, on: bool) {
        self.recalc_on_resort = on;
    }

    /// Phase 30: KH 完全 comparer (δ値 + amount tie-break) を有効化する．
    /// `set_move_evals` の **前** に呼ぶこと (re-sort がこの flag を参照するため)．
    pub(super) fn set_kh_full_comparer(&mut self, on: bool) {
        self.kh_full_comparer = on;
    }

    /// Phase 31: kh_full_comparer を AND ノードのみに限定する (δ tie-break を OR で無効化)．
    pub(super) fn set_kh_full_and_only(&mut self, on: bool) {
        self.kh_full_and_only = on;
    }

    /// 現フレームで有効な kh_full 値 (and_only なら AND ノードのみ true)．
    #[inline]
    fn effective_kh_full(&self) -> bool {
        self.kh_full_comparer && (!self.kh_full_and_only || !self.or_node)
    }

    /// Phase 21: has_old_child を !is_first_visit ベースで再計算 (旧ロジック)．
    pub(super) fn recompute_has_old_child_any_revisit(&mut self) {
        self.has_old_child = self.results.iter().any(|r| !r.is_first_visit);
    }

    /// Phase 21: deferred count (= moves.len() - idx.len())．
    pub(super) fn deferred_count(&self) -> usize {
        self.moves.len().saturating_sub(self.idx.len())
    }

    /// Phase 14: position_fh accessor．
    pub(super) fn position_fh(&self) -> u64 {
        self.position_fh
    }

    /// Phase 14: or_node accessor．
    pub(super) fn or_node_value(&self) -> bool {
        self.or_node
    }

    /// Phase 14: 指定 move の sum_mask bit を off にする (= max 集約に切替)．
    /// KH `ResolveDoubleCountIfBranchRoot` の child 探索 + sum_mask reset 相当．
    /// 指定 move_to_match に一致する child を探し，見つかったら sum_mask off + recalc_delta．
    /// `move_to_match` は idx に含まれる手の中で `to_move16` 比較で同定．
    pub(super) fn reset_sum_mask_for_move(&mut self, move_to_match: u16) -> bool {
        for &i_raw in &self.idx {
            let mv = self.moves[i_raw as usize];
            if mv.to_move16() == move_to_match {
                let bit = 1u64 << (i_raw as u64);
                if (self.sum_mask & bit) != 0 {
                    self.sum_mask &= !bit;
                    self.recalc_delta();
                    return true;
                }
                return false;
            }
        }
        false
    }

    /// 空か (= empty)．
    pub(super) fn empty(&self) -> bool {
        self.idx.is_empty() || self.excluded_moves >= self.idx.len()
    }

    /// Phase 12 (v0.91.0) 診断用: idx_ サイズ．
    pub(super) fn idx_len(&self) -> usize {
        self.idx.len()
    }

    /// Phase 12 診断用: 全 moves サイズ．
    pub(super) fn moves_len(&self) -> usize {
        self.moves.len()
    }

    /// Phase 14 (v0.93.0): DAG correction 用．現 best_move (idx[excluded_moves]) の
    /// sum_mask を OFF にして max 集約に切替える．KH `local_expansion.hpp:398
    /// sum_mask_.Reset(idx_.front())` 相当．二重カウント検出時に呼ぶ．
    pub(super) fn reset_sum_mask_for_best(&mut self) -> bool {
        if self.empty() {
            return false;
        }
        let i_raw = self.idx[self.excluded_moves] as usize;
        let bit = 1u64 << (i_raw as u64);
        if (self.sum_mask & bit) != 0 {
            self.sum_mask &= !bit;
            self.recalc_delta();
            true
        } else {
            false
        }
    }

    /// Phase 27: KH `IsSumDeltaNode`=false の child を max 集約に切替える．
    /// `force_max_raw` は sum→max にすべき move の raw index 列 (`super::is_sum_delta_node`
    /// が `false` を返した手)．KH `local_expansion.hpp:177` の `!IsSumDeltaNode(...)` 分岐相当で，
    /// **非 final** の child のみ sum_mask を reset する．1 件でも reset したら `recalc_delta` する．
    /// solver 側で board を参照して force-max 集合を計算し，`set_move_evals` の **後** に呼ぶ前提
    /// (recalc が最終 idx 順序を反映するため)．
    pub(super) fn apply_force_max(&mut self, force_max_raw: &[u32]) -> bool {
        let mut changed = false;
        for &i in force_max_raw {
            let iu = i as usize;
            if iu >= self.results.len() {
                continue;
            }
            if self.results[iu].is_final() {
                continue;
            }
            let bit = 1u64 << (i as u64);
            if (self.sum_mask & bit) != 0 {
                self.sum_mask &= !bit;
                changed = true;
            }
        }
        if changed {
            self.recalc_delta();
        }
        changed
    }

    /// BestMove: `idx_[excluded_moves_]` の手を返す．
    pub(super) fn best_move(&self) -> Move {
        let raw = self.idx[self.excluded_moves] as usize;
        self.moves[raw]
    }

    /// Phase 31 診断: idx 順 (探索順) 上位 k 子の (move, pn, dn, move_eval) を返す．
    pub(super) fn trace_children(&self, k: usize) -> Vec<(Move, u32, u32, i32)> {
        self.idx
            .iter()
            .skip(self.excluded_moves)
            .take(k)
            .map(|&i| {
                let r = i as usize;
                (self.moves[r], self.results[r].pn, self.results[r].dn, self.move_evals[r])
            })
            .collect()
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

    /// 計測専用 (V3THX): 閾値伝播の micro-lever 確定用 internal accessor．
    pub(super) fn dbg_second_phi(&self) -> u32 {
        self.get_second_phi()
    }
    pub(super) fn dbg_sum_delta_except_best(&self) -> u32 {
        self.sum_delta_except_best
    }
    pub(super) fn dbg_max_delta_except_best(&self) -> u32 {
        self.max_delta_except_best
    }
    pub(super) fn dbg_active_count(&self) -> usize {
        self.idx.len().saturating_sub(self.excluded_moves)
    }
    pub(super) fn dbg_deferred_count(&self) -> usize {
        self.moves.len().saturating_sub(self.idx.len())
    }
    /// 計測専用: active children を idx (sorted) 順で (move, phi, delta, is_first) で返す．
    pub(super) fn dbg_children(&self) -> Vec<(Move, u32, u32, bool)> {
        self.idx[self.excluded_moves..]
            .iter()
            .map(|&i| {
                let r = &self.results[i as usize];
                (self.moves[i as usize], r.phi(self.or_node), r.delta(self.or_node), r.is_first_visit)
            })
            .collect()
    }

    /// AND proven 時の max mate_distance を計算する．
    /// AND の場合 defender は max-resistance を選ぶので max(child.mate_distance)+1．
    pub(super) fn max_mate_distance_over_children(&self) -> u16 {
        let mut max_md: u16 = 0;
        for &i_raw in &self.idx {
            let r = &self.results[i_raw as usize];
            if r.pn == 0 {
                // proven child
                if r.mate_distance > max_md {
                    max_md = r.mate_distance;
                }
            }
        }
        max_md.saturating_add(1)
    }

    /// Phase 17 (v0.96.0): OR proven without main loop iteration の場合，
    /// children の中で proven (pn=0) の min mate_distance + 1 と対応 move を返す．
    /// main loop が回らなかった場合に呼ぶ (proof_best_move 未設定)．
    pub(super) fn min_proven_or_child(&self) -> (u16, u16) {
        let mut best_md: u16 = u16::MAX;
        let mut best_bm: u16 = 0;
        for &i_raw in &self.idx {
            let r = &self.results[i_raw as usize];
            if r.pn == 0 && r.mate_distance < best_md {
                best_md = r.mate_distance;
                best_bm = self.moves[i_raw as usize].to_move16();
            }
        }
        if best_md == u16::MAX {
            (0, 0)
        } else {
            (best_bm, best_md.saturating_add(1))
        }
    }

    /// Phase 11 (v0.90.0): AND proven 時の max-resistance defender move + その mate_distance + 1 を返す．
    /// PV 抽出時に「defender が実際に選ぶ手」として TT に store するため．
    /// 旧 `best_move()` は phi=dn 昇順の先頭 (= 任意の proven defender) を返していたが，
    /// DML 導入で順序が変わり 17 手 mate path が選ばれなくなったため明示的に max-md 選択へ．
    pub(super) fn max_resistance_defender(&self) -> (u16, u16) {
        let mut best_md: u16 = 0;
        let mut best_bm: u16 = 0;
        for &i_raw in &self.idx {
            let r = &self.results[i_raw as usize];
            if r.pn == 0 && r.mate_distance >= best_md {
                best_md = r.mate_distance;
                best_bm = self.moves[i_raw as usize].to_move16();
            }
        }
        (best_bm, best_md.saturating_add(1))
    }

    /// OR proven 時の min mate_distance を計算 (= shortest mate via children)．
    pub(super) fn min_mate_distance_over_proven_children(&self) -> u16 {
        let mut min_md: u16 = u16::MAX;
        for &i_raw in &self.idx {
            let r = &self.results[i_raw as usize];
            if r.pn == 0 && r.mate_distance < min_md {
                min_md = r.mate_distance;
            }
        }
        if min_md == u16::MAX { 0 } else { min_md.saturating_add(1) }
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

        // Phase 11 (v0.90.0): final 化した child の dml_next が残っていれば復活．
        // KH `local_expansion.hpp:319-340` 相当．
        if search_result.is_final() {
            let mut cur = self.dml_next[old_raw];
            while cur >= 0 {
                let cur_raw = cur as usize;
                self.idx.push(cur_raw as u32);
                // 復活した手の delta が 0 (= 既に final) なら chain を進める
                if self.results[cur_raw].delta(self.or_node) > 0 {
                    break;
                }
                cur = self.dml_next[cur_raw];
            }
        }

        // re-sort: 現 best の評価が変わったので idx_ を更新
        // Phase 23 (G4 強化): KH SearchResultComparer 準拠 (phi → delta → len → eval)．
        let or_node = self.or_node;
        let kh_full = self.effective_kh_full();
        let results = &self.results;
        let me_evals = &self.move_evals;
        if self.excluded_moves < self.idx.len() {
            self.idx[self.excluded_moves..].sort_by(|&i, &j| {
                child_ordering_ex(
                    or_node,
                    &results[i as usize], me_evals[i as usize],
                    &results[j as usize], me_evals[j as usize],
                    kh_full,
                )
            });
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
        let child_thphi = thphi.min(second_phi.saturating_add(self.threshold_epsilon));
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
        // KH local_expansion.hpp:513-514: deferred moves penalty
        if self.deferred_penalty_denom > 0 && self.moves.len() > self.idx.len() {
            let raw_penalty = (self.moves.len() - self.idx.len()) / self.deferred_penalty_denom as usize;
            let penalty = if self.deferred_penalty_floor {
                raw_penalty.max(1) as u32
            } else {
                raw_penalty as u32
            };
            delta_except_best = delta_except_best.saturating_add(penalty);
        }
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
    /// KH 風: excluded_moves > 0 (= proof/refutation を発見済み) かつ front が INF なら
    /// 0 を返す (= 全 children proven for OR / 全 refuted for AND)．
    fn get_phi(&self) -> u32 {
        let front_phi = if self.excluded_moves < self.idx.len() {
            self.front_result().phi(self.or_node)
        } else {
            u32::MAX
        };
        if front_phi >= u32::MAX && self.excluded_moves > 0 {
            return 0;
        }
        front_phi
    }

    /// 現フレームの delta: OR ノードなら sum(dn)，AND ノードなら sum(pn)．
    /// KH `GetDelta` (local_expansion.hpp:466) 移植．
    /// 旧版は max-aggregate branch で best_delta を落とす bug 有り (Phase 10 で修正)．
    fn get_delta(&self) -> u32 {
        if self.empty() {
            return 0;
        }
        let raw = self.idx[self.excluded_moves] as usize;
        let best_delta = self.results[raw].delta(self.or_node);
        let mut sum_delta = self.sum_delta_except_best;
        let mut max_delta = self.max_delta_except_best;
        if (self.sum_mask >> (raw as u64)) & 1 == 1 {
            sum_delta = sum_delta.saturating_add(best_delta);
        } else {
            if best_delta > max_delta {
                max_delta = best_delta;
            }
        }
        // KH local_expansion.hpp:485-488: deferred moves penalty
        if self.deferred_penalty_denom > 0 && self.moves.len() > self.idx.len() {
            let raw_penalty = (self.moves.len() - self.idx.len()) / self.deferred_penalty_denom as usize;
            let penalty = if self.deferred_penalty_floor {
                raw_penalty.max(1) as u32
            } else {
                raw_penalty as u32
            };
            sum_delta = sum_delta.saturating_add(penalty);
        }
        let raw_delta = sum_delta.saturating_add(max_delta);
        if self.excluded_moves > 0 && raw_delta == 0 {
            return u32::MAX;
        }
        raw_delta
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

/// Phase 23 (G4 強化): KH `SearchResultComparer` 移植 (`search_result.hpp:258`)．
///
/// 2 つの child を比較し，「良い (= idx の先頭に来るべき)」方が `Less` になる順序を返す．
/// 比較基準 (KH と同順):
/// 1. φ値 (OR=pn, AND=dn) 昇順
/// 2. δ値 (OR=dn, AND=pn) 昇順
/// 3. proven (pn==0) のとき詰み手数: OR は短い順 / AND は長い順
/// 4. (千日手 repetition_start: maou 未トラッキングのため skip)
/// 5. amount 昇順
/// 6. `move_eval` (KH `MoveBriefEvaluation`) 昇順 — 最終 tie-break
///
/// 旧実装は `(phi, move_eval)` のみで δ値・詰み手数を無視していた．特に基準 3 の
/// 欠落により proven 兄弟の中で短い詰みが先頭に来ず，PV 品質 (29 vs 31 手) に
/// 影響していた．
#[inline]
fn child_ordering(
    or_node: bool,
    a: &MidSearchResult,
    ea: i32,
    b: &MidSearchResult,
    eb: i32,
) -> std::cmp::Ordering {
    child_ordering_ex(or_node, a, ea, b, eb, false)
}

/// Phase 30: `kh_full` で KH `SearchResultComparer` 完全準拠 (δ値 + amount tie-break)．
///
/// `kh_full=false` (旧 default): φ値 → proven 詰み手数 → move_eval．δ値を省略する
/// (PN_UNIT=16 スケールでは move 選択を悪化させたため: 29te 初回 190K→242K)．
///
/// `kh_full=true` (KH coherent, unit=2 スケール併用前提): KH 基準どおり
/// φ値 (1) → δ値 (2) → proven 詰み手数 (3) → amount (5) → move_eval．unit=2 では
/// δ値が小さい整数となり KH 本来の細かい ordering を再現する．
#[inline]
fn child_ordering_ex(
    or_node: bool,
    a: &MidSearchResult,
    ea: i32,
    b: &MidSearchResult,
    eb: i32,
    kh_full: bool,
) -> std::cmp::Ordering {
    // 1. φ値
    let (pa, pb) = (a.phi(or_node), b.phi(or_node));
    if pa != pb {
        return pa.cmp(&pb);
    }
    // 2. δ値 (KH 基準 2)．kh_full のみ．
    if kh_full {
        let (da, db) = (a.delta(or_node), b.delta(or_node));
        if da != db {
            return da.cmp(&db);
        }
    }
    // 3. proven (pn==0) の詰み手数優先 (KH 基準 3)．
    if a.pn == 0 && b.pn == 0 && a.mate_distance != b.mate_distance {
        return if or_node {
            a.mate_distance.cmp(&b.mate_distance) // OR: 短い詰みを優先
        } else {
            b.mate_distance.cmp(&a.mate_distance) // AND: 長い抵抗を優先
        };
    }
    // 4. KH 基準 4 (disproven dn==0 の repetition_start 優先) は maou では未採用:
    //    実測で一部テストが非終了化 (selection 変化で探索が発散) する一方 29te には無効だったため．
    // 5. amount (KH 基準 5)．kh_full のみ．
    if kh_full && a.amount != b.amount {
        return a.amount.cmp(&b.amount);
    }
    // 6. move_eval (KH `MoveBriefEvaluation`) tie-break
    ea.cmp(&eb)
}

/// Phase 11 (v0.90.0): KH `DelayedMoveList` 移植．Phase 12 (v0.91.0) で再有効化．
///
/// `kh_dml=false` (旧挙動): AND ノードの同 to_sq drops のみ chain 化．
/// `kh_dml=true` (Phase 26, KH parity): KH `delayed_move_list.hpp` を忠実移植．
///   - 駒打ち: AND ノードでのみ delay (同 to_sq で chain)．
///   - 非駒打ち 成/不成: 成れる駒 (歩/角/飛 + 香 rank2/8) を OR/AND 両方で chain 化
///     ((from,to) 同一の成・不成ペアを束ね，先頭のみ展開・残りは prev final 待ち)．
/// chain 上の prev が final になるまで next を idx_ に push しない．
pub(super) fn build_delayed_chain(
    moves: &[Move],
    or_node: bool,
    kh_dml: bool,
    us_is_black: bool,
) -> (Vec<i32>, Vec<i32>) {
    build_delayed_chain_chuai(moves, or_node, kh_dml, us_is_black, None)
}

/// `build_delayed_chain` の拡張版．`chuai` を渡すと KH `IsSame` の cross-square「無意味な中合い」
/// 束ね (delayed_move_list.hpp:151-156) を有効化する．`chuai[i] == true` = move i が
/// 「support 0 かつ 逆王手でない drop 中合い」= KH が後回しにすべきと判定する手．
pub(super) fn build_delayed_chain_chuai(
    moves: &[Move],
    or_node: bool,
    kh_dml: bool,
    us_is_black: bool,
    chuai: Option<&[bool]>,
) -> (Vec<i32>, Vec<i32>) {
    let mut prev = Vec::new();
    let mut next = Vec::new();
    build_delayed_chain_chuai_into(moves, or_node, kh_dml, us_is_black, chuai, &mut prev, &mut next);
    (prev, next)
}

/// `build_delayed_chain_chuai` の in-place 版 (pool 再利用用)．
/// `prev`/`next` を clear して同一内容を構築する (capacity 維持でアロケーション回避)．
pub(super) fn build_delayed_chain_chuai_into(
    moves: &[Move],
    or_node: bool,
    kh_dml: bool,
    us_is_black: bool,
    chuai: Option<&[bool]>,
    prev: &mut Vec<i32>,
    next: &mut Vec<i32>,
) {
    let n = moves.len();
    prev.clear();
    prev.resize(n, -1i32);
    next.clear();
    next.resize(n, -1i32);

    if !kh_dml {
        // 旧挙動: AND ノードの同 to_sq drops のみ chain．
        if or_node {
            return;
        }
        let mut last_of_to: [u32; 81] = [0; 81];
        for (i, m) in moves.iter().enumerate() {
            if !m.is_drop() {
                continue;
            }
            let to = m.to_sq();
            let to_idx = (to.col() as usize) * 9 + (to.row() as usize);
            if to_idx >= 81 {
                continue;
            }
            let last = last_of_to[to_idx];
            if last > 0 {
                let last_i = (last - 1) as usize;
                prev[i] = last_i as i32;
                next[last_i] = i as i32;
            }
            last_of_to[to_idx] = (i + 1) as u32;
        }
        return;
    }

    // KH parity (Phase 26): delayed_move_list.hpp の double-linked list 構築．
    // 各 chain の先頭 (representative move, raw index) を heads に保持 (KH kMaxLen=10)．
    const MAX_HEADS: usize = 10;
    let mut heads: Vec<(Move, usize)> = Vec::with_capacity(MAX_HEADS);
    for (i, &m) in moves.iter().enumerate() {
        if !dml_is_delayable(m, or_node, us_is_black) {
            continue;
        }
        let mut found = false;
        for h in heads.iter_mut() {
            let same = if let Some(c) = chuai {
                dml_is_same_chuai(h.0, m, c[h.1], c[i])
            } else {
                dml_is_same(h.0, m)
            };
            if same {
                next[h.1] = i as i32;
                prev[i] = h.1 as i32;
                *h = (m, i);
                found = true;
                break;
            }
        }
        if !found && heads.len() < MAX_HEADS {
            heads.push((m, i));
        }
    }
}

/// KH `DelayedMoveList::IsDelayable` 移植 (delayed_move_list.hpp:108)．
///
/// 「すぐ展開する必要のない手」= true．
/// - 駒打ち: AND ノードでのみ delay (合駒は他の手の結果を見てから読む)．
/// - 非駒打ち: from/to が敵陣 (成れる) かつ 歩/角/飛 → delay．香は rank2(黒)/rank8(白) のみ．
#[inline]
fn dml_is_delayable(m: Move, or_node: bool, us_is_black: bool) -> bool {
    if m.is_drop() {
        return !or_node;
    }
    let from = m.from_sq();
    let to = m.to_sq();
    let in_enemy = |sq: crate::types::Square| -> bool {
        let r = sq.row();
        if us_is_black {
            r <= 2
        } else {
            r >= 6
        }
    };
    if !(in_enemy(from) || in_enemy(to)) {
        return false;
    }
    match m.moving_piece_type_raw() {
        1 | 5 | 6 => true, // Pawn / Bishop / Rook
        2 => {
            // Lance: KH black=RANK_2(row 1), white=RANK_8(row 7)
            if us_is_black {
                to.row() == 1
            } else {
                to.row() == 7
            }
        }
        _ => false,
    }
}

/// KH `DelayedMoveList::IsSame` 移植 (delayed_move_list.hpp:143)．
///
/// 両者 delayable 前提．どちらかを後回しにすべきなら true．
/// - 駒打ち同士: 同 to_sq (Phase 1; KH の「無意味な中合い」cross-square 束ねは未実装)．
/// - 非駒打ち同士: (from,to) が同一 = 成/不成ペア．
#[inline]
fn dml_is_same(m1: Move, m2: Move) -> bool {
    if m1.is_drop() && m2.is_drop() {
        m1.to_sq() == m2.to_sq()
    } else if !m1.is_drop() && !m2.is_drop() {
        m1.from_sq() == m2.from_sq() && m1.to_sq() == m2.to_sq()
    } else {
        false
    }
}

/// `dml_is_same` の KH 完全版 (delayed_move_list.hpp:143-164)．
/// `chuai1`/`chuai2` = それぞれの手が「support 0 かつ 逆王手でない drop 中合い」か．
/// 駒打ち同士は，**同 to_sq** または **両者が無意味な中合い (cross-square)** なら後回し対象．
#[inline]
fn dml_is_same_chuai(m1: Move, m2: Move, chuai1: bool, chuai2: bool) -> bool {
    if m1.is_drop() && m2.is_drop() {
        m1.to_sq() == m2.to_sq() || (chuai1 && chuai2)
    } else if !m1.is_drop() && !m2.is_drop() {
        m1.from_sq() == m2.from_sq() && m1.to_sq() == m2.to_sq()
    } else {
        false
    }
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

}
