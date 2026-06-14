//! mid_v4: KH `SearchImpl` の忠実移植による探索エンジン (module 5 + 3b 配線)．
//!
//! 忠実 component を結線して動く探索を構成する:
//! - [`super::kh_local_expansion::LocalExpansion`] (3a: RecalcDelta-after-resort 込の δ/閾値/選択)
//! - [`super::tt_v4::TranspositionTable`] (module 4: len-aware + cross-hand)
//! - [`super::mate_len::MateLen`] threading (全ノードへ len-1 を伝播)
//!
//! ## single-thread / 近道排除
//! - tt は **local の `&mut TranspositionTable`** を再帰へ渡す (DfPnSolver field を増やさず，
//!   `&mut self` (movegen) と `&mut tt` を disjoint に保つ)．
//! - 並列化なし (KH single-thread)．
//!
//! ## 移植済
//! - **1 手詰先読み (CheckObviousFinalOrNode, v2.11.0)**: AND first-visit child を do_move し OR 子の
//!   1 手詰/詰み無を先読みし proven/disproven seed (81,496→48,662, -40%; [`check_obvious_final_or_node_v4`])．
//! - **proof hand 極小化 (KH HandSet, v2.11.0)**: OR proof は `before_hand`，AND proof は `ProofHandSet` max 集約，
//!   look-ahead/terminal も極小化 ([`super::proof_hand`])．**29te node には非効果** (= breadth/selectivity が
//!   真因; hand 値でないことを mid_v4 で実測確認．Phase 28 と整合) だが KH 忠実かつ sound なので維持．
//! - **DML deferral (v2.12.0)**: `DelayedMoveList` を build_v4_expansion に配線．同マス合駒/成不成ペアの
//!   prev chain が未 final の手を idx から除外 (後回し) し，final 化で `update_best_child` が revival
//!   (48,662→36,768, -24%)．KH ctor の `i_is_skipped` を忠実再現．
//!
//! ## 本版で未移植 (sound だが効率劣; TODO)
//! - **EliminateDoubleCount (DAG double-count 抑止)**: KH は SearchImpl 毎に ancestor 展開 stack を walk するが
//!   mid_v4 は再帰 stack で明示 list を持たない (要構造変更)．残 gap (KH 3.95×) の主候補．
//! - 無意味中合いの cross-square DML (support 0 + no-check; mid_v3 では mate-len 退行のため要注意)．
//! - disproof hand 集約の極小化 / STRICT PV replay の v4 版．

use super::delayed_move_list::DelayedMoveList;
use super::kh_local_expansion::LocalExpansion;
use super::mate_len::{MateLen, DEPTH_MAX_MATE_LEN, ZERO_MATE_LEN};
use super::path_key::path_key_after;
use super::search_result::{
    extend_search_threshold, BitSet64, PnDn, SearchResult, K_INFINITE_PN_DN,
};
use super::solver::{DfPnSolver, TsumeResult};
use super::tt_v4::TranspositionTable;
use crate::board::Board;

/// init_pn_dn (unit-16) を KH `kPnDnUnit=2` へ縮約する除数 (mid_v3 `PN_UNIT_SCALE` と同値)．
const DIV: u64 = 8;

impl DfPnSolver {
    /// mid_v4 探索の root (KH `SearchEntry` 相当の IDS + 診断)．`V3_V4ENG=1` で起動する．
    pub(super) fn solve_via_v4(&mut self, board: &mut Board) -> TsumeResult {
        self.attacker = board.turn;
        self.v3_nodes = 0;
        self.v3_path.clear();
        self.timed_out = false;
        self.start_time = std::time::Instant::now();

        // len-aware TT (local; 再帰へ &mut で渡す)．サイズは budget 比例 + 大きめ確保で
        // cluster 満杯 (GC 未移植) を回避．
        let size = ((self.max_nodes as usize).saturating_mul(2)).clamp(1 << 18, 1 << 23);
        let mut tt = TranspositionTable::new(size);

        let mut thpn: PnDn = 1;
        let mut thdn: PnDn = 1;
        let mut last = SearchResult::make_unknown(1, 1, DEPTH_MAX_MATE_LEN, 1, BitSet64::full());
        loop {
            if self.v3_nodes >= self.max_nodes || self.is_timed_out() {
                self.timed_out = self.is_timed_out();
                break;
            }
            let mut inc_flag = 0u32;
            last = self.search_impl_v4(
                &mut tt,
                board,
                thpn,
                thdn,
                DEPTH_MAX_MATE_LEN,
                0,
                0u64,
                true,
                &mut inc_flag,
            );
            if last.pn() == 0 || last.dn() == 0 {
                break;
            }
            if last.pn() >= K_INFINITE_PN_DN || last.dn() >= K_INFINITE_PN_DN {
                break;
            }
            // NextPnDnThresholds: th = max(curr, val*1.7+1)．
            let cap = K_INFINITE_PN_DN - 1;
            let ntpn = thpn.max((last.pn() as f64 * 1.7) as PnDn + 1).min(cap);
            let ntdn = thdn.max((last.dn() as f64 * 1.7) as PnDn + 1).min(cap);
            if ntpn == thpn && ntdn == thdn {
                thpn = thpn.saturating_mul(2).saturating_add(1).min(cap);
                thdn = thdn.saturating_mul(2).saturating_add(1).min(cap);
                if thpn >= cap && thdn >= cap {
                    break;
                }
            } else {
                thpn = ntpn;
                thdn = ntdn;
            }
        }

        eprintln!(
            "[v4] root pn={} dn={} mate_len={} nodes={} tt_cap={}",
            last.pn(),
            last.dn(),
            last.len().len(),
            self.v3_nodes,
            tt.capacity()
        );

        if last.pn() == 0 {
            // proven．STRICT PV replay の v4 版は未配線のため，健全性は root mate_len で当面評価する
            // (29te → mate-29 が期待値)．PV は未構築 (空)．
            TsumeResult::Checkmate {
                moves: Vec::new(),
                nodes_searched: self.v3_nodes,
            }
        } else if last.dn() == 0 {
            TsumeResult::NoCheckmate {
                nodes_searched: self.v3_nodes,
            }
        } else {
            TsumeResult::Unknown {
                nodes_searched: self.v3_nodes,
            }
        }
    }

    /// KH `SearchImpl` (komoring_heights.cpp:389) の忠実移植．len threading 込み．
    #[allow(clippy::too_many_arguments)]
    fn search_impl_v4(
        &mut self,
        tt: &mut TranspositionTable,
        board: &mut Board,
        mut thpn: PnDn,
        mut thdn: PnDn,
        len: MateLen,
        depth: u32,
        path_key: u64,
        first_search: bool,
        inc_flag: &mut u32,
    ) -> SearchResult {
        self.v3_nodes += 1;
        if self.v3_nodes >= self.max_nodes
            || (self.v3_nodes & 0x3FF == 0 && self.is_timed_out())
        {
            self.timed_out = self.timed_out || self.is_timed_out();
            // budget 切れは非 final unknown で unwind (root の IDS が次で break)．
            return SearchResult::make_first_visit(1, 1, len, 1);
        }

        let attacker_hand = board.hand[self.attacker.index()];
        let mut exp = match self.build_v4_expansion(tt, board, len, depth, path_key, first_search) {
            Ok(e) => e,
            Err(terminal) => return terminal,
        };

        let orig_thpn = thpn;
        let orig_thdn = thdn;
        let orig_inc = *inc_flag;

        let mut curr = exp.current_result(board, depth as i32);
        if curr.is_final() {
            return curr;
        }
        if exp.does_have_old_child() {
            *inc_flag += 1;
        }
        if *inc_flag > 0 {
            extend_search_threshold(curr, &mut thpn, &mut thdn);
        }

        self.v3_path.insert(board.hash, depth);

        while curr.pn() < thpn && curr.dn() < thdn {
            if self.v3_nodes >= self.max_nodes || self.timed_out {
                break;
            }
            let best_move = exp.best_move();
            let is_first = exp.front_is_first_visit();
            let (cthpn, cthdn) = exp.front_pn_dn_thresholds(thpn, thdn);
            let best_raw = exp.front_raw();
            let child_query = exp.query_at(best_raw);

            let captured = board.do_move(best_move);
            let child_pk = path_key_after(path_key, best_move, depth as usize);

            let child_result = if is_first {
                let initial = exp.front_result();
                if *inc_flag > 0 {
                    *inc_flag -= 1;
                }
                if initial.is_final() || initial.pn() >= cthpn || initial.dn() >= cthdn {
                    initial
                } else {
                    self.search_impl_v4(
                        tt, board, cthpn, cthdn, len.sub(1), depth + 1, child_pk, is_first, inc_flag,
                    )
                }
            } else {
                self.search_impl_v4(
                    tt, board, cthpn, cthdn, len.sub(1), depth + 1, child_pk, is_first, inc_flag,
                )
            };

            board.undo_move(best_move, captured);

            exp.update_best_child(child_result);
            // KH UpdateBestChild 内 query.SetResult: 子結果を子 query で TT へ (親 = 本ノード)．
            tt.set_result(&child_query, child_result, (board.hash, attacker_hand));

            curr = exp.current_result(board, depth as i32);

            thpn = orig_thpn;
            thdn = orig_thdn;
            if *inc_flag > 0 {
                extend_search_threshold(curr, &mut thpn, &mut thdn);
            } else if *inc_flag == 0 && orig_inc > 0 {
                break;
            }
        }

        self.v3_path.remove(&board.hash);
        *inc_flag = (*inc_flag).min(orig_inc);
        curr
    }

    /// KH `LocalExpansion` ctor (3b 配線): movegen + 各子 seed (faithful TT LookUp) + 千日手．
    /// `Err(result)` = 終局 (合法手なし) の即時結果．`Ok(exp)` = 展開済ノード．
    fn build_v4_expansion(
        &mut self,
        tt: &mut TranspositionTable,
        board: &mut Board,
        len: MateLen,
        depth: u32,
        path_key: u64,
        first_search: bool,
    ) -> Result<LocalExpansion, SearchResult> {
        let attacker = self.attacker;
        let or_node = board.turn == attacker;

        let mut moves: Vec<crate::moves::Move> = Vec::new();
        if or_node {
            self.check_moves_into(board, &mut moves);
        } else {
            let av = self.generate_defense_moves_inner(board, false);
            moves.extend_from_slice(av.as_slice());
        }

        if moves.is_empty() {
            // OR (王手なし) = 不詰 disproven / AND (受けなし=詰み) = proven (詰み完了 len 0)．
            // 終端 hand は KH `HandSet{...}.Get` の極小/極大版を使う (cross-hand 再利用効率)．
            let r = if or_node {
                let dh = super::proof_hand::disproof_hand_terminal_or(board);
                SearchResult::make_final(false, dh, DEPTH_MAX_MATE_LEN, 1)
            } else {
                let ph = super::proof_hand::proof_hand_terminal_and(board);
                SearchResult::make_final(true, ph, ZERO_MATE_LEN, 1)
            };
            return Err(r);
        }

        // KH `delayed_move_list_{n, mp_}` (local_expansion.hpp:155)．同マス合駒/成不成ペアを
        // 双方向 chain 化し，prev が未 final の手は idx から除外 (= 後回し) する．
        let dml = DelayedMoveList::build(&moves, or_node);

        let defender_king = board.king_square(attacker.opponent());
        let n = moves.len();
        let mut evals: Vec<i32> = Vec::with_capacity(n);
        let mut results: Vec<SearchResult> = Vec::with_capacity(n);
        let mut queries = Vec::with_capacity(n);
        let mut idx: Vec<u32> = Vec::with_capacity(n);
        let mut does_have_old = false;

        for (i, &m) in moves.iter().enumerate() {
            // eval / seed は親局面で計算 (KH: MoveBriefEvaluation / InitialPnDn(n, move))．
            let eval = match defender_king {
                Some(k) => super::move_brief_eval(m, k, board),
                None => 0,
            };
            let (sp, sd) = if or_node {
                super::init_pn_dn_or_kh(board, m, attacker)
            } else {
                super::init_pn_dn_and_kh(board, m, attacker)
            };
            let seed = ((sp as u64 / DIV).max(1), (sd as u64 / DIV).max(1));

            let captured = board.do_move(m);
            let ch = board.hash;
            let child_hand = board.hand[attacker.index()];
            let child_pk = path_key_after(path_key, m, depth as usize);
            let q = tt.build_query(child_pk, ch, child_hand, (depth + 1) as i32);

            let mut r = if let Some(&anc_ply) = self.v3_path.get(&ch) {
                // path 上の同一局面 = 千日手 (KH IsRepetitionAfter)．
                SearchResult::make_repetition(child_hand, len, 1, anc_ply as i32)
            } else {
                let mut dhoc = false;
                let res = tt.look_up(&q, len.sub(1), &mut dhoc, || seed);
                does_have_old = does_have_old || dhoc;
                res
            };
            // KH DML skip 判定 (local_expansion.hpp:194-203): 非 final かつ prev chain に未 final の
            // 先行手があれば後回し (idx に積まない)．prev は i より前に処理済なので results が揃っている．
            let is_skipped =
                !r.is_final() && dml.has_unresolved_prev(i, |j| results[j].is_final());

            // KH `CheckObviousFinalOrNode` 先読み (local_expansion.hpp:217-221)．non-skipped のみ．
            // AND node の first-visit child は子が OR node (攻め方手番)．board は do_move 済 (= 子局面)
            // なので攻め方の 1 手詰／詰み無を先読みし，proven/disproven を seed する．これにより
            // 「詰む応手」を展開せず除外，「逃れる応手」を即 disproof し breadth を抑える．
            // 子結果が final になったら KH `query.SetResult` 相当で TT へ格納する (PV/伝播の整合)．
            if !is_skipped && !or_node && first_search && r.is_first_visit() {
                if let Some(res) = self.check_obvious_final_or_node_v4(board) {
                    tt.set_result(&q, res, (ch, child_hand));
                    r = res;
                }
            }
            board.undo_move(m, captured);

            evals.push(eval);
            results.push(r);
            queries.push(q);
            if !is_skipped {
                idx.push(i as u32);
            }
        }

        // KH revival 用 next chain (delayed_move_list_.Next)．後回し手が final 化したら
        // update_best_child が dml_next を辿って idx へ復活させる．
        let mut dml_next = vec![-1i32; n];
        for (i, slot) in dml_next.iter_mut().enumerate() {
            if let Some(nx) = dml.next(i) {
                *slot = nx as i32;
            }
        }

        Ok(LocalExpansion::from_parts(
            or_node,
            len,
            moves,
            evals,
            queries,
            results,
            idx,
            BitSet64::full(),
            does_have_old,
            dml_next,
            1,
        ))
    }

    /// KH `detail::CheckObviousFinalOrNode` (local_expansion.hpp:47) の忠実移植．
    ///
    /// `board` は子 OR node (攻め方手番) へ do_move 済前提．末端の固定深さ探索により
    /// 自明な詰み (1 手詰) / 不詰 (王手手段なし) を**展開せず**検知する:
    /// - 王手手段なし (`!DoesHaveMatePossibility`) → disproven (不詰確定)．
    /// - 1 手詰あり (`CheckMate1Ply`) → proven (mate-1)．
    ///
    /// proof/disproof hand は KH `HandSet{...}.Get` の極小/極大版を使う (cross-hand 再利用効率)．
    /// この極小化が無いと，look-ahead 葉が full hand を store し，AND 集約 (`ProofHandSet` max) で
    /// full hand が上方伝播してしまい cross-hand Superior/Inferior が効かない (= breadth)．
    fn check_obvious_final_or_node_v4(&self, board: &mut Board) -> Option<SearchResult> {
        let (mm_opt, has_checks) = self.mate1ply_with_cached_checks(board);
        if !has_checks {
            // 攻め方に王手手段なし → 詰み不可能 → 不詰 (KH MakeFinal<false>, kDepthMaxMateLen)．
            // 反証 hand = KH `HandSet{DisproofHandTag}.Get` の極小化版．
            let dh = super::proof_hand::disproof_hand_terminal_or(board);
            Some(SearchResult::make_final(false, dh, DEPTH_MAX_MATE_LEN, 1))
        } else if let Some(mm) = mm_opt {
            // 1 手詰 → 詰み proven (mate-1)．極小証明駒 = 詰め上がり局面の終端証明駒
            // (`proof_hand_terminal_and`) を mate 着手の手前へ `before_hand` で逆算する (KH BeforeHand)．
            let cap = board.do_move(mm);
            let term = super::proof_hand::proof_hand_terminal_and(board);
            board.undo_move(mm, cap);
            let ph = super::proof_hand::before_hand(board, mm, term);
            Some(SearchResult::make_final(true, ph, MateLen::from_len(1), 1))
        } else {
            None
        }
    }
}
