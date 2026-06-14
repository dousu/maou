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
//! ## 移植済 (29te: 81,496 → 18,399, -77%; KH 19,270 並に到達)
//! - **1 手詰先読み (CheckObviousFinalOrNode, v2.11.0)**: AND first-visit child を do_move し OR 子の
//!   1 手詰/詰み無を先読みし proven/disproven seed (81,496→48,662, -40%; [`check_obvious_final_or_node_v4`])．
//! - **DML deferral (v2.12.0)**: `DelayedMoveList` を build_v4_expansion に配線．同マス合駒/成不成ペアの
//!   prev chain が未 final の手を idx から除外 (後回し) し，final 化で `update_best_child` が revival
//!   (48,662→36,768, -24%)．KH ctor の `i_is_skipped` を忠実再現．
//! - **🎯 position-only TT board_key (v2.13.0)**: TT を `position_key` (board_hash, 持駒除外; KH BoardKey)
//!   で索引し，hand は entry に別管理する．従来は `board.hash` (持駒込) を board_key にしていたため
//!   cross-hand Superior/Inferior が**全く発火していなかった**．修正で cross-hand 再利用が有効化
//!   (36,768→18,399, -50%, mate-29 健全)．**これが KH parity 到達の本丸**．
//!
//! ## 反証 / 未移植
//! - ❌ **proof hand 極小化 (KH HandSet) は unsound**: cross-hand 有効化後に minimal proof hand を使うと
//!   過剰な Superior 再利用で **mate-39 偽証明**を生む (v2.11.0 で実装→v2.13.0 で撤回, full hand は sound)．
//!   KH では sound なので maou 移植側に bug あり (`before_hand`/`ProofHandSet`/`add_if`)．要 debug (低優先)．
//! - **EliminateDoubleCount (DAG)**: KH は SearchImpl 毎に ancestor 展開 stack を walk (mid_v4 は再帰 stack)．
//! - STRICT PV replay の v4 版 / 無意味中合いの cross-square DML．

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

        // len-aware TT (local; 再帰へ &mut で渡す)．サイズは budget 比例で確保し，満杯時は
        // GC (maybe_collect_garbage) で低 amount entry を間引く．`V4_TTSIZE` で entry 数を上書き可．
        let size = if let Ok(s) = std::env::var("V4_TTSIZE") {
            s.parse::<usize>().unwrap_or(1 << 23).max(1 << 12)
        } else {
            ((self.max_nodes as usize).saturating_mul(2)).clamp(1 << 18, 1 << 23)
        };
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
            "[v4] root pn={} dn={} mate_len={} nodes={} tt_cap={} gc={}",
            last.pn(),
            last.dn(),
            last.len().len(),
            self.v3_nodes,
            tt.capacity(),
            tt.gc_count()
        );

        if last.pn() == 0 {
            // STRICT PV replay (v4 版): proven 証明木を実際に replay し，OR は proven child を，
            // AND は **全合法防御** を列挙して詰みに帰着するか厳密検証する (TT pn/dn を信用しない)．
            // Some(d) = sound mate-d / None = 偽証明 or 不完全 (budget 枯渇は別表示)．
            let mut path: Vec<u64> = Vec::new();
            let mut memo: std::collections::HashMap<u64, Option<u16>> =
                std::collections::HashMap::new();
            let mut budget: u64 = 80_000_000;
            let verified = self.verify_v4_proof(&mut tt, board, &mut path, &mut memo, &mut budget);
            match verified {
                Some(d) => eprintln!(
                    "[v4] STRICT VERIFY Some({}) (root mate_len={}, budget_left={})",
                    d,
                    last.len().len(),
                    budget
                ),
                None if budget == 0 => {
                    eprintln!("[v4] STRICT VERIFY INCONCLUSIVE (budget exhausted)")
                }
                None => eprintln!("[v4] STRICT VERIFY None — UNSOUND or incomplete proof tree"),
            }
            // PV は memo (検証済距離) を辿って復元 (OR=最短 proven, AND=max-resistance)．
            let pv = self.build_v4_pv(board, &mut tt, &memo, last.len().len() as usize + 8);
            TsumeResult::Checkmate {
                moves: pv,
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

        // KH 流 GC (komoring_heights.cpp:446): hashfull が 50% を超えたら低 amount entry を
        // 間引く．これを怠るとテーブル満杯時に `look_up` の probe が O(cap) へ退化し，39te の
        // ような大規模探索で per-node 時間が爆発する (KH は探索中つねに <50% を維持)．
        // 4096 node 毎にだけ確認 (KH `kHashfullCheeckSkipRatio`)．節点が少ない局面 (29te) では
        // 50% に到達しないため no-op = 既存挙動不変．
        if self.v3_nodes & 0xFFF == 0 {
            tt.maybe_collect_garbage();
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
            // 親 board_key は **position-only** (KH BoardKeyHandPair; cross-hand のため hand は別管理)．
            tt.set_result(
                &child_query,
                child_result,
                (super::position_key(board), attacker_hand),
            );

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
            // hand は full attacker_hand (KH HandSet 極小化は cross-hand 有効時 unsound のため不使用)．
            let attacker_hand = board.hand[attacker.index()];
            let r = if or_node {
                SearchResult::make_final(false, attacker_hand, DEPTH_MAX_MATE_LEN, 1)
            } else {
                SearchResult::make_final(true, attacker_hand, ZERO_MATE_LEN, 1)
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
            // **TT board_key は position-only** (`position_key`; KH BoardKey)．hand は entry に別管理し
            // cross-hand Superior/Inferior を効かせる．千日手判定だけは full hash (持駒込みの同一局面)．
            let ch_full = board.hash;
            let ch_pos = super::position_key(board);
            let child_hand = board.hand[attacker.index()];
            let child_pk = path_key_after(path_key, m, depth as usize);
            let q = tt.build_query(child_pk, ch_pos, child_hand, (depth + 1) as i32);

            let mut r = if let Some(&anc_ply) = self.v3_path.get(&ch_full) {
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
                    tt.set_result(&q, res, (ch_pos, child_hand));
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
    /// proof/disproof hand は **full attacker_hand** (= 子 OR node の手番側持駒) を使う．KH `HandSet`
    /// 極小化は cross-hand TT 有効時に偽証明 (mate-39) を生むため不使用 (sound 優先)．
    fn check_obvious_final_or_node_v4(&self, board: &mut Board) -> Option<SearchResult> {
        let or_hand = board.hand[board.turn.index()];
        let (mm_opt, has_checks) = self.mate1ply_with_cached_checks(board);
        if !has_checks {
            // 攻め方に王手手段なし → 詰み不可能 → 不詰 (KH MakeFinal<false>, kDepthMaxMateLen)．
            Some(SearchResult::make_final(false, or_hand, DEPTH_MAX_MATE_LEN, 1))
        } else if mm_opt.is_some() {
            // 1 手詰 → 詰み proven (mate-1)．
            Some(SearchResult::make_final(true, or_hand, MateLen::from_len(1), 1))
        } else {
            None
        }
    }

    /// STRICT PV replay (mid_v3 `verify_v3_proof` mid_v3.rs:660 の v4 版)．
    ///
    /// proven 後の TT を辿り，証明木が **完全な強制詰み** か実際の手の replay で厳密検証する:
    /// - **OR (攻め)**: ① 直接 1 手詰 (look-ahead leaf は AND-grandchild を TT 格納しないため
    ///   TT-only 選択では取りこぼす → 先に `mate1ply` で検出) → ② TT-proven child を proven_len
    ///   昇順で replay 検証．いずれかが詰みに帰着すれば `Some(d+1)`．
    /// - **AND (受け)**: `generate_defense_moves_inner` で **全合法防御**を列挙し，各々が詰みに
    ///   帰着するか確認 (futile filter は探索と同一 move set)．1 つでも逃れれば `None` (偽証明)．
    /// - 末端: AND 手なし & 王手 = 詰み `Some(0)`．OR proven child 無し = 不完全 `None`．
    /// - path 上の同一局面 = 千日手 = 受け方脱出 = `None`．`memo` で局面を重複検証しない．
    fn verify_v4_proof(
        &mut self,
        tt: &mut TranspositionTable,
        board: &mut Board,
        path: &mut Vec<u64>,
        memo: &mut std::collections::HashMap<u64, Option<u16>>,
        budget: &mut u64,
    ) -> Option<u16> {
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        let h = board.hash;
        if path.contains(&h) {
            return None; // 千日手 = 受け方脱出 = 不詰
        }
        if let Some(&r) = memo.get(&h) {
            return r;
        }
        let attacker = self.attacker;
        let or_node = board.turn == attacker;
        let result = if or_node {
            // (1) 直接 1 手詰 (look-ahead で seed された leaf を確実に拾う)．
            let mut or_res: Option<u16> = None;
            let (mm, _has_checks) = self.mate1ply_with_cached_checks(board);
            if let Some(mv) = mm {
                let cap = board.do_move(mv);
                let mated = self.generate_defense_moves_inner(board, false).is_empty()
                    && board.is_in_check(board.turn);
                board.undo_move(mv, cap);
                if mated {
                    or_res = Some(1);
                }
            }
            // (2) TT-proven child を proven_len 昇順で replay 検証．
            if or_res.is_none() {
                let mut moves: Vec<crate::moves::Move> = Vec::new();
                self.check_moves_into(board, &mut moves);
                let mut cands: Vec<(crate::moves::Move, u16)> = Vec::new();
                for &m in &moves {
                    let cap = board.do_move(m);
                    let ch_pos = super::position_key(board);
                    let child_hand = board.hand[attacker.index()];
                    let q = tt.build_query(0, ch_pos, child_hand, 0);
                    let mut dhoc = false;
                    let r = tt.look_up(&q, DEPTH_MAX_MATE_LEN, &mut dhoc, || {
                        (K_INFINITE_PN_DN, K_INFINITE_PN_DN)
                    });
                    board.undo_move(m, cap);
                    if r.pn() == 0 {
                        cands.push((m, r.len().len() as u16));
                    }
                }
                cands.sort_by_key(|&(_, l)| l);
                path.push(h);
                for (mv, _) in cands {
                    let cap = board.do_move(mv);
                    let r = self.verify_v4_proof(tt, board, path, memo, budget);
                    board.undo_move(mv, cap);
                    if let Some(d) = r {
                        or_res = Some(d + 1);
                        break;
                    }
                }
                path.pop();
            }
            or_res
        } else {
            // AND: 全合法防御を列挙し各々が詰みへ帰着するか (max-resistance)．
            let legal: Vec<crate::moves::Move> = self
                .generate_defense_moves_inner(board, false)
                .as_slice()
                .to_vec();
            if legal.is_empty() {
                if board.is_in_check(board.turn) {
                    Some(0) // 受けなし & 王手 = 詰み
                } else {
                    None // stalemate 非王手 = 詰みでない
                }
            } else {
                path.push(h);
                let mut maxd = 0u16;
                let mut ok = true;
                for m in &legal {
                    let cap = board.do_move(*m);
                    let r = self.verify_v4_proof(tt, board, path, memo, budget);
                    board.undo_move(*m, cap);
                    match r {
                        Some(d) => maxd = maxd.max(d + 1),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                path.pop();
                if ok {
                    Some(maxd)
                } else {
                    None
                }
            }
        };
        memo.insert(h, result);
        result
    }

    /// 検証済 `memo` (verify_v4_proof の局面別詰み距離) を辿って PV を復元する．
    /// OR=最短詰み手 (1 手詰優先 / memo 距離最小)，AND=max-resistance (memo 距離最大) を選ぶ．
    /// `board` は破壊しないよう clone 上で前進する．
    fn build_v4_pv(
        &mut self,
        board: &Board,
        tt: &mut TranspositionTable,
        memo: &std::collections::HashMap<u64, Option<u16>>,
        max_steps: usize,
    ) -> Vec<crate::moves::Move> {
        let attacker = self.attacker;
        let mut b = board.clone();
        let mut pv: Vec<crate::moves::Move> = Vec::new();
        for _ in 0..max_steps {
            let or_node = b.turn == attacker;
            let chosen = if or_node {
                // 1 手詰優先 (look-ahead leaf)．
                let (mm, _) = self.mate1ply_with_cached_checks(&mut b);
                let mut pick: Option<crate::moves::Move> = None;
                if let Some(mv) = mm {
                    let cap = b.do_move(mv);
                    let mated = self.generate_defense_moves_inner(&mut b, false).is_empty()
                        && b.is_in_check(b.turn);
                    b.undo_move(mv, cap);
                    if mated {
                        pick = Some(mv);
                    }
                }
                if pick.is_none() {
                    let mut moves: Vec<crate::moves::Move> = Vec::new();
                    self.check_moves_into(&mut b, &mut moves);
                    let mut best: Option<(crate::moves::Move, u16)> = None;
                    for &m in &moves {
                        let cap = b.do_move(m);
                        let dist = memo.get(&b.hash).copied().flatten();
                        // memo に無くても TT-proven なら proven_len で代替評価．
                        let proven_len = if dist.is_none() {
                            let q = tt.build_query(
                                0,
                                super::position_key(&b),
                                b.hand[attacker.index()],
                                0,
                            );
                            let mut dhoc = false;
                            let r = tt.look_up(&q, DEPTH_MAX_MATE_LEN, &mut dhoc, || {
                                (K_INFINITE_PN_DN, K_INFINITE_PN_DN)
                            });
                            if r.pn() == 0 {
                                Some(r.len().len() as u16)
                            } else {
                                None
                            }
                        } else {
                            dist
                        };
                        b.undo_move(m, cap);
                        if let Some(d) = proven_len {
                            if best.map_or(true, |(_, bd)| d < bd) {
                                best = Some((m, d));
                            }
                        }
                    }
                    pick = best.map(|(m, _)| m);
                }
                pick
            } else {
                // AND: max-resistance defense (memo 距離最大)．手なし=詰み終端．
                let legal: Vec<crate::moves::Move> = self
                    .generate_defense_moves_inner(&mut b, false)
                    .as_slice()
                    .to_vec();
                if legal.is_empty() {
                    None
                } else {
                    let mut best: Option<(crate::moves::Move, u16)> = None;
                    for &m in &legal {
                        let cap = b.do_move(m);
                        let dist = memo.get(&b.hash).copied().flatten().unwrap_or(0);
                        b.undo_move(m, cap);
                        if best.map_or(true, |(_, bd)| dist > bd) {
                            best = Some((m, dist));
                        }
                    }
                    best.map(|(m, _)| m)
                }
            };
            match chosen {
                Some(mv) => {
                    pv.push(mv);
                    b.do_move(mv);
                }
                None => break,
            }
        }
        pv
    }
}
