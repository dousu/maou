//! STRICT PV replay と PV 復元．
//!
//! proven 後の TT を辿り，証明木が完全な強制詰みかを実手 replay で厳密検証し
//! (`verify_proof`)，検証済 memo から PV 手順を復元する (`build_pv`)．
//! いずれも探索本体 (`solve_impl`) の証明後フェーズから呼ばれる．

use crate::board::Board;

use super::mate_len::DEPTH_MAX_MATE_LEN;
use super::search_result::K_INFINITE_PN_DN;
use super::solver::DfPnSolver;
use super::tt::TranspositionTable;

impl DfPnSolver {
    /// STRICT PV replay．
    ///
    /// proven 後の TT を辿り，証明木が **完全な強制詰み** か実際の手の replay で厳密検証する:
    /// - **OR (攻め)**: ① 直接 1 手詰 (look-ahead leaf は AND-grandchild を TT 格納しないため
    ///   TT-only 選択では取りこぼす → 先に `mate1ply` で検出) → ② TT-proven child を proven_len
    ///   昇順で replay 検証．いずれかが詰みに帰着すれば `Some(d+1)`．
    /// - **AND (受け)**: `generate_defense_moves_inner` で **全合法防御**を列挙し，各々が詰みに
    ///   帰着するか確認 (futile filter は探索と同一 move set)．1 つでも逃れれば `None` (偽証明)．
    /// - 末端: AND 手なし & 王手 = 詰み `Some(0)`．OR proven child 無し = 不完全 `None`．
    /// - path 上の同一局面 = 千日手 = 受け方脱出 = `None`．`memo` で局面を重複検証しない．
    pub(super) fn verify_proof(
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
                    let r = self.verify_proof(tt, board, path, memo, budget);
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
                    let r = self.verify_proof(tt, board, path, memo, budget);
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

    /// 検証済 `memo` (verify_proof の局面別詰み距離) を辿って PV を復元する．
    /// OR=最短詰み手 (1 手詰優先 / memo 距離最小)，AND=max-resistance (memo 距離最大) を選ぶ．
    /// `board` は破壊しないよう clone 上で前進する．
    pub(super) fn build_pv(
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
