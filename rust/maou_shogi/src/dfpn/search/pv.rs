//! STRICT PV replay と PV 復元．
//!
//! proven 後の TT を辿り，証明木が完全な強制詰みかを実手 replay で厳密検証し
//! (`verify_proof`)，検証済 memo から PV 手順を復元する (`build_pv`)．
//! いずれも探索本体 (`solve_impl`) の証明後フェーズから呼ばれる．

use crate::board::Board;
use crate::moves::Move;
use std::collections::HashMap;

use crate::dfpn::mate_len::DEPTH_MAX_MATE_LEN;
use crate::dfpn::search_result::K_INFINITE_PN_DN;
use crate::dfpn::solver::DfPnSolver;
use crate::dfpn::tt::TranspositionTable;

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
        memo: &mut HashMap<u64, Option<u16>>,
        pv_choice: &mut HashMap<u64, Move>,
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
                    pv_choice.insert(h, mv);
                    or_res = Some(1);
                }
            }
            // (2) TT-proven child を全件 replay 検証し，**無駄合い除外後の最短** child を選ぶ
            //     (= 攻め方は最短詰みを選ぶ)．proven_len 昇順で評価しつつ全件 verify する．
            if or_res.is_none() {
                let mut moves: Vec<Move> = Vec::new();
                self.check_moves_into(board, &mut moves);
                let mut cands: Vec<(Move, u16)> = Vec::new();
                for &m in &moves {
                    let cap = board.do_move(m);
                    let ch_pos = crate::dfpn::position_key(board);
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
                let mut best: Option<(Move, u16)> = None;
                for (mv, _) in cands {
                    let cap = board.do_move(mv);
                    let r = self.verify_proof(tt, board, path, memo, pv_choice, budget);
                    board.undo_move(mv, cap);
                    if let Some(d) = r {
                        let dist = d + 1;
                        if best.map_or(true, |(_, bd)| dist < bd) {
                            best = Some((mv, dist));
                        }
                    }
                }
                path.pop();
                if let Some((mv, dist)) = best {
                    pv_choice.insert(h, mv);
                    or_res = Some(dist);
                }
            }
            or_res
        } else {
            // AND: 全合法防御を列挙し各々が詰みへ帰着するか確認 (max-resistance, 無駄合い除外)．
            // **無駄合い (合い駒で詰み手数が中合いの手数を除いて伸びないもの) は手数に数えない**．
            // length-based 一般判定: 王手中の AND node で drop は必ず合駒 (中合い)．合駒 m が無駄合いなのは
            //   (A) 取り返し透過: child OR の最適手が合駒マス X を取り返し，取り返し後 M' でチェッカーが
            //       生き残り mate を継続する (= 捨て取りされて消えるだけ; 連続中合いは再帰で collapse)，または
            //   (B) 非伸長: 合駒の詰み距離 (d+1) が **非合駒防御 (玉移動/王手駒取り) の最長 base** を超えない
            //       (= 玉移動と同等以下しか粘れない; 攻め方に無視される合駒もここで除外される)．
            // (A)/(B) いずれにも当たらない合駒は **net で base を超えて伸びる有効中合い** (例: S*6i→…→6h6i+
            // でチェッカーを取り返す筋) なので手数に数える．無駄合いを除いても base が max-resistance を
            // 担保するため **除外は sound** (偽詰みを生まない)．確信が持てない場合は数える保守側に倒す．
            let legal: Vec<Move> = self
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
                // pass 1: 全防御を verify し (距離, drop か, 取り返し透過無駄合いか) を収集．
                // base = 非 drop (玉移動/王手駒取り) 防御の最長詰み距離 (d+1)．
                let mut ok = true;
                let mut infos: Vec<(Move, u16, bool, bool)> = Vec::with_capacity(legal.len());
                let mut base = 0u16;
                for m in &legal {
                    let x = m.to_sq();
                    let is_drop = m.is_drop();
                    let cap = board.do_move(*m);
                    let child_h = board.hash;
                    let r = self.verify_proof(tt, board, path, memo, pv_choice, budget);
                    // (A) 取り返し透過の判定．board は verify 後 child OR 局面 (m do_move 済)．
                    let mut recapture_transparent = false;
                    if is_drop {
                        if let Some(&amove) = pv_choice.get(&child_h) {
                            if amove.to_sq() == x {
                                let cap2 = board.do_move(amove);
                                let gh = board.hash;
                                let checker_survives =
                                    pv_choice.get(&gh).map_or(true, |dm| dm.to_sq() != x);
                                board.undo_move(amove, cap2);
                                recapture_transparent = checker_survives;
                            }
                        }
                    }
                    board.undo_move(*m, cap);
                    match r {
                        Some(d) => {
                            if !is_drop {
                                base = base.max(d + 1);
                            }
                            infos.push((*m, d, is_drop, recapture_transparent));
                        }
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                path.pop();
                if ok {
                    // pass 2: 無駄合いを除外して max-resistance を取る．
                    let mut maxd = 0u16;
                    let mut best_move: Option<Move> = None;
                    let mut best_is_drop = true;
                    for &(m, d, is_drop, recapture_transparent) in &infos {
                        let contrib = d + 1;
                        // 無駄合い: (A) 取り返し透過，または (B) 非伸長 (base を超えない合駒)．
                        let mudaai = is_drop && (recapture_transparent || contrib <= base);
                        if mudaai {
                            continue;
                        }
                        if contrib > maxd
                            || best_move.is_none()
                            || (contrib == maxd && best_is_drop && !is_drop)
                        {
                            maxd = contrib;
                            best_move = Some(m);
                            best_is_drop = is_drop;
                        }
                    }
                    if let Some(bm) = best_move {
                        pv_choice.insert(h, bm);
                    }
                    // 全防御が無駄合い (best_move なし) なら maxd=0 = 王手詰み (無駄合いを除けば受けなし)．
                    Some(maxd)
                } else {
                    None
                }
            }
        };
        memo.insert(h, result);
        result
    }

    /// `verify_proof` が記録した局面別の最適手 (`pv_choice`) を辿って PV を復元する．
    /// OR=無駄合い除外後の最短詰み手，AND=max-resistance (無駄合いは除外済) を選ぶ．
    /// 無駄合いは `verify_proof` の集計で除外されているため，PV にも現れない (= 手数に数えない)．
    /// `board` は破壊しないよう clone 上で前進する．
    pub(super) fn build_pv(
        &mut self,
        board: &Board,
        pv_choice: &HashMap<u64, Move>,
        max_steps: usize,
    ) -> Vec<Move> {
        let mut b = board.clone();
        let mut pv: Vec<Move> = Vec::new();
        for _ in 0..max_steps {
            match pv_choice.get(&b.hash) {
                Some(&mv) => {
                    pv.push(mv);
                    b.do_move(mv);
                }
                None => break,
            }
        }
        pv
    }
}
