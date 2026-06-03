//! mid_v3: KomoringHeights 探索コアの ground-up 移植 (Phase 32)．
//!
//! mid_v2 は per-node 機構 (seed/ordering/threshold formula) は KH 等価だが，
//! scale (unit-16) + cumulative extend + maou TT の **emergent dynamics** で footprint が
//! KH の 14× になる (診断で確定: reuse 1.89× vs KH 9.2×, best-child flicker)．
//!
//! mid_v3 は KH の **coherent whole** を別経路で再現する:
//!   - unit-2 scale (`V3_U=2`, `V3_INF`)．
//!   - KH `InitialPnDn` seed を unit-2 へ縮約．
//!   - KH `FrontPnDnThresholds` (second_phi+1) + `ExtendSearchThreshold` (**非累積** max(th,pn+1))．
//!   - KH `NextPnDnThresholds` (root IDS, ×1.7) — fine-grained re-descent で focused set へ収束．
//!   - **clean exact-match TT** (board.hash key) — mid_v2 TT の eviction/depth-filter baggage を排す．
//!
//! 全経路は `solve_via_v3` でのみ起動し，既存 mid_v2/baseline には一切影響しない (gated)．

use crate::board::Board;
use crate::moves::Move;
use super::solver::{DfPnSolver, TsumeResult};

/// kPnDnUnit 相当．
pub(super) const V3_U: u64 = 2;
/// kInfinitePnDn 相当 (u64::MAX/2-1)．
pub(super) const V3_INF: u64 = u64::MAX / 2 - 1;
/// 探索深さ上限 (千日手保険)．
const V3_MAX_PLY: u32 = 127;

/// Phase 33 (LE path) の INF sentinel．`MidLocalExpansion` が u32 + saturating_add で
/// 動くため LE path は全て u32 unit-2 で動作する．clean TT (`V3Entry`) には u64 として
/// 格納するが LE 値は常に `V3_INF_U` 以下なので往復で正確．
const V3_INF_U: u32 = u32::MAX;
/// LE path の「未確定 (budget 切れ)」を表す小さな unit 値．
const V3_U_U32: u32 = V3_U as u32;

/// mid_v3 の exact-match TT エントリ．
#[derive(Clone, Copy, Default)]
pub(super) struct V3Entry {
    pub pn: u64,
    pub dn: u64,
    /// proven 時の詰み手数 (mate distance)．
    pub len: u16,
    /// best child の move16．
    pub best: u16,
    /// この局面が格納された最も浅い ply (KH `min_depth_`)．child lookup 時 `min_depth < child_depth`
    /// なら is_shallow=true (= 浅い transposition) で TCA を発火させる (LE path のみ使用)．
    pub min_depth: u16,
}

impl V3Entry {
    #[inline]
    fn is_final(&self) -> bool {
        self.pn == 0 || self.dn == 0
    }
    /// or_node から見た phi (= 進めたい側の数値)．OR は pn, AND は dn．
    #[inline]
    fn phi(&self, or_node: bool) -> u64 {
        if or_node { self.pn } else { self.dn }
    }
    /// or_node から見た delta (= 相手側の数値)．OR は dn, AND は pn．
    #[inline]
    fn delta(&self, or_node: bool) -> u64 {
        if or_node { self.dn } else { self.pn }
    }
}

/// 探索中の子の状態 (LocalExpansion の results 相当)．
#[derive(Clone, Copy)]
struct V3Child {
    mv: Move,
    pn: u64,
    dn: u64,
    len: u16,
    is_first: bool,
    is_final: bool,
    /// この子の現在値が依存する千日手の最浅祖先 ply (なければ u32::MAX)．
    rep_min: u32,
    /// KH `MoveBriefEvaluation` (小さいほど良い手)．coherent LocalExpansion port 時に tie-break で使う．
    #[allow(dead_code)]
    eval: i32,
    /// KH DML: この子が active (展開対象) か．deferred な間は集約/選択から除外する (現状常に true)．
    active: bool,
    /// DML chain の次の手 (raw index)．この子が final 化したら next を activate する．-1 = 末尾．
    next_in_chain: i32,
    /// δ を sum で計上するか (KH sum_mask)．false = max 集約 (IsSumDeltaNode=false の near-dup)．
    sum_delta: bool,
}

impl V3Child {
    #[inline]
    fn phi(&self, or_node: bool) -> u64 {
        if or_node { self.pn } else { self.dn }
    }
    #[inline]
    fn delta(&self, or_node: bool) -> u64 {
        if or_node { self.dn } else { self.pn }
    }
}

#[inline]
fn clamp_inf(v: u64) -> u64 {
    if v > V3_INF { V3_INF } else { v }
}

/// KH `ExtendSearchThreshold`: 非累積 extend．thpn = max(thpn, pn+1)．
#[inline]
fn v3_extend(pn: u64, dn: u64, thpn: &mut u64, thdn: &mut u64) {
    // final (pn=0 or dn=0) では延長しない (curr が即解決のため意味なし)．
    if pn != 0 && dn != 0 {
        if pn < V3_INF {
            *thpn = (*thpn).max(pn + 1);
        }
        if dn < V3_INF {
            *thdn = (*thdn).max(dn + 1);
        }
    }
}

impl DfPnSolver {
    /// mid_v3 エントリ: KH SearchEntry 風 root IDS で 29te を解く．
    pub fn solve_via_v3(&mut self, board: &mut Board) -> TsumeResult {
        self.v3_tt.clear();
        self.v3_path.clear();
        self.v3_nodes = 0;
        self.attacker = board.turn;
        self.start_time = std::time::Instant::now();
        self.timed_out = false;

        // Phase 33 (案②): 検証済 MidLocalExpansion を per-node に駆動する LE path．
        if self.param_v3_local_exp {
            return self.solve_via_v3_le(board);
        }

        // KH SearchEntry: len=任意長 なので thpn/thdn は小さく始め NextPnDnThresholds で ×1.7 成長．
        let mut thpn: u64 = 1;
        let mut thdn: u64 = 1;
        let mut last = (V3_INF, V3_INF, 0u16, u32::MAX);
        loop {
            if self.v3_nodes >= self.max_nodes || self.is_timed_out() {
                self.timed_out = self.is_timed_out();
                break;
            }
            let mut inc_flag = 0u32;
            let r = self.search_v3(board, thpn, thdn, 0, &mut inc_flag);
            last = r;
            let (pn, dn, _len, _rep) = r;
            if pn == 0 || dn == 0 {
                break; // 解決 (詰み or 不詰)．
            }
            if pn >= V3_INF || dn >= V3_INF {
                break; // overflow / 未解決．
            }
            // NextPnDnThresholds: th = max(curr_th, pn*1.7+1)．
            let ntpn = clamp_inf((thpn).max((pn as f64 * 1.7) as u64 + 1));
            let ntdn = clamp_inf((thdn).max((dn as f64 * 1.7) as u64 + 1));
            if ntpn == thpn && ntdn == thdn {
                // 成長しない (両者 INF 等) → 終了．
                thpn = clamp_inf(thpn * 2 + 1);
                thdn = clamp_inf(thdn * 2 + 1);
                if thpn >= V3_INF && thdn >= V3_INF {
                    break;
                }
            } else {
                thpn = ntpn;
                thdn = ntdn;
            }
        }

        let (pn, _dn, _len, _rep) = last;
        if pn == 0 {
            if std::env::var("V3_DIAG").is_ok() {
                let mut path = Vec::new();
                let mut memo = std::collections::HashMap::new();
                let mut budget = 5_000_000u64;
                let v = self.verify_v3_proof(board, &mut path, &mut memo, &mut budget);
                eprintln!("[v3classic] STRICT VERIFY: {:?} (budget_left={budget})", v);
            }
            let pv = self.extract_pv_v3(board);
            let nodes = self.v3_nodes;
            TsumeResult::Checkmate {
                moves: pv,
                nodes_searched: nodes,
            }
        } else {
            TsumeResult::NoCheckmate {
                nodes_searched: self.v3_nodes,
            }
        }
    }

    /// KH `SearchImpl` の移植 (unit-2)．戻り値 (pn, dn, mate_len, rep_min)．
    ///
    /// `rep_min` = この結果が依存する千日手が参照する**最も浅い祖先 ply** (なければ `u32::MAX`)．
    /// GHI soundness: 結果が path 上流 (rep_min < 自 ply) の祖先に依存する場合，その結果は
    /// path-dependent なので TT へ absolute 格納してはならない (KH RepetitionTable の本質)．
    fn search_v3(
        &mut self,
        board: &mut Board,
        mut thpn: u64,
        mut thdn: u64,
        ply: u32,
        inc_flag: &mut u32,
    ) -> (u64, u64, u16, u32) {
        self.v3_nodes += 1;
        if self.v3_nodes >= self.max_nodes || (self.v3_nodes & 0x3FF == 0 && self.is_timed_out()) {
            return (V3_U, V3_U, 0, u32::MAX); // budget 切れ: 未知扱い．
        }

        let hash = board.hash;
        // 千日手: パス上に同一局面 → 進展なし = 不詰 (pn=INF, dn=0)．参照祖先 ply を rep_min に．
        if let Some(&anc_ply) = self.v3_path.get(&hash) {
            return (V3_INF, 0, 0, anc_ply);
        }
        if ply >= V3_MAX_PLY {
            return (V3_INF, 0, 0, ply);
        }

        // TT: final 済なら即返す (格納されているのは absolute なものだけ → rep_min=MAX)．
        if let Some(e) = self.v3_tt.get(&hash) {
            if e.is_final() {
                return (e.pn, e.dn, e.len, u32::MAX);
            }
        }

        let or_node = board.turn == self.attacker;

        // 子の生成: OR=王手, AND=受け (無駄合い filter 付き)．
        let moves: Vec<Move> = if or_node {
            self.generate_check_moves_cached(board).into_iter().collect()
        } else {
            self.generate_defense_moves_inner(board, false)
                .into_iter()
                .collect()
        };

        // 終端: OR で王手なし → 不詰 (pn=INF, dn=0)．AND で受けなし → 詰み (pn=0, dn=INF, len=0)．
        // 終端は path 非依存なので absolute (rep_min=MAX)．
        if moves.is_empty() {
            let r = if or_node {
                (V3_INF, 0u64, 0u16)
            } else {
                (0u64, V3_INF, 0u16)
            };
            self.v3_tt.insert(hash, V3Entry { pn: r.0, dn: r.1, len: r.2, best: 0, min_depth: 0 });
            return (r.0, r.1, r.2, u32::MAX);
        }

        // OR ノード 1 手詰判定 (KH CheckObviousFinalOrNode 相当)．absolute．
        if or_node {
            let checks_av = self.generate_check_moves_cached(board);
            if let Some(mm) = board.mate_move_in_1ply(checks_av.as_slice(), board.turn) {
                self.v3_tt.insert(hash, V3Entry { pn: 0, dn: V3_INF, len: 1, best: mm.to_move16(), min_depth: 0 });
                return (0, V3_INF, 1, u32::MAX);
            }
        }

        // KH MoveBriefEvaluation 用の玉 = 常に受け方 (詰まされる側) の玉 (n.KingSquare 相当)．
        let us_is_black = board.turn == crate::types::Color::Black;
        let defender_king = board.king_square(self.attacker.opponent());
        // KH DML: 同一マス合駒 / 成不成ペアの chain を構築．head のみ active で開始．
        let (dml_prev, dml_next) = super::mid_v2::build_delayed_chain(&moves, or_node, true, us_is_black);

        // 子の初期化: seed (unit-2) または TT 値．
        let mut children: Vec<V3Child> = Vec::with_capacity(moves.len());
        for (mi, &m) in moves.iter().enumerate() {
            // seed は board=parent / move=m で計算 (KH InitialPnDn と同じ呼び方)．
            let (sp, sd) = if or_node {
                super::init_pn_dn_or_kh(board, m, self.attacker)
            } else {
                super::init_pn_dn_and_kh(board, m, self.attacker)
            };
            let div = (PN_UNIT_SCALE).max(1);
            let seed_pn = ((sp as u64) / div).max(1);
            let seed_dn = ((sd as u64) / div).max(1);
            let eval = match defender_king {
                Some(ksq) => super::move_brief_eval(m, ksq, board),
                None => 0,
            };

            let captured = board.do_move(m);
            let ch = board.hash;
            let (cpn, cdn, clen, is_first, is_final, rep_min) =
                if let Some(&anc_ply) = self.v3_path.get(&ch) {
                    // 子が千日手 → 不詰扱い (dn=0)．参照祖先 ply を rep_min に．
                    (V3_INF, 0u64, 0u16, false, true, anc_ply)
                } else if let Some(e) = self.v3_tt.get(&ch) {
                    (e.pn, e.dn, e.len, false, e.is_final(), u32::MAX)
                } else {
                    (seed_pn, seed_dn, 0u16, true, false, u32::MAX)
                };
            board.undo_move(m, captured);
            // Phase 32: DML/sum_mask/eval を coherent 移植したが 29te が再潜行 thrash で 5M 未解決に
            // 退行 (deferred penalty/activation の δ 過小評価が非単調 re-descent を誘発)．KH DML semantics
            // の精密 debug が必要なため一旦無効化 (active=true, sum_delta=true) = sound 181K baseline 維持．
            // chain (dml_prev/next) / is_sum_delta_node は将来の careful port 用に計算だけ残す．
            let _ = (dml_prev[mi], dml_next[mi], defender_king, us_is_black);
            children.push(V3Child {
                mv: m, pn: cpn, dn: cdn, len: clen, is_first, is_final, rep_min, eval,
                active: true,
                next_in_chain: dml_next[mi],
                sum_delta: true,
            });
        }

        let orig_thpn = thpn;
        let orig_thdn = thdn;
        let orig_inc_flag = *inc_flag;

        // DoesHaveOldChild: 既訪問 (is_first=false) かつ non-final な子があれば TCA 延長．
        let does_have_old = children.iter().any(|c| !c.is_first && !c.is_final);
        if does_have_old {
            *inc_flag = inc_flag.saturating_add(1);
        }

        let (mut cur_pn, mut cur_dn, _) = v3_aggregate(&children, or_node);
        if *inc_flag > 0 {
            v3_extend(cur_pn, cur_dn, &mut thpn, &mut thdn);
        }

        self.v3_path.insert(hash, ply);

        while cur_pn < thpn && cur_dn < thdn {
            if self.v3_nodes >= self.max_nodes || self.timed_out {
                break;
            }
            // best = phi 最小の子．
            let best_idx = v3_best_idx(&children, or_node);
            let second_phi = v3_second_phi(&children, or_node, best_idx);

            // FrontPnDnThresholds (KH: active 子 + sum_mask + deferred penalty)．
            let (thphi, thdelta) = if or_node { (thpn, thdn) } else { (thdn, thpn) };
            let child_thphi = thphi.min(clamp_inf(second_phi.saturating_add(1)));
            let sum_delta = v3_sum_delta_except_best(&children, or_node, best_idx);
            let child_thdelta = if thdelta > sum_delta { thdelta - sum_delta } else { 0 };
            let (child_thpn, child_thdn) = if or_node {
                (child_thphi, child_thdelta)
            } else {
                (child_thdelta, child_thphi)
            };

            let best_mv = children[best_idx].mv;
            let captured = board.do_move(best_mv);

            let (cpn, cdn, clen, crep) = if children[best_idx].is_first {
                if *inc_flag > 0 {
                    *inc_flag -= 1;
                }
                let sp = children[best_idx].pn;
                let sd = children[best_idx].dn;
                if sp >= child_thpn || sd >= child_thdn {
                    (sp, sd, children[best_idx].len, u32::MAX)
                } else {
                    self.search_v3(board, child_thpn, child_thdn, ply + 1, inc_flag)
                }
            } else {
                self.search_v3(board, child_thpn, child_thdn, ply + 1, inc_flag)
            };

            board.undo_move(best_mv, captured);

            let c = &mut children[best_idx];
            c.pn = cpn;
            c.dn = cdn;
            c.len = clen;
            c.is_first = false;
            let became_final = cpn == 0 || cdn == 0;
            c.is_final = became_final;
            c.rep_min = crep;
            // KH DML: best が final 化したら chain の次手を activate (deferred 展開)．
            if became_final {
                let nxt = children[best_idx].next_in_chain;
                if nxt >= 0 {
                    children[nxt as usize].active = true;
                }
            }

            let agg = v3_aggregate(&children, or_node);
            cur_pn = agg.0;
            cur_dn = agg.1;

            thpn = orig_thpn;
            thdn = orig_thdn;
            if *inc_flag > 0 {
                v3_extend(cur_pn, cur_dn, &mut thpn, &mut thdn);
            } else if *inc_flag == 0 && orig_inc_flag > 0 {
                break;
            }
        }

        self.v3_path.remove(&hash);
        *inc_flag = (*inc_flag).min(orig_inc_flag);

        // mate_len + best を確定．
        let (cur_pn2, cur_dn2, _agg_best) = v3_aggregate(&children, or_node);
        let len = v3_mate_len(&children, or_node, cur_pn2);
        // PV 用 best: OR は最短 proven 子，AND は最長抵抗 (max-len) 子 (canonical mate 手順)．
        // 探索の best_idx (min-phi) とは別 — PV は game-theoretic な最善応手を辿る．
        let best_move16 = v3_pv_best(&children, or_node, cur_pn2);

        // GHI soundness: 結果が依存する千日手の最浅祖先 ply (保守的に全子の min)．
        let node_rep_min = children.iter().map(|c| c.rep_min).min().unwrap_or(u32::MAX);

        // TT へ格納するのは absolute な結果のみ:
        //   - 千日手非依存 (node_rep_min == MAX)，または
        //   - 参照祖先が自 ply 以深 (node_rep_min >= ply; 千日手が自 subtree 内に閉じている)．
        // path 上流に依存する結果 (node_rep_min < ply) は path-dependent なので格納しない
        // (= KH RepetitionTable: 千日手依存の結果は absolute にキャッシュしない)．
        if node_rep_min == u32::MAX || node_rep_min >= ply {
            self.v3_tt.insert(
                hash,
                V3Entry { pn: cur_pn2, dn: cur_dn2, len, best: best_move16, min_depth: 0 },
            );
        }
        (cur_pn2, cur_dn2, len, node_rep_min)
    }

    /// 証明済 root から best move を辿って PV を復元する．
    /// move16 を合法手リストと突合して完全な Move を得る．
    fn extract_pv_v3(&self, board: &mut Board) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut cur = board.clone();
        for step in 0..256 {
            let e = match self.v3_tt.get(&cur.hash) {
                Some(e) => *e,
                None => {
                    eprintln!("[v3pv] step {step}: TT miss (pn unknown) — break");
                    break;
                }
            };
            if e.pn != 0 {
                eprintln!("[v3pv] step {step}: entry pn={} (not proven) len={} — break", e.pn, e.len);
                break;
            }
            if e.best == 0 {
                eprintln!("[v3pv] step {step}: best=0 (proven len={}) — break", e.len);
                break;
            }
            let legal = crate::movegen::generate_legal_moves(&mut cur);
            let mv = match legal.iter().copied().find(|m| m.to_move16() == e.best) {
                Some(m) => m,
                None => {
                    eprintln!("[v3pv] step {step}: best move16={:#06x} not in {} legal — break", e.best, legal.len());
                    break;
                }
            };
            pv.push(mv);
            cur.do_move(mv);
        }
        pv
    }

    /// Phase 33 診断: v3_tt の証明木が **完全な強制詰み** か厳密検証する．
    /// OR (攻め) は格納 best 手を辿り，AND (受け) は **全合法手** を列挙して各々が
    /// 詰みに帰着するか確認する (futile filter を信用しない strict check)．戻り値は
    /// 真の詰み手数 (max-resistance)，未カバー/千日手脱出があれば `None` (= unsound)．
    fn verify_v3_proof(
        &mut self,
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
        let or_node = board.turn == self.attacker;
        let result = if or_node {
            match self.v3_tt.get(&h).copied() {
                Some(e) if e.pn == 0 && e.best != 0 => {
                    let legal = crate::movegen::generate_legal_moves(board);
                    match legal.iter().copied().find(|m| m.to_move16() == e.best) {
                        Some(mv) => {
                            path.push(h);
                            let cap = board.do_move(mv);
                            let r = self.verify_v3_proof(board, path, memo, budget);
                            board.undo_move(mv, cap);
                            path.pop();
                            r.map(|d| d + 1)
                        }
                        None => None,
                    }
                }
                _ => None,
            }
        } else {
            // 受け方: 探索と同じ move set (futile 合駒 filter 込み) で全手を列挙し，
            // 各手が詰みに帰着するか確認する．
            let legal: Vec<Move> = self
                .generate_defense_moves_inner(board, false)
                .into_iter()
                .collect();
            if legal.is_empty() {
                if board.is_in_check(board.turn()) {
                    Some(0)
                } else {
                    None
                }
            } else {
                path.push(h);
                let mut maxd = 0u16;
                let mut ok = true;
                for m in &legal {
                    let cap = board.do_move(*m);
                    let r = self.verify_v3_proof(board, path, memo, budget);
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

    /// Phase 33 (案②): LE path の root IDS．u32 unit-2 で `search_v3_le` を駆動する．
    fn solve_via_v3_le(&mut self, board: &mut Board) -> TsumeResult {
        // Phase 33k 実験用 env override (seed-fix 後の dominance/proof_hand 再測定)．
        if std::env::var("V3_DOM").is_ok() {
            self.param_v3_dominance = true;
        }
        if std::env::var("V3_PH").is_ok() {
            self.param_v3_proof_hand = true;
        }
        // proof_hand hand-aware reuse は `set_v3_proof_hand(true)` で有効化 (default off)．
        // Phase 33g: per-ply 計測リセット (KHPLY 比較診断)．
        self.v3_ply_total = [0; 64];
        self.v3_ply_unique = [0; 64];
        self.v3_ply_seen.clear();
        self.v3_sel_dumped = [0; 8];
        self.v3_trace_cnt = 0;
        // LE は KH ExpansionStack を mid_expansion_stack 上に積むため毎 solve で空に戻す．
        self.mid_expansion_stack.clear();
        self.mid_frame_moves.clear();
        // RepetitionMemo (cycle cache; KH RepetitionTable) を毎 solve でリセット．
        if self.param_v3_dominance || self.param_v3_rep_cache {
            self.v3_rep_memo.clear();
            self.v3_dom_fires = 0;
            self.v3_rep_inserts = 0;
            self.v3_rep_hits = 0;
        }
        // dominance は `set_v3_dominance(true)` で有効化 (default off; path 配列を使う)．
        if self.param_v3_dominance {
            self.path_len = 0;
        }
        // DAG double-count 補正用 parent_map をリセット．
        if self.param_v3_dag {
            self.parent_map.clear();
            self.v3_dag_resets = 0;
        }
        if self.param_v3_proof_hand {
            self.v3_proven.clear();
            self.v3_disproven.clear();
            self.v3_ph_hits = 0;
        }
        let mut thpn: u32 = 1;
        let mut thdn: u32 = 1;
        let mut last_pn: u32 = V3_INF_U;
        loop {
            if self.v3_nodes >= self.max_nodes || self.is_timed_out() {
                self.timed_out = self.is_timed_out();
                break;
            }
            let mut inc_flag = 0u32;
            let (pn, dn, _len, _rep) = self.search_v3_le(board, thpn, thdn, 0, 0u64, &mut inc_flag);
            last_pn = pn;
            if std::env::var("V3_DIAG").is_ok() {
                let rlen = self.v3_tt.get(&board.hash).map(|e| e.len).unwrap_or(0);
                eprintln!("[v3le] th=({thpn},{thdn}) -> pn={pn} dn={dn} rootlen={rlen} nodes={} tt={}",
                    self.v3_nodes, self.v3_tt.len());
            }
            if pn == 0 || dn == 0 {
                break;
            }
            if pn >= V3_INF_U || dn >= V3_INF_U {
                break;
            }
            // NextPnDnThresholds: th = max(curr_th, pn*1.7+1)．
            let cap = V3_INF_U - 1;
            let ntpn = (thpn.max((pn as f64 * 1.7) as u32 + 1)).min(cap);
            let ntdn = (thdn.max((dn as f64 * 1.7) as u32 + 1)).min(cap);
            if ntpn == thpn && ntdn == thdn {
                thpn = (thpn.saturating_mul(2).saturating_add(1)).min(cap);
                thdn = (thdn.saturating_mul(2).saturating_add(1)).min(cap);
                if thpn >= cap && thdn >= cap {
                    break;
                }
            } else {
                thpn = ntpn;
                thdn = ntdn;
            }
        }

        if std::env::var("V3_DIAG").is_ok() {
            eprintln!("[v3dom] dag={} ph={} dom={} dag_resets={} ph_hits={} dom_fires={} rep_hits={} nodes={}",
                self.param_v3_dag, self.param_v3_proof_hand, self.param_v3_dominance,
                self.v3_dag_resets, self.v3_ph_hits, self.v3_dom_fires, self.v3_rep_hits, self.v3_nodes);
        }
        if std::env::var("V3_PLY").is_ok() {
            let (mut tt, mut tu) = (0u64, 0u64);
            for d in 1..40 {
                if self.v3_ply_total[d] > 0 {
                    eprintln!("V3PLY d={:>2} total={:>8} unique={:>8}", d, self.v3_ply_total[d], self.v3_ply_unique[d]);
                }
                tt += self.v3_ply_total[d];
                tu += self.v3_ply_unique[d];
            }
            eprintln!("V3PLY TOTAL total={} unique={}", tt, tu);
        }
        if last_pn == 0 {
            if std::env::var("V3_DIAG").is_ok() {
                let mut path = Vec::new();
                let mut memo = std::collections::HashMap::new();
                let mut budget = 5_000_000u64;
                let v = self.verify_v3_proof(board, &mut path, &mut memo, &mut budget);
                eprintln!(
                    "[v3le] STRICT VERIFY: {:?} (budget_left={budget}) — Some(d)=sound mate-d, None=UNSOUND",
                    v
                );
            }
            let pv = self.extract_pv_v3(board);
            TsumeResult::Checkmate {
                moves: pv,
                nodes_searched: self.v3_nodes,
            }
        } else {
            TsumeResult::NoCheckmate {
                nodes_searched: self.v3_nodes,
            }
        }
    }

    /// Phase 33 (案②): 検証済 `MidLocalExpansion` を per-node に駆動する SearchImpl．
    /// mid_v2 production loop (solver.rs:7292) の **コアループのみ** を抽出し，
    /// maou TT/proof-hand/DAG baggage を排して mid_v3 の clean TT + 非累積 extend +
    /// root IDS + GHI (rep_min) と合成する．戻り値 (pn, dn, mate_len, rep_min) は u32 unit-2．
    fn search_v3_le(
        &mut self,
        board: &mut Board,
        thpn: u32,
        thdn: u32,
        ply: u32,
        path_key: u64,
        inc_flag: &mut u32,
    ) -> (u32, u32, u16, u32) {
        use super::mid_v2::{MidLocalExpansion, MidSearchResult, REPETITION_NONE};

        self.v3_nodes += 1;
        if self.v3_nodes >= self.max_nodes || (self.v3_nodes & 0x3FF == 0 && self.is_timed_out()) {
            return (V3_U_U32, V3_U_U32, 0, REPETITION_NONE);
        }

        let hash = board.hash;
        // Phase 33g: per-ply total/unique 訪問計測 (KHPLY trace 比較用)．
        {
            let p = (ply as usize).min(63);
            self.v3_ply_total[p] += 1;
            if self.v3_ply_seen.insert((ply, hash)) {
                self.v3_ply_unique[p] += 1;
            }
        }
        if let Some(&anc_ply) = self.v3_path.get(&hash) {
            return (V3_INF_U, 0, 0, anc_ply);
        }
        if ply >= V3_MAX_PLY {
            return (V3_INF_U, 0, 0, ply);
        }
        // KH RepetitionTable: この経路 (path_key) が過去に repetition 不詰と判明していれば再利用．
        // dominance / rep_cache のいずれか有効時．戻り値 rep_min = 記録された repetition_start．
        if self.param_v3_dominance || self.param_v3_rep_cache {
            if let Some((rep_depth, _len)) = self.v3_rep_memo.contains(path_key, 0) {
                self.v3_rep_hits += 1;
                return (V3_INF_U, 0, 0, rep_depth);
            }
        }
        if let Some(e) = self.v3_tt.get(&hash) {
            if e.is_final() {
                return (tt_pn_u32(e), tt_dn_u32(e), e.len, REPETITION_NONE);
            }
        }
        // Phase 33d: hand-aware reuse — 同一盤面 (pos_key) を異なる持駒で先に解いていれば再利用．
        if self.param_v3_proof_hand {
            let pk = super::position_key(board);
            let att_hand = board.hand[self.attacker.index()];
            if let Some((len, best)) = self.v3_proof_lookup(pk, &att_hand) {
                self.v3_ph_hits += 1;
                // PV 抽出用に exact key へも書き戻す．
                let md = self.v3_min_depth(hash, ply);
                self.v3_tt.insert(hash, V3Entry { pn: 0, dn: V3_INF_U as u64, len, best, min_depth: md });
                return (0, V3_INF_U, len, REPETITION_NONE);
            }
            if self.v3_disproof_lookup(pk, &att_hand) {
                self.v3_ph_hits += 1;
                let md = self.v3_min_depth(hash, ply);
                self.v3_tt.insert(hash, V3Entry { pn: V3_INF_U as u64, dn: 0, len: 0, best: 0, min_depth: md });
                return (V3_INF_U, 0, 0, REPETITION_NONE);
            }
        }

        let or_node = board.turn == self.attacker;
        // NOTE (Phase 33j): df-pn の探索は move 生成順に acutely sensitive (実測: normal 75K /
        // reversed 5M+ 未解決 / to-square 82K / move16 2M = ~66× span)．maou の native 生成順が
        // 単純順では最良．KH の 19K は YaneuraOu movegen の特定順由来で，完全一致には movegen 順の
        // bit 再現が要る (df-pn 公式の外)．これが残 gap の正体 (emergent でなく movegen 順)．
        let moves: Vec<Move> = if or_node {
            self.generate_check_moves_cached(board).into_iter().collect()
        } else {
            self.generate_defense_moves_inner(board, false)
                .into_iter()
                .collect()
        };

        if moves.is_empty() {
            let (p, d, l) = if or_node {
                (V3_INF_U, 0u32, 0u16)
            } else {
                (0u32, V3_INF_U, 0u16)
            };
            let md = self.v3_min_depth(hash, ply);
            self.v3_tt.insert(hash, V3Entry { pn: p as u64, dn: d as u64, len: l, best: 0, min_depth: md });
            return (p, d, l, REPETITION_NONE);
        }

        if or_node {
            let checks_av = self.generate_check_moves_cached(board);
            if let Some(mm) = board.mate_move_in_1ply(checks_av.as_slice(), board.turn) {
                let md = self.v3_min_depth(hash, ply);
                self.v3_tt.insert(
                    hash,
                    V3Entry { pn: 0, dn: V3_INF_U as u64, len: 1, best: mm.to_move16(), min_depth: md },
                );
                return (0, V3_INF_U, 1, REPETITION_NONE);
            }
        }

        let us_is_black = board.turn == crate::types::Color::Black;
        let defender_king = board.king_square(self.attacker.opponent());

        // 子の初期化: seed (unit-2) / clean TT 値 / 千日手．
        let div = PN_UNIT_SCALE.max(1);
        let mut initial_results: Vec<MidSearchResult> = Vec::with_capacity(moves.len());
        let mut evals: Vec<i32> = Vec::with_capacity(moves.len());
        // Phase 33m: KH DelayedMoveList の cross-square「無意味な中合い」検出 (delayed_move_list.hpp:151)．
        // AND ノードで，support 0 (受け方の他駒に守られない) かつ 逆王手でない drop = 後回し対象．
        // 実験 (V3_CHUAI) 時のみ計算 (default は overhead 回避のため skip)．
        let want_chuai = !or_node && std::env::var("V3_CHUAI").is_ok();
        let defender_for_chuai = self.attacker.opponent();
        let mut chuai: Vec<bool> = Vec::with_capacity(moves.len());
        for &m in &moves {
            let (sp, sd) = if or_node {
                super::init_pn_dn_or_kh(board, m, self.attacker)
            } else {
                super::init_pn_dn_and_kh(board, m, self.attacker)
            };
            let seed_pn = ((sp as u64) / div).max(1) as u32;
            let seed_dn = ((sd as u64) / div).max(1) as u32;
            // KH 中合い判定: AND ノードの drop で，受け方 support=0 のもの (support は do_move 前に計算)．
            let chuai_support0 = want_chuai
                && m.is_drop()
                && board.compute_checkers_at(m.to_sq(), defender_for_chuai).count()
                    + super::king_supports(board, m.to_sq(), defender_for_chuai)
                    == 0;
            // KH `MoveBriefEvaluation` は常に **受け方 (詰まされる側) の玉** からの距離を使う
            // (`n.KingSquare() = king_square(AndColor())`)．board.turn の玉だと OR ノードで攻め方
            // 自玉になり距離項が反転する (S*7i の eval が 8 番手→最下位にずれる原因だった)．
            let eval = match defender_king {
                Some(ksq) => {
                    let mut e = super::move_brief_eval(m, ksq, board);
                    // Phase 33n 実験: KH movegen は同点で board-move を drop より前に出す傾向．
                    // eval は 10 刻みなので +1 は完全同点のみ崩し drop を後ろへ送る．
                    if std::env::var("V3_DROPLAST").is_ok() && m.is_drop() {
                        e += 1;
                    }
                    e
                }
                None => 0,
            };
            let captured = board.do_move(m);
            let ch = board.hash;
            // 逆王手判定: drop 後に攻め方が王手されていれば逆王手 (中合いではなく有効手 → 後回ししない)．
            if chuai_support0 {
                chuai.push(!board.is_in_check(self.attacker));
            } else {
                chuai.push(false);
            }
            // KH `IsRepetitionOrInferiorAfter` (local_expansion.hpp:160): 子が
            //   (1) path 上の同一局面 (千日手),
            //   (2) [dominance ON] RepetitionMemo の千日手経路 / 祖先優等局面 (IsInferior),
            // なら MakeRepetition (dn=0, repetition_start=祖先 ply)．sound GHI は path_key
            // RepetitionMemo routing が担保 (clean TT は absolute のみ)．
            let dom_rep: Option<u32> = if self.param_v3_dominance {
                let child_path_key = super::path_key::path_key_after(path_key, m, ply as usize);
                if let Some((d, _)) = self.v3_rep_memo.contains(child_path_key, 0) {
                    self.v3_rep_hits += 1;
                    Some(d)
                } else if let Some(d) = self.is_dominated_in_path(
                    super::position_key(board),
                    &board.hand[self.attacker.index()],
                ) {
                    self.v3_dom_fires += 1;
                    Some(d)
                } else {
                    None
                }
            } else {
                None
            };
            // Phase 33e: hand-aware seeding — clean TT exact miss でも，同一 pos_key を
            // 別持駒で既に解いていれば child を proven/disproven で seed する (KH 主 TT 統合相当)．
            // entry 検査より前 (= 選択前) に効くため探索順序と coherent．
            let ph_seed: Option<MidSearchResult> = if self.param_v3_proof_hand
                && !self.v3_tt.contains_key(&ch)
            {
                let cpk = super::position_key(board);
                let chand = board.hand[self.attacker.index()];
                if let Some((clen, _cbest)) = self.v3_proof_lookup(cpk, &chand) {
                    self.v3_ph_hits += 1;
                    let mut r = MidSearchResult::new_win(clen);
                    r.is_first_visit = false;
                    Some(r)
                } else if self.v3_disproof_lookup(cpk, &chand) {
                    self.v3_ph_hits += 1;
                    let mut r = MidSearchResult::new_lose(0);
                    r.is_first_visit = false;
                    Some(r)
                } else {
                    None
                }
            } else {
                None
            };
            let r = if let Some(&anc_ply) = self.v3_path.get(&ch) {
                MidSearchResult::new_repetition(anc_ply)
            } else if let Some(dom_ply) = dom_rep {
                MidSearchResult::new_repetition(dom_ply)
            } else if let Some(e) = self.v3_tt.get(&ch) {
                let cpn = tt_pn_u32(e);
                let cdn = tt_dn_u32(e);
                // KH min_depth: この child が以前より浅い ply で格納されていれば is_shallow → TCA 発火対象．
                let shallow = (e.min_depth as u32) < ply + 1;
                if cpn == 0 {
                    let mut r = MidSearchResult::new_win(e.len);
                    r.is_first_visit = false;
                    r.is_shallow = shallow;
                    r
                } else if cdn == 0 {
                    let mut r = MidSearchResult::new_lose(e.len);
                    r.is_first_visit = false;
                    r.is_shallow = shallow;
                    r
                } else {
                    let mut r = MidSearchResult::new_unknown(cpn, cdn);
                    r.is_first_visit = false;
                    r.mate_distance = e.len;
                    r.is_shallow = shallow;
                    r
                }
            } else if let Some(r) = ph_seed {
                r
            } else {
                let mut r = MidSearchResult::new_unknown(seed_pn, seed_dn);
                r.is_first_visit = true;
                r
            };
            board.undo_move(m, captured);
            initial_results.push(r);
            evals.push(eval);
        }

        // KH coherent (unit-2) 構成で MidLocalExpansion を構築 (DML on)．
        // Phase 33m: cross-square「無意味な中合い」束ね (KH delayed_move_list.hpp:151)．
        // 実測では 29te で効果なし (−0.3%) かつ mate-29→31 退行 — d3-d7 の breadth は
        // AND(中合い) だけでなく OR(王手) でも膨らむため．V3_CHUAI で実験有効化 (default off)．
        let chuai_opt: Option<&[bool]> = if std::env::var("V3_CHUAI").is_ok() {
            Some(chuai.as_slice())
        } else {
            None
        };
        let mut expansion = MidLocalExpansion::new_with_fh_dml_chuai(
            or_node,
            moves,
            initial_results,
            hash,
            true,
            us_is_black,
            chuai_opt,
        );
        expansion.set_kh_full_comparer(true);
        expansion.set_move_evals(evals);
        expansion.set_threshold_epsilon(1);
        expansion.set_deferred_penalty_denom(8);
        expansion.set_deferred_penalty_floor(true);
        // KH IsSumDeltaNode: OR の near-duplicate (香成/不成等) を max 集約へ．
        if or_node {
            let force_max: Vec<u32> = expansion
                .moves
                .iter()
                .enumerate()
                .filter(|(_, &m)| {
                    !super::is_sum_delta_node(m, or_node, us_is_black, defender_king, board)
                })
                .map(|(i, _)| i as u32)
                .collect();
            if !force_max.is_empty() {
                expansion.apply_force_max(&force_max);
            }
        }
        // TCA trigger (Phase 33h): KH `min_depth` 流に，浅い transposition (is_shallow) を持つ child が
        // あるときのみ has_old_child=true (= MidLocalExpansion::new の is_shallow 集約をそのまま使う)．
        // 従来の「任意の再訪」基準 (recompute_has_old_child_any_revisit) は TCA を過剰発火させ
        // threshold を過延長 → per-node over-branching の一因だった (per-ply 比較で判明)．

        // 診断: root の children を sorted 順で dump (KH KHROOT と比較)．
        if ply == 0 && std::env::var("KHDUMP").is_ok() {
            for (k, &ir) in expansion.idx.iter().enumerate() {
                let r = &expansion.results[ir as usize];
                eprintln!("V3ROOT {} {} pn={} dn={}", k, expansion.moves[ir as usize].to_usi(), r.pn, r.dn);
            }
        }
        // Phase 33k: per-ply 1 階更新ごとの node selection 比較 (ユーザ指示)．
        // 各 ply の **初回 first-visit** node の children seed を sorted 順で dump．
        // sfen で KH の同一局面と突き合わせる (KHSEL と比較)．
        if std::env::var("KHSEL").is_ok() && (ply as usize) <= 7 && self.v3_sel_dumped[ply as usize] == 0 {
            self.v3_sel_dumped[ply as usize] = 1;
            eprintln!("V3SEL ply={} or={} sfen={}", ply, or_node, board.sfen());
            for (k, &ir) in expansion.idx.iter().enumerate().take(12) {
                let r = &expansion.results[ir as usize];
                eprintln!(
                    "V3SEL   {} {} pn={} dn={}",
                    k,
                    expansion.moves[ir as usize].to_usi(),
                    r.pn,
                    r.dn,
                );
            }
        }
        // V3TRACE: chronological per-expansion trace (KH KHTRACE と 1 構築ごとに突き合わせる)．
        if std::env::var("V3TRACE").is_ok() && self.v3_trace_cnt < 80 {
            self.v3_trace_cnt += 1;
            let bi = expansion.idx[0] as usize;
            let r = &expansion.results[bi];
            eprintln!(
                "V3TRACE {} ply={} or={} best={} bpn={} bdn={} sfen={}",
                self.v3_trace_cnt, ply, if or_node { 1 } else { 0 },
                expansion.moves[bi].to_usi(), r.pn, r.dn, board.sfen()
            );
        }
        // KH ExpansionStack: double-count 補正で祖先の sum_mask を reset できるよう共有スタックへ push．
        // mid_v2 と LE は同時実行しないため mid_expansion_stack を共用 (solve 開始時に clear 済)．
        self.mid_expansion_stack.push(expansion);
        self.mid_frame_moves.push(0);
        let stack_idx = self.mid_expansion_stack.len() - 1;

        let orig_thpn = thpn;
        let orig_thdn = thdn;
        let orig_inc_flag = *inc_flag;
        let mut cur_thpn = thpn;
        let mut cur_thdn = thdn;

        let mut curr = self.mid_expansion_stack[stack_idx].current_result();
        if self.mid_expansion_stack[stack_idx].does_have_old_child() {
            *inc_flag = inc_flag.saturating_add(1);
        }
        if *inc_flag > 0 {
            v3_extend_u32(curr.pn, curr.dn, &mut cur_thpn, &mut cur_thdn);
        }

        self.v3_path.insert(hash, ply);

        // KH VisitHistory: dominance 有効時のみ自局面 (pos_key, 攻め方持駒) を path に push
        // (capacity 内)．子孫の IsInferior 判定で祖先として参照される．
        let dom_pushed = if self.param_v3_dominance && self.path_len < super::solver::PATH_CAPACITY {
            self.path_pos_key[self.path_len] = super::position_key(board);
            self.path_hand[self.path_len] = board.hand[self.attacker.index()];
            self.path_len += 1;
            true
        } else {
            false
        };

        // OR proven 時の best 手 + 詰み手数 (KH multi_pv=1 の immediate proof break 用)．
        let mut proof_best_move: u16 = 0;
        let mut proof_md: u16 = 0;

        while curr.pn < cur_thpn && curr.dn < cur_thdn {
            if self.v3_nodes >= self.max_nodes || self.timed_out {
                break;
            }
            if self.mid_expansion_stack[stack_idx].empty() {
                break;
            }
            let best_mv = self.mid_expansion_stack[stack_idx].best_move();
            let is_first = self.mid_expansion_stack[stack_idx].front_is_first_visit();
            let (cthpn, cthdn) =
                self.mid_expansion_stack[stack_idx].front_pn_dn_thresholds(cur_thpn, cur_thdn);
            let child_pk = super::path_key::path_key_after(path_key, best_mv, ply as usize);
            // double-count 補正用に branch move を記録．
            self.mid_frame_moves[stack_idx] = best_mv.to_move16();

            let captured = board.do_move(best_mv);
            // KH EliminateDoubleCount: child が祖先 frame から到達可能 (DAG) なら，その祖先の
            // branch sum_mask を max へ reset し δ 二重計上を防ぐ．
            if self.param_v3_dag {
                let child_fh = board.hash;
                self.eliminate_double_count_v3(stack_idx, child_fh, hash);
            }
            let child_result = if is_first {
                let initial = *self.mid_expansion_stack[stack_idx].front_result();
                if *inc_flag > 0 {
                    *inc_flag -= 1;
                }
                if initial.is_final() || initial.pn >= cthpn || initial.dn >= cthdn {
                    initial
                } else {
                    let (cp, cd, cl, cr) = self.search_v3_le(board, cthpn, cthdn, ply + 1, child_pk, inc_flag);
                    mk_result_u32(cp, cd, cl, cr)
                }
            } else {
                let cached = *self.mid_expansion_stack[stack_idx].front_result();
                // 非 first child: final (pn=0/dn=0) は確定値なので再探索不要．
                // それ以外は **常に再探索** する (classic mid_v3 と同様)．clean TT では
                // 過去値が repetition 依存の偽 pn=0 を含み得るため，threshold 超過による
                // cached shortcut (mid_v2 の scoped TT 前提) は GHI 偽証明を生む．
                if cached.is_final() {
                    cached
                } else {
                    let (cp, cd, cl, cr) = self.search_v3_le(board, cthpn, cthdn, ply + 1, child_pk, inc_flag);
                    mk_result_u32(cp, cd, cl, cr)
                }
            };
            board.undo_move(best_mv, captured);

            // KH multi_pv=1 相当: child の phi が 0 になった瞬間に即 proof/disproof 確定して break．
            // これが無いと AND ノードで「全 defense が escape (dn=0) → excluded_moves が
            // idx.len() に達し current_result が empty() の degenerate (0,0) を win と誤読」する
            // GHI 偽証明が起きる (mid_v2 solver.rs:7805 と同じ break)．
            let child_phi = child_result.phi(or_node);
            self.mid_expansion_stack[stack_idx].update_best_child(child_result);
            if child_phi == 0 {
                curr = if or_node {
                    proof_best_move = best_mv.to_move16();
                    proof_md = child_result.mate_distance.saturating_add(1);
                    MidSearchResult::new_win(proof_md)
                } else if child_result.is_repetition() {
                    MidSearchResult::new_repetition(child_result.repetition_start)
                } else {
                    MidSearchResult::new_lose(0)
                };
                break;
            }
            curr = self.mid_expansion_stack[stack_idx].current_result();

            cur_thpn = orig_thpn;
            cur_thdn = orig_thdn;
            if *inc_flag > 0 {
                v3_extend_u32(curr.pn, curr.dn, &mut cur_thpn, &mut cur_thdn);
            } else if orig_inc_flag > 0 {
                break;
            }
        }

        self.v3_path.remove(&hash);
        if dom_pushed {
            self.path_len -= 1;
        }
        *inc_flag = (*inc_flag).min(orig_inc_flag);

        // curr は break 時に確定済み (win/lose/repetition)，通常 exit 時はループ最後の
        // current_result．ここで再計算すると AND degenerate (0,0) を踏むため再計算しない．
        let (pn, dn) = (curr.pn, curr.dn);
        let (best16, len) = if pn == 0 {
            if or_node {
                if proof_best_move != 0 {
                    (proof_best_move, proof_md)
                } else {
                    self.mid_expansion_stack[stack_idx].min_proven_or_child()
                }
            } else {
                self.mid_expansion_stack[stack_idx].max_resistance_defender()
            }
        } else {
            (0u16, 0u16)
        };

        // GHI: 結果が依存する千日手の最浅祖先 ply (全子の min)．
        let node_rep_min = self.mid_expansion_stack[stack_idx]
            .results
            .iter()
            .map(|r| r.repetition_start)
            .min()
            .unwrap_or(REPETITION_NONE);

        // KH ExpansionStack: frame を pop (push と対称)．
        self.mid_expansion_stack.pop();
        self.mid_frame_moves.pop();

        // KH 流ルーティング: 結果は **absolute** か **repetition** のいずれか．
        //   - proof (pn=0): 常に absolute (詰みは千日手依存になり得ない) → clean TT．
        //   - disproof (dn=0): determining repetition が path 上流 (node_rep_min < ply) なら
        //     path 依存 → **RepetitionMemo[path_key]** へ (clean TT には入れない)．
        //     subtree 内で resolve (>= ply) なら absolute → clean TT．
        //   - unknown: 非 taint (>= ply / NONE) のみ clean TT．taint 付きは skip (再計算)．
        let md_self = self.v3_min_depth(hash, ply);
        let ret_rep_min;
        if pn == 0 {
            self.v3_tt
                .insert(hash, V3Entry { pn: pn as u64, dn: dn as u64, len, best: best16, min_depth: md_self });
            // Phase 33d: absolute proof を hand-aware store にも記録 (actual hand; Stage 1)．
            if self.param_v3_proof_hand {
                let pk = super::position_key(board);
                let h = board.hand[self.attacker.index()];
                self.v3_store_proof(pk, h, len, best16);
            }
            ret_rep_min = REPETITION_NONE;
        } else if dn == 0 {
            if node_rep_min != REPETITION_NONE && node_rep_min < ply {
                // repetition 依存 disproof → path_key でキャッシュ (dominance/rep_cache 時; len gating なし)．
                if self.param_v3_dominance || self.param_v3_rep_cache {
                    self.v3_rep_memo.insert(path_key, node_rep_min, 0);
                    self.v3_rep_inserts += 1;
                }
                ret_rep_min = node_rep_min;
            } else {
                self.v3_tt
                    .insert(hash, V3Entry { pn: pn as u64, dn: dn as u64, len, best: best16, min_depth: md_self });
                // Phase 33d: absolute disproof を hand-aware store にも記録．
                if self.param_v3_proof_hand {
                    let pk = super::position_key(board);
                    let h = board.hand[self.attacker.index()];
                    self.v3_store_disproof(pk, h);
                }
                ret_rep_min = REPETITION_NONE;
            }
        } else {
            if node_rep_min == REPETITION_NONE || node_rep_min >= ply {
                self.v3_tt
                    .insert(hash, V3Entry { pn: pn as u64, dn: dn as u64, len, best: best16, min_depth: md_self });
                ret_rep_min = REPETITION_NONE;
            } else {
                ret_rep_min = node_rep_min;
            }
        }
        (pn, dn, len, ret_rep_min)
    }

    /// Phase 33h: KH `min_depth_`．この hash を ply で格納する際の min_depth (既存 entry と min)．
    #[inline]
    fn v3_min_depth(&self, hash: u64, ply: u32) -> u16 {
        let p = ply.min(u16::MAX as u32) as u16;
        match self.v3_tt.get(&hash) {
            Some(e) => e.min_depth.min(p),
            None => p,
        }
    }

    /// Phase 33d: hand-aware proof reuse．pos_key の保存済証明で `proof_hand <= hand` のものがあれば
    /// (len, best) を返す (攻め方が要求以上の駒を持つので同じ証明が成立)．
    fn v3_proof_lookup(&self, pos_key: u64, hand: &[u8; super::HAND_KINDS]) -> Option<(u16, u16)> {
        let v = self.v3_proven.get(&pos_key)?;
        for (ph, len, best) in v {
            if super::hand_gte_forward_chain(hand, ph) {
                return Some((*len, *best));
            }
        }
        None
    }

    /// Phase 33d: hand-aware disproof reuse．`disproof_hand >= hand` の保存済反証があれば不詰．
    fn v3_disproof_lookup(&self, pos_key: u64, hand: &[u8; super::HAND_KINDS]) -> bool {
        match self.v3_disproven.get(&pos_key) {
            Some(v) => v.iter().any(|dh| super::hand_gte_forward_chain(dh, hand)),
            None => false,
        }
    }

    /// Phase 33d: proof_hand を antichain (極小集合) で格納．新 hand を支配する既存があれば skip，
    /// 新 hand が支配する既存は除去する．
    fn v3_store_proof(&mut self, pos_key: u64, hand: [u8; super::HAND_KINDS], len: u16, best: u16) {
        let v = self.v3_proven.entry(pos_key).or_default();
        for (ph, _, _) in v.iter() {
            if super::hand_gte_forward_chain(&hand, ph) {
                return; // 既存 ph <= hand: 既存の方が一般的 → 新規不要．
            }
        }
        v.retain(|(ph, _, _)| !super::hand_gte_forward_chain(ph, &hand)); // 新 hand <= 既存 ph を除去．
        v.push((hand, len, best));
        if v.len() > 16 {
            v.remove(0);
        }
    }

    /// Phase 33d: disproof_hand を antichain (極大集合) で格納．
    fn v3_store_disproof(&mut self, pos_key: u64, hand: [u8; super::HAND_KINDS]) {
        let v = self.v3_disproven.entry(pos_key).or_default();
        for dh in v.iter() {
            if super::hand_gte_forward_chain(dh, &hand) {
                return; // 既存 dh >= hand: 既存の方が一般的．
            }
        }
        v.retain(|dh| !super::hand_gte_forward_chain(&hand, dh)); // 新 hand >= 既存 dh を除去．
        v.push(hand);
        if v.len() > 16 {
            v.remove(0);
        }
    }

    /// Phase 33c: KH `EliminateDoubleCount` の LE 版 (mid_v2 solver.rs:8060 移植)．
    ///
    /// 現フレーム (`stack_idx`) の best_move を実行直後 (`board.hash == child_fh`) に呼ぶ．
    /// child が parent_map 上で **immediate parent 以外の祖先** から到達可能 (DAG transposition) なら，
    /// その祖先フレームの branch move の sum_mask を off (max 集約) にして δ 二重計上を防ぐ．
    /// 発散判定 (祖先 chain の pn/dn が急増していたら別系統とみなす) は clean TT (`v3_tt`) を参照する．
    fn eliminate_double_count_v3(&mut self, stack_idx: usize, child_fh: u64, immediate_parent_fh: u64) {
        // child_fh の最初の parent を記録 (既存は上書きしない)．
        self.parent_map.entry(child_fh).or_insert(immediate_parent_fh);

        // KH `kAncestorSearchThreshold = 3 * kPnDnUnit` (unit-2)．
        const ANCESTOR_SEARCH_THRESHOLD: u32 = 3 * V3_U_U32;
        const MAX_DAG_LOOKBACK: usize = 16;

        let mut last_pn = u32::MAX;
        let mut last_dn = u32::MAX;
        let mut pn_flag = true;
        let mut dn_flag = true;
        let mut cur_fh = child_fh;

        for _ in 0..MAX_DAG_LOOKBACK {
            let parent_fh = match self.parent_map.get(&cur_fh) {
                Some(&p) => p,
                None => return,
            };
            // 初訪問: step0 で parent が immediate parent なら通常展開 (double-count なし)．
            if cur_fh == child_fh && parent_fh == immediate_parent_fh {
                return;
            }
            // cur の pn/dn を clean TT から取得 (無ければ unknown 小値)．
            let (cur_pn, cur_dn) = match self.v3_tt.get(&cur_fh) {
                Some(e) => (tt_pn_u32(e), tt_dn_u32(e)),
                None => (V3_U_U32, V3_U_U32),
            };
            if cur_dn > last_dn.saturating_add(ANCESTOR_SEARCH_THRESHOLD) {
                dn_flag = false;
            }
            if cur_pn > last_pn.saturating_add(ANCESTOR_SEARCH_THRESHOLD) {
                pn_flag = false;
            }
            // 祖先 stack に parent_fh があるか?
            for anc_idx in (0..stack_idx).rev() {
                if self.mid_expansion_stack[anc_idx].position_fh() == parent_fh {
                    let branch_or_node = self.mid_expansion_stack[anc_idx].or_node_value();
                    // OR node は dn の，AND node は pn の double-count を補正する．
                    let allowed = (branch_or_node && dn_flag) || (!branch_or_node && pn_flag);
                    if allowed {
                        let branch_move = self.mid_frame_moves[anc_idx];
                        if branch_move != 0
                            && self.mid_expansion_stack[anc_idx].reset_sum_mask_for_move(branch_move)
                        {
                            self.v3_dag_resets += 1;
                        }
                    }
                    return;
                }
            }
            last_pn = cur_pn;
            last_dn = cur_dn;
            cur_fh = parent_fh;
        }
    }
}

/// LE path: clean TT (u64) → u32 (INF clamp)．
#[inline]
fn tt_pn_u32(e: &V3Entry) -> u32 {
    e.pn.min(V3_INF_U as u64) as u32
}
#[inline]
fn tt_dn_u32(e: &V3Entry) -> u32 {
    e.dn.min(V3_INF_U as u64) as u32
}

/// LE path: 子 recursion の戻り (pn, dn, len, rep) を `MidSearchResult` へ変換．
/// rep != NONE の disproof は repetition，それ以外の dn==0 は通常 lose．unknown には
/// rep taint を伝播させ (GHI), node_rep_min が上へ届くようにする．
#[inline]
fn mk_result_u32(cp: u32, cd: u32, cl: u16, cr: u32) -> super::mid_v2::MidSearchResult {
    use super::mid_v2::{MidSearchResult, REPETITION_NONE};
    // win (cp==0) は absolute (tsume の詰みは千日手依存になり得ない) なので taint を持たせない．
    // disproof (cd==0) と unknown は cycle 依存 (cr<NONE) を保持し，GHI caching を制御する．
    if cp == 0 {
        MidSearchResult::new_win(cl)
    } else if cd == 0 {
        if cr != REPETITION_NONE {
            MidSearchResult::new_repetition(cr)
        } else {
            MidSearchResult::new_lose(cl)
        }
    } else {
        let mut u = MidSearchResult::new_unknown(cp, cd);
        u.is_first_visit = false;
        u.mate_distance = cl;
        u.repetition_start = cr;
        u
    }
}

/// LE path: KH `ExtendSearchThreshold` 非累積版 (mid_v3 と同じ; u32)．
#[inline]
fn v3_extend_u32(pn: u32, dn: u32, thpn: &mut u32, thdn: &mut u32) {
    if pn != 0 && dn != 0 {
        if pn < V3_INF_U {
            *thpn = (*thpn).max(pn + 1);
        }
        if dn < V3_INF_U {
            *thdn = (*thdn).max(dn + 1);
        }
    }
}

/// deferred (非 active) 子数に基づく KH deferred penalty (δ += deferred/8)．
fn v3_deferred_penalty(children: &[V3Child]) -> u64 {
    let deferred = children.iter().filter(|c| !c.active).count() as u64;
    deferred / 8
}

/// (pn, dn, best_move16) を **active 子のみ** から集約する (KH LocalExpansion)．
/// phi = min(active phi); delta = Σ_{active,sum} δ + max_{active,!sum} δ + deferred_penalty．
fn v3_aggregate(children: &[V3Child], or_node: bool) -> (u64, u64, u16) {
    let mut min_phi = V3_INF;
    let mut sum_d = 0u64;
    let mut max_d = 0u64;
    let mut best_mv16 = 0u16;
    let mut best_set = false;
    for c in children.iter().filter(|c| c.active) {
        let phi = c.phi(or_node);
        let delta = c.delta(or_node);
        if !best_set || phi < min_phi {
            min_phi = phi;
            best_mv16 = c.mv.to_move16();
            best_set = true;
        }
        if c.sum_delta {
            sum_d = clamp_inf(sum_d.saturating_add(delta));
        } else if delta > max_d {
            max_d = delta;
        }
    }
    let delta = clamp_inf(sum_d.saturating_add(max_d).saturating_add(v3_deferred_penalty(children)));
    if best_mv16 == 0 {
        // active が空 (異常) の保険．
        best_mv16 = children.first().map(|c| c.mv.to_move16()).unwrap_or(0);
    }
    let (pn, dn) = if or_node { (min_phi, delta) } else { (delta, min_phi) };
    (pn, dn, best_mv16)
}

/// PV 用 best move16: OR=最短 proven 子 / AND=最長抵抗 (max-len) 子．
/// node が未解決 (node_pn>0 かつ dn>0) の場合は min-phi 子 (探索 best) を返す．
fn v3_pv_best(children: &[V3Child], or_node: bool, node_pn: u64) -> u16 {
    if or_node {
        if node_pn == 0 {
            // proven OR: 最短 proven 子．
            let mut bl = u16::MAX;
            let mut bm = 0u16;
            for c in children {
                if c.pn == 0 && c.len <= bl {
                    bl = c.len;
                    bm = c.mv.to_move16();
                }
            }
            if bm != 0 {
                return bm;
            }
        }
    } else if node_pn == 0 {
        // proven AND (全子 proven): 最長抵抗 (max-len) 子．
        let mut bl: i32 = -1;
        let mut bm = 0u16;
        for c in children {
            if c.len as i32 >= bl {
                bl = c.len as i32;
                bm = c.mv.to_move16();
            }
        }
        if bm != 0 {
            return bm;
        }
    }
    // fallback: min-phi 子．
    let idx = v3_best_idx(children, or_node);
    children[idx].mv.to_move16()
}

/// best 子の index (active 子のみ): phi 最小．
/// NOTE (Phase 32): eval tie-break は DML/sum_mask 無しでは 181K→250K 悪化のため一旦無効
/// (coherent LocalExpansion port が安定したら再導入)．現状 active は常に true．
fn v3_best_idx(children: &[V3Child], or_node: bool) -> usize {
    let mut best = 0usize;
    let mut best_phi = V3_INF;
    let mut found = false;
    for (i, c) in children.iter().enumerate() {
        if !c.active {
            continue;
        }
        let phi = c.phi(or_node);
        if !found || phi < best_phi {
            best_phi = phi;
            best = i;
            found = true;
        }
    }
    best
}

/// best を除いた active 子の phi 最小値 (FrontThresholds の second_phi)．
fn v3_second_phi(children: &[V3Child], or_node: bool, best_idx: usize) -> u64 {
    let mut second = V3_INF;
    for (i, c) in children.iter().enumerate() {
        if i == best_idx || !c.active {
            continue;
        }
        let phi = c.phi(or_node);
        if phi < second {
            second = phi;
        }
    }
    second
}

/// best を除いた active 子の δ 集約 (Σ_{sum} δ + max_{!sum} δ) + deferred penalty．
/// FrontThresholds の child_thdelta = thdelta - これ．
fn v3_sum_delta_except_best(children: &[V3Child], or_node: bool, best_idx: usize) -> u64 {
    let mut sum_d = 0u64;
    let mut max_d = 0u64;
    for (i, c) in children.iter().enumerate() {
        if i == best_idx || !c.active {
            continue;
        }
        let d = c.delta(or_node);
        if c.sum_delta {
            sum_d = clamp_inf(sum_d.saturating_add(d));
        } else if d > max_d {
            max_d = d;
        }
    }
    clamp_inf(sum_d.saturating_add(max_d).saturating_add(v3_deferred_penalty(children)))
}

/// proven (pn==0) 時の mate_len．OR=best proven 子の len+1 / AND=max 子 len+1．
fn v3_mate_len(children: &[V3Child], or_node: bool, node_pn: u64) -> u16 {
    if node_pn != 0 {
        return 0;
    }
    if or_node {
        // 最短 proven 子 + 1．
        let mut best = u16::MAX;
        for c in children {
            if c.pn == 0 {
                best = best.min(c.len);
            }
        }
        best.saturating_add(1)
    } else {
        // 全子 proven のはず → max + 1．
        let mut mx = 0u16;
        for c in children {
            mx = mx.max(c.len);
        }
        mx.saturating_add(1)
    }
}

/// init_pn_dn_*_kh は unit-16 ベース (PN_UNIT=16) を返すため unit-2 へ縮約する除数．
const PN_UNIT_SCALE: u64 = 8;
