//! 手生成，PNS (Proof Number Search)，PV 復元，公開 API．

use std::collections::VecDeque;

use rustc_hash::FxHashSet;
#[cfg(feature = "profile")]
use std::time::Instant;

use crate::board::Board;
use crate::moves::Move;
use crate::types::{Piece, HAND_KINDS};

use super::entry::{PnsNode, PNS_INIT_CAPACITY_CAP};
use super::solver::{DfPnSolver, TsumeResult};
use super::{
    adjust_hand_for_move, edge_cost_and, edge_cost_or, heuristic_dn_from_pn, heuristic_or_dn,
    position_key, propagate_nm_remaining, sacrifice_check_boost,
    INF, PN_UNIT, REMAINING_INFINITE, WPN_GAMMA_SHIFT,
};

impl DfPnSolver {


    /// PV パス上の OR ノードで未証明の子ノードを追加証明する．
    ///
    /// Df-Pn は OR ノードで1つの子ノードが証明されると他を未探索のまま残す．
    /// PV 抽出で正確な最短詰み手数を計算するため，PV 上の OR ノードで
    /// 未証明の王手を追加証明する．反復的に PV を更新し収束させる．
    ///
    /// # 制限事項
    ///
    /// - 反復回数は2回固定．長手数(29手詰め等)で PV 上の未証明子が多い場合，
    ///   2回の反復では PV が収束せず `CheckmateNoPv` になることがある．
    ///   この場合 `pv_nodes_per_child` を増やしても改善されない．
    /// - `extract_pv_recursive` の深度制限(`depth * 2`)により PV 復元が
    ///   打ち切られるケースでも `CheckmateNoPv` になる(後述)．
    pub(super) fn complete_or_proofs(&mut self, board: &mut Board) {
        let saved_max = self.max_nodes;
        // 証明完了フェーズのノード予算:
        //   主探索ノード数と pv_nodes_per_child*8 の小さい方を追加予算とする．
        //   ただし短手数の詰将棋 (少ノードで解けた場合) でも PV 復元に
        //   十分なノードを確保するため，最低 pv_nodes_per_child ノードを保証する．
        let mid_nodes = self.nodes_searched;
        let budget_cap = self.pv_nodes_per_child.saturating_mul(8);

        // Phase 1: PV を抽出 → PV 上の OR ノードを完成 → 再抽出
        // find_shortest のときは最大 4 回反復:
        //   追加証明で PV が変化 → 新 PV 上の未証明 OR 子がさらにある →
        //   もう 1 回証明 → PV が安定，という収束サイクルを回す．
        // changed == false で早期終了するため，収束済みの場合は追加コストなし．
        self.max_nodes =
            self.nodes_searched.saturating_add(
                mid_nodes.min(budget_cap).max(self.pv_nodes_per_child),
            );

        // Phase 1: PV を抽出 → PV 上の OR ノードを完成 → 再抽出
        // 2回固定: 1回目で新たに証明された子が PV を短縮する可能性があるため
        // 2回目を実行する．changed == false で早期終了するため，
        // 収束済みの場合は追加コストなし．
        for _ in 0..2 {
            if self.is_timed_out()
                || self.nodes_searched >= self.max_nodes
            {
                break;
            }
            let pv = self.extract_pv_limited(board, 100_000);
            if pv.is_empty() {
                break;
            }
            let changed =
                self.complete_pv_or_nodes(board, &pv);
            if !changed {
                break;
            }
        }

        self.max_nodes = saved_max;
    }

    /// PV 上の各 OR ノードで未証明子ノードの追加証明を試みる．
    ///
    /// クローンした盤面で PV を辿り，各 OR ノードで未証明の王手を追加証明する．
    /// 返り値: 新たに証明された子ノードがあれば true．
    pub(super) fn complete_pv_or_nodes(&mut self, board: &mut Board, pv: &[Move]) -> bool {
        let mut board_clone = board.clone();
        let mut any_changed = false;

        for (i, pv_move) in pv.iter().enumerate() {
            let or_node = i % 2 == 0;
            let ply = i as u32;

            if or_node {
                let moves = self.generate_check_moves(&mut board_clone);
                for m in &moves {
                    if self.nodes_searched >= self.max_nodes {
                        break;
                    }
                    let captured = board_clone.do_move(*m);
                    let (cpn, cdn) = self.look_up_board(&board_clone);

                    if cpn != 0 && cdn != 0 {
                        let saved = self.max_nodes;
                        // 1子あたり pv_nodes_per_child ノード上限．
                        // PV 沿いの各未証明子に対する追加証明の予算．
                        self.max_nodes = self
                            .nodes_searched
                            .saturating_add(self.pv_nodes_per_child)
                            .min(saved);
                        let _snda_pns1 = self.prev_attacker_move;
                        self.prev_attacker_move = Move(0);
                        self.mid(&mut board_clone, INF - 1, INF - 1, ply + 1, false);
                        self.prev_attacker_move = _snda_pns1;
                        self.max_nodes = saved;

                        if self.look_up_board(&board_clone).0 == 0 {
                            any_changed = true;
                        }
                    }

                    board_clone.undo_move(*m, captured);
                }
            }

            // PV に沿って盤面を進める(前進専用: undo_move は不要)
            let _captured = board_clone.do_move(*pv_move);
        }

        any_changed
    }

    /// 訪問数制限付き PV 復元．
    pub(super) fn extract_pv_limited(&mut self, board: &mut Board, max_visits: u64) -> Vec<Move> {
        let mut board_clone = board.clone();
        let mut visits = 0u64;
        // PV 抽出の incomplete フラグをリセット．AND ノードで全 defender
        // 子を評価し切れなかった場合に extract_pv_recursive_inner が
        // self.pv_extraction_incomplete を true に設定する．
        self.pv_extraction_incomplete = false;
        self.extract_pv_recursive_inner(
            &mut board_clone,
            true,
            &mut FxHashSet::default(),
            0,
            false,
            &mut visits,
            max_visits,
        )
    }

    /// PV 内の "無駄合 (useless interposition)" pair の数を数える．
    ///
    /// 無駄合 pair の定義: 連続する手列 (defender_drop, attacker_capture, [next_defender])
    /// において:
    /// - `defender_drop` は駒打ち (`is_drop()`)
    /// - `attacker_capture` は **同じマス** での駒取り
    /// - 次の defender 手は同じマスでの recapture **でない**
    ///   (recapture なら犠牲交換 = 無駄合 ではない)
    ///
    /// この pair は機械的に詰みを 2 手延ばすだけで真の防御リソースを買っていない
    /// ため，AND ノードでの最長抵抗比較から除外することで設計ドキュメント
    /// (`docs/design/tsume-solver/aigoma-optimization.md`) の意図する
    /// 「PV(最長) = 真の最長抵抗」を復元する．
    ///
    /// # 前提条件
    ///
    /// - `pv[0]` が **defender 手** (AND ノードが選ぶ手) であること．
    ///   OR-node 起点の PV (attacker 先頭) を渡すと意味のない結果になる．
    ///   呼び出し元は `extract_pv_recursive_inner` の AND ノード分岐のみ．
    ///
    /// # i += 2 ステップの意味
    ///
    /// チェーン内でペアが重なることはない (非重複 pairing)．
    /// (defender_drop, attacker_capture) を消費したら次は (next_defender,
    /// next_attacker) のペアを見る．これにより，例えばマルチレベル連続
    /// チェーン `DC DC DC ...` (D=drop, C=capture) は 3 pair と数えられる．
    /// 途中の recapture は次の pair の対象にはならず重複計上を避ける．
    pub(super) fn count_useless_interpose_pairs(pv: &[Move]) -> usize {
        // 前提条件の契約: pv[0] は defender 手でなければならない．
        // 手そのものから side は判らないが，defender_drop + attacker_capture が
        // 同一マスで発生する特殊パターンは OR 始点では解釈不能 (OR 始点だと
        // pv[0]=attacker_drop となり pair セマンティクスが崩れる)．
        // 呼び出しは `extract_pv_recursive_inner` の AND 分岐のみ．
        let mut count = 0;
        let mut i = 0;
        while i + 1 < pv.len() {
            let def = pv[i];
            let att = pv[i + 1];
            if def.is_drop()
                && att.captured_piece_raw() != 0
                && att.to_sq() == def.to_sq()
            {
                // defender の次の手 (i+2) が同じマスへの recapture か？
                let recaptured = if i + 2 < pv.len() {
                    let next_def = pv[i + 2];
                    next_def.captured_piece_raw() != 0
                        && next_def.to_sq() == def.to_sq()
                } else {
                    false
                };
                if !recaptured {
                    count += 1;
                }
            }
            i += 2;
        }
        count
    }

    /// PV 復元の再帰実装．
    ///
    /// 各ノードで全候補手のサブPVを生成し，攻め方は最短，玉方は最長を選ぶ．
    /// ループ検出にはフルハッシュ，TT 参照には位置キー＋持ち駒を使用する．
    ///
    /// # 深度制限
    ///
    /// 再帰深度は `self.depth * 2` で打ち切る．OR/AND が交互に呼ばれるため，
    /// depth=31 なら最大62手の PV を復元できる．理論上，AND ノードで
    /// 複数の応手候補を評価する際に ply が詰み手数の2倍を超えうるが，
    /// TT に証明済みエントリがあれば即座に返るため，実用上は十分な深さである．
    ///
    /// ただし `complete_pv_or_nodes` で新たに証明された子が PV を変化させた場合，
    /// AND ノードの応手評価で ply が急増し，深度制限に達して空リストが返ることがある．
    /// この場合 `pv_nodes_per_child` を増やしても改善されず，`depth` の増加が必要．
    pub(super) fn extract_pv_recursive(
        &mut self,
        board: &mut Board,
        or_node: bool,
        visited: &mut FxHashSet<u64>,
        ply: u32,
    ) -> Vec<Move> {
        let mut visits = 0u64;
        self.extract_pv_recursive_inner(board, or_node, visited, ply, false, &mut visits, u64::MAX)
    }

    /// `extract_pv_recursive` の内部実装．
    ///
    /// `diag` が true の場合，各plyでの候補手・sub_pv長・選択理由を
    /// 標準エラーに出力する．
    /// `visits` は TT 参照回数を計測し，`max_visits` を超えると空リストを返す．
    pub(super) fn extract_pv_recursive_inner(
        &mut self,
        board: &mut Board,
        or_node: bool,
        visited: &mut FxHashSet<u64>,
        ply: u32,
        diag: bool,
        visits: &mut u64,
        max_visits: u64,
    ) -> Vec<Move> {
        // 訪問数制限チェック
        *visits += 1;
        if *visits > max_visits {
            return Vec::new();
        }

        // スタックオーバーフロー防止: 探索手数の2倍を再帰深度の上限とする
        if ply >= self.depth.saturating_mul(2) {
            if diag {
                verbose_eprintln!("[PV diag] ply={} depth_limit reached (max={})", ply, self.depth.saturating_mul(2));
            }
            return Vec::new();
        }
        let full_hash = board.hash;

        // ループ検出(フルハッシュ)
        if visited.contains(&full_hash) {
            if diag {
                verbose_eprintln!("[PV diag] ply={} loop detected hash={:#x}", ply, full_hash);
            }
            return Vec::new();
        }

        let (node_pn, _node_dn) = self.look_up_board_for_pv(board);

        if or_node {
            if node_pn != 0 {
                if diag {
                    verbose_eprintln!("[PV diag] ply={} OR node unproven pn={}", ply, node_pn);
                }
                return Vec::new();
            }

            let moves = self.generate_check_moves_cached(board);
            if moves.is_empty() {
                if diag {
                    verbose_eprintln!("[PV diag] ply={} OR node no check moves", ply);
                }
                return Vec::new();
            }

            if diag {
                verbose_eprintln!("[PV diag] ply={} OR node, {} check moves", ply, moves.len());
            }

            let mut best_pv: Option<Vec<Move>> = None;

            for m in &moves {
                if *visits > max_visits {
                    break;
                }
                let captured = board.do_move(*m);
                let (child_pn, _) = self.look_up_board_for_pv(board);

                // TT で proof が見つからなくても，応手が存在しなければ
                // 即詰み(leaf node)．部分集合クラスタ走査でも proof を
                // 見つけられないケースの安全策として残す．
                let is_proven = if child_pn == 0 {
                    true
                } else {
                    let defense = self.generate_defense_moves(board);
                    let leaf_mate = defense.is_empty();
                    if leaf_mate && diag {
                        verbose_eprintln!(
                            "[PV diag] ply={} OR child {} pn={} but no defense (leaf mate)",
                            ply, m.to_usi(), child_pn
                        );
                    }
                    leaf_mate
                };

                if is_proven {
                    visited.insert(full_hash);
                    let sub_pv =
                        self.extract_pv_recursive_inner(
                            board, false, visited, ply + 1, diag,
                            visits, max_visits,
                        );
                    visited.remove(&full_hash);

                    let total_len = 1 + sub_pv.len();
                    // OR ノードの PV は奇数長でなければならない
                    // (攻め方の手で始まり攻め方の手で終わる)
                    if total_len % 2 == 0 && !sub_pv.is_empty() {
                        if diag {
                            verbose_eprintln!(
                                "[PV diag] ply={} OR skip {} (even len={}, sub={})",
                                ply, m.to_usi(), total_len, sub_pv.len()
                            );
                        }
                        board.undo_move(*m, captured);
                        continue;
                    }
                    let is_better = match &best_pv {
                        None => true,
                        Some(prev) => total_len < prev.len(),
                    };

                    if diag {
                        verbose_eprintln!(
                            "[PV diag] ply={} OR candidate {} len={} better={}{}",
                            ply, m.to_usi(), total_len, is_better,
                            if let Some(prev) = &best_pv {
                                format!(" (prev_best={})", prev.len())
                            } else {
                                String::new()
                            }
                        );
                    }

                    if is_better {
                        let mut pv = vec![*m];
                        pv.extend(sub_pv);
                        best_pv = Some(pv);
                    }
                } else if diag {
                    verbose_eprintln!(
                        "[PV diag] ply={} OR child {} unproven pn={}",
                        ply, m.to_usi(), child_pn
                    );
                }

                board.undo_move(*m, captured);
            }

            #[cfg(feature = "verbose")]
            if diag {
                if let Some(ref pv) = best_pv {
                    eprintln!(
                        "[PV diag] ply={} OR CHOSEN {} (pv_len={})",
                        ply, pv[0].to_usi(), pv.len()
                    );
                } else {
                    eprintln!("[PV diag] ply={} OR no valid PV found", ply);
                }
            }

            best_pv.unwrap_or_default()
        } else {
            // AND ノード: 呼び出し側の OR ノードで child_pn == 0 を
            // 確認した後のみ再帰するため，この局面は証明済みのはず．
            let moves = self.generate_defense_moves(board);
            if moves.is_empty() {
                if diag {
                    verbose_eprintln!("[PV diag] ply={} AND node no defense (checkmate)", ply);
                }
                return Vec::new();
            }

            if diag {
                verbose_eprintln!("[PV diag] ply={} AND node, {} defense moves", ply, moves.len());
            }

            // 【不採用】v0.24.23 で導入された TT distance ベースの fast path は
            // unsound であった: TT の `mate_distance` は **raw** 距離を保存しており，
            // 無駄合 chain によって inflate された値を返す．fast path はその raw
            // 距離で longest resistance を選んでいたため，chain drop 子を誤って
            // 選択し Mate(21) を返す可能性があった (39手詰め ply 24 の既知バグ)．
            //
            // 修正: 全 defender 子を再帰評価し，`effective_len = total_len -
            // 2 * useless_pairs` で比較する slow path を常に使う．訪問予算は
            // `visits`/`max_visits` で制限され，予算超過時は `all_evaluated = false`
            // を立てて呼び出し側で `CheckmateNoPv` に変換させる．
            //
            // 性能: fast path 廃止により深い aigoma 問題で PV 抽出コストが増える
            // が，実測では `test_tsume_39te_ply22_no_pns` は visit 予算 10M 内で
            // 完了する (ply 22 の Mate(17) を正しく発見)．

            let mut best_pv: Option<Vec<Move>> = None;
            let mut best_is_capture = false;
            let mut best_is_drop = false;
            // best_effective_len: 効果長 (useless interpose pair を除外した長さ)．
            // 定義は `count_useless_interpose_pairs` を参照．
            let mut best_effective_len: usize = 0;
            // soundness 用フラグ: 全 defender 子を評価し切ったか追跡する．
            // visit budget 不足で途中 break した場合，PV 抽出は最長抵抗を
            // 検証できていないので空 PV を返して呼び出し側で
            // CheckmateNoPv 扱いにする．
            let mut all_evaluated = true;

            for m in &moves {
                if *visits > max_visits {
                    all_evaluated = false;
                    break;
                }
                let captured = board.do_move(*m);
                let (child_pn, _) = self.look_up_board_for_pv(board);

                if child_pn == 0 {
                    visited.insert(full_hash);
                    let sub_pv =
                        self.extract_pv_recursive_inner(
                            board, true, visited, ply + 1, diag,
                            visits, max_visits,
                        );
                    visited.remove(&full_hash);

                    let total_len = 1 + sub_pv.len();
                    // AND ノードの PV は偶数長でなければならない
                    // (玉方の手で始まり攻め方の手で終わる)
                    if total_len % 2 == 1 {
                        if diag {
                            verbose_eprintln!(
                                "[PV diag] ply={} AND skip {} (odd len={}, sub={})",
                                ply, m.to_usi(), total_len, sub_pv.len()
                            );
                        }
                        board.undo_move(*m, captured);
                        continue;
                    }
                    // この候補手の full_pv = [m] ++ sub_pv
                    // 無駄合 pair を数えて効果長を計算する
                    let mut full_pv: Vec<Move> = Vec::with_capacity(total_len);
                    full_pv.push(*m);
                    full_pv.extend_from_slice(&sub_pv);
                    let useless_pairs = Self::count_useless_interpose_pairs(&full_pv);
                    let effective_len = total_len.saturating_sub(2 * useless_pairs);

                    let is_capture = m.captured_piece_raw() > 0;
                    let is_drop = m.is_drop();
                    let is_better = match &best_pv {
                        None => true,
                        Some(prev) => {
                            // 第一基準: 効果長 (無駄合除外後の真の resistance) が長い
                            // 第二基準: 効果長同率なら raw length が短い (chain inflation の少ない) PV を優先
                            // 第三基準: 同率なら駒取りを優先
                            // 第四基準: 駒取り状況も同じなら合駒 (打ち駒) を優先
                            if effective_len > best_effective_len {
                                true
                            } else if effective_len < best_effective_len {
                                false
                            } else if total_len < prev.len() {
                                true
                            } else if total_len > prev.len() {
                                false
                            } else if is_capture && !best_is_capture {
                                true
                            } else if is_drop
                                && !best_is_drop
                                && is_capture == best_is_capture
                            {
                                true
                            } else {
                                false
                            }
                        }
                    };

                    if diag {
                        verbose_eprintln!(
                            "[PV diag] ply={} AND candidate {} len={} eff={} pairs={} capture={} drop={} better={}",
                            ply, m.to_usi(), total_len, effective_len, useless_pairs,
                            is_capture, is_drop, is_better,
                        );
                    }

                    if is_better {
                        best_pv = Some(full_pv);
                        best_is_capture = is_capture;
                        best_is_drop = is_drop;
                        best_effective_len = effective_len;
                    }
                } else if diag {
                    verbose_eprintln!(
                        "[PV diag] ply={} AND child {} unproven pn={}",
                        ply, m.to_usi(), child_pn
                    );
                }

                board.undo_move(*m, captured);
            }

            #[cfg(feature = "verbose")]
            if diag {
                if let Some(ref pv) = best_pv {
                    eprintln!(
                        "[PV diag] ply={} AND CHOSEN {} (pv_len={})",
                        ply, pv[0].to_usi(), pv.len()
                    );
                } else {
                    eprintln!("[PV diag] ply={} AND no valid PV found", ply);
                }
            }

            // Soundness: visit 予算が尽きて全 defender を評価し切れなかった場合，
            // 返す PV は真の longest resistance である保証がない．
            // `pv_extraction_incomplete` フラグを立て，呼び出し側 (`solve()`) で
            // `CheckmateNoPv` に変換させる．暫定 PV は参考値として保持．
            if !all_evaluated {
                self.pv_extraction_incomplete = true;
            }
            best_pv.unwrap_or_default()
        }
    }

    /// 探索ノード数を返す．
    pub fn nodes_searched(&self) -> u64 {
        self.nodes_searched
    }

    /// IDS-dfpn (MID) フォールバック．
    ///
    /// PNS がアリーナ上限に達した場合に呼び出される．
    /// PNS で蓄積された TT エントリ(証明・中間値)を引き継ぎ，
    /// 残りのノード予算で IDS-dfpn を実行する．
    pub(super) fn mid_fallback(&mut self, board: &mut Board) {
        let pk = position_key(board);
        let att_hand = board.hand[self.attacker.index()];
        self.diag_root_pk = pk;
        self.diag_root_hand = att_hand;
        let saved_depth = self.depth;
        // saved_depth_for_epsilon は full-depth 用に saved_depth を保持する．
        self.saved_depth_for_epsilon = saved_depth;
        // 診断 (v0.71.2): `param_skip_ids_shallow=true` で IDS の浅い iter を skip．
        let mut ids_depth: u32 = if self.param_skip_ids_shallow {
            saved_depth
        } else {
            2
        };
        let total_max_nodes = self.max_nodes;
        // PNS で蓄積された中間エントリ(pn>0, dn>0)を除去し proof のみ保持する．
        // 中間エントリを保持すると以下の問題が発生する:
        //   1. HashMap サイズ増大により CPU キャッシュ効率が低下し NPS が半減する
        //      (338K entries → ~126 NPS vs 12K entries → ~194 NPS)．
        //   2. MID の child init で cpn>1/cdn>1 として扱われ，
        //      底辺の簡単な詰みを再発見する機会が失われる．
        self.table.retain_proofs_only();

        #[cfg(feature = "tt_diag")]
        eprintln!("[mid_fallback] after TT cleanup: TT_pos={} proven_pos={} nodes_so_far={} total_budget={} confirmed_dis={}",
            self.table.len(), self.table.proven_len(),
            self.nodes_searched, total_max_nodes,
            self.table.count_working_confirmed_disproofs());

        // 停滞検出用: 前回の IDS 反復終了時の root pn/dn を保持する．
        // IDS 反復後に root_pn/dn が変化しなかった場合，MID が
        // dn 閾値カスケード縮退により進捗不能と判断する．
        let mut prev_root_pn: u32 = 0;
        let mut prev_root_dn: u32 = 0;

        // 合駒チェーン最適化の IDS 反復間デルタ追跡
        #[cfg(feature = "tt_diag")]
        let mut prev_prefilter_hits = self.prefilter_hits;
        #[cfg(feature = "tt_diag")]
        let mut prev_cross_deduce = self.diag_cross_deduce_hits;
        #[cfg(feature = "tt_diag")]
        let mut prev_deferred_act = self.diag_mid_deferred_activations;
        #[cfg(feature = "tt_diag")]
        let mut prev_prefilter_skip = self.diag_prefilter_skip_remaining;
        #[cfg(feature = "tt_diag")]
        let mut prev_prefilter_miss = self.diag_prefilter_miss;
        #[cfg(feature = "tt_diag")]
        let mut prev_reverse_disproof = self.diag_reverse_disproof_hits;

        loop {
            if ids_depth > saved_depth {
                ids_depth = saved_depth;
            }
            self.depth = ids_depth;
            self.table.current_ids_depth = ids_depth;
            self.path_len = 0;
            self.path_set.clear();
            let remaining = ids_depth as u16;
            let (root_pn, _, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
            verbose_eprintln!("[ids] depth={}/{} root_pn={} nodes={} time={:.1}s",
                ids_depth, saved_depth, root_pn, self.nodes_searched,
                self.start_time.elapsed().as_secs_f64());
            if root_pn == 0 {
                verbose_eprintln!("[ids] root proved, break");
                break;
            }
            let _budget = if ids_depth < saved_depth {
                // 中間 IDS step の予算配分:
                // remaining_budget / (remaining_steps + 1) の均等割り．
                //
                // remaining_steps は (saved_depth - ids_depth) / 2 + 1 で
                // 概算．浅い step (depth=2,4) は allocated budget の
                // ごく一部しか消費せず，未消費分が後続 step に蓄積
                // されるため，深い step ほど実質的に多くの budget を得る．
                let remaining_budget =
                    total_max_nodes.saturating_sub(self.nodes_searched);
                let remaining_steps =
                    ((saved_depth.saturating_sub(ids_depth)) / 2)
                        .max(1) as u64 + 1;
                let b = (remaining_budget / (remaining_steps + 1))
                    .max(1024);
                self.max_nodes = self.nodes_searched.saturating_add(b);
                b
            } else {
                // フルデプス: 残り予算の全てを割り当て．
                self.max_nodes = total_max_nodes;
                total_max_nodes.saturating_sub(self.nodes_searched)
            };

            #[cfg(feature = "tt_diag")]
            let pre_nodes = self.nodes_searched;
            #[cfg(feature = "tt_diag")]
            let _pre_max_ply = self.max_ply;
            // IDS 反復ごとに max_ply をリセットし，各反復の到達深さを追跡する
            self.max_ply = 0;

            if ids_depth == saved_depth {
                // フルデプス: MID 先行(動的予算) + Frontier Variant フォールバック．
                //
                // 方針B(§10.2): MID を固定 1/2 予算ではなくチャンク分割し，
                // TT エントリ成長を監視する．MID が新しい TT エントリを
                // 生成しなくなった時点で停滞と判定し，残り予算を動的に
                // Frontier に回す．方針A で MID の閾値余裕が拡大したため，
                // MID が進捗できる範囲をまず効率的に処理し，
                // 閾値飢餓で停滞した時点で速やかに Frontier に切り替える．
                //
                // TT 清掃なしでシームレスに遷移: MID が蓄積した
                // 証明・反証・中間エントリを Frontier がそのまま活用する．
                #[cfg(feature = "tt_diag")]
                eprintln!("[ids_final] ids_depth={} proven_pos={} working_pos={} nodes={}",
                    ids_depth, self.table.proven_len(), self.table.len(),
                    self.nodes_searched);

                let remaining_budget =
                    total_max_nodes.saturating_sub(self.nodes_searched);

                let mid_max_budget = remaining_budget / 2;
                // チャンクサイズ: 1M (TT 停滞の早期検出のため固定)
                let chunk_size: u64 = 1_000_000;
                let mid_deadline = self.nodes_searched.saturating_add(mid_max_budget);
                // 案4 (v0.55.9): FrontierTT も含めた TT 成長チェック．
                // FRONTIER_REMAINING_THRESHOLD > 0 の場合，remaining ≤ threshold の
                // intermediate は WorkingTT ではなく FrontierTT に格納されるため，
                // WorkingTT のみを見る `len()` では停滞を誤検知する．
                let mut prev_tt_len = self.table.len() + self.table.frontier_len();

                while self.nodes_searched < mid_deadline && !self.timed_out {
                    let chunk_end = (self.nodes_searched + chunk_size).min(mid_deadline);
                    self.max_nodes = chunk_end;
                    let (root_pn, root_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                    if root_pn == 0 || root_dn == 0 {
                        break; // 証明/反証完了
                    }
                    let _snda_pns2 = self.prev_attacker_move;
                    self.prev_attacker_move = Move(0);
                    self.mid(board, INF - 1, INF - 1, 0, true);
                    self.prev_attacker_move = _snda_pns2;

                    let (r_pn, r_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                    if r_pn == 0 || r_dn == 0 {
                        break; // 証明/反証完了
                    }

                    // TT 成長チェック: 新規エントリが生成されていなければ停滞．
                    // 意図的動作: Periodic GC (solver.rs) が mid() 内部で発火し
                    // エントリを削除した場合も curr_tt_len < prev_tt_len となり
                    // Frontier 遷移がトリガーされる．GC は TT 容量 80% 超過時
                    // のみ発火するため，TT 圧迫下での Frontier 遷移は合理的．
                    let curr_tt_len = self.table.len() + self.table.frontier_len();
                    if curr_tt_len <= prev_tt_len {
                        verbose_eprintln!("[ids] MID stagnation: TT {} → {}, shifting to Frontier",
                            prev_tt_len, curr_tt_len);
                        break;
                    }
                    prev_tt_len = curr_tt_len;
                }

                // MID で未解決 → Frontier Variant にフォールバック
                let (r_pn, r_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                if r_pn != 0 && r_dn != 0
                    && self.nodes_searched < total_max_nodes
                    && !self.timed_out
                {
                    self.max_nodes = total_max_nodes;
                    self.frontier_variant(board, total_max_nodes);
                }
            } else {
                // 浅い反復: 通常の MID で TT をウォームアップ
                let (root_pn, root_dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                if root_pn != 0 && root_dn != 0
                    && self.nodes_searched < self.max_nodes
                    && !self.timed_out
                {
                    verbose_eprintln!("[ids] calling MID: depth={} root_pn={} root_dn={} nodes={}",
                        ids_depth, root_pn, root_dn, self.nodes_searched);
                    #[cfg(feature = "profile")]
                    let _mid_wall_start = Instant::now();
                    let _snda_pns3 = self.prev_attacker_move;
                    self.prev_attacker_move = Move(0);
                    self.mid(board, INF - 1, INF - 1, 0, true);
                    self.prev_attacker_move = _snda_pns3;
                    verbose_eprintln!("[ids] MID returned: depth={} nodes={} time={:.1}s",
                        ids_depth, self.nodes_searched, self.start_time.elapsed().as_secs_f64());
                    #[cfg(feature = "profile")]
                    {
                        self.profile_stats.mid_total_ns += _mid_wall_start.elapsed().as_nanos() as u64;
                        self.profile_stats.mid_total_count += 1;
                    }
                }
            }

            #[cfg(feature = "tt_diag")]
            {
                let used = self.nodes_searched - pre_nodes;
                let (pn, dn, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
                let d_prefilter = self.prefilter_hits - prev_prefilter_hits;
                let d_cross = self.diag_cross_deduce_hits - prev_cross_deduce;
                let d_deferred = self.diag_mid_deferred_activations - prev_deferred_act;
                let d_pf_skip = self.diag_prefilter_skip_remaining - prev_prefilter_skip;
                let d_pf_miss = self.diag_prefilter_miss - prev_prefilter_miss;
                let d_rev_dis = self.diag_reverse_disproof_hits - prev_reverse_disproof;
                let ids_elapsed = self.start_time.elapsed().as_secs_f64();
                verbose_eprintln!("[ids_diag] depth={}/{} budget={} used={} TT_pos={} root_pn={} root_dn={} max_ply={} \
                    prefilter_hit={} prefilter_skip_rem={} prefilter_miss={} cross={} rev_dis={} act={} \
                    cap_tt={}/{} thr_exits={} term_exits={} in_path={} lb_prov={} lb_thr={} lb_nodes={} \
                    init_and_dis={} single_ch={} node_lim={} time={:.2}s",
                    ids_depth, saved_depth, _budget, used,
                    self.table.len(), pn, dn,
                    self.max_ply,
                    d_prefilter, d_pf_skip, d_pf_miss,
                    d_cross, d_rev_dis, d_deferred,
                    self.diag_capture_tt_hits, self.diag_capture_tt_calls,
                    self.diag_threshold_exits,
                    self.diag_terminal_exits,
                    self.diag_in_path_exits,
                    self.diag_loop_break_proved,
                    self.diag_loop_break_threshold,
                    self.diag_loop_break_nodes,
                    self.diag_init_and_disproof_exits,
                    self.diag_single_child_exits,
                    self.diag_node_limit_exits,
                    ids_elapsed);
                // ply ヒストグラム出力(非ゼロ ply のみ)
                let mut ply_str = String::new();
                for p in 0..64 {
                    if self.diag_ply_visits[p] > 0 {
                        if !ply_str.is_empty() { ply_str.push_str(", "); }
                        ply_str.push_str(&format!("{}:{}", p, self.diag_ply_visits[p]));
                    }
                }
                if !ply_str.is_empty() {
                    verbose_eprintln!("[ids_diag] ply_visits: {}", ply_str);
                }
                // single-child counter (repurposed diag_ply_proofs)
                let mut sc_str = String::new();
                for p in 0..64 {
                    if self.diag_ply_proofs[p] > 0 {
                        if !sc_str.is_empty() { sc_str.push_str(", "); }
                        sc_str.push_str(&format!("{}:{}", p, self.diag_ply_proofs[p]));
                    }
                }
                if !sc_str.is_empty() {
                    verbose_eprintln!("[ids_diag] single_child: {}", sc_str);
                }
                // リセット
                self.diag_ply_visits = [0u64; 64];
                self.diag_ply_proofs = [0u64; 64];
                self.diag_capture_tt_calls = 0;
                self.diag_capture_tt_hits = 0;
                self.diag_threshold_exits = 0;
                self.diag_terminal_exits = 0;
                self.diag_loop_break_proved = 0;
                self.diag_loop_break_threshold = 0;
                self.diag_loop_break_nodes = 0;
                self.diag_in_path_exits = 0;
                self.diag_init_and_disproof_exits = 0;
                self.diag_single_child_exits = 0;
                self.diag_node_limit_exits = 0;
                prev_prefilter_hits = self.prefilter_hits;
                prev_cross_deduce = self.diag_cross_deduce_hits;
                prev_deferred_act = self.diag_mid_deferred_activations;
                prev_prefilter_skip = self.diag_prefilter_skip_remaining;
                prev_prefilter_miss = self.diag_prefilter_miss;
                prev_reverse_disproof = self.diag_reverse_disproof_hits;
            }
            let (root_pn2, root_dn2, _) = self.look_up_pn_dn(pk, &att_hand, remaining);
            verbose_eprintln!("[ids] after MID: depth={} root_pn={} root_dn={} nodes={} time={:.1}s",
                ids_depth, root_pn2, root_dn2, self.nodes_searched,
                self.start_time.elapsed().as_secs_f64());
            if root_pn2 == 0 {
                // (v0.24.71) tag-aware IDS break guard: FILTER_DEPENDENT proof で
                // IDS を終了すると false Checkmate が返る．root の proof が
                // ABSOLUTE の場合のみ break する．
                // v0.24.72 で施策α 不採用後は FILTER_DEPENDENT proof が生成
                // されないため本 guard は常に ABSOLUTE 経路．実質 no-op で
                // 効果を持たないが，tag infrastructure が残存する限り保持．
                // (削除候補: proof_tag インフラ一括除去時に同時に削除可能)
                let root_tag = self.table.get_proof_tag(pk, &att_hand);
                if root_tag == super::entry::PROOF_TAG_ABSOLUTE {
                    verbose_eprintln!("[ids] proved (ABSOLUTE) at depth={}, break", ids_depth);
                    break;
                }
                verbose_eprintln!(
                    "[ids] proved (tag={}) at depth={}, continuing IDS for re-verification",
                    root_tag, ids_depth);
            }
            // IDS NM 判定: 構造的判定のみ信頼する．
            //
            // 1. NM remaining が REMAINING_INFINITE **かつ** 現在の IDS depth が
            //    残り手数以上なら真の不詰として打ち切る．
            //    MID が深さ制限なしに全パスを網羅して NM を確定した場合にのみ
            //    REMAINING_INFINITE が伝搬される．
            //    ただし ids_depth < remaining の場合は，前の IDS サイクルが
            //    蓄積した ProvenTT の confirmed disproof が浅い depth の
            //    look_up でヒットしただけの可能性がある．
            //    この場合は depth-limited 仮 NM と同等に扱い昇格しない．
            //    (v0.24.63: ply 2 NoMate バグの修正)
            // 2. 全王手が再帰的に反証可能なら REMAINING_INFINITE に昇格して打ち切る．
            //
            // depth 制限由来の仮 NM (remaining < REMAINING_INFINITE) は昇格しない．
            // 深い詰みが存在する局面でも浅い IDS 深さでは NM になるのが当然であり，
            // これを真の不詰と判定すると偽陽性が発生する(例: 39手詰め)．
            if root_dn2 == 0 {
                let root_nm_rem = self.table.get_disproof_remaining(pk, &att_hand);
                // v0.24.66: NM 昇格判定の depth ガード．
                // saved_depth と outer_solve_depth (solve() の真の depth) の最大値を使用し，
                // IDS の浅いステップでの false NM 昇格を防止する．
                let nm_guard_depth = saved_depth.max(self.outer_solve_depth);
                verbose_eprintln!("[ids] NM: depth={}/{} (guard={}) nm_rem={} REMAINING_INFINITE={}",
                    ids_depth, saved_depth, nm_guard_depth, root_nm_rem, REMAINING_INFINITE);
                if root_nm_rem == REMAINING_INFINITE
                    && ids_depth >= nm_guard_depth
                {
                    verbose_eprintln!("[ids] NM INFINITE (ids_depth >= guard_depth), break");
                    break;
                }
                let checks = self.generate_check_moves_cached(board);
                let refutable = if checks.is_empty() {
                    true
                } else {
                    self.depth_limit_all_checks_refutable(board, &checks)
                };
                verbose_eprintln!("[ids] refutable={} checks={} ids_depth={}/{} guard={}",
                    refutable, checks.len(), ids_depth, saved_depth, nm_guard_depth);
                if (checks.is_empty() || refutable) && ids_depth >= nm_guard_depth {
                    verbose_eprintln!("[ids] NM promoted to INFINITE (ids_depth >= guard_depth), break");
                    // att_hand で保存(TT ヒット率最大化)
                    self.table.store(
                        pk, att_hand, INF, 0, REMAINING_INFINITE, pk as u32,
                    );
                    break;
                }
                // depth 制限由来の仮 NM → IDS を続行してより深い探索で再検証．
            }
            if self.nodes_searched >= total_max_nodes || self.timed_out {
                break;
            }
            if ids_depth >= saved_depth {
                break;
            }

            // 停滞検出: 予算を使い切ったのに root pn/dn が変化していない場合，
            // dn 閾値カスケード縮退により MID が深部に到達できていない．
            let stagnated = root_pn2 > 0
                && root_dn2 > 0
                && root_pn2 == prev_root_pn
                && root_dn2 == prev_root_dn;

            // 時間ベースの反復カットオフ:
            // 浅い IDS 反復が残り時間の 1/4 以上消費した場合，
            // この深さでの NPS が低く後続の深さに時間を回せないため
            // 最大深さへ即座にジャンプする．
            let total_elapsed = self.start_time.elapsed().as_secs_f64();
            let remaining_time = self.timeout.as_secs_f64() - total_elapsed;
            let time_exceeded = ids_depth < saved_depth
                && remaining_time < self.timeout.as_secs_f64() * 0.25;

            // IDS 反復間での depth 進行判断．
            // 施策 I (v0.24.45): WorkingTT の全クリアは廃止し，intermediate
            // エントリを選択的に保持する．保持時は remaining を depth 差分だけ
            // シフトし，新 depth での初期下限値として再利用する．
            let prev_ids_depth = ids_depth;

            if stagnated || time_exceeded {
                ids_depth = saved_depth;
            } else {
                // 深さ進行:
                //
                // depth < 16: 倍増 (2→4→8→16)
                // depth ≥ 16: +4 刻みスキップなし (16→20→24→28→32→36→40→...)
                //
                // saved_depth ≤ 19: 直接ジャンプ (2→4→saved_depth)
                //   浅い問題 (remaining ≤ 17) は depth=16 暖機のコスト (~460K nodes) が
                //   直接ジャンプの損失を上回るため直接ジャンプが最効率．
                //   (実測: ply 24, saved_depth=17 で直接ジャンプ 683K < 段階的 718K)
                //
                //   FrontierTT (v0.55.9+, remaining ≤ 24) 環境では段階的 IDS の優位が
                //   消失する: 1M ノード予算で warmup に ~460K 消費すると full-depth 予算が
                //   不足し ply 22 (saved_depth=19) の探索精度が低下した．
                //   直接ジャンプで full-depth 予算を最大化することで回帰を解消する．
                //   (旧計測: saved_depth=19 で直接ジャンプ 9.29M → 段階的 7.63M, -19.2%
                //    は ~9M ノード予算時の値; 1M 予算では warmup 比率が逆転する)
                //
                // saved_depth > 19: 段階的 IDS
                //   倍増フェーズ (2→4→8→16) + IDS-17 中間ステップ + +4 刻みで
                //   TT を段階的に構築してから最終深さに到達する．
                let next = if ids_depth >= 16 {
                    ids_depth + 4
                } else {
                    ids_depth.saturating_mul(2).max(ids_depth + 2)
                };
                if saved_depth <= 19 && next > 4 && next < saved_depth {
                    ids_depth = saved_depth;
                } else if ids_depth == 16 && next > 17 && saved_depth > 19 && saved_depth <= 26
                    && !self.param_no_ids17
                {
                    // ids_depth=16 は 2→4→8→16 の倍増パスで到達する．
                    // saved_depth=21-26 のとき +4 で 20 にジャンプする前に
                    // 17 を挟んで TT 暖機を追加する．
                    ids_depth = 17;
                } else {
                    ids_depth = next.min(saved_depth);
                }
            }

            // 施策 I (v0.24.45): WorkingTT から intermediate を選択的に保持．
            //
            // `test_tsume_39te_ply25_gap_diagnosis` (v0.24.44) で「depth=17 で
            // 75K intermediate → depth=21 で 0」という中間進捗の全消去が観測され，
            // 合駒チェーンの不詰部分証明が毎 IDS step でゼロから再構築される
            // 問題の原因となっていた．
            //
            // 保持条件 (`retain_working_intermediates` 参照):
            //   pn>0 && dn>0 && pn<INF && !path_dependent && remaining < INFINITE
            //
            // 保持時は `remaining` に depth 差分 (delta) を加算する．
            // 旧 depth で計算された pn/dn は新 depth での初期下限値として
            // 安全に再利用される (より多くの remaining = より多くの探索が
            // 必要 → 旧 pn/dn は常に真の値以下)．
            //
            // delta は通常 2〜4 (段階的 IDS)．stagnation で saved_depth に
            // 直接ジャンプする場合は delta が大きくなりうるが，
            // shift 後の remaining が REMAINING_INFINITE を超えない限り
            // 安全に保持される (超える場合は個別に除去される)．
            // IDS depth 遷移前: 現在の WorkingTT 分布をスナップショットとして記録する．
            // TT 遷移 (retain_working_intermediates / clear_working) の直前であり，
            // この depth での探索が完了した時点の完全な分布が得られる．
            {
                let elapsed_secs = self.start_time.elapsed().as_secs_f64();
                let (pd_pn, pd_dn, pd_joint) = self.table.collect_working_pn_dn_dist();
                self.pn_dn_per_depth.push((prev_ids_depth, self.nodes_searched, elapsed_secs, pd_pn, pd_dn, pd_joint));
            }

            if ids_depth > prev_ids_depth {
                let delta = (ids_depth - prev_ids_depth) as u16;
                // min_remaining=0: 全ての非 path-dep intermediate を保持
                #[allow(unused_variables)]
                let retained = self.table.retain_working_intermediates(0, delta);
                verbose_eprintln!("[ids] retain_intermediates: prev={} new={} delta={} kept={}",
                    prev_ids_depth, ids_depth, delta, retained);
            } else {
                // depth が進まない (stagnated で prev == saved_depth) or
                // depth が減る (起きえないが防御的に) → 従来通り全クリア
                self.table.clear_working();
            }

            // ProvenTT の浅い refutable disproof を選択的に除去する．
            // confirmed disproof (remaining=REMAINING_INFINITE) は深さ非依存の
            // 永続エントリのため除去しない (v0.55.28 バグ#2 修正)．
            // refutable disproof のみ次の IDS depth の半分未満のものを除去．
            self.table.clear_proven_disproofs_below(ids_depth / 2);

            prev_root_pn = root_pn2;
            prev_root_dn = root_dn2;
        }
        self.depth = saved_depth;
        self.max_nodes = total_max_nodes;
    }

    /// Frontier Variant: PNS→局所 MID サイクル．
    ///
    /// IDS のフルデプス反復で使用される．浅い IDS 反復で蓄積された
    /// TT 証明・確定反証を活用しつつ，PNS のグローバル最適選択で
    /// MID の閾値飢餓(§10.4)を回避する．
    ///
    /// ### PNS→MID サイクルの相乗効果
    ///
    /// - PNS の TT 書き込みが MID の child init で TT ヒット率を向上させる．
    /// - MID の証明蓄積が次の PNS サイクルでのフロンティア選択精度を向上させる．
    /// - 各サイクルで MID 後に `retain_proofs()` を呼び，
    ///   中間エントリの蓄積による PNS フロンティア選択の汚染を防止する．
    ///   証明(pn=0)と確定反証(dn=0, 非経路依存)は保持される．
    fn frontier_variant(&mut self, board: &mut Board, total_max_nodes: u64) {
        let pk = position_key(board);
        let att_hand = board.hand[self.attacker.index()];

        // アリーナを1回確保し，サイクル間で再利用する(方針C)．
        // arena.clear() は内部の PnsNode を drop するが，外側 Vec の
        // capacity は保持されるため再確保コストを回避できる．
        // 初期確保は INIT_CAPACITY_CAP で抑え，必要に応じて Vec 自動拡張に任せる．
        let mut arena: Vec<PnsNode> = Vec::with_capacity(
            self.param_pns_arena_max.min(PNS_INIT_CAPACITY_CAP),
        );

        let mut frontier_iters = 0u32;
        const MAX_FRONTIER_ITERS: u32 = 50;
        // Zero-proof early skip: PNS が連続して proof=0 を返した場合，
        // PNS サイクルをスキップして MID に全予算を回す．
        // 1 回だけ proof=0 は偶然の可能性があるため 2 連続で判定する．
        let mut consecutive_zero_proofs: u32 = 0;

        while frontier_iters < MAX_FRONTIER_ITERS
            && self.nodes_searched < total_max_nodes
            && !self.timed_out
        {
            frontier_iters += 1;
            let remaining_budget = total_max_nodes.saturating_sub(self.nodes_searched);
            if remaining_budget < 10_000 {
                break;
            }

            // Zero-proof early skip: 2 サイクル連続で PNS が proof=0 なら
            // PNS フェーズをスキップし，MID に全予算を集中する．
            //
            // 注: これは一方向ラッチである．一度スキップ状態に入ると
            // `consecutive_zero_proofs` は更新されないため，当該 Frontier Variant
            // の残りのイテレーションで PNS は再起動されない．MID が新たな進捗を
            // 出しても PNS は実行されない設計選択であり，PNS が非生産的と
            // 判定された後は MID に全予算を集中することを優先する．
            let skip_pns = consecutive_zero_proofs >= 2;

            if !skip_pns {
                // PNS フェーズ: TT を更新しフロンティアを特定
                // 予算の動的調整: 前サイクルで proof を生産していた場合は
                // 予算を拡大(remaining/10 = 10%)し，より多くの proof を蓄積して
                // MID の TT ヒット率を向上させる．
                // 初回(last_pns_proof_stores 未設定)または proof=0 の場合は
                // 従来どおり remaining/20 = 5%．
                let pns_ratio = if self.last_pns_proof_stores > 0 { 10 } else { 20 };
                let pns_budget = (remaining_budget / pns_ratio).max(10_000).min(50_000);
                self.max_nodes = self.nodes_searched.saturating_add(pns_budget);
                #[cfg(feature = "verbose")]
                let (proofs_before, growth_before, spin_before, changed_before) = (
                    self.dbg_pns_proof_stores,
                    self.dbg_pns_arena_growth,
                    self.dbg_pns_spin_iters,
                    self.dbg_pns_changed_iters,
                );
                self.skip_refutable_disproof = true;
                let _pv = self.pns_main_with_arena(board, &mut arena);
                self.skip_refutable_disproof = false;

                // Zero-proof 判定: 直前の PNS サイクルの proof store 数を確認
                if self.last_pns_proof_stores == 0 {
                    consecutive_zero_proofs += 1;
                } else {
                    consecutive_zero_proofs = 0;
                }

                #[cfg(feature = "verbose")]
                {
                    let cycle_proofs = self.dbg_pns_proof_stores - proofs_before;
                    let cycle_growth = self.dbg_pns_arena_growth - growth_before;
                    let cycle_spin = self.dbg_pns_spin_iters - spin_before;
                    let cycle_changed = self.dbg_pns_changed_iters - changed_before;
                    let cycle_total = cycle_spin + cycle_changed;
                    let cycle_spin_pct = if cycle_total > 0 {
                        cycle_spin as f64 / cycle_total as f64 * 100.0
                    } else { 0.0 };
                    verbose_eprintln!(
                        "[fv] iter {} pns: proofs={} arena_growth={} spin={:.1}% ({}/{}) budget={}",
                        frontier_iters, cycle_proofs, cycle_growth,
                        cycle_spin_pct, cycle_spin, cycle_total, pns_budget,
                    );
                }

                let (r_pn, r_dn, _) = self.look_up_pn_dn(pk, &att_hand, self.depth as u16);
                if r_pn == 0 || r_dn == 0 {
                    break; // PNS で証明または反証完了
                }
            } else {
                verbose_eprintln!(
                    "[fv] iter {} pns: SKIPPED (zero-proof early skip, {} consecutive)",
                    frontier_iters, consecutive_zero_proofs,
                );
            }

            // MID フェーズ: PNS で更新された TT を活用して局所探索
            let remaining_budget2 = total_max_nodes.saturating_sub(self.nodes_searched);

            if skip_pns {
                // PNS SKIPPED サイクル: 残余 budget 全量を 1 回の MID に渡して終了する．
                //
                // 理由: budget を分割すると各 MID 呼び出しが depth=31 の探索途中で
                // 予算切れになり，root (ply=0) まで戻れず root TT エントリが更新されない．
                // root TT が空のまま次の MID が始まるため収束しない．
                // 残余を全量渡せば MID が完走して root エントリを正しく更新できる．
                //
                // retain_proofs() は PNS アリーナが存在しないため不要．
                self.max_nodes = total_max_nodes;
                self.path_len = 0;
                self.path_set.clear();
                let (r_pn2, r_dn2, _) = self.look_up_pn_dn(pk, &att_hand, self.depth as u16);
                if r_pn2 != 0 && r_dn2 != 0 {
                    let _snda_pns4 = self.prev_attacker_move;
                    self.prev_attacker_move = Move(0);
                    self.mid(board, INF - 1, INF - 1, 0, true);
                    self.prev_attacker_move = _snda_pns4;
                }
                break;
            }

            let mid_budget = (remaining_budget2 / 4).max(50_000).min(remaining_budget2);
            self.max_nodes = self.nodes_searched.saturating_add(mid_budget);
            self.path_len = 0;
            self.path_set.clear();

            let (r_pn2, r_dn2, _) = self.look_up_pn_dn(pk, &att_hand, self.depth as u16);
            if r_pn2 == 0 || r_dn2 == 0 {
                break;
            }
            let _snda_pns5 = self.prev_attacker_move;
            self.prev_attacker_move = Move(0);
            self.mid(board, INF - 1, INF - 1, 0, true);
            self.prev_attacker_move = _snda_pns5;

            let (r_pn3, r_dn3, _) = self.look_up_pn_dn(pk, &att_hand, self.depth as u16);
            if r_pn3 == 0 || r_dn3 == 0 {
                break;
            }

            // サイクル間 TT 清掃: PNS アリーナが破棄される境界でのみ実施する．
            // PNS を実行したサイクルのみ: MID が蓄積した中間エントリを除去し，
            // 次の PNS サイクルが新鮮な状態でフロンティアを選択できるようにする．
            // 証明(pn=0)と確定反証(dn=0, 非経路依存)は保持する．
            //
            // NOTE (v0.24.45): 当初は `retain_proofs_and_intermediates()` で
            // 非 path-dep intermediate も保持する案を試みたが，
            // `test_no_checkmate_gold_interposition` で soundness 違反が発生した．
            self.table.retain_proofs();
        }
        self.max_nodes = total_max_nodes;
        #[cfg(feature = "verbose")]
        verbose_eprintln!(
            "[fv] done: {} iters, total proofs={} arena_growth={} cycles={}",
            frontier_iters, self.dbg_pns_proof_stores, self.dbg_pns_arena_growth, self.dbg_pns_cycles,
        );
    }

    // ================================================================
    // Best-First Proof Number Search (PNS)
    // ================================================================

    /// Best-First PNS メインループ．
    ///
    /// 明示的な探索木(アリーナ)上で most-proving node を選択・展開し，
    /// pn/dn をルートまでバックアップする．df-pn の閾値制御を必要とせず，
    /// グローバルに最適なノード選択を行う．
    ///
    /// アリーナが `param_pns_arena_max` (デフォルト 5M) に達した場合は探索を
    /// 打ち切り，呼び出し元で MID ベースの探索にフォールバックする．
    pub(super) fn pns_main(&mut self, board: &mut Board) -> Option<Vec<Move>> {
        let mut arena: Vec<PnsNode> = Vec::with_capacity(
            self.param_pns_arena_max.min(PNS_INIT_CAPACITY_CAP),
        );
        self.skip_refutable_disproof = true;
        self.in_initial_pns_phase = true;
        let result = self.pns_main_with_arena(board, &mut arena);
        self.in_initial_pns_phase = false;
        self.skip_refutable_disproof = false;
        result
    }

    /// Best-First PNS メインループ(アリーナ再利用版)．
    ///
    /// `frontier_variant()` から繰り返し呼ばれる際にアリーナの
    /// 外側 Vec の再確保を回避し NPS を改善する．
    pub(super) fn pns_main_with_arena(
        &mut self,
        board: &mut Board,
        arena: &mut Vec<PnsNode>,
    ) -> Option<Vec<Move>> {
        arena.clear();

        // ルートノード生成
        let pk = position_key(board);
        let fh = board.hash;
        let hand = board.hand[self.attacker.index()];
        arena.push(PnsNode {
            pos_key: pk,
            full_hash: fh,
            hand,
            pn: PN_UNIT,
            dn: PN_UNIT,
            parent: u32::MAX,
            move_from_parent: Move(0),
            or_node: true,
            expanded: false,
            children: Vec::new(),
            cached_best: u32::MAX,
            remaining: self.depth as u16,
            deferred_drops: VecDeque::new(),
        });

        // 再利用バッファ(ループ内のアロケーション回避)
        let max_path = self.depth as usize + 2;
        let mut path: Vec<u32> = Vec::with_capacity(max_path);
        let mut captures: Vec<Piece> = Vec::with_capacity(max_path);
        // PNS 選択ウォークの最大深さ．depth(最大 41) + deferred drops 活性化分の余裕．
        const ANCESTORS_CAP: usize = 65;
        let mut ancestors_buf = [0u64; ANCESTORS_CAP];
        let mut ancestors_len: usize;


        // PNS メインループ
        let mut pns_iters: u64 = 0;
        // PNS 収束検出: root_pn が一定反復数改善しなければ打ち切る．
        // PNS はアリーナ飽和後に選択ウォークだけを繰り返し
        // 予算を消費するため，早期打ち切りで MID に予算を回す．
        let mut best_root_pn: u32 = u32::MAX;
        let mut iters_since_improvement: u64 = 0;
        const PNS_STAGNATION_LIMIT: u64 = 500_000;
        // P4: アリーナ成長率監視による適応的早期終了
        const GROWTH_CHECK_INTERVAL: u64 = 10_000;
        // 10 回連続の GROWTH_CHECK_INTERVAL (合計 100K PNS イテレーション) で
        // アリーナ成長ゼロなら打ち切る．
        const GROWTH_STALL_LIMIT: u32 = 10;
        let mut prev_arena_size: usize = 1; // ルートノード分
        let mut growth_stall_count: u32 = 0;
        #[cfg(feature = "verbose")]
        let mut spin_iters_local: u64 = 0;
        #[cfg(feature = "verbose")]
        let mut changed_iters_local: u64 = 0;
        #[cfg(feature = "verbose")]
        let arena_size_at_entry: usize = arena.len();
        loop {
            pns_iters += 1;
            #[cfg(feature = "verbose")]
            let (root_pn_before, root_dn_before) = (arena[0].pn, arena[0].dn);
            // 終了条件: ルート証明/反証
            if arena[0].pn == 0 || arena[0].dn == 0 {
                break;
            }
            // 終了条件: ノード制限・タイムアウト
            if self.nodes_searched >= self.max_nodes || self.timed_out {
                break;
            }
            // 終了条件: アリーナ満杯
            if arena.len() >= self.param_pns_arena_max {
                break;
            }
            // 終了条件: PNS 収束停滞(root_pn ベース)
            if arena[0].pn < best_root_pn {
                best_root_pn = arena[0].pn;
                iters_since_improvement = 0;
            } else {
                iters_since_improvement += 1;
                if iters_since_improvement >= PNS_STAGNATION_LIMIT {
                    #[cfg(feature = "tt_diag")]
                    eprintln!("[pns_diag] stagnation: root_pn={} no improvement for {} iters, stopping",
                        arena[0].pn, PNS_STAGNATION_LIMIT);
                    break;
                }
            }
            // 終了条件: アリーナ成長停止(適応的打ち切り)
            if pns_iters % GROWTH_CHECK_INTERVAL == 0 {
                let current_size = arena.len();
                if current_size == prev_arena_size {
                    growth_stall_count += 1;
                    if growth_stall_count >= GROWTH_STALL_LIMIT {
                        #[cfg(feature = "tt_diag")]
                        eprintln!("[pns_diag] arena growth stalled: size={} for {}K iters, stopping",
                            current_size, growth_stall_count as u64 * GROWTH_CHECK_INTERVAL / 1000);
                        break;
                    }
                } else {
                    growth_stall_count = 0;
                    prev_arena_size = current_size;
                }
            }
            // 定期タイムアウトチェック
            if pns_iters & 0xFF == 0 && self.is_timed_out() {
                self.timed_out = true;
                break;
            }

            // Most-proving node 選択 + 盤面復元
            path.clear();
            captures.clear();
            path.push(0);
            ancestors_buf[0] = arena[0].full_hash;
            ancestors_len = 1;
            let mut current = 0u32;

            let mut skip_expand = false;

            while arena[current as usize].expanded {
                let ci = current as usize;

                // AND ノード: 全子証明済み + deferred_drops → 逐次活性化
                if !arena[ci].or_node && !arena[ci].deferred_drops.is_empty() {
                    let all_proven = arena[ci].children.iter()
                        .all(|&c| arena[c as usize].pn == 0);
                    if all_proven {
                        // 証明済み子の直接 OR ノードのみ TT に flush．
                        // 中間ノード(子ありの AND/OR)は actual hand で格納するため
                        // 既存の中間エントリを不正に evict し MID 収束を妨げる恐れがある．
                        // 直接 OR 子ノードのみ flush することで，try_prefilter_block が
                        // 「合駒後の OR ノードが証明済みか」を判定できるようにする．
                        for &child_idx in &arena[ci].children {
                            let child = &arena[child_idx as usize];
                            if child.pn == 0 && child.or_node {
                                self.store(
                                    child.pos_key, child.hand, 0, INF,
                                    REMAINING_INFINITE, child.pos_key as u32,
                                );
                            }
                        }

                        let and_remaining = arena[ci].remaining;
                        let mut and_proof = [0u8; HAND_KINDS];

                        let mut activated_unproven = false;
                        while !arena[ci].deferred_drops.is_empty() {
                            let next_drop = arena[ci].deferred_drops.pop_front().unwrap();
                            #[cfg(feature = "tt_diag")]
                            { self.diag_pns_deferred_activations += 1; }
                            #[cfg(feature = "tt_diag")]
                            eprintln!(
                                "[pns_seq] AND node idx={}: activate drop {} (deferred_remaining={}), tt_entries={}",
                                ci, next_drop.to_usi(), arena[ci].deferred_drops.len(),
                                self.table.total_entries(),
                            );

                            // TT ベースプレフィルタ: 合駒の捕獲後局面が
                            // 既に TT で証明済みなら展開不要．
                            // PNS flush で先行子の証明が TT に反映された後に
                            // チェックすることで，同一マスの兄弟合駒を
                            // TT ヒットで即座に証明できる．
                            if and_remaining >= 3 {
                                let cap_pre = board.do_move(next_drop);
                                let child_hand_pre = board.hand[self.attacker.index()];
                                board.undo_move(next_drop, cap_pre);
                                if self.try_prefilter_block(
                                    board, next_drop, &child_hand_pre,
                                    and_remaining, &mut and_proof,
                                ) {
                                    self.prefilter_hits += 1;
                                    #[cfg(feature = "tt_diag")]
                                    {
                                        self.diag_pns_deferred_already_proven += 1;
                                        verbose_eprintln!(
                                            "[pns_seq]   prefilter hit for {} → proven",
                                            next_drop.to_usi(),
                                        );
                                    }
                                    // アリーナにも証明済みとして追加
                                    let cap_pf = board.do_move(next_drop);
                                    let child_pk_pf = position_key(board);
                                    let child_fh_pf = board.hash;
                                    let child_hand_pf =
                                        board.hand[self.attacker.index()];
                                    let child_remaining_pf =
                                        and_remaining.saturating_sub(1);
                                    board.undo_move(next_drop, cap_pf);
                                    let pf_idx = arena.len() as u32;
                                    arena.push(PnsNode {
                                        pos_key: child_pk_pf,
                                        full_hash: child_fh_pf,
                                        hand: child_hand_pf,
                                        pn: 0,
                                        dn: INF,
                                        parent: current,
                                        move_from_parent: next_drop,
                                        or_node: true,
                                        expanded: true,
                                        children: Vec::new(),
                                        cached_best: u32::MAX,
                                        remaining: child_remaining_pf,
                                        deferred_drops: VecDeque::new(),
                                    });
                                    arena[ci].children.push(pf_idx);
                                    continue;
                                }
                            }

                            let cap = board.do_move(next_drop);
                            let child_fh = board.hash;
                            let child_pk = position_key(board);
                            let child_hand = board.hand[self.attacker.index()];
                            let child_remaining =
                                arena[ci].remaining.saturating_sub(1);

                            let is_loop = ancestors_buf[..ancestors_len].contains(&child_fh);
                            let (cpn, cdn) = if is_loop {
                                (INF, 0u32)
                            } else {
                                let (p, d, _) = self.look_up_pn_dn(
                                    child_pk, &child_hand, child_remaining,
                                );
                                (p, d)
                            };

                            board.undo_move(next_drop, cap);

                            // 反証(防御成功) → AND ノード反証
                            if cdn == 0 {
                                arena[ci].pn = INF;
                                arena[ci].dn = 0;
                                skip_expand = true;
                                activated_unproven = true;
                                break;
                            }

                            // 子ノード生成
                            let child_idx = arena.len() as u32;
                            arena.push(PnsNode {
                                pos_key: child_pk,
                                full_hash: child_fh,
                                hand: child_hand,
                                pn: cpn,
                                dn: cdn,
                                parent: current,
                                move_from_parent: next_drop,
                                or_node: true,
                                expanded: cpn == 0 || cdn == 0,
                                children: Vec::new(),
                                cached_best: u32::MAX,
                                remaining: child_remaining,
                                deferred_drops: VecDeque::new(),
                            });
                            arena[ci].children.push(child_idx);

                            // 証明済み → 次の deferred drop へ
                            if cpn == 0 {
                                #[cfg(feature = "tt_diag")]
                                { self.diag_pns_deferred_already_proven += 1; }
                                continue;
                            }

                            // 未証明 → この子を MPN として選択
                            let cap = board.do_move(next_drop);
                            captures.push(cap);
                            path.push(child_idx);
                            debug_assert!(ancestors_len < ANCESTORS_CAP, "ancestors overflow");
                            ancestors_buf[ancestors_len] = child_fh;
                            ancestors_len += 1;
                            current = child_idx;
                            activated_unproven = true;
                            break;
                        }

                        if activated_unproven {
                            break; // MPN 選択ループ終了
                        }

                        // 全 deferred が証明済み → AND ノード証明
                        arena[ci].pn = 0;
                        arena[ci].dn = INF;
                        skip_expand = true;
                        break;
                    }
                }

                // 通常の子ノード選択
                if arena[ci].children.is_empty() {
                    // 子ノードなし(展開済みだが全消去等) → リーフとして再展開
                    break;
                }
                let best_child = if arena[ci].cached_best != u32::MAX {
                    arena[ci].cached_best
                } else if arena[ci].or_node {
                    *arena[ci].children.iter()
                        .min_by_key(|&&c| (arena[c as usize].pn, arena[c as usize].dn))
                        .expect("children non-empty (guarded above)")
                } else {
                    *arena[ci].children.iter()
                        .min_by_key(|&&c| (arena[c as usize].dn, arena[c as usize].pn))
                        .expect("children non-empty (guarded above)")
                };
                let child_move = arena[best_child as usize].move_from_parent;
                let captured = board.do_move(child_move);
                captures.push(captured);
                path.push(best_child);
                debug_assert!(ancestors_len < ANCESTORS_CAP, "ancestors overflow");
                ancestors_buf[ancestors_len] = arena[best_child as usize].full_hash;
                ancestors_len += 1;
                current = best_child;
            }

            // リーフ展開(逐次活性化で解決済みの場合はスキップ)
            if !skip_expand {
                let ply = (path.len() - 1) as u32;
                self.pns_expand(board, arena, current, ply, &ancestors_buf[..ancestors_len]);
            }

            // 盤面をルートに戻す
            for i in (1..path.len()).rev() {
                let child_move = arena[path[i] as usize].move_from_parent;
                board.undo_move(child_move, captures[i - 1]);
            }

            // バックアップ: 展開ノードからルートまで pn/dn を更新
            Self::pns_backup(arena, current);

            // 空回り(pn/dn 不変)検出
            #[cfg(feature = "verbose")]
            {
                if arena[0].pn == root_pn_before && arena[0].dn == root_dn_before {
                    spin_iters_local += 1;
                } else {
                    changed_iters_local += 1;
                }
            }
        }

        #[cfg(feature = "verbose")]
        {
            self.dbg_pns_spin_iters += spin_iters_local;
            self.dbg_pns_changed_iters += changed_iters_local;
            self.dbg_pns_arena_growth += (arena.len() - arena_size_at_entry) as u64;
            self.dbg_pns_cycles += 1;
        }

        // 診断: PNS 終了時の状態
        #[cfg(feature = "tt_diag")]
        {
            let pns_nodes_used = self.nodes_searched;
            let root_pn = arena[0].pn;
            let root_dn = arena[0].dn;
            let pns_elapsed = self.start_time.elapsed().as_secs_f64();
            verbose_eprintln!("[pns_diag] arena={}/{} iters={} nodes_used={} root_pn={} root_dn={} TT_pos={} time={:.2}s",
                arena.len(), self.param_pns_arena_max, pns_iters, pns_nodes_used, root_pn, root_dn,
                self.table.len(), pns_elapsed);

            // (v0.24.74 診断) root.dn==0 (false NM 疑い) のとき子ノードの状態をダンプ
            if root_dn == 0 && root_pn != 0 && arena[0].expanded {
                verbose_eprintln!("[pns_false_nm] root.dn=0 detected! children:");
                for &ci in &arena[0].children {
                    let child = &arena[ci as usize];
                    let src = if child.expanded { "expanded" } else { "leaf" };
                    verbose_eprintln!(
                        "  child[{}] move={} or={} pn={} dn={} expanded={} remaining={} pk={:#x} hand={:?} src={}",
                        ci, child.move_from_parent.to_usi(), child.or_node,
                        child.pn, child.dn, child.expanded, child.remaining,
                        child.pos_key, child.hand, src,
                    );
                    // dn=0 の子: TT にどのような NM エントリがあるか確認
                    if child.dn == 0 {
                        let has_proof = self.table.has_proof(child.pos_key, &child.hand);
                        let (tt_pn, tt_dn, _) = self.table.look_up(
                            child.pos_key, &child.hand, REMAINING_INFINITE, true);
                        verbose_eprintln!(
                            "    TT lookup: pn={} dn={} has_proof={}", tt_pn, tt_dn, has_proof);
                        // ProvenTT のエントリを直接ダンプ
                        self.table.dump_entries(child.pos_key, &child.hand);
                    }
                }
            }
        }

        // 証明/反証結果を TT に格納(PV 抽出用)
        self.last_pns_proof_stores = self.pns_store_to_tt(&arena);

        // デバッグ: 証明ツリーの整合性チェック
        #[cfg(debug_assertions)]
        if arena[0].pn == 0 {
            Self::validate_pns_proof(&arena, 0);
        }

        // ルートが証明済みならアリーナから直接 PV を抽出
        if arena[0].pn == 0 {
            let mut visited: FxHashSet<u64> = FxHashSet::default();
            let pv = self.pns_extract_pv(board, &arena, 0, &mut visited);
            if !pv.is_empty() && pv.len() % 2 == 1 {
                return Some(pv);
            }
        }
        None
    }

    /// PNS ノード展開: リーフノードの子を生成し初期 pn/dn を設定する．
    ///
    /// 子ノードの初期化では既存 TT エントリおよびヒューリスティック
    /// (DFPN-E エッジコスト，Deep df-pn 深さバイアス，静的詰め判定)を利用する．
    pub(super) fn pns_expand(
        &mut self,
        board: &mut Board,
        arena: &mut Vec<PnsNode>,
        node_idx: u32,
        ply: u32,
        ancestors: &[u64],
    ) {
        self.nodes_searched += 1;
        if ply > self.max_ply {
            self.max_ply = ply;
        }

        let or_node = arena[node_idx as usize].or_node;
        let remaining = arena[node_idx as usize].remaining;
        let pos_key = arena[node_idx as usize].pos_key;
        let att_hand = arena[node_idx as usize].hand;

        // 終端: 深さ制限 / 手数制限
        if remaining == 0 || ply >= self.depth || board.ply() as u32 >= self.draw_ply {
            arena[node_idx as usize].pn = INF;
            arena[node_idx as usize].dn = 0;
            arena[node_idx as usize].expanded = true;
            if or_node {
                // OR ノードの深さ制限: 王手が0手なら真の不詰(REMAINING_INFINITE)．
                // 王手がある場合の NM 昇格判定はハイブリッドで実施する．
                //
                // 1. まず `all_checks_refutable_by_tt` (~2µs/王手) で TT ベース
                //    の高速判定．TT に `REMAINING_INFINITE` エントリが既にあれば
                //    これだけで決着する．
                // 2. TT 判定が false の場合，`refutable_check_failed` キャッシュを
                //    確認．既に false 確定なら skip．
                // 3. それ以外は `depth_limit_all_checks_refutable` (再帰 movegen)
                //    を実行し，false なら結果を memoize する．
                //
                // v0.24.31 以前は常に `depth_limit_all_checks_refutable` を
                // 呼んでおり，39手詰め ply 20 では 10,000 回上限到達が
                // 99.9% で PNS 時間の 99.97% を消費していた．
                // ハイブリッド化 + memoization で共通ケースを ~2µs に短縮しつつ
                // 29手詰めの NM 昇格率を維持する．
                let checks = self.generate_check_moves_cached(board);
                if checks.is_empty() {
                    self.store(pos_key, att_hand, INF, 0,
                        REMAINING_INFINITE, pos_key as u32);
                } else if self.refutable_check_with_cache(board, pos_key, &checks) {
                    // (v0.24.75) refutable disproof として格納．通常 lookup からは
                    // 不可視とし PNS の arena-limited false NM を防止する．
                    self.table.store_refutable_disproof(pos_key, att_hand);
                } else {
                    self.store(pos_key, att_hand, INF, 0, 0, pos_key as u32);
                }
            } else {
                self.store(pos_key, att_hand, INF, 0, 0, pos_key as u32);
            }
            return;
        }

        // 合法手生成
        let moves = if or_node {
            self.generate_check_moves_cached(board)
        } else {
            self.generate_defense_moves(board)
        };

        if moves.is_empty() {
            if or_node {
                // 王手手段なし → 不詰
                arena[node_idx as usize].pn = INF;
                arena[node_idx as usize].dn = 0;
                self.store(pos_key, att_hand, INF, 0, REMAINING_INFINITE, pos_key as u32);
            } else {
                // 応手なし → 詰み
                arena[node_idx as usize].pn = 0;
                arena[node_idx as usize].dn = INF;
                self.store(pos_key, [0; HAND_KINDS], 0, INF, REMAINING_INFINITE, pos_key as u32);
            }
            arena[node_idx as usize].expanded = true;
            return;
        }

        // DFPN-E: 守備側玉の位置(OR ノードのエッジコスト計算用)
        let defender_king_sq = if or_node {
            board.king_square(board.turn.opponent())
        } else {
            None
        };

        let child_remaining = remaining.saturating_sub(1);
        let child_or_node = !or_node;
        let mut or_nm_min_remaining: u16 = REMAINING_INFINITE;
        // AND ノードの合駒逐次活性化: 最初の unproven drop のみ子ノード生成
        let mut first_unproven_drop_added = or_node; // OR ノードでは無効

        for m in &moves {
            let captured = board.do_move(*m);
            let child_fh = board.hash;
            let child_pk = position_key(board);
            let child_hand = board.hand[self.attacker.index()];

            // ループ検出: 祖先と同一局面なら即座に不詰/無限ループ扱い
            let is_loop = ancestors.contains(&child_fh);

            let (mut cpn, mut cdn) = if is_loop {
                (INF, 0u32)
            } else {
                let (p, d, _) = self.look_up_pn_dn(child_pk, &child_hand, child_remaining);
                (p, d)
            };

            // TT に初期値(1,1)しかない場合: ヒューリスティック初期化
            if cpn == PN_UNIT && cdn == PN_UNIT && !is_loop {
                if child_or_node {
                    // 子は OR ノード(攻め方手番): 王手数ベース
                    let checks = self.generate_check_moves_cached(board);
                    if checks.is_empty() {
                        cpn = INF;
                        cdn = 0;
                        self.store(child_pk, child_hand, INF, 0, REMAINING_INFINITE, child_pk as u32);
                    } else if self.has_mate_in_1_with(board, &checks) {
                        cpn = 0;
                        cdn = INF;
                        self.store(child_pk, child_hand, 0, INF, REMAINING_INFINITE, child_pk as u32);
                    } else {
                        let nc = checks.len() as u32;
                        let (or_pn, or_se) = self.heuristic_or_pn(board, nc, child_pk);
                        cpn = or_pn
                            .saturating_add(edge_cost_and(*m))
                            .saturating_add(sacrifice_check_boost(board, &checks));
                        let att_in_check = board.is_in_check(board.turn);
                        cdn = heuristic_or_dn(or_se, nc, att_in_check);
                        self.store(child_pk, child_hand, cpn, cdn, child_remaining, child_pk as u32);
                    }
                } else {
                    // 子は AND ノード(玉方手番): 応手数ベース
                    let defenses = self.generate_defense_moves(board);
                    if defenses.is_empty() {
                        cpn = 0;
                        cdn = INF;
                        self.store(child_pk, [0; HAND_KINDS], 0, INF, REMAINING_INFINITE, child_pk as u32);
                    } else {
                        let n = defenses.len() as u32;
                        cpn = self.heuristic_and_pn(board, n);
                        if let Some(ksq) = defender_king_sq {
                            cpn = cpn.saturating_add(edge_cost_or(*m, ksq));
                        }
                        cdn = heuristic_dn_from_pn(cpn);
                        self.store(child_pk, child_hand, cpn, cdn, child_remaining, child_pk as u32);
                    }
                }
            }

            board.undo_move(*m, captured);

            // OR ノードで子が即座に証明 → 親を証明して終了
            if or_node && cpn == 0 {
                let child_ph = self.table.get_proof_hand(child_pk, &child_hand);
                let mut proof = adjust_hand_for_move(*m, &child_ph);
                for k in 0..HAND_KINDS {
                    proof[k] = proof[k].min(att_hand[k]);
                }
                arena[node_idx as usize].pn = 0;
                arena[node_idx as usize].dn = INF;
                arena[node_idx as usize].expanded = true;
                self.store(pos_key, proof, 0, INF, REMAINING_INFINITE, pos_key as u32);
                return;
            }
            // AND ノードで子が即座に反証 → 親を反証して終了
            if !or_node && cdn == 0 {
                arena[node_idx as usize].pn = INF;
                arena[node_idx as usize].dn = 0;
                arena[node_idx as usize].expanded = true;
                let child_rem = self.table.get_effective_disproof_info(
                    child_pk, &child_hand, child_remaining,
                ).map(|(r, _)| r).unwrap_or(0);
                let prop_rem = propagate_nm_remaining(child_rem, remaining);
                // 反証駒最適化: 子の ProvenTT 反証駒 (DH_C ≥ att_hand) を AND-node に伝播．
                // AND-node では守備側着手で att_hand 不変 (child_hand = att_hand)．
                let dh = self.table.get_disproof_hand(child_pk, &child_hand);
                self.store(pos_key, dh, INF, 0, prop_rem, pos_key as u32);
                return;
            }
            // OR ノードで子が反証済み → 子を追加せずスキップ
            if or_node && cdn == 0 {
                let child_rem = self.table.get_effective_disproof_info(
                    child_pk, &child_hand, child_remaining,
                ).map(|(r, _)| r).unwrap_or(0);
                or_nm_min_remaining = or_nm_min_remaining.min(child_rem);
                continue;
            }

            // AND ノードの合駒逐次活性化:
            // 未証明の drop を1つだけ子ノードとして展開し，残りは deferred_drops に格納．
            // 弱い駒から順に証明 → TT 蓄積 → 強い駒の探索で援用する．
            // 既に証明済み(cpn=0)の drop はそのまま子ノードとして追加する．
            if !or_node && m.is_drop() && cpn > 0 {
                if first_unproven_drop_added {
                    #[cfg(feature = "tt_diag")]
                    eprintln!(
                        "[pns_seq] pns_expand: defer drop {} at AND node idx={}",
                        m.to_usi(), node_idx,
                    );
                    arena[node_idx as usize].deferred_drops.push_back(*m);
                    continue;
                }
                first_unproven_drop_added = true;
            }

            let child_idx = arena.len() as u32;
            arena.push(PnsNode {
                pos_key: child_pk,
                full_hash: child_fh,
                hand: child_hand,
                pn: cpn,
                dn: cdn,
                parent: node_idx,
                move_from_parent: *m,
                or_node: child_or_node,
                expanded: cpn == 0 || cdn == 0,
                children: Vec::new(),
                cached_best: u32::MAX,
                remaining: child_remaining,
                deferred_drops: VecDeque::new(),
            });
            arena[node_idx as usize].children.push(child_idx);
        }

        // OR ノードで全子が反証済み(children 空)
        if or_node && arena[node_idx as usize].children.is_empty() {
            arena[node_idx as usize].pn = INF;
            arena[node_idx as usize].dn = 0;
            arena[node_idx as usize].expanded = true;
            let mut prop_rem = propagate_nm_remaining(or_nm_min_remaining, remaining);
            // REMAINING_INFINITE 昇格: ハイブリッド判定 (上述と同じ経路)．
            if prop_rem != REMAINING_INFINITE {
                let checks = self.generate_check_moves_cached(board);
                if checks.is_empty() {
                    prop_rem = REMAINING_INFINITE;
                } else if self.refutable_check_with_cache(board, pos_key, &checks) {
                    // (v0.24.75) refutable disproof として格納
                    self.table.store_refutable_disproof(pos_key, att_hand);
                    return;
                }
            }
            self.store(pos_key, att_hand, INF, 0, prop_rem, pos_key as u32);
            return;
        }

        arena[node_idx as usize].expanded = true;
    }

    /// PNS バックアップ: 展開ノードからルートまで pn/dn を再計算する．
    ///
    /// OR ノード: pn = min(child_pn), dn = sum(child_dn)
    /// AND ノード: WPN = max(child_pn) + (unproven_count - 1), dn = min(child_dn)
    /// pn/dn が変化しなくなった時点で伝播を打ち切る．
    pub(super) fn pns_backup(arena: &mut [PnsNode], start_idx: u32) {
        let mut current = start_idx;
        loop {
            let ni = current as usize;

            if !arena[ni].expanded || arena[ni].children.is_empty() {
                // 終端ノード(子なし): pn/dn は展開時に設定済み
                if arena[ni].parent == u32::MAX {
                    return;
                }
                current = arena[ni].parent;
                continue;
            }

            // AND ノードが展開時の早期脱出で反証済み(dn=0)の場合，
            // 部分的な子しかアリーナにないため再計算をスキップする．
            // (再計算すると部分的な子だけから pn=0 と誤判定される)
            if !arena[ni].or_node && arena[ni].dn == 0 {
                if arena[ni].parent == u32::MAX {
                    return;
                }
                current = arena[ni].parent;
                continue;
            }

            let old_pn = arena[ni].pn;
            let old_dn = arena[ni].dn;

            let (new_pn, new_dn, best) = if arena[ni].or_node {
                // OR ノード: pn = min(child_pn), dn = WPN-scaled sum(child_dn)
                // best_child: min by (pn, dn)
                let mut min_pn = INF;
                let mut sum_dn: u64 = 0;
                let mut max_dn: u32 = 0;
                let mut best_idx = arena[ni].children[0];
                let mut best_key = (u32::MAX, u32::MAX);
                let num_children = arena[ni].children.len();
                for i in 0..num_children {
                    let child = arena[ni].children[i];
                    let ci = child as usize;
                    let key = (arena[ci].pn, arena[ci].dn);
                    if key < best_key {
                        best_key = key;
                        best_idx = child;
                    }
                    if arena[ci].pn < min_pn {
                        min_pn = arena[ci].pn;
                    }
                    if arena[ci].dn > max_dn { max_dn = arena[ci].dn; }
                    sum_dn = sum_dn.saturating_add(arena[ci].dn as u64);
                }
                let sum_other = sum_dn.saturating_sub(max_dn as u64);
                let dn = (max_dn as u64)
                    .saturating_add(sum_other >> WPN_GAMMA_SHIFT)
                    .min(INF as u64) as u32;
                (min_pn, dn, best_idx)
            } else {
                // AND ノード: WPN, dn = min(child_dn)
                // best_child: min by (dn, pn) among unproven children
                let mut max_pn: u32 = 0;
                let mut sum_pn: u64 = 0;
                let mut min_dn = INF;
                let mut best_idx = arena[ni].children[0];
                let mut best_key = (u32::MAX, u32::MAX);
                let mut unproven: u32 = 0;
                let mut disproved = false;
                let num_children = arena[ni].children.len();
                for i in 0..num_children {
                    let child = arena[ni].children[i];
                    let ci = child as usize;
                    if arena[ci].dn == 0 {
                        disproved = true;
                        break;
                    }
                    if arena[ci].pn == 0 {
                        // VPN: 証明済み子を pn 合計から除外
                        continue;
                    }
                    let key = (arena[ci].dn, arena[ci].pn);
                    if key < best_key {
                        best_key = key;
                        best_idx = child;
                    }
                    if arena[ci].pn > max_pn {
                        max_pn = arena[ci].pn;
                    }
                    sum_pn += arena[ci].pn as u64;
                    if arena[ci].dn < min_dn {
                        min_dn = arena[ci].dn;
                    }
                    unproven += 1;
                }
                if disproved {
                    (INF, 0u32, best_idx)
                } else if unproven == 0 && arena[ni].deferred_drops.is_empty() {
                    // 全子証明済み + deferred なし → AND ノード証明
                    (0u32, INF, best_idx)
                } else if unproven == 0 {
                    // 全子証明済みだが deferred_drops 残り → 未完了
                    // MPN 選択時に次の合駒を活性化するため pn=1, dn=1 で保持
                    (PN_UNIT, PN_UNIT, best_idx)
                } else {
                    let sum_other = sum_pn.saturating_sub(max_pn as u64);
                    let pn = (max_pn as u64)
                        .saturating_add(sum_other >> WPN_GAMMA_SHIFT)
                        .min(INF as u64) as u32;
                    (pn, min_dn, best_idx)
                }
            };

            arena[ni].pn = new_pn;
            arena[ni].dn = new_dn;
            arena[ni].cached_best = best;

            // pn/dn が変化しなければ伝播打ち切り
            if new_pn == old_pn && new_dn == old_dn {
                return;
            }
            if arena[ni].parent == u32::MAX {
                return;
            }
            current = arena[ni].parent;
        }
    }

    /// PNS 証明ツリーの整合性を検証する(デバッグ用)．
    ///
    /// OR ノード(pn=0): 少なくとも1つの子が pn=0
    /// AND ノード(pn=0): 全ての子が pn=0(展開済みの場合)
    #[cfg(debug_assertions)]
    pub(super) fn validate_pns_proof(arena: &[PnsNode], idx: u32) {
        let node = &arena[idx as usize];
        if node.pn != 0 {
            return;
        }
        if !node.expanded || node.children.is_empty() {
            // リーフ: TT/ヒューリスティックから取得した pn=0
            verbose_eprintln!("  PNS leaf proven: idx={}, or={}, pk={:#x}, move={}",
                idx, node.or_node, node.pos_key,
                if idx == 0 { "root".to_string() } else { node.move_from_parent.to_usi() });
            return;
        }
        if node.or_node {
            // OR: 少なくとも1つの子が pn=0
            let has_proven = node.children.iter().any(|&c| arena[c as usize].pn == 0);
            assert!(has_proven,
                "PNS BUG: OR node {} (pk={:#x}) is proven but no child has pn=0. children: {:?}",
                idx, node.pos_key,
                node.children.iter().map(|&c| {
                    let ch = &arena[c as usize];
                    format!("{}(pn={},dn={},or={},exp={})",
                        ch.move_from_parent.to_usi(), ch.pn, ch.dn, ch.or_node, ch.expanded)
                }).collect::<Vec<_>>());
            // 証明された子を表示
            #[cfg(feature = "verbose")]
            {
                let proven_child = node.children.iter()
                    .find(|&&c| arena[c as usize].pn == 0).unwrap();
                eprintln!("  PNS OR proven: idx={}, pk={:#x}, best_child={} ({})",
                    idx, node.pos_key, proven_child,
                    arena[*proven_child as usize].move_from_parent.to_usi());
            }
            // 再帰: 証明された子のみ
            for &c in &node.children {
                if arena[c as usize].pn == 0 {
                    Self::validate_pns_proof(arena, c);
                }
            }
        } else {
            verbose_eprintln!("  PNS AND proven: idx={}, pk={:#x}, move={}, {} children",
                idx, node.pos_key, node.move_from_parent.to_usi(), node.children.len());
            // AND: 全子が pn=0
            for &c in &node.children {
                assert!(arena[c as usize].pn == 0,
                    "PNS BUG: AND node {} (pk={:#x}) is proven but child {} ({}) has pn={}",
                    idx, node.pos_key, c,
                    arena[c as usize].move_from_parent.to_usi(),
                    arena[c as usize].pn);
                Self::validate_pns_proof(arena, c);
            }
        }
    }

    /// PNS アリーナから直接 PV を抽出する．
    ///
    /// TT ベースの `extract_pv` と異なり，PNS が構築した明示的な探索木を
    /// 辿るため，PNS が証明した経路を正確に復元できる．
    /// 展開されていないリーフ(TT から pn=0 を取得した子)では
    /// TT ベースの `extract_pv_recursive` にフォールバックする．
    ///
    /// - OR ノード: 証明済み子のうち最短 PV を選択
    /// - AND ノード: 証明済み子のうち最長 PV を選択(最長抵抗)
    pub(super) fn pns_extract_pv(
        &mut self,
        board: &mut Board,
        arena: &[PnsNode],
        node_idx: u32,
        visited: &mut FxHashSet<u64>,
    ) -> Vec<Move> {
        let node = &arena[node_idx as usize];

        // 未証明ノード → PV なし
        if node.pn != 0 {
            return Vec::new();
        }

        // ループ検出
        if visited.contains(&node.full_hash) {
            return Vec::new();
        }

        // 未展開リーフまたは子なし(終端) → TT フォールバック
        if !node.expanded || node.children.is_empty() {
            return self.extract_pv_recursive(board, node.or_node, visited, 0);
        }

        if node.or_node {
            // OR ノード: 証明済み子から最短 PV を選択
            let mut best_pv: Option<Vec<Move>> = None;

            for &ci in &node.children {
                let child = &arena[ci as usize];
                if child.pn != 0 {
                    continue;
                }

                let captured = board.do_move(child.move_from_parent);
                visited.insert(node.full_hash);
                let sub_pv = self.pns_extract_pv(board, arena, ci, visited);
                visited.remove(&node.full_hash);
                board.undo_move(child.move_from_parent, captured);

                // sub_pv が空でないか，AND 終端(応手なし=詰み)なら有効
                let total_len = 1 + sub_pv.len();
                // 奇数長(攻め方の手で終わる)のみ有効な PV
                if total_len % 2 == 0 && !sub_pv.is_empty() {
                    continue;
                }
                let is_better = match &best_pv {
                    None => true,
                    Some(prev) => total_len < prev.len(),
                };
                if is_better {
                    let mut pv = vec![child.move_from_parent];
                    pv.extend(sub_pv);
                    best_pv = Some(pv);
                }
            }

            best_pv.unwrap_or_default()
        } else {
            // AND ノード: 全子が証明済み，最長 PV を選択(最長抵抗)．
            // 無駄合 chain による raw length 膨張を除外するため，
            // extract_pv_recursive_inner と同じく effective length で比較する．
            let mut best_pv: Option<Vec<Move>> = None;
            let mut best_is_capture = false;
            let mut best_effective_len: usize = 0;

            for &ci in &node.children {
                let child = &arena[ci as usize];
                if child.pn != 0 {
                    continue;
                }

                let captured = board.do_move(child.move_from_parent);
                visited.insert(node.full_hash);
                let sub_pv = self.pns_extract_pv(board, arena, ci, visited);
                visited.remove(&node.full_hash);
                board.undo_move(child.move_from_parent, captured);

                let total_len = 1 + sub_pv.len();
                // AND ノードの PV は偶数長でなければならない
                if total_len % 2 == 1 {
                    continue;
                }
                // full_pv = [child_move] ++ sub_pv で effective length を計算
                let mut full_pv: Vec<Move> = Vec::with_capacity(total_len);
                full_pv.push(child.move_from_parent);
                full_pv.extend_from_slice(&sub_pv);
                let useless_pairs = Self::count_useless_interpose_pairs(&full_pv);
                let effective_len = total_len.saturating_sub(2 * useless_pairs);

                let is_capture = child.move_from_parent.captured_piece_raw() > 0;
                // Phase 1 (PNS arena-based) の AND ノード選択基準．
                // `extract_pv_recursive_inner` (Phase 2, TT-based) と同じ順序だが，
                // arena では is_drop tiebreaker は省略してある (全子が評価済みで
                // 無駄合判定が効果長に既に反映されているため)．
                let is_better = match &best_pv {
                    None => true,
                    Some(prev) => {
                        // 第一基準: 効果長 (無駄合除外後の真の resistance) が長い
                        // 第二基準: 同率なら raw length が短い PV を優先
                        // 第三基準: 同率なら駒取りを優先
                        if effective_len > best_effective_len {
                            true
                        } else if effective_len < best_effective_len {
                            false
                        } else if total_len < prev.len() {
                            true
                        } else if total_len > prev.len() {
                            false
                        } else if is_capture && !best_is_capture {
                            true
                        } else {
                            false
                        }
                    }
                };
                if is_better {
                    best_pv = Some(full_pv);
                    best_is_capture = is_capture;
                    best_effective_len = effective_len;
                }
            }

            best_pv.unwrap_or_default()
        }
    }

    /// PNS アリーナの証明済みノードを TT に格納する．
    ///
    /// 証明済み(pn=0)の中間ノードのみを格納し，反証(dn=0)は格納しない．
    /// 格納された proof エントリ数を返す．
    pub(super) fn pns_store_to_tt(&mut self, arena: &[PnsNode]) -> u64 {
        let mut proof_store_count: u64 = 0;
        for node in arena {
            if node.pn == 0 && node.expanded && !node.children.is_empty() {
                // 証明済み中間ノード
                if node.or_node {
                    // OR 証明: 証明子の手を TT Best Move に記録
                    let best_child = node.children.iter()
                        .find(|&&c| arena[c as usize].pn == 0);
                    if let Some(&ci) = best_child {
                        let best_move16 = arena[ci as usize].move_from_parent.to_move16();
                        self.store_with_best_move(
                            node.pos_key, node.hand, 0, INF,
                            REMAINING_INFINITE, node.pos_key as u32, best_move16,
                        );
                        proof_store_count += 1;
                    }
                } else {
                    // AND 証明: 全子が証明済み
                    self.store(
                        node.pos_key, node.hand, 0, INF,
                        REMAINING_INFINITE, node.pos_key as u32,
                    );
                    proof_store_count += 1;
                }
            } else if node.dn == 0 && node.expanded {
                // PNS の反証(NM)は TT にバックプロパゲーションしない．
                // 理由: PNS はアリーナサイズに制限された最良優先探索であり，
                // 探索木を完全には展開しない．そのため PNS の NM は
                // 「アリーナ内で反証された」という意味でしかなく，
                // MID(DFS) の NM のように「深さ R 以内で完全に反証された」
                // とは保証できない．PNS NM を remaining 付きで TT に格納すると，
                // 後続の mid_fallback が NM エントリをヒットして探索をスキップし，
                // 偽 NoCheckmate を引き起こす．
                // 展開フェーズ(expand_pns_node)で各ノードの NM は既に
                // TT に個別に記録済みなので，backprop での追加格納は不要．
            }
        }
        #[cfg(feature = "verbose")]
        {
            self.dbg_pns_proof_stores += proof_store_count;
        }
        #[cfg(feature = "tt_diag")]
        {
            for node in arena {
                if node.pn == 0 && node.expanded && !node.children.is_empty() {
                    let ply = self.depth.saturating_sub(node.remaining as u32) as usize;
                    if ply < 64 {
                        self.diag_pns_proof_ply[ply] += 1;
                    }
                }
            }
        }
        proof_store_count
    }
}


/// デフォルトタイムアウト(300秒)で詰将棋を解く便利関数．
///
/// [`solve_tsume_with_timeout`] のラッパーで，タイムアウト・PV ノード予算・
/// TT GC 閾値にはデフォルト値を使用する．
///
/// # 引数
///
/// - `sfen`: 局面の SFEN 文字列．
/// - `depth`: 最大探索手数(`None` でデフォルト 31)．
/// - `nodes`: 最大ノード数(`None` でデフォルト 1,048,576)．
/// - `draw_ply`: 引き分け手数(`None` でデフォルト 32767)．
///
/// # 戻り値
///
/// [`TsumeResult`] を返す．SFEN パースエラー時は `Err` を返す．
pub fn solve_tsume(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
) -> Result<TsumeResult, crate::board::SfenError> {
    solve_tsume_with_timeout(sfen, depth, nodes, draw_ply, None, None, None, None)
}

/// タイムアウト指定付きで詰将棋を解く便利関数．
///
/// # 戻り値
///
/// 詰みが証明された場合でも，PV 復元フェーズ(`complete_pv_or_nodes`)の
/// ノード予算が不足すると [`TsumeResult::CheckmateNoPv`] が返ることがある．
/// 特に長手数(17手以上)の詰将棋では，PV 沿いの各未証明子に対する
/// 追加証明の1子あたり予算(デフォルト 1024 ノード)が不足しやすい．
/// `pv_nodes_per_child` を増やすことで改善できる．
///
/// # 引数
///
/// - `find_shortest`: 最短手数探索を行うか(None でデフォルト true)．
///   false にすると `complete_or_proofs()` による追加探索をスキップし，
///   最初に見つかった詰み手順をそのまま返す．ノード数は削減されるが，
///   返される手順が最短とは限らない．
/// - `pv_nodes_per_child`: PV 復元時の1子あたりノード予算(None でデフォルト 1024)．
///   長手数の詰将棋で `CheckmateNoPv` が返る場合に増やすと効果的．
pub fn solve_tsume_with_timeout(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
    timeout_secs: Option<u64>,
    find_shortest: Option<bool>,
    pv_nodes_per_child: Option<u64>,
    tt_gc_threshold: Option<usize>,
) -> Result<TsumeResult, crate::board::SfenError> {
    let mut board = Board::empty();
    board.set_sfen(sfen)?;

    let mut solver = DfPnSolver::with_timeout(
        depth.unwrap_or(31),
        nodes.unwrap_or(1_048_576),
        draw_ply.unwrap_or(32767),
        timeout_secs.unwrap_or(300),
    );
    solver.set_find_shortest(find_shortest.unwrap_or(true));
    if let Some(budget) = pv_nodes_per_child {
        solver.set_pv_nodes_per_child(budget);
    }
    if let Some(gc) = tt_gc_threshold {
        solver.set_tt_gc_threshold(gc);
    }

    Ok(solver.solve(&mut board))
}

/// 詰将棋を解き，探索終了時の WorkingTT pn/dn 分布を返す (分析用)．
///
/// 返り値: `(TsumeResult, pn_hist, dn_hist, joint_hist)`
/// - pn_hist: pn 値の log2 ヒストグラム (32 バケット)
/// - dn_hist: dn 値の log2 ヒストグラム (32 バケット)
/// - joint_hist: (pn バケット × dn バケット) の 2D ヒストグラム (32×32 = 1024 要素)
/// - per_depth: IDS 各 depth の `(ids_depth, nodes, elapsed_secs, pn_hist, dn_hist, joint)`
pub fn solve_tsume_and_collect_pn_dn_dist(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
    timeout_secs: Option<u64>,
    find_shortest: Option<bool>,
    pv_nodes_per_child: Option<u64>,
    tt_gc_threshold: Option<usize>,
) -> Result<
    (
        TsumeResult,
        [u64; 32],
        [u64; 32],
        Vec<u64>,
        Vec<(u32, u64, f64, [u64; 32], [u64; 32], Vec<u64>)>,
    ),
    crate::board::SfenError,
> {
    let mut board = Board::empty();
    board.set_sfen(sfen)?;

    let mut solver = DfPnSolver::with_timeout(
        depth.unwrap_or(31),
        nodes.unwrap_or(1_048_576),
        draw_ply.unwrap_or(32767),
        timeout_secs.unwrap_or(300),
    );
    solver.set_find_shortest(find_shortest.unwrap_or(true));
    if let Some(budget) = pv_nodes_per_child {
        solver.set_pv_nodes_per_child(budget);
    }
    if let Some(gc) = tt_gc_threshold {
        solver.set_tt_gc_threshold(gc);
    }

    let result = solver.solve(&mut board);
    let (pn_hist, dn_hist, joint_hist) = solver.collect_pn_dn_dist();
    let per_depth = solver.collect_pn_dn_dist_per_depth().to_vec();
    Ok((result, pn_hist, dn_hist, joint_hist, per_depth))
}
