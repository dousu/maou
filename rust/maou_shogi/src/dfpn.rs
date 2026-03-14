//! Df-Pn (Depth-First Proof-Number Search) による詰将棋ソルバー．
//!
//! cshogi と同じ Df-Pn アルゴリズムを採用し，攻め方(先手)が玉方(後手)を
//! 詰ませる最善手順を求める．
//!
//! # 引数
//!
//! - `depth`: 最大探索手数(デフォルト 31)．無限ループ防止用．
//! - `nodes`: 最大ノード数(デフォルト 1,048,576 = 2^20)．計算時間・メモリ制限用．
//! - `draw_ply`: 引き分け手数(デフォルト 32767)．

use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::attack;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movegen;
use crate::moves::Move;
use crate::types::{Color, PieceType, Square};

/// 証明数・反証数の無限大を表す定数．
const INF: u32 = u32::MAX;

/// 詰将棋の探索結果．
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TsumeResult {
    /// 詰みが見つかった場合．手順を含む．
    Checkmate {
        moves: Vec<Move>,
        nodes_searched: u64,
    },
    /// 不詰の場合．
    NoCheckmate { nodes_searched: u64 },
    /// 探索制限に達した場合(nodes上限 or depth上限)．
    Unknown { nodes_searched: u64 },
}

/// 固定サイズ転置表のエントリ．
///
/// - key: Zobrist ハッシュの全 64 ビット(衝突検証用)
/// - pn, dn: 証明数・反証数
#[derive(Debug, Clone, Copy)]
struct DfPnEntry {
    key: u64,
    pn: u32,
    dn: u32,
}

/// 固定サイズの転置表(2-way set associative)．
///
/// HashMap の代わりに固定長 Vec を使用し，キャッシュ効率を最大化する．
/// 各スロットに 2 エントリを格納し，衝突時は pn=0(証明済み)を優先保持，
/// それ以外は上書き．
struct TranspositionTable {
    /// 各スロットに 2 エントリ(偶数=slot0, 奇数=slot1)
    entries: Vec<DfPnEntry>,
    mask: u64,
}

impl TranspositionTable {
    /// 指定サイズ(2のべき乗に切り上げ)で転置表を生成する．
    fn new(min_size: usize) -> Self {
        let size = min_size.max(1024).next_power_of_two();
        let empty = DfPnEntry {
            key: 0,
            pn: 0,
            dn: 0,
        };
        TranspositionTable {
            entries: vec![empty; size * 2], // 2-way
            mask: (size - 1) as u64,
        }
    }

    /// 転置表を参照する．未登録なら pn=1, dn=1 を返す．
    #[inline]
    fn look_up(&self, hash: u64) -> (u32, u32) {
        let base = ((hash & self.mask) as usize) * 2;
        let e0 = &self.entries[base];
        if e0.key == hash && (e0.pn | e0.dn) != 0 {
            return (e0.pn, e0.dn);
        }
        let e1 = &self.entries[base + 1];
        if e1.key == hash && (e1.pn | e1.dn) != 0 {
            return (e1.pn, e1.dn);
        }
        (1, 1) // デフォルト: 未探索
    }

    /// 転置表を更新する．
    ///
    /// 置換戦略: 空スロット > 非証明済みエントリ > slot1 を上書き．
    /// pn=0(証明済み)のエントリは可能な限り保持する．
    #[inline]
    fn store(&mut self, hash: u64, pn: u32, dn: u32) {
        let base = ((hash & self.mask) as usize) * 2;
        let new_entry = DfPnEntry { key: hash, pn, dn };

        // 同一ハッシュの既存エントリを更新
        if self.entries[base].key == hash {
            self.entries[base] = new_entry;
            return;
        }
        if self.entries[base + 1].key == hash {
            self.entries[base + 1] = new_entry;
            return;
        }

        // 空スロットに格納
        if (self.entries[base].pn | self.entries[base].dn) == 0 {
            self.entries[base] = new_entry;
            return;
        }
        if (self.entries[base + 1].pn | self.entries[base + 1].dn) == 0 {
            self.entries[base + 1] = new_entry;
            return;
        }

        // 証明済み(pn=0)でない方を上書き
        if self.entries[base].pn != 0 {
            self.entries[base] = new_entry;
        } else if self.entries[base + 1].pn != 0 {
            self.entries[base + 1] = new_entry;
        } else {
            // 両方証明済み → slot1 を上書き
            self.entries[base + 1] = new_entry;
        }
    }

    /// 全エントリをクリアする．
    fn clear(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = DfPnEntry {
                key: 0,
                pn: 0,
                dn: 0,
            };
        }
    }
}

/// Df-Pn ソルバー．
pub struct DfPnSolver {
    /// 最大探索手数．
    depth: u32,
    /// 最大ノード数．
    max_nodes: u64,
    /// 引き分け手数．
    draw_ply: u32,
    /// 実行時間制限．
    timeout: Duration,
    /// 転置表(固定サイズ)．
    table: TranspositionTable,
    /// 探索ノード数．
    nodes_searched: u64,
    /// 探索中の最大ply(デバッグ用)．
    max_ply: u32,
    /// 探索中のパス(ループ検出用)．
    path: HashSet<u64>,
    /// 探索開始時刻．
    start_time: Instant,
    /// タイムアウトしたかどうか．
    timed_out: bool,
}

impl DfPnSolver {
    /// 新しいソルバーを生成する(タイムアウト 300 秒)．
    pub fn new(depth: u32, max_nodes: u64, draw_ply: u32) -> Self {
        Self::with_timeout(depth, max_nodes, draw_ply, 300)
    }

    /// タイムアウト指定付きでソルバーを生成する．
    pub fn with_timeout(depth: u32, max_nodes: u64, draw_ply: u32, timeout_secs: u64) -> Self {
        // TT サイズ: 詰将棋の作業集合は通常数千〜数万局面のため，
        // max_nodes に比例させず固定上限で十分．大きすぎるとアロケーション
        // オーバーヘッドが支配的になる(5M→200ms)．
        // 最大 512K エントリ(2-way で 1M エントリ ≈ 16MB)，最小 64K．
        let tt_size = (max_nodes as usize).min(524_288).max(65_536);
        DfPnSolver {
            depth,
            max_nodes,
            draw_ply,
            timeout: Duration::from_secs(timeout_secs),
            table: TranspositionTable::new(tt_size),
            nodes_searched: 0,
            max_ply: 0,
            path: HashSet::new(),
            start_time: Instant::now(),
            timed_out: false,
        }
    }

    /// デフォルトパラメータでソルバーを生成する．
    pub fn default_solver() -> Self {
        Self::new(31, 1_048_576, 32767)
    }

    /// タイムアウトしたかどうかを返す．
    #[inline]
    fn is_timed_out(&self) -> bool {
        self.timed_out || self.start_time.elapsed() >= self.timeout
    }

    /// 転置表を参照する．
    #[inline]
    fn look_up_pn_dn(&self, hash: u64) -> (u32, u32) {
        self.table.look_up(hash)
    }

    /// 転置表を更新する．
    #[inline]
    fn store(&mut self, hash: u64, pn: u32, dn: u32) {
        self.table.store(hash, pn, dn);
    }

    /// 詰将棋を解く(反復深化 Df-Pn)．
    ///
    /// `board` は攻め方の手番から開始する局面．
    /// 片玉局面(攻め方に玉がない)を想定するが，両玉でも動作する．
    ///
    /// 反復深化により，短い手順から順に探索する．
    /// 深い行き止まりに無駄なノードを費やすことを防ぎ，
    /// 浅い詰み手順を優先的に発見する．
    pub fn solve(&mut self, board: &mut Board) -> TsumeResult {
        self.table.clear();
        self.nodes_searched = 0;
        self.max_ply = 0;
        self.path.clear();
        self.start_time = Instant::now();
        self.timed_out = false;

        let root_hash = board.hash;

        // 単一パス Df-Pn: 反復深化なしで全深さを一度に探索．
        // 証明数・反証数が自然に探索を最も有望な手順に誘導する．
        self.mid(board, INF - 1, INF - 1, 0, true);

        let (root_pn, root_dn) = self.look_up_pn_dn(root_hash);

        if root_pn == 0 {
            self.complete_or_proofs(board);
            let moves = self.extract_pv(board);
            TsumeResult::Checkmate {
                moves,
                nodes_searched: self.nodes_searched,
            }
        } else if root_dn == 0 {
            TsumeResult::NoCheckmate {
                nodes_searched: self.nodes_searched,
            }
        } else {
            TsumeResult::Unknown {
                nodes_searched: self.nodes_searched,
            }
        }
    }

    /// OR/AND ノードの Df-Pn 探索．
    ///
    /// `or_node` が true のとき OR ノード(攻め方)，false のとき AND ノード(玉方)．
    fn mid(
        &mut self,
        board: &mut Board,
        pn_threshold: u32,
        dn_threshold: u32,
        ply: u32,
        or_node: bool,
    ) {
        // ノード制限・タイムアウトチェック
        if self.nodes_searched >= self.max_nodes {
            return;
        }
        // 1024 ノードごとにタイマーをチェック(毎回チェックはオーバーヘッドが大きい)
        if self.nodes_searched & 0x3FF == 0 && self.is_timed_out() {
            self.timed_out = true;
            return;
        }
        self.nodes_searched += 1;
        if ply > self.max_ply {
            self.max_ply = ply;
        }

        let hash = board.hash;

        // ループ検出: 現在の探索パス上に同一局面があれば千日手
        // TT には書き込まず即座に返す(GHI 対策)
        if self.path.contains(&hash) {
            return;
        }

        // 終端条件: 深さ制限・手数制限
        if ply >= self.depth || board.ply() as u32 >= self.draw_ply {
            self.store(hash, INF, 0);
            return;
        }

        // 合法手生成
        let moves = if or_node {
            self.generate_check_moves(board)
        } else {
            self.generate_defense_moves(board)
        };

        // 終端条件チェック
        if moves.is_empty() {
            if or_node {
                // 攻め方に王手手段がない → 不詰
                self.store(hash, INF, 0);
            } else {
                // 玉方に応手がない → 詰み
                self.store(hash, 0, INF);
            }
            return;
        }

        // 子ノードのハッシュを事前計算し，同時に浅い探索で短手数詰め・不詰を検出する．
        //
        // cshogi 互換の最適化:
        // - OR ノード: 1手詰め(応手 0) / 3手詰め(全応手に1手詰め王手あり) /
        //   source-based init (pn = dn = 応手数)
        // - AND ノード: 不詰(王手 0) / 1手詰め検出(孫ノード) /
        //   source-based init (pn = 1, dn = 王手数)
        let mut children: Vec<(Move, u64)> = Vec::with_capacity(moves.len());
        for m in &moves {
            let captured = board.do_move(*m);
            let child_hash = board.hash;

            let (cpn, cdn) = self.look_up_pn_dn(child_hash);
            if cpn == 1 && cdn == 1 {
                if or_node {
                    // 子は AND ノード: 応手を生成
                    let defenses = self.generate_defense_moves(board);
                    if defenses.is_empty() {
                        // 1手詰め
                        self.store(child_hash, 0, INF);
                    } else if ply + 2 < self.depth {
                        // 3手詰めチェック: 全応手に対し1手詰め王手が存在するか
                        let mut all_mated = true;
                        for d in &defenses {
                            let cap_d = board.do_move(*d);
                            let gc_hash = board.hash;
                            let mate = self.has_mate_in_1(board);
                            if mate {
                                // 孫 OR ノードを証明済みとして記録
                                self.store(gc_hash, 0, INF);
                            }
                            board.undo_move(*d, cap_d);
                            if !mate {
                                all_mated = false;
                                break;
                            }
                        }
                        if all_mated {
                            // 3手詰め確定
                            self.store(child_hash, 0, INF);
                        } else {
                            // Source-based init: pn = dn = 応手数
                            let n = defenses.len() as u32;
                            self.store(child_hash, n, n);
                        }
                    } else {
                        // 深さ制限近く: source-based init のみ
                        let n = defenses.len() as u32;
                        self.store(child_hash, n, n);
                    }
                } else {
                    // 子は OR ノード: 王手手を生成
                    let checks = self.generate_check_moves(board);
                    if checks.is_empty() {
                        // 不詰(王手手なし)
                        self.store(child_hash, INF, 0);
                    } else if ply + 2 < self.depth && self.has_mate_in_1(board) {
                        // 子 OR ノードに1手詰めが存在 → 証明済み
                        self.store(child_hash, 0, INF);
                    } else {
                        // Source-based init: pn = 1, dn = 王手数
                        self.store(child_hash, 1, checks.len() as u32);
                    }
                }
            }

            board.undo_move(*m, captured);

            // 即座に解決チェック: 子の証明/反証が親の解決を導く場合，
            // MID ループに入らず早期リターンする
            let (cpn_now, cdn_now) = self.look_up_pn_dn(child_hash);
            if or_node && cpn_now == 0 {
                // OR ノード: 子が証明済み → 親も証明済み
                self.store(hash, 0, INF);
                return;
            }
            if !or_node && cdn_now == 0 {
                // AND ノード: 子が反証済み → 親も反証済み
                self.store(hash, INF, 0);
                return;
            }

            children.push((*m, child_hash));
        }

        // パスに追加
        self.path.insert(hash);

        // MID ループ
        loop {
            // 各子ノードの pn/dn を収集(パス上の子はループとして扱う)
            let child_pn_dn: Vec<(u32, u32)> = children
                .iter()
                .map(|&(_, child_hash)| {
                    if self.path.contains(&child_hash) {
                        (INF, 0)
                    } else {
                        self.look_up_pn_dn(child_hash)
                    }
                })
                .collect();

            // OR ノード: pn = min(children.pn), dn = sum(children.dn)
            // AND ノード: pn = sum(children.pn), dn = min(children.dn)
            let (current_pn, current_dn) = if or_node {
                let min_pn = child_pn_dn.iter().map(|e| e.0).min().unwrap_or(INF);
                let sum_dn = child_pn_dn
                    .iter()
                    .map(|e| e.1 as u64)
                    .sum::<u64>()
                    .min(INF as u64) as u32;
                (min_pn, sum_dn)
            } else {
                let sum_pn = child_pn_dn
                    .iter()
                    .map(|e| e.0 as u64)
                    .sum::<u64>()
                    .min(INF as u64) as u32;
                let min_dn = child_pn_dn.iter().map(|e| e.1).min().unwrap_or(INF);
                (sum_pn, min_dn)
            };

            // 転置表を更新
            self.store(hash, current_pn, current_dn);

            // 閾値チェック
            if current_pn >= pn_threshold || current_dn >= dn_threshold {
                break;
            }

            // ノード制限・タイムアウトチェック
            if self.nodes_searched >= self.max_nodes || self.timed_out {
                break;
            }

            // 最良の子ノードを選択し，閾値を計算
            let (best_idx, child_pn_th, child_dn_th) = if or_node {
                let (idx, second_pn) = self.select_best_or(&child_pn_dn);
                let child_dn_th = dn_threshold
                    .saturating_sub(current_dn)
                    .saturating_add(child_pn_dn[idx].1)
                    .min(INF - 1);
                let child_pn_th = pn_threshold
                    .min(second_pn.saturating_add(1))
                    .min(INF - 1);
                (idx, child_pn_th, child_dn_th)
            } else {
                let (idx, second_dn) = self.select_best_and(&child_pn_dn);
                let child_pn_th = pn_threshold
                    .saturating_sub(current_pn)
                    .saturating_add(child_pn_dn[idx].0)
                    .min(INF - 1);
                let child_dn_th = dn_threshold
                    .min(second_dn.saturating_add(1))
                    .min(INF - 1);
                (idx, child_pn_th, child_dn_th)
            };

            // 子ノードを探索
            let (m, _) = children[best_idx];
            let captured = board.do_move(m);
            self.mid(board, child_pn_th, child_dn_th, ply + 1, !or_node);
            board.undo_move(m, captured);

        }

        // パスから除去
        self.path.remove(&hash);
    }

    /// 現在の局面(攻め方の手番)に1手詰めがあるか判定する．
    ///
    /// 王手を生成し，応手が0の王手があれば1手詰め．
    /// 詰みの局面(応手なし AND ノード)を TT に記録する．
    fn has_mate_in_1(&mut self, board: &mut Board) -> bool {
        let checks = self.generate_check_moves(board);
        for m in &checks {
            let captured = board.do_move(*m);
            let child_hash = board.hash;
            let defenses = self.generate_defense_moves(board);
            if defenses.is_empty() {
                // 詰み局面を TT に記録(PV 復元で必要)
                self.store(child_hash, 0, INF);
                board.undo_move(*m, captured);
                return true;
            }
            board.undo_move(*m, captured);
        }
        false
    }

    /// OR ノードの最良子ノードを選択する．
    ///
    /// 返り値: (最良インデックス, 2番目に小さい pn)
    fn select_best_or(&self, entries: &[(u32, u32)]) -> (usize, u32) {
        let mut best_idx = 0;
        let mut best_pn = INF;
        let mut second_pn = INF;

        for (i, &(pn, _dn)) in entries.iter().enumerate() {
            if pn < best_pn {
                second_pn = best_pn;
                best_pn = pn;
                best_idx = i;
            } else if pn < second_pn {
                second_pn = pn;
            }
        }

        (best_idx, second_pn)
    }

    /// AND ノードの最良子ノードを選択する．
    ///
    /// 返り値: (最良インデックス, 2番目に小さい dn)
    fn select_best_and(&self, entries: &[(u32, u32)]) -> (usize, u32) {
        let mut best_idx = 0;
        let mut best_dn = INF;
        let mut second_dn = INF;

        for (i, &(_pn, dn)) in entries.iter().enumerate() {
            if dn < best_dn {
                second_dn = best_dn;
                best_dn = dn;
                best_idx = i;
            } else if dn < second_dn {
                second_dn = dn;
            }
        }

        (best_idx, second_dn)
    }

    /// 玉方の王手回避手を生成する(合い効かずを除外)．
    ///
    /// 全合法手生成の代わりに回避手のみを直接生成する:
    /// 1. 玉の移動(攻め方に利かれていないマスへ)
    /// 2. 王手駒の捕獲(ピンされていない駒による)
    /// 3. 合い駒(飛び駒の王手の場合，間のマスへ移動または打つ)
    ///
    /// 合い効かず(futile interposition)もフィルタする．
    fn generate_defense_moves(&self, board: &mut Board) -> Vec<Move> {
        let defender = board.turn;
        let attacker = defender.opponent();

        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => return movegen::generate_legal_moves(board),
        };

        // 王手している駒を特定
        let checkers = self.find_checkers(board, king_sq, attacker);
        if checkers.is_empty() {
            // 王手されていない(通常ありえないが安全策)
            return movegen::generate_legal_moves(board);
        }

        let all_occ = board.all_occupied();
        let our_occ = board.occupied[defender.index()];
        let mut moves = Vec::with_capacity(32);

        // --- 1. 玉の移動 ---
        let king_attacks = attack::step_attacks(defender, PieceType::King, king_sq);
        let king_targets = king_attacks & !our_occ;
        for to in king_targets {
            let captured_piece = board.squares[to.index()];
            let captured_raw = captured_piece.0;
            let m = Move::new_move(king_sq, to, false, captured_raw, PieceType::King as u8);
            // 移動先が安全か(攻め方に利かれていないか)チェック
            let captured = board.do_move(m);
            let safe = !board.is_in_check(defender);
            board.undo_move(m, captured);
            if safe {
                moves.push(m);
            }
        }

        // 両王手の場合，玉移動のみ可能
        if checkers.count() > 1 {
            return moves;
        }

        // 単一の王手駒
        let checker_sq = checkers.lsb().unwrap();

        // --- 2. 王手駒の捕獲(玉以外の駒で) ---
        self.generate_capture_checker(
            board, &mut moves, checker_sq, king_sq, defender, all_occ, our_occ,
        );

        // --- 3. 合い駒(飛び駒の王手の場合) ---
        let sliding_checker = self.find_sliding_checker(board, king_sq, attacker);
        if sliding_checker.is_some() {
            let between = attack::between_bb(checker_sq, king_sq);
            if between.is_not_empty() {
                // 合い効かずマスを計算
                let futile = self.compute_futile_squares(
                    board, &between, king_sq, checker_sq, defender, attacker,
                );
                // 間のマスへの合い駒
                self.generate_interpositions(
                    board, &mut moves, &between, &futile, king_sq, defender, all_occ, our_occ,
                );
            }
        }

        moves
    }

    /// 全ての王手駒を検出する．
    fn find_checkers(&self, board: &Board, king_sq: Square, attacker: Color) -> Bitboard {
        let occ = board.all_occupied();
        let att = attacker.index();
        let defender = attacker.opponent();

        let mut checkers = Bitboard::EMPTY;

        // 歩
        checkers = checkers
            | (attack::step_attacks(defender, PieceType::Pawn, king_sq)
                & board.piece_bb[att][PieceType::Pawn as usize]);
        // 桂
        checkers = checkers
            | (attack::step_attacks(defender, PieceType::Knight, king_sq)
                & board.piece_bb[att][PieceType::Knight as usize]);
        // 銀
        checkers = checkers
            | (attack::step_attacks(defender, PieceType::Silver, king_sq)
                & board.piece_bb[att][PieceType::Silver as usize]);
        // 金 + 成駒
        let gold_like = board.piece_bb[att][PieceType::Gold as usize]
            | board.piece_bb[att][PieceType::ProPawn as usize]
            | board.piece_bb[att][PieceType::ProLance as usize]
            | board.piece_bb[att][PieceType::ProKnight as usize]
            | board.piece_bb[att][PieceType::ProSilver as usize];
        checkers = checkers
            | (attack::step_attacks(defender, PieceType::Gold, king_sq) & gold_like);
        // 馬・龍(ステップ部分)
        let king_step = attack::step_attacks(defender, PieceType::King, king_sq);
        checkers = checkers
            | (king_step
                & (board.piece_bb[att][PieceType::Horse as usize]
                    | board.piece_bb[att][PieceType::Dragon as usize]));
        // 香
        checkers = checkers
            | (attack::lance_attacks(defender, king_sq, occ)
                & board.piece_bb[att][PieceType::Lance as usize]);
        // 角・馬
        checkers = checkers
            | (attack::bishop_attacks(king_sq, occ)
                & (board.piece_bb[att][PieceType::Bishop as usize]
                    | board.piece_bb[att][PieceType::Horse as usize]));
        // 飛・龍
        checkers = checkers
            | (attack::rook_attacks(king_sq, occ)
                & (board.piece_bb[att][PieceType::Rook as usize]
                    | board.piece_bb[att][PieceType::Dragon as usize]));

        checkers
    }

    /// 王手駒を玉以外の駒で捕獲する手を生成する．
    fn generate_capture_checker(
        &self,
        board: &mut Board,
        moves: &mut Vec<Move>,
        checker_sq: Square,
        king_sq: Square,
        defender: Color,
        all_occ: Bitboard,
        our_occ: Bitboard,
    ) {
        // 玉以外の自駒で checker_sq に利いている駒を探す
        let mut our_bb = our_occ;
        while our_bb.is_not_empty() {
            let from = our_bb.pop_lsb();
            if from == king_sq {
                continue; // 玉は上で処理済み
            }
            let piece = board.squares[from.index()];
            let pt = piece.piece_type().unwrap();
            let attacks = attack::piece_attacks(defender, pt, from, all_occ);
            if !attacks.contains(checker_sq) {
                continue;
            }

            let captured_raw = board.squares[checker_sq.index()].0;
            let in_promo_zone =
                checker_sq.is_promotion_zone(defender) || from.is_promotion_zone(defender);

            if pt.can_promote() && in_promo_zone {
                let m = Move::new_move(from, checker_sq, true, captured_raw, pt as u8);
                if self.is_evasion_legal(board, m, defender) {
                    moves.push(m);
                }
                if !movegen::must_promote(defender, pt, checker_sq) {
                    let m = Move::new_move(from, checker_sq, false, captured_raw, pt as u8);
                    if self.is_evasion_legal(board, m, defender) {
                        moves.push(m);
                    }
                }
            } else if !movegen::must_promote(defender, pt, checker_sq) {
                let m = Move::new_move(from, checker_sq, false, captured_raw, pt as u8);
                if self.is_evasion_legal(board, m, defender) {
                    moves.push(m);
                }
            }
        }
    }

    /// 合い効かずマスを計算する．
    fn compute_futile_squares(
        &self,
        board: &Board,
        between: &Bitboard,
        king_sq: Square,
        checker_sq: Square,
        defender: Color,
        attacker: Color,
    ) -> Bitboard {
        let king_step = attack::step_attacks(
            defender.opponent(),
            PieceType::King,
            king_sq,
        );
        let mut futile = Bitboard::EMPTY;

        for sq in *between {
            if self.is_attacked_by_non_king(board, sq, defender) {
                continue;
            }
            if king_step.contains(sq)
                && !self.is_attacked_by_excluding(board, sq, attacker, checker_sq)
            {
                continue;
            }
            futile.set(sq);
        }
        futile
    }

    /// 間のマスへの合い駒手を生成する(移動・打ち)．
    fn generate_interpositions(
        &self,
        board: &mut Board,
        moves: &mut Vec<Move>,
        between: &Bitboard,
        futile: &Bitboard,
        king_sq: Square,
        defender: Color,
        all_occ: Bitboard,
        our_occ: Bitboard,
    ) {
        for to in *between {
            // --- 駒移動による合い駒 ---
            let mut our_bb = our_occ;
            while our_bb.is_not_empty() {
                let from = our_bb.pop_lsb();
                if from == king_sq {
                    continue;
                }
                let piece = board.squares[from.index()];
                let pt = piece.piece_type().unwrap();
                let attacks = attack::piece_attacks(defender, pt, from, all_occ);
                if !attacks.contains(to) {
                    continue;
                }
                let captured_raw = board.squares[to.index()].0;
                let in_promo_zone =
                    to.is_promotion_zone(defender) || from.is_promotion_zone(defender);

                if pt.can_promote() && in_promo_zone {
                    let m = Move::new_move(from, to, true, captured_raw, pt as u8);
                    if self.is_evasion_legal(board, m, defender) {
                        moves.push(m);
                    }
                    if !movegen::must_promote(defender, pt, to) {
                        let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                        if self.is_evasion_legal(board, m, defender) {
                            moves.push(m);
                        }
                    }
                } else if !movegen::must_promote(defender, pt, to) {
                    let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                    if self.is_evasion_legal(board, m, defender) {
                        moves.push(m);
                    }
                }
            }

            // --- 駒打ちによる合い駒(合い効かずでないマスのみ) ---
            if futile.contains(to) {
                continue;
            }
            for (hand_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
                if board.hand[defender.index()][hand_idx] == 0 {
                    continue;
                }
                // 行き所のない駒チェック
                if movegen::must_promote(defender, pt, to) {
                    continue;
                }
                // 二歩チェック
                if pt == PieceType::Pawn {
                    let our_pawns =
                        board.piece_bb[defender.index()][PieceType::Pawn as usize];
                    let col = to.col();
                    if (our_pawns & Bitboard::file_mask(col)).is_not_empty() {
                        continue;
                    }
                }
                let m = Move::new_drop(to, pt);
                // 打ち歩詰めチェック
                if pt == PieceType::Pawn && movegen::is_pawn_drop_mate(board, m) {
                    continue;
                }
                moves.push(m);
            }
        }
    }

    /// 回避手の合法性チェック(ピンの確認)．
    #[inline]
    fn is_evasion_legal(&self, board: &mut Board, m: Move, defender: Color) -> bool {
        let captured = board.do_move(m);
        let in_check = board.is_in_check(defender);
        board.undo_move(m, captured);
        !in_check
    }

    /// 飛び駒で王手している駒のマスを返す(単一の場合のみ)．
    ///
    /// 飛び駒が複数(両王手)の場合は None を返す(合い駒不可のため)．
    /// 飛び駒の王手がない場合も None を返す．
    fn find_sliding_checker(
        &self,
        board: &Board,
        king_sq: Square,
        attacker: Color,
    ) -> Option<Square> {
        let occ = board.all_occupied();
        let att = attacker.index();

        let mut checkers = attack::rook_attacks(king_sq, occ)
            & (board.piece_bb[att][PieceType::Rook as usize]
                | board.piece_bb[att][PieceType::Dragon as usize]);
        checkers = checkers
            | (attack::bishop_attacks(king_sq, occ)
                & (board.piece_bb[att][PieceType::Bishop as usize]
                    | board.piece_bb[att][PieceType::Horse as usize]));
        checkers = checkers
            | (attack::lance_attacks(board.turn, king_sq, occ)
                & board.piece_bb[att][PieceType::Lance as usize]);

        // 単一の飛び駒のみ対象
        if checkers.count() == 1 {
            checkers.lsb()
        } else {
            None
        }
    }

    /// 指定マスに対して指定色の駒の効きがあるか判定する(除外条件付き)．
    ///
    /// - `exclude_king`: 玉の効きを除外する(合い効かず: 玉はライン上に移動できないため)
    /// - `excluded_sq`: 特定マスの駒を除外する(飛び駒の移動元を除外するため)
    fn is_attacked_by_filtered(
        &self,
        board: &Board,
        sq: Square,
        attacker_color: Color,
        exclude_king: bool,
        excluded_sq: Option<Square>,
    ) -> bool {
        let mut occ = board.all_occupied();
        let att = attacker_color.index();
        let defender = attacker_color.opponent();

        // 除外マスのマスクを計算
        // occ からも除外して，スライド駒のレイ計算を正確にする
        let mask = match excluded_sq {
            Some(esq) => {
                let mut m = crate::bitboard::Bitboard::EMPTY;
                m.set(esq);
                occ = occ & !m;
                !m
            }
            None => !crate::bitboard::Bitboard::EMPTY, // 全ビット1
        };

        // 歩
        if (attack::step_attacks(defender, PieceType::Pawn, sq)
            & board.piece_bb[att][PieceType::Pawn as usize]
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 桂
        if (attack::step_attacks(defender, PieceType::Knight, sq)
            & board.piece_bb[att][PieceType::Knight as usize]
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 銀
        if (attack::step_attacks(defender, PieceType::Silver, sq)
            & board.piece_bb[att][PieceType::Silver as usize]
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 金 + 成駒
        let gold_movers = (board.piece_bb[att][PieceType::Gold as usize]
            | board.piece_bb[att][PieceType::ProPawn as usize]
            | board.piece_bb[att][PieceType::ProLance as usize]
            | board.piece_bb[att][PieceType::ProKnight as usize]
            | board.piece_bb[att][PieceType::ProSilver as usize])
            & mask;
        if (attack::step_attacks(defender, PieceType::Gold, sq) & gold_movers).is_not_empty() {
            return true;
        }
        // 玉・馬・龍(ステップ部分)
        let king_step = attack::step_attacks(defender, PieceType::King, sq);
        let mut step_pieces = board.piece_bb[att][PieceType::Horse as usize]
            | board.piece_bb[att][PieceType::Dragon as usize];
        if !exclude_king {
            step_pieces = step_pieces | board.piece_bb[att][PieceType::King as usize];
        }
        if (king_step & step_pieces & mask).is_not_empty() {
            return true;
        }
        // 香
        if (attack::lance_attacks(defender, sq, occ)
            & board.piece_bb[att][PieceType::Lance as usize]
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 角・馬(スライド部分)
        if (attack::bishop_attacks(sq, occ)
            & (board.piece_bb[att][PieceType::Bishop as usize]
                | board.piece_bb[att][PieceType::Horse as usize])
            & mask)
            .is_not_empty()
        {
            return true;
        }
        // 飛・龍(スライド部分)
        if (attack::rook_attacks(sq, occ)
            & (board.piece_bb[att][PieceType::Rook as usize]
                | board.piece_bb[att][PieceType::Dragon as usize])
            & mask)
            .is_not_empty()
        {
            return true;
        }
        false
    }

    /// 指定マスに対して指定色の玉以外の駒の効きがあるか判定する．
    fn is_attacked_by_non_king(
        &self,
        board: &Board,
        sq: Square,
        attacker_color: Color,
    ) -> bool {
        self.is_attacked_by_filtered(board, sq, attacker_color, true, None)
    }

    /// 指定マスに対して指定色の駒の効きがあるか判定する(特定マスの駒を除外)．
    fn is_attacked_by_excluding(
        &self,
        board: &Board,
        sq: Square,
        attacker_color: Color,
        excluded_sq: Square,
    ) -> bool {
        self.is_attacked_by_filtered(board, sq, attacker_color, false, Some(excluded_sq))
    }

    /// 攻め方の王手になる手を生成する．
    ///
    /// 最適化: 玉方の玉に王手がかかる手のみを直接生成する．
    /// 全合法手を生成してからフィルタする方式と比べ，生成候補を大幅に削減する．
    fn generate_check_moves(&self, board: &mut Board) -> Vec<Move> {
        let us = board.turn;
        let them = us.opponent();
        let has_own_king = board.king_square(us).is_some();

        let king_sq = match board.king_square(them) {
            Some(sq) => sq,
            None => return Vec::new(),
        };

        let our_occ = board.occupied[us.index()];
        let all_occ = board.all_occupied();
        let empty = !all_occ;

        // 各駒種について「このマスに置くと玉に王手がかかる」ターゲットを事前計算
        // step_attacks(them, pt, king_sq) は「玉から見た逆利き」= 王手元になれるマス

        let mut moves = Vec::with_capacity(64);

        // --- 1. 駒打ち: ターゲットマスのみに打つ ---
        for (hand_idx, &pt) in PieceType::HAND_PIECES.iter().enumerate() {
            if board.hand[us.index()][hand_idx] == 0 {
                continue;
            }

            // この駒種で王手になるマスを計算
            let check_targets = match pt {
                PieceType::Lance => attack::lance_attacks(them, king_sq, all_occ),
                PieceType::Bishop => attack::bishop_attacks(king_sq, all_occ),
                PieceType::Rook => attack::rook_attacks(king_sq, all_occ),
                _ => attack::step_attacks(them, pt, king_sq),
            };
            let mut drop_targets = check_targets & empty;

            // 二歩チェック
            if pt == PieceType::Pawn {
                let our_pawns = board.piece_bb[us.index()][PieceType::Pawn as usize];
                for col in 0..9u8 {
                    let file = Bitboard::file_mask(col);
                    if (our_pawns & file).is_not_empty() {
                        drop_targets &= !file;
                    }
                }
            }

            // 行き所のない駒の制限
            match pt {
                PieceType::Pawn | PieceType::Lance => {
                    let forbidden = match us {
                        Color::Black => Bitboard::rank_mask(0),
                        Color::White => Bitboard::rank_mask(8),
                    };
                    drop_targets &= !forbidden;
                }
                PieceType::Knight => {
                    let forbidden = match us {
                        Color::Black => Bitboard::rank_mask(0) | Bitboard::rank_mask(1),
                        Color::White => Bitboard::rank_mask(7) | Bitboard::rank_mask(8),
                    };
                    drop_targets &= !forbidden;
                }
                _ => {}
            }

            for to in drop_targets {
                let m = Move::new_drop(to, pt);
                // 打ち歩詰めチェック
                if pt == PieceType::Pawn && movegen::is_pawn_drop_mate(board, m) {
                    continue;
                }
                // 駒打ちは自玉への王手放置にならない(片玉でも両玉でも)
                moves.push(m);
            }
        }

        // --- 2. 盤上の駒の移動 ---
        // 直接王手: 移動先から玉に利きがある手
        // 開き王手: 駒が移動することで背後のスライド駒から玉に利きが通る手

        // 開き王手の候補を事前計算:
        // 玉からのレイ上にいる自駒で，その間に他の駒がない場合，
        // そこから移動すると開き王手になりうる
        let discoverers = self.compute_discoverers(board, us, king_sq);

        let mut our_bb = our_occ;
        while our_bb.is_not_empty() {
            let from = our_bb.pop_lsb();
            let piece = board.squares[from.index()];
            let pt = piece.piece_type().unwrap();
            let attacks = attack::piece_attacks(us, pt, from, all_occ);
            let targets = attacks & !our_occ;

            let is_discoverer = discoverers.contains(from);

            for to in targets {
                let captured_piece = board.squares[to.index()];
                let captured_raw = captured_piece.0;
                let in_promo_zone = to.is_promotion_zone(us) || from.is_promotion_zone(us);

                // 成り先の駒種での王手チェック
                if pt.can_promote() && in_promo_zone {
                    let promoted_pt = pt.promoted().unwrap();
                    let gives_direct = self.attacks_square(us, promoted_pt, to, all_occ, king_sq);
                    if gives_direct || is_discoverer {
                        let m = Move::new_move(from, to, true, captured_raw, pt as u8);
                        if self.is_legal_quick(board, m, has_own_king) {
                            moves.push(m);
                        }
                    }

                    // 不成
                    if !movegen::must_promote(us, pt, to) {
                        let gives_direct =
                            self.attacks_square(us, pt, to, all_occ, king_sq);
                        if gives_direct || is_discoverer {
                            let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                            if self.is_legal_quick(board, m, has_own_king) {
                                moves.push(m);
                            }
                        }
                    }
                } else if !movegen::must_promote(us, pt, to) {
                    let gives_direct = self.attacks_square(us, pt, to, all_occ, king_sq);
                    if gives_direct || is_discoverer {
                        let m = Move::new_move(from, to, false, captured_raw, pt as u8);
                        if self.is_legal_quick(board, m, has_own_king) {
                            moves.push(m);
                        }
                    }
                }
            }
        }

        // 手順序: 成り駒 > 駒取り > その他(DFPN の同 pn 時の tie-break に効く)
        moves.sort_by_key(|m| {
            let promo = m.is_promotion();
            let capture = m.captured_piece_raw() > 0;
            match (promo, capture) {
                (true, true) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 3,
            }
        });

        moves
    }

    /// 指定マスに置いた駒が玉のマスに利いているか判定する．
    #[inline]
    fn attacks_square(
        &self,
        color: Color,
        pt: PieceType,
        from: Square,
        occ: Bitboard,
        target: Square,
    ) -> bool {
        attack::piece_attacks(color, pt, from, occ).contains(target)
    }

    /// 開き王手の元になりうる自駒を計算する．
    ///
    /// 玉からの飛び利きレイ上にいる自駒で，玉との間に他の駒がない場合，
    /// その駒はレイ方向以外への移動で開き王手になる．
    fn compute_discoverers(&self, board: &Board, us: Color, king_sq: Square) -> Bitboard {
        let all_occ = board.all_occupied();
        let our_occ = board.occupied[us.index()];
        let mut discoverers = Bitboard::EMPTY;

        // 飛車・龍のレイ(十字方向)
        let our_rook_like = board.piece_bb[us.index()][PieceType::Rook as usize]
            | board.piece_bb[us.index()][PieceType::Dragon as usize];
        if our_rook_like.is_not_empty() {
            // レイ上の最近接駒が自駒で，その先に飛・龍がある場合
            for dir_attacks in [
                attack::rook_attacks(king_sq, all_occ),
            ] {
                let nearest = dir_attacks & all_occ;
                for sq in nearest {
                    if !our_occ.contains(sq) {
                        continue;
                    }
                    // この駒の先に飛・龍があるか
                    let beyond_occ = all_occ & {
                        let mut tmp = Bitboard::EMPTY;
                        tmp.set(sq);
                        !tmp
                    };
                    let beyond = attack::rook_attacks(king_sq, beyond_occ) & !attack::rook_attacks(king_sq, all_occ);
                    if (beyond & our_rook_like).is_not_empty() {
                        discoverers.set(sq);
                    }
                }
            }
        }

        // 角・馬のレイ(斜め方向)
        let our_bishop_like = board.piece_bb[us.index()][PieceType::Bishop as usize]
            | board.piece_bb[us.index()][PieceType::Horse as usize];
        if our_bishop_like.is_not_empty() {
            for dir_attacks in [
                attack::bishop_attacks(king_sq, all_occ),
            ] {
                let nearest = dir_attacks & all_occ;
                for sq in nearest {
                    if !our_occ.contains(sq) {
                        continue;
                    }
                    let beyond_occ = all_occ & {
                        let mut tmp = Bitboard::EMPTY;
                        tmp.set(sq);
                        !tmp
                    };
                    let beyond = attack::bishop_attacks(king_sq, beyond_occ) & !attack::bishop_attacks(king_sq, all_occ);
                    if (beyond & our_bishop_like).is_not_empty() {
                        discoverers.set(sq);
                    }
                }
            }
        }

        // 香のレイ(縦方向: 玉から見て香の利く方向)
        let our_lance = board.piece_bb[us.index()][PieceType::Lance as usize];
        if our_lance.is_not_empty() {
            // 香は前方のみ攻撃する飛び駒．
            // 玉から見て，攻め方の香の前方にある自駒が開き王手元
            let lance_ray = attack::lance_attacks(us.opponent(), king_sq, all_occ);
            let nearest = lance_ray & all_occ;
            for sq in nearest {
                if !our_occ.contains(sq) {
                    continue;
                }
                let beyond_occ = all_occ & {
                    let mut tmp = Bitboard::EMPTY;
                    tmp.set(sq);
                    !tmp
                };
                let beyond = attack::lance_attacks(us.opponent(), king_sq, beyond_occ)
                    & !attack::lance_attacks(us.opponent(), king_sq, all_occ);
                if (beyond & our_lance).is_not_empty() {
                    discoverers.set(sq);
                }
            }
        }

        discoverers
    }

    /// 合法性の簡易チェック(自玉の王手放置のみ)．
    ///
    /// 片玉の場合(自玉なし)は常に合法．
    #[inline]
    fn is_legal_quick(&self, board: &mut Board, m: Move, has_own_king: bool) -> bool {
        if !has_own_king {
            return true;
        }
        let us = board.turn;
        let captured = board.do_move(m);
        let in_check = board.is_in_check(us);
        board.undo_move(m, captured);
        !in_check
    }

    /// PV パス上の OR ノードで未証明の子ノードを追加証明する．
    ///
    /// Df-Pn は OR ノードで1つの子ノードが証明されると他を未探索のまま残す．
    /// PV 抽出で正確な最短詰み手数を計算するため，PV 上の OR ノードで
    /// 未証明の王手を追加証明する．反復的に PV を更新し収束させる．
    fn complete_or_proofs(&mut self, board: &mut Board) {
        let saved_max = self.max_nodes;
        // 証明完了フェーズに元の max_nodes と同じ追加予算を与えるため，
        // 全体で最大 max_nodes の約2倍のノードを消費する可能性がある．
        self.max_nodes = self.nodes_searched.saturating_add(saved_max);

        // 反復: PV を抽出 → PV 上の OR ノードを完成 → 再抽出
        for _ in 0..5 {
            if self.is_timed_out() {
                break;
            }
            let pv = self.extract_pv(board);
            if pv.is_empty() {
                break;
            }
            let changed = self.complete_pv_or_nodes(board, &pv);
            if !changed {
                break; // 新たに証明された子ノードがなければ収束
            }
        }

        self.max_nodes = saved_max;
    }

    /// PV 上の各 OR ノードで未証明子ノードの追加証明を試みる．
    ///
    /// クローンした盤面で PV を辿り，各 OR ノードで未証明の王手を追加証明する．
    /// 返り値: 新たに証明された子ノードがあれば true．
    fn complete_pv_or_nodes(&mut self, board: &mut Board, pv: &[Move]) -> bool {
        let mut board_clone = board.clone();
        let mut any_changed = false;

        for (i, pv_move) in pv.iter().enumerate() {
            let or_node = i % 2 == 0; // 偶数 ply = 攻め方(OR)
            let ply = i as u32;

            if or_node {
                // OR ノード: 未証明の王手を追加証明
                let moves = self.generate_check_moves(&mut board_clone);
                for m in &moves {
                    if self.nodes_searched >= self.max_nodes {
                        break;
                    }
                    let captured = board_clone.do_move(*m);
                    let (cpn, cdn) = self.look_up_pn_dn(board_clone.hash);

                    if cpn != 0 && cdn != 0 {
                        // 各子ノードに最大10万ノードの追加予算を割り当てる．
                        // saved(= 初期ノード + max_nodes)を上限とし，
                        // 残予算が少ない場合は自然に打ち切られる．
                        let saved = self.max_nodes;
                        self.max_nodes = self
                            .nodes_searched
                            .saturating_add(100_000)
                            .min(saved);
                        self.mid(&mut board_clone, INF - 1, INF - 1, ply + 1, false);
                        self.max_nodes = saved;

                        if self.look_up_pn_dn(board_clone.hash).0 == 0 {
                            any_changed = true;
                        }
                    }

                    board_clone.undo_move(*m, captured);
                }
            }

            // PV に沿って盤面を進める
            let _ = board_clone.do_move(*pv_move);
        }

        any_changed
    }

    /// 詰み手順(PV)を復元する．
    ///
    /// 攻め方(OR): 証明済み子ノードの中で最短手順を選択．
    /// 玉方(AND): 証明済み子ノードの中で最長抵抗を選択．
    fn extract_pv(&self, board: &mut Board) -> Vec<Move> {
        let mut board_clone = board.clone();
        self.extract_pv_recursive(&mut board_clone, true, &mut HashSet::new())
    }

    /// PV 復元の再帰実装．
    ///
    /// 各ノードで全候補手のサブPVを生成し，攻め方は最短，玉方は最長を選ぶ．
    fn extract_pv_recursive(
        &self,
        board: &mut Board,
        or_node: bool,
        visited: &mut HashSet<u64>,
    ) -> Vec<Move> {
        let hash = board.hash;

        // ループ検出
        if visited.contains(&hash) {
            return Vec::new();
        }

        let (node_pn, _node_dn) = self.look_up_pn_dn(hash);

        if or_node {
            // 攻め方: pn == 0 でなければ未証明
            if node_pn != 0 {
                return Vec::new();
            }

            let moves = self.generate_check_moves(board);
            if moves.is_empty() {
                return Vec::new();
            }

            // 全証明済み子ノードのサブPVを比較し，最短を選ぶ
            let mut best_pv: Option<Vec<Move>> = None;

            for m in &moves {
                let captured = board.do_move(*m);
                let (child_pn, _child_dn) = self.look_up_pn_dn(board.hash);

                if child_pn == 0 {
                    visited.insert(hash);
                    let sub_pv = self.extract_pv_recursive(board, false, visited);
                    visited.remove(&hash);

                    let total_len = 1 + sub_pv.len();
                    let is_better = match &best_pv {
                        None => true,
                        Some(prev) => total_len < prev.len(),
                    };

                    if is_better {
                        let mut pv = vec![*m];
                        pv.extend(sub_pv);
                        best_pv = Some(pv);
                    }
                }

                board.undo_move(*m, captured);
            }

            best_pv.unwrap_or_default()
        } else {
            // 玉方: 応手なし = 詰み(PV終端)
            // 無駄合いを除外した応手を使用
            let moves = self.generate_defense_moves(board);
            if moves.is_empty() {
                return Vec::new();
            }

            // 全証明済み子ノードのサブPVを比較し，最長抵抗を選ぶ．
            // 同手数の場合は駒を取る手(captured_piece_raw > 0)を優先する(詰将棋の慣例)．
            let mut best_pv: Option<Vec<Move>> = None;
            let mut best_is_capture = false;

            for m in &moves {
                let captured = board.do_move(*m);
                let (child_pn, _child_dn) = self.look_up_pn_dn(board.hash);

                if child_pn == 0 {
                    visited.insert(hash);
                    let sub_pv = self.extract_pv_recursive(board, true, visited);
                    visited.remove(&hash);

                    let total_len = 1 + sub_pv.len();
                    let is_capture = m.captured_piece_raw() > 0;
                    let is_better = match &best_pv {
                        None => true,
                        Some(prev) => {
                            if total_len > prev.len() {
                                true
                            } else if total_len == prev.len() && is_capture && !best_is_capture {
                                true
                            } else {
                                false
                            }
                        }
                    };

                    if is_better {
                        let mut pv = vec![*m];
                        pv.extend(sub_pv);
                        best_pv = Some(pv);
                        best_is_capture = is_capture;
                    }
                }

                board.undo_move(*m, captured);
            }

            best_pv.unwrap_or_default()
        }
    }

    /// 探索ノード数を返す．
    pub fn nodes_searched(&self) -> u64 {
        self.nodes_searched
    }
}

/// 詰将棋を解く便利関数．
///
/// # 引数
///
/// - `sfen`: 局面のSFEN文字列．
/// - `depth`: 最大探索手数(None でデフォルト 31)．
/// - `nodes`: 最大ノード数(None でデフォルト 1,048,576)．
/// - `draw_ply`: 引き分け手数(None でデフォルト 32767)．
/// - `timeout_secs`: 実行時間制限(秒)(None でデフォルト 300 秒)．
pub fn solve_tsume(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
) -> Result<TsumeResult, crate::board::SfenError> {
    solve_tsume_with_timeout(sfen, depth, nodes, draw_ply, None)
}

/// タイムアウト指定付きで詰将棋を解く便利関数．
pub fn solve_tsume_with_timeout(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
    timeout_secs: Option<u64>,
) -> Result<TsumeResult, crate::board::SfenError> {
    let mut board = Board::empty();
    board.set_sfen(sfen)?;

    let mut solver = DfPnSolver::with_timeout(
        depth.unwrap_or(31),
        nodes.unwrap_or(1_048_576),
        draw_ply.unwrap_or(32767),
        timeout_secs.unwrap_or(300),
    );

    Ok(solver.solve(&mut board))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 詰将棋画像のテストケース: 小阪昇作，9手詰
    ///
    /// 局面: 後手玉1四，先手角2四・3四，後手銀3一・後手香3二
    /// 先手持ち駒: 飛，歩
    /// 後手持ち駒: 飛，金4，銀3，桂4，香3，歩17
    ///
    /// 正解手順: 1三角成，同玉，2三飛打，1二玉，1三歩打，1一玉，2一飛成，同玉，1二歩成
    #[test]
    fn test_tsume_9te_kosaka() {
        let sfen = "6s2/6l2/9/6BBk/9/9/9/9/9 b RPr4g3s4n3l17p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(15, 1_048_576, 32767);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(),
                    9,
                    "should be 9-move checkmate, got: {:?}",
                    usi_moves
                );

                // 正解手順を検証(USI形式)
                assert_eq!(usi_moves[0], "2d1c+", "move 1: 1三角成");
                assert_eq!(usi_moves[1], "1d1c", "move 2: 同玉");
                assert_eq!(usi_moves[2], "R*2c", "move 3: 2三飛打");
                assert_eq!(usi_moves[3], "1c1b", "move 4: 1二玉");
                assert_eq!(usi_moves[4], "P*1c", "move 5: 1三歩打");
                assert_eq!(usi_moves[5], "1b1a", "move 6: 1一玉");
                assert_eq!(usi_moves[6], "2c2a+", "move 7: 2一飛成");
                assert_eq!(usi_moves[7], "1a2a", "move 8: 同玉");
                assert_eq!(usi_moves[8], "1c1b+", "move 9: 1二歩成");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 簡単な1手詰め．
    #[test]
    fn test_tsume_1te() {
        // 後手玉1一，先手金2三，先手持ち駒: 金
        // G*1b(1二金打)で詰み
        let sfen = "8k/9/7G1/9/9/9/9/9/9 b G 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(3, 100_000, 32767);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                assert_eq!(moves.len(), 1);
                assert_eq!(moves[0].to_usi(), "G*1b", "1手詰め: G*1b(1二金打)");
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 3手詰め: 後手玉1一，先手飛3三，先手持ち駒: 金
    ///
    /// 正解: 1三飛成，2一玉，2二金打 まで3手詰
    #[test]
    fn test_tsume_3te() {
        let sfen = "8k/9/6R2/9/9/9/9/9/9 b G 1";
        let result = solve_tsume(sfen, Some(7), Some(1_048_576), None).unwrap();

        let expected = ["3c1c+", "1a2a", "G*2b"];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 3,
                    "expected 3 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 不詰のケース．
    #[test]
    fn test_no_checkmate() {
        // 後手玉5一，先手持ち駒: 歩 → 歩では詰まない
        let sfen = "4k4/9/9/9/9/9/9/9/9 b P 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(5, 100_000, 32767);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::NoCheckmate { .. } => {}
            other => panic!("expected NoCheckmate, got {:?}", other),
        }
    }

    /// solve_tsume 便利関数のテスト．
    #[test]
    fn test_solve_tsume_convenience() {
        let result = solve_tsume(
            "8k/9/7G1/9/9/9/9/9/9 b G 1",
            Some(3),
            Some(100_000),
            None,
        )
        .unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                assert_eq!(moves.len(), 1);
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    #[test]
    fn test_tsume_image2() {
        // 盤面: 5一と，2一王，1一香，2二銀，4三飛，2四角，先手持駒: 金桂
        // 11手詰め: 32金打，同玉，42角成，21玉，31馬，同銀，23飛成，22銀，33桂打，31玉，41と
        let sfen = "4+P2kl/7s1/5R3/7B1/9/9/9/9/9 b GNrb3g3s3n3l17p 1";
        let result = solve_tsume(sfen, Some(31), Some(1_048_576), None).unwrap();

        let expected = [
            "G*3b", "2a3b", "2d4b+", "3b2a", "4b3a", "2b3a",
            "4c2c+", "3a2b", "N*3c", "2a3a", "5a4a",
        ];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(moves.len(), 11, "expected 11 moves, got {}: {:?}", moves.len(), usi_moves);
                assert_eq!(
                    usi_moves,
                    expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// between_bb ヘルパーのテスト．
    #[test]
    fn test_between_bb() {
        // 飛5一(col=4,row=0) と 1一(col=0,row=0) の間: 2一,3一,4一
        let between = attack::between_bb(Square::new(4, 0), Square::new(0, 0));
        assert_eq!(between.count(), 3);
        assert!(between.contains(Square::new(1, 0))); // 2一
        assert!(between.contains(Square::new(2, 0))); // 3一
        assert!(between.contains(Square::new(3, 0))); // 4一

        // 隣接マス(間なし)
        let between2 = attack::between_bb(Square::new(0, 0), Square::new(1, 0));
        assert!(between2.is_empty());

        // 斜め
        let between3 = attack::between_bb(Square::new(0, 0), Square::new(3, 3));
        assert_eq!(between3.count(), 2);
        assert!(between3.contains(Square::new(1, 1)));
        assert!(between3.contains(Square::new(2, 2)));
    }

    /// タイムアウト機能のテスト．
    #[test]
    fn test_timeout() {
        // 不詰の局面を極短タイムアウトで探索 → Unknown が返る
        let sfen = "4k4/9/9/9/9/9/9/9/9 b P 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, u64::MAX, 32767, 0);
        // timeout=0 なので即タイムアウト(ただし最初の1024ノードは走る)
        let result = solver.solve(&mut board);

        // NoCheckmate か Unknown のどちらか(歩1枚では詰まない)
        match &result {
            TsumeResult::NoCheckmate { .. } | TsumeResult::Unknown { .. } => {}
            other => panic!("expected NoCheckmate or Unknown, got {:?}", other),
        }
    }

    /// 詰将棋画像3のテストケース．
    ///
    /// 局面: 後手玉2三，後手桂2一，後手香1一，後手飛5四，後手歩1三，後手と1六
    ///       先手歩1五，先手桂3四，先手金2六，先手香3六
    /// 先手持駒: 飛
    /// 後手持駒: 歩15，香2，桂2，銀4，金3，角2
    #[test]
    fn test_tsume_image3() {
        let sfen = "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/9 b R2b3g4s2n2l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        let expected = [
            "3d2b+", "2c2b", "R*4b", "2b2c", "4b3b+", "2c2d",
            "2f2e", "2d2e", "3b3e",
        ];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 9,
                    "expected 9 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// 2一龍後に2三歩打で詰まないことを検証する．
    ///
    /// 無駄合いフィルタが誤って2三歩打を除外していた問題の回帰テスト．
    /// 2二桂成，同玉，4二飛打，2三玉，3二飛成，2四玉，2一龍 の後，
    /// 後手は2三歩打で合い駒でき，これは無駄合いではない．
    #[test]
    fn test_image3_ryu_2a_not_checkmate() {
        // 2一龍後の局面を作成
        // 初期局面から 3d2b+, 2c2b, R*4b, 2b2c, 4b3b+, 2c2d, 3b2a を実行
        let sfen = "7nl/9/7kp/4r1N2/8P/6LG+p/9/9/9 b R2b3g4s2n2l15p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let moves_usi = ["3d2b+", "2c2b", "R*4b", "2b2c", "4b3b+", "2c2d", "3b2a"];
        for usi in &moves_usi {
            let m = board.move_from_usi(usi).expect(&format!("invalid USI: {}", usi));
            board.do_move(m);
        }

        // ここは AND ノード(後手番)．2三歩打(P*2c)が合法手に含まれることを検証
        let defenses = movegen::generate_legal_moves(&mut board);
        let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();
        // debug: defenses

        // P*2c (2三歩打) が合法手に含まれること
        assert!(
            usi_defenses.contains(&"P*2c".to_string()),
            "P*2c should be a legal defense, got: {:?}", usi_defenses
        );

        // 2三歩打後の局面は詰みではないことを確認
        let p2c = board.move_from_usi("P*2c").unwrap();
        let cap = board.do_move(p2c);

        // 先手の王手手段を確認
        let _checks: Vec<String> = movegen::generate_legal_moves(&mut board)
            .into_iter()
            .filter(|m| {
                let c = board.do_move(*m);
                let gives_check = board.is_in_check(board.turn);
                board.undo_move(*m, c);
                gives_check
            })
            .map(|m| m.to_usi())
            .collect();

        board.undo_move(p2c, cap);
    }

    /// 詰将棋テストケース4．
    ///
    /// 局面: 7nk/9/5R3/8p/6P2/9/9/9/9 b SNPr2b4g3s2n4l15p
    /// 11手詰め(合い効かずにより3二歩打を除外): 2二銀打，1二玉，1三歩打，
    /// 2二玉，3四桂打，1一玉，1二と，同玉，4二飛成，1一玉，2二桂成
    #[test]
    fn test_tsume_4() {
        let sfen = "7nk/9/5R3/8p/6P2/9/9/9/9 b SNPr2b4g3s2n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        let expected = [
            "S*2b", "1a1b", "P*1c", "1b2b", "N*3d", "2b1a",
            "1c1b+", "1a1b", "4c4b+", "1b1a", "3d2b+",
        ];

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 11,
                    "expected 11 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                assert_eq!(
                    usi_moves, expected,
                    "PV mismatch:\n  got:      {:?}\n  expected: {:?}",
                    usi_moves, expected,
                );
            }
            other => panic!("expected Checkmate, got {:?}", other),
        }
    }

    /// generate_check_moves の結果を brute-force と比較する．
    #[test]
    fn test_check_moves_completeness() {
        use std::collections::BTreeSet;
        let test_positions = [
            // 17手詰めの初期局面(OR node)
            "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1",
            // image2
            "4+P2kl/7s1/5R3/7B1/9/9/9/9/9 b GNrb3g3s3n3l17p 1",
            // image3
            "l1k6/9/1pB6/9/9/9/9/9/9 b RGrb4g4s4n3l16p 1",
            // 9te kosaka
            "6s2/6l2/9/6BBk/9/9/9/9/9 b RPr4g3s4n3l17p 1",
        ];

        let solver = DfPnSolver::default_solver();
        for sfen in &test_positions {
            let mut board = Board::new();
            board.set_sfen(sfen).unwrap();

            // Brute-force: 全合法手 → フィルタ
            let brute_checks: BTreeSet<String> = movegen::generate_legal_moves(&mut board)
                .into_iter()
                .filter(|m| {
                    let c = board.do_move(*m);
                    let gives_check = board.is_in_check(board.turn);
                    board.undo_move(*m, c);
                    gives_check
                })
                .map(|m| m.to_usi())
                .collect();

            let optimized_checks: BTreeSet<String> = solver
                .generate_check_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            let missing: BTreeSet<_> = brute_checks.difference(&optimized_checks).collect();
            let extra: BTreeSet<_> = optimized_checks.difference(&brute_checks).collect();

            assert!(
                missing.is_empty() && extra.is_empty(),
                "check moves mismatch for sfen: {}\n  missing: {:?}\n  extra: {:?}\n  brute: {:?}\n  opt: {:?}",
                sfen, missing, extra, brute_checks, optimized_checks
            );
        }
    }

    /// generate_defense_moves と generate_legal_moves の結果を比較する．
    ///
    /// 王手がかかっている局面で，回避手生成(evasion)が
    /// 全合法手のサブセットであり，かつ全合法手を漏れなく含むことを検証する．
    #[test]
    fn test_defense_moves_completeness() {
        use std::collections::BTreeSet;
        // テスト局面: 攻め方が王手した直後の局面をいくつか用意
        let test_positions = [
            // S*4a で王手 → 玉方の応手
            "9/3S1Pk2/9/8R/8B/9/9/9/9 w rb4g2s4n4l17p 2",
            // 飛車で王手(スライディング)
            "9/5Pk2/9/5R3/8B/9/9/9/9 w 2Srb4g2s4n4l17p 2",
            // 角で王手
            "9/5Pk2/9/8R/5B3/9/9/9/9 w 2Srb4g2s4n4l17p 2",
            // test_tsume_image3 の中間局面(R*2a 後)
            "l1k6/R8/1pB6/9/9/9/9/9/9 w rb4g4s4n3l16p 2",
        ];

        let solver = DfPnSolver::default_solver();

        for sfen in &test_positions {
            let mut board = Board::new();
            if board.set_sfen(sfen).is_err() {
                continue;
            }

            // まず王手されているか確認
            if !board.is_in_check(board.turn) {
                continue;
            }

            let defense_moves: BTreeSet<String> = solver
                .generate_defense_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            let legal_moves: BTreeSet<String> = movegen::generate_legal_moves(&mut board)
                .iter()
                .map(|m| m.to_usi())
                .collect();

            // defense_moves は legal_moves のサブセットであること
            let extra: BTreeSet<_> = defense_moves.difference(&legal_moves).collect();
            assert!(
                extra.is_empty(),
                "defense has extra moves not in legal: {:?}\nsfen: {}",
                extra, sfen
            );

            // legal_moves は defense_moves のサブセットであること(漏れがない)
            let missing: BTreeSet<_> = legal_moves.difference(&defense_moves).collect();
            assert!(
                missing.is_empty(),
                "defense is missing legal moves: {:?}\nsfen: {}\ndefense: {:?}\nlegal: {:?}",
                missing, sfen, defense_moves, legal_moves
            );
        }
    }

    /// 17手詰め局面の solve() テスト．
    #[test]
    fn test_17te_solve() {
        let sfen = "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::with_timeout(31, 5_000_000, 32767, 60);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate {
                moves,
                nodes_searched,
            } => {
                let pv: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                eprintln!(
                    "17te: CHECKMATE in {} moves, {} nodes: {}",
                    pv.len(),
                    nodes_searched,
                    pv.join(" ")
                );
            }
            other => panic!("expected Checkmate for 17te, got {:?}", other),
        }
    }

    /// brute-force 詰み判定(DFPN との結果比較用)．
    #[test]
    #[ignore] // 5M ノードを使う重いテスト．明示的に `cargo test -- --ignored` で実行．
    fn test_17te_bruteforce() {
        let sfen = "9/5Pk2/9/8R/8B/9/9/9/9 b 2Srb4g2s4n4l17p 1";
        let mut board = Board::new();
        board.set_sfen(sfen).unwrap();

        let solver = DfPnSolver::default_solver();

        // 深さ制限付き brute-force
        fn is_checkmate(
            board: &mut Board,
            solver: &DfPnSolver,
            depth: u32,
            or_node: bool,
            nodes: &mut u64,
        ) -> bool {
            if depth == 0 {
                return false;
            }
            *nodes += 1;
            if *nodes > 5_000_000 {
                return false; // 打ち切り
            }

            let moves = if or_node {
                solver.generate_check_moves(board)
            } else {
                movegen::generate_legal_moves(board)
            };

            if moves.is_empty() {
                return !or_node; // OR: 王手なし=不詰, AND: 応手なし=詰み
            }

            if or_node {
                // OR: いずれかの子が詰みなら true
                for m in &moves {
                    let captured = board.do_move(*m);
                    let result = is_checkmate(board, solver, depth - 1, false, nodes);
                    board.undo_move(*m, captured);
                    if result {
                        return true;
                    }
                }
                false
            } else {
                // AND: 全ての子が詰みなら true
                for m in &moves {
                    let captured = board.do_move(*m);
                    let result = is_checkmate(board, solver, depth - 1, true, nodes);
                    board.undo_move(*m, captured);
                    if !result {
                        return false;
                    }
                }
                true
            }
        }

        for depth in (1..=21).step_by(2) {
            let mut nodes = 0u64;
            let result = is_checkmate(&mut board, &solver, depth, true, &mut nodes);
            if result {
                break;
            }
        }
    }
}
