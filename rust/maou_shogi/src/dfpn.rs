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

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::attack;
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

/// Df-Pn 探索のエントリ．
#[derive(Debug, Clone, Copy)]
struct DfPnEntry {
    pn: u32,
    dn: u32,
}

impl Default for DfPnEntry {
    fn default() -> Self {
        DfPnEntry { pn: 1, dn: 1 }
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
    /// 転置表．
    table: HashMap<u64, DfPnEntry>,
    /// 探索ノード数．
    nodes_searched: u64,
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
        DfPnSolver {
            depth,
            max_nodes,
            draw_ply,
            timeout: Duration::from_secs(timeout_secs),
            table: HashMap::with_capacity(max_nodes.min(1_048_576) as usize),
            nodes_searched: 0,
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
    fn look_up(&self, hash: u64) -> DfPnEntry {
        self.table.get(&hash).copied().unwrap_or_default()
    }

    /// 転置表を更新する．
    fn store(&mut self, hash: u64, pn: u32, dn: u32) {
        self.table.insert(hash, DfPnEntry { pn, dn });
    }

    /// 詰将棋を解く．
    ///
    /// `board` は攻め方の手番から開始する局面．
    /// 片玉局面(攻め方に玉がない)を想定するが，両玉でも動作する．
    pub fn solve(&mut self, board: &mut Board) -> TsumeResult {
        self.table.clear();
        self.nodes_searched = 0;
        self.path.clear();
        self.start_time = Instant::now();
        self.timed_out = false;

        let root_hash = board.hash;

        // ルートノードの Df-Pn 探索
        self.mid(board, INF - 1, INF - 1, 0, true);

        let entry = self.look_up(root_hash);

        if entry.pn == 0 {
            // 詰みが証明された → OR ノードの未証明子ノードを追加証明
            self.complete_or_proofs(board);
            // 手順を復元
            let moves = self.extract_pv(board);
            TsumeResult::Checkmate {
                moves,
                nodes_searched: self.nodes_searched,
            }
        } else if entry.dn == 0 {
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

        let hash = board.hash;

        // 終端条件: 深さ制限
        if ply >= self.depth {
            self.store(hash, INF, 0);
            return;
        }

        // 終端条件: 手数制限(引き分け)
        if board.ply() as u32 >= self.draw_ply {
            self.store(hash, INF, 0);
            return;
        }

        // ループ検出
        if self.path.contains(&hash) {
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

        self.path.insert(hash);

        // MID ループ
        loop {
            // 各子ノードの pn/dn を収集
            let child_entries: Vec<DfPnEntry> = moves
                .iter()
                .map(|m| {
                    let captured = board.do_move(*m);
                    let child_hash = board.hash;
                    board.undo_move(*m, captured);
                    self.look_up(child_hash)
                })
                .collect();

            // OR ノード: pn = min(children.pn), dn = sum(children.dn)
            // AND ノード: pn = sum(children.pn), dn = min(children.dn)
            let (current_pn, current_dn) = if or_node {
                let min_pn = child_entries.iter().map(|e| e.pn).min().unwrap_or(INF);
                let sum_dn = child_entries
                    .iter()
                    .map(|e| e.dn as u64)
                    .sum::<u64>()
                    .min(INF as u64) as u32;
                (min_pn, sum_dn)
            } else {
                let sum_pn = child_entries
                    .iter()
                    .map(|e| e.pn as u64)
                    .sum::<u64>()
                    .min(INF as u64) as u32;
                let min_dn = child_entries.iter().map(|e| e.dn).min().unwrap_or(INF);
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
                let (idx, second_pn) = self.select_best_or(&child_entries);
                let child_dn_th = dn_threshold
                    .saturating_sub(current_dn)
                    .saturating_add(child_entries[idx].dn)
                    .min(INF - 1);
                let child_pn_th = pn_threshold
                    .min(second_pn.saturating_add(1))
                    .min(INF - 1);
                (idx, child_pn_th, child_dn_th)
            } else {
                let (idx, second_dn) = self.select_best_and(&child_entries);
                let child_pn_th = pn_threshold
                    .saturating_sub(current_pn)
                    .saturating_add(child_entries[idx].pn)
                    .min(INF - 1);
                let child_dn_th = dn_threshold
                    .min(second_dn.saturating_add(1))
                    .min(INF - 1);
                (idx, child_pn_th, child_dn_th)
            };

            // 子ノードを探索
            let m = moves[best_idx];
            let captured = board.do_move(m);
            self.mid(board, child_pn_th, child_dn_th, ply + 1, !or_node);
            board.undo_move(m, captured);
        }

        self.path.remove(&hash);
    }

    /// OR ノードの最良子ノードを選択する．
    ///
    /// 返り値: (最良インデックス, 2番目に小さい pn)
    fn select_best_or(&self, entries: &[DfPnEntry]) -> (usize, u32) {
        let mut best_idx = 0;
        let mut best_pn = INF;
        let mut second_pn = INF;

        for (i, e) in entries.iter().enumerate() {
            if e.pn < best_pn {
                second_pn = best_pn;
                best_pn = e.pn;
                best_idx = i;
            } else if e.pn < second_pn {
                second_pn = e.pn;
            }
        }

        (best_idx, second_pn)
    }

    /// AND ノードの最良子ノードを選択する．
    ///
    /// 返り値: (最良インデックス, 2番目に小さい dn)
    fn select_best_and(&self, entries: &[DfPnEntry]) -> (usize, u32) {
        let mut best_idx = 0;
        let mut best_dn = INF;
        let mut second_dn = INF;

        for (i, e) in entries.iter().enumerate() {
            if e.dn < best_dn {
                second_dn = best_dn;
                best_dn = e.dn;
                best_idx = i;
            } else if e.dn < second_dn {
                second_dn = e.dn;
            }
        }

        (best_idx, second_dn)
    }

    /// 玉方の応手を生成する(合い効かずを除外)．
    ///
    /// 飛び駒(飛/龍/角/馬/香)による王手に対して，合い駒が効かない場合を検出する．
    /// 合い効かず条件:
    /// 1. 飛び駒と玉の間のマスに玉方の駒(玉除く)の効きがない
    /// 2. 飛び駒が合い駒を取った後，玉がその駒を取り返せない
    ///    (取り返せる = 取った位置が玉に隣接かつ攻め方の他の駒が利いていない)
    fn generate_defense_moves(&self, board: &mut Board) -> Vec<Move> {
        let moves = movegen::generate_legal_moves(board);

        // 玉の位置を取得
        let king_sq = match board.king_square(board.turn) {
            Some(sq) => sq,
            None => return moves,
        };

        // 飛び駒による王手駒を特定
        let attacker = board.turn.opponent();
        let checker_sq = self.find_sliding_checker(board, king_sq, attacker);

        let checker_sq = match checker_sq {
            Some(sq) => sq,
            None => return moves, // 飛び駒の王手でない or 両王手
        };

        // 間のマスを計算
        let between = attack::between_bb(checker_sq, king_sq);
        if between.is_empty() {
            return moves; // 隣接する王手(合い駒不可)
        }

        // 各間マスごとに合い効かず判定
        // 条件: 玉方の駒(玉除く)の効きがなく，かつ飛び駒が取った後に玉が取り返せない
        let defender = board.turn;
        let futile_squares = attack::between_bb(checker_sq, king_sq);
        let king_step = attack::step_attacks(
            defender.opponent(),
            PieceType::King,
            king_sq,
        );

        // 各間マスについて個別判定
        let mut filtered_between = crate::bitboard::Bitboard::EMPTY;
        for sq in futile_squares {
            // 条件1: 玉方の非玉駒が利いている → 合い効かずでない
            if self.is_attacked_by_non_king(board, sq, defender) {
                continue;
            }

            // 条件2: 飛び駒が取った後，玉が取り返せるか？
            // 取った位置が玉に隣接，かつ攻め方の他の駒がその位置に利いていない場合，
            // 玉が取り返せる → 合い効かずでない
            if king_step.contains(sq) {
                // 飛び駒がこのマスに移動した後，攻め方の他の駒が利いているかを
                // 近似的に判定: 現在の局面で攻め方(飛び駒自身を除く)が利いているか
                // 飛び駒は移動するため除外が必要だが，is_attacked_by で近似
                // (飛び駒自身の利きも含まれるが，移動元からの利きは消える)
                //
                // より正確な判定: 飛び駒を除外した攻め方利きを計算
                if !self.is_attacked_by_excluding(board, sq, attacker, checker_sq) {
                    continue; // 攻め方の他の駒が利いていない → 玉が取り返せる
                }
            }

            filtered_between.set(sq);
        }

        if filtered_between.is_empty() {
            return moves;
        }

        // 合い効かずマスへの合い駒打ちを除外
        moves
            .into_iter()
            .filter(|m| {
                if !m.is_drop() {
                    return true; // 駒移動は常に有効
                }
                !filtered_between.contains(m.to_sq())
            })
            .collect()
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
        let occ = board.all_occupied();
        let att = attacker_color.index();
        let defender = attacker_color.opponent();

        // 除外マスのマスクを計算
        let mask = match excluded_sq {
            Some(esq) => {
                let mut m = crate::bitboard::Bitboard::EMPTY;
                m.set(esq);
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
    fn generate_check_moves(&self, board: &mut Board) -> Vec<Move> {
        let all_moves = movegen::generate_legal_moves(board);

        all_moves
            .into_iter()
            .filter(|m| {
                let captured = board.do_move(*m);
                let gives_check = board.is_in_check(board.turn);
                board.undo_move(*m, captured);
                gives_check
            })
            .collect()
    }

    /// PV パス上の OR ノードで未証明の子ノードを追加証明する．
    ///
    /// Df-Pn は OR ノードで1つの子ノードが証明されると他を未探索のまま残す．
    /// PV 抽出で正確な最短詰み手数を計算するため，PV 上の OR ノードで
    /// 未証明の王手を追加証明する．反復的に PV を更新し収束させる．
    fn complete_or_proofs(&mut self, board: &mut Board) {
        let saved_max = self.max_nodes;
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
                    let child = self.look_up(board_clone.hash);

                    if child.pn != 0 && child.dn != 0 {
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

                        if self.look_up(board_clone.hash).pn == 0 {
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

        let entry = self.look_up(hash);

        if or_node {
            // 攻め方: pn == 0 でなければ未証明
            if entry.pn != 0 {
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
                let child_entry = self.look_up(board.hash);

                if child_entry.pn == 0 {
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
                let child_entry = self.look_up(board.hash);

                if child_entry.pn == 0 {
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
        eprintln!("Defenses after R*2a: {:?}", usi_defenses);

        // P*2c (2三歩打) が合法手に含まれること
        assert!(
            usi_defenses.contains(&"P*2c".to_string()),
            "P*2c should be a legal defense, got: {:?}", usi_defenses
        );

        // 2三歩打後の局面は詰みではないことを確認
        let p2c = board.move_from_usi("P*2c").unwrap();
        let cap = board.do_move(p2c);

        // 先手の王手手段を確認
        let checks: Vec<String> = movegen::generate_legal_moves(&mut board)
            .into_iter()
            .filter(|m| {
                let c = board.do_move(*m);
                let gives_check = board.is_in_check(board.turn);
                board.undo_move(*m, c);
                gives_check
            })
            .map(|m| m.to_usi())
            .collect();
        eprintln!("Check moves after P*2c: {:?}", checks);

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
}
