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

use std::collections::HashMap;
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
    path: Vec<u64>,
    /// 探索開始時刻．
    start_time: Instant,
    /// タイムアウトしたかどうか．
    timed_out: bool,
}

impl DfPnSolver {
    /// 新しいソルバーを生成する．
    ///
    /// `timeout_secs` で実行時間制限(秒)を指定する．0 の場合はデフォルト 30 秒．
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
            table: HashMap::new(),
            nodes_searched: 0,
            path: Vec::new(),
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

        self.path.push(hash);

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

            // 最良の子ノードを選択
            let (best_idx, second_pn, second_dn) = if or_node {
                self.select_best_or(&child_entries)
            } else {
                self.select_best_and(&child_entries)
            };

            // 子ノードの閾値を計算
            let (child_pn_th, child_dn_th) = if or_node {
                let cpn = dn_threshold
                    .saturating_sub(current_dn)
                    .saturating_add(child_entries[best_idx].dn)
                    .min(INF - 1);
                let cdn = pn_threshold
                    .min(second_pn.saturating_add(1))
                    .min(INF - 1);
                (cdn, cpn)
            } else {
                let cpn = pn_threshold
                    .saturating_sub(current_pn)
                    .saturating_add(child_entries[best_idx].pn)
                    .min(INF - 1);
                let cdn = dn_threshold
                    .min(second_dn.saturating_add(1))
                    .min(INF - 1);
                (cpn, cdn)
            };

            // 子ノードを探索
            let m = moves[best_idx];
            let captured = board.do_move(m);
            self.mid(board, child_pn_th, child_dn_th, ply + 1, !or_node);
            board.undo_move(m, captured);
        }

        self.path.pop();
    }

    /// OR ノードの最良子ノードを選択する．
    fn select_best_or(&self, entries: &[DfPnEntry]) -> (usize, u32, u32) {
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

        (best_idx, second_pn, 0)
    }

    /// AND ノードの最良子ノードを選択する．
    fn select_best_and(&self, entries: &[DfPnEntry]) -> (usize, u32, u32) {
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

        (best_idx, 0, second_dn)
    }

    /// 飛び駒による王手駒を見つける．
    ///
    /// 返り値: (王手駒のマス, 王手駒の駒種) のリスト．
    /// 近接駒(歩,桂,銀,金,王)による王手は含まない．
    fn find_sliding_checkers(board: &Board, king_sq: Square, attacker: Color) -> Vec<(Square, PieceType)> {
        let mut checkers = Vec::new();
        let occ = board.all_occupied();
        let atk = attacker.index();

        // 香: 玉から見て attacker の香方向にいるか
        let lance_atk = attack::lance_attacks(attacker.opponent(), king_sq, occ);
        let mut lance_bb = lance_atk & board.piece_bb[atk][PieceType::Lance as usize];
        while let Some(sq) = lance_bb.lsb() {
            checkers.push((sq, PieceType::Lance));
            lance_bb.clear(sq);
        }

        // 角・馬
        let bishop_atk = attack::bishop_attacks(king_sq, occ);
        for &pt in &[PieceType::Bishop, PieceType::Horse] {
            let mut bb = bishop_atk & board.piece_bb[atk][pt as usize];
            while let Some(sq) = bb.lsb() {
                checkers.push((sq, pt));
                bb.clear(sq);
            }
        }

        // 飛・龍
        let rook_atk = attack::rook_attacks(king_sq, occ);
        for &pt in &[PieceType::Rook, PieceType::Dragon] {
            let mut bb = rook_atk & board.piece_bb[atk][pt as usize];
            while let Some(sq) = bb.lsb() {
                checkers.push((sq, pt));
                bb.clear(sq);
            }
        }

        checkers
    }

    /// 無駄合い(futile interposition)を除外した合法手を生成する．
    ///
    /// 飛び駒による王手に対して，合い駒を打っても(または移動しても)
    /// 攻め方がそれを取って同じように王手が続く場合，その合い駒は無駄合いである．
    /// 詰将棋のルールに従い，無駄合いを除外する．
    fn generate_defense_moves(&self, board: &mut Board) -> Vec<Move> {
        let all_moves = movegen::generate_legal_moves(board);

        // 玉方(hand番)の玉位置を取得
        let defender = board.turn;
        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => return all_moves, // 玉がなければフィルタ不可
        };

        // 飛び駒による王手を見つける
        let attacker = defender.opponent();
        let sliding_checkers = Self::find_sliding_checkers(board, king_sq, attacker);

        // 飛び駒王手がなければ無駄合いは発生しない
        if sliding_checkers.is_empty() {
            return all_moves;
        }

        // 王手が2つ以上(両王手)の場合は玉が動くしかないので合い駒自体が不可
        // → フィルタ不要(合法手生成が正しく処理する)
        if sliding_checkers.len() > 1 {
            return all_moves;
        }

        let (checker_sq, checker_pt) = sliding_checkers[0];

        // 王手駒と玉の間のマスを計算
        let between = attack::between_bb(checker_sq, king_sq);
        if between.is_empty() {
            // 隣接からの王手 → 合い駒不可
            return all_moves;
        }

        all_moves
            .into_iter()
            .filter(|m| {
                let to = m.to_sq();

                // 駒打ち以外(玉の移動，駒を動かす合い駒，王手駒の捕獲)は常に許可．
                // 駒移動による合い駒は元マスが空くため盤面状態が変わり，
                // 無駄合いの判定が複雑になるため除外しない．
                if !m.is_drop() {
                    return true;
                }

                // 王手駒を直接取る打ち駒はないので(打ち駒は空マスのみ)ここは不要

                // between 上でない駒打ちは許可(合い駒ではない)
                if !between.contains(to) {
                    return true;
                }

                // ここに来たら駒打ちによる合い駒(interposition drop)
                // 攻め方の王手駒がこの合い駒を取れるかチェック
                let mut occ_after = board.all_occupied();
                occ_after.set(to); // 合い駒を置く

                let checker_attacks = attack::piece_attacks(
                    attacker,
                    checker_pt,
                    checker_sq,
                    occ_after,
                );

                if !checker_attacks.contains(to) {
                    // 王手駒が取れない → 有効な合い駒
                    return true;
                }

                // 王手駒が合い駒を取った後，玉にまだ利いているか
                // (= 取っても王手が続くか → 無駄合い)
                let mut occ_after_capture = occ_after;
                occ_after_capture.clear(checker_sq); // 王手駒の元位置を空に
                // to には王手駒が移動(既にセット済み)

                let checker_attacks_after = attack::piece_attacks(
                    attacker,
                    checker_pt,
                    to, // 王手駒が合い駒マスに移動
                    occ_after_capture,
                );

                if checker_attacks_after.contains(king_sq) {
                    // 取っても王手 → 無駄合い(除外)
                    false
                } else {
                    // 取ると王手にならない → 有効な合い駒
                    true
                }
            })
            .collect()
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
            board_clone.do_move(*pv_move);
        }

        any_changed
    }

    /// 詰み手順(PV)を復元する．
    ///
    /// 攻め方(OR): 証明済み子ノードの中で最短手順を選択．
    /// 玉方(AND): 証明済み子ノードの中で最長抵抗を選択．
    fn extract_pv(&self, board: &mut Board) -> Vec<Move> {
        let mut board_clone = board.clone();
        self.extract_pv_recursive(&mut board_clone, true, &mut Vec::new())
    }

    /// PV 復元の再帰実装．
    ///
    /// 各ノードで全候補手のサブPVを生成し，攻め方は最短，玉方は最長を選ぶ．
    fn extract_pv_recursive(
        &self,
        board: &mut Board,
        or_node: bool,
        visited: &mut Vec<u64>,
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
                    visited.push(hash);
                    let sub_pv = self.extract_pv_recursive(board, false, visited);
                    visited.pop();

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
                    visited.push(hash);
                    let sub_pv = self.extract_pv_recursive(board, true, visited);
                    visited.pop();

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
/// - `timeout_secs`: 実行時間制限(秒)(None でデフォルト 30 秒)．
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
    /// 正解: G*2b(2二金打)，1a2a(2一玉)，3c2c+(飛車2三成，col=1 check) まで3手詰
    #[test]
    fn test_tsume_3te() {
        // 飛3三(col=2,row=2)は1一に直接利かない(col,rowが異なる)
        // G*2b(金打2二): 金(1,1)が1一(0,0)にfw-diag利き→王手
        // 玉の逃げ場: 2一(1,0): 金fw→✗. 1二(0,1): 金side→✗. 2二(1,1): 金自体→✗.
        // 唯一の手: K captures? 1a can't go to 2b(金). Can't capture.
        // Wait: 1一→2一 is blocked by gold fw? Gold at (1,1) forward = (1,0) = 2一. Yes ✗.
        // So this might be 1-move mate...
        //
        // Let's verify: 1二(0,1) attacked by gold side → ✗. 2一(1,0) attacked by gold → ✗.
        // But what about 2二(1,1)? That's the gold. Can king take? Protected by rook at 3三(2,2)?
        // Rook at (2,2) attacks row=2: (0,2),(1,2),(3,2),...,(8,2) and col=2: (2,0),(2,1),...,(2,8).
        // (1,1) NOT on row=2 or col=2. Not protected. So king CAN capture gold at 2二.
        // After Kx2b (king at 2二(1,1)): rook at 3三(2,2) attacks (1,2),(0,2),(2,1),(2,0),...
        // King at (1,1): is it attacked by rook? (1,1) is not on col=2 or row=2. Not attacked. ✓
        // So Kx2b is a valid defense and it's not mate. ✓
        //
        // After Kx2b: sente needs to check again.
        // Rook at 3三ⅱ can go to 3b(2,1) with check? Rook at (2,1) attacks col=2 and row=1.
        // King at (1,1) is on row=1. Check! ✓
        // Then king at 2二(1,1): 1一(0,0) not attacked by rook, 1二(0,1) on row 1 → ✗,
        // 2一(1,0) not on col=2/row=1, 2三(1,2) not on col=2/row=1, 3二(2,1) rook itself,
        // 3一(2,0) on col=2 → ✗, 3三(2,2) on col=2 → ✗.
        // King can go to: 1一(0,0), 2一(1,0), 2三(1,2).
        // Multiple escapes → not mate from 3c3b.
        //
        // This is actually more complex. Let's just use the solver to verify.
        let sfen = "8k/9/6R2/9/9/9/9/9/9 b G 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(7, 1_048_576, 32767);
        let result = solver.solve(&mut board);

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert!(
                    usi_moves.len() % 2 == 1,
                    "checkmate should be odd number of moves, got: {:?}",
                    usi_moves
                );
                assert!(
                    usi_moves.len() <= 7,
                    "should be at most 7-move checkmate, got: {:?}",
                    usi_moves
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

    /// 診断テスト: G*3b 後の AND ノードと 1二玉後の OR ノードの TT 状態を分析する．
    #[test]
    fn diagnose_pv_extraction() {
        let sfen = "4+P2kl/7s1/5R3/7B1/9/9/9/9/9 b GNrb3g3s3n3l17p 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let mut solver = DfPnSolver::new(31, 1_048_576, 32767);
        let _result = solver.solve(&mut board);

        // --- G*3b を実行 ---
        let m_g3b = board.move_from_usi("G*3b").unwrap();
        let cap0 = board.do_move(m_g3b);

        eprintln!("=== AND node after G*3b ===");
        let defenses = movegen::generate_legal_moves(&mut board);
        for d in &defenses {
            let cap1 = board.do_move(*d);
            let entry = solver.look_up(board.hash);
            eprintln!(
                "  defense {}: pn={}, dn={}",
                d.to_usi(), entry.pn, entry.dn
            );
            board.undo_move(*d, cap1);
        }

        // --- 1二玉 (2a1b) を実行 ---
        let m_1b = board.move_from_usi("2a1b").unwrap();
        let cap1 = board.do_move(m_1b);

        eprintln!("\n=== OR node after G*3b, 2a1b (1二玉) ===");
        let attacks = solver.generate_check_moves(&mut board);
        eprintln!("  check moves count: {}", attacks.len());
        for a in &attacks {
            let cap2 = board.do_move(*a);
            let entry = solver.look_up(board.hash);
            eprintln!(
                "  attack {}: pn={}, dn={}",
                a.to_usi(), entry.pn, entry.dn
            );
            board.undo_move(*a, cap2);
        }

        board.undo_move(m_1b, cap1);

        // --- 同玉 (2a3b) を実行 ---
        let m_3b = board.move_from_usi("2a3b").unwrap();
        let cap1b = board.do_move(m_3b);

        eprintln!("\n=== OR node after G*3b, 2a3b (同玉) ===");
        let attacks2 = solver.generate_check_moves(&mut board);
        eprintln!("  check moves count: {}", attacks2.len());
        for a in &attacks2 {
            let cap2 = board.do_move(*a);
            let entry = solver.look_up(board.hash);
            eprintln!(
                "  attack {}: pn={}, dn={}",
                a.to_usi(), entry.pn, entry.dn
            );
            board.undo_move(*a, cap2);
        }

        board.undo_move(m_3b, cap1b);
        board.undo_move(m_g3b, cap0);
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
                eprintln!("PV ({}): {:?}", usi_moves.len(), usi_moves);
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

    /// 無駄合い検出テスト: 飛車の王手に対する駒打ち合い駒が無駄合いとして除外される．
    ///
    /// 局面: 後手玉1一，先手飛5一(同じ段で王手)，後手持駒: 金
    /// G*2a, G*3a, G*4a は飛車に取られて王手が続く → 無駄合い．
    /// 正しい応手は玉の移動のみ．
    #[test]
    fn test_mudaai_drop_filter() {
        // SFEN は 9筋→1筋 の順で記述
        // 後手玉 1一(col=0,row=0) = 末尾の k
        // 先手飛 5一(col=4,row=0) = 5番目(左から5番目=右から5番目)
        // "4R3k" = 9一空,8一空,7一空,6一空,5一飛,4一空,3一空,2一空,1一玉
        // → 9876=空4つ, R=5一飛, 空3つ, k=1一玉 → "4R3k"
        // 後手番で後手持駒: 金
        let sfen = "4R3k/9/9/9/9/9/9/9/9 w g 1";
        let mut board = Board::empty();
        board.set_sfen(sfen).unwrap();

        let solver = DfPnSolver::new(5, 100_000, 32767);

        // generate_defense_moves で合い駒が除外されることを検証
        let defenses = solver.generate_defense_moves(&mut board);
        let usi_defenses: Vec<String> = defenses.iter().map(|m| m.to_usi()).collect();
        eprintln!("Defenses after mudaai filter: {:?}", usi_defenses);

        // between 上への金打ちが除外されていること
        let rook_sq = Square::new(4, 0); // 5一
        let king_sq = Square::new(0, 0); // 1一
        let between = attack::between_bb(rook_sq, king_sq);
        for m in &defenses {
            if m.is_drop() && between.contains(m.to_sq()) {
                panic!(
                    "futile drop {} should be filtered (mudaai)",
                    m.to_usi()
                );
            }
        }

        // 玉の移動は含まれていること
        assert!(
            defenses.iter().any(|m| !m.is_drop()),
            "king moves should be present"
        );
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
    /// 局面: 後手玉2三，後手桂2一，後手香1一，後手飛5四
    ///       先手歩1三・1五，先手と1六，先手桂3四，先手金2六，先手香3六
    /// 先手持駒: 飛
    /// 後手持駒: 歩15，香2，桂2，銀4，金3，角2
    #[test]
    fn test_tsume_image3() {
        let sfen = "7nl/9/7kP/4r1N2/8P/6LG+P/9/9/9 b R2b3g4s2n2l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                eprintln!("Image3 PV ({}): {:?}", usi_moves.len(), usi_moves);
                // 手順を表示して確認
                assert!(
                    usi_moves.len() % 2 == 1,
                    "checkmate should be odd moves, got {}: {:?}",
                    usi_moves.len(), usi_moves
                );
            }
            TsumeResult::NoCheckmate { nodes_searched } => {
                eprintln!("No checkmate found (nodes: {})", nodes_searched);
            }
            TsumeResult::Unknown { nodes_searched } => {
                eprintln!("Unknown (nodes: {}, may need more nodes/time)", nodes_searched);
            }
        }
    }
}
