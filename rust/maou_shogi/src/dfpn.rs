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

use arrayvec::ArrayVec;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use std::time::{Duration, Instant};

use crate::attack;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::movegen;
use crate::moves::Move;
use crate::types::{Color, PieceType, Square};
use crate::zobrist::ZOBRIST;

/// 王手手/応手の最大数．
/// 将棋の合法手上限は593だが，詰将棋では王手・応手ともに
/// これより十分少ないため 320 で十分．
const MAX_MOVES: usize = 320;

/// 証明数・反証数の無限大を表す定数．
const INF: u32 = u32::MAX;

/// 持ち駒の種類数(歩・香・桂・銀・金・角・飛)．
const HAND_KINDS_TT: usize = 7;

/// 反証駒の最大値(任意の持ち駒で不詰の場合に使用)．
/// 各駒種の最大枚数: 歩18，香4，桂4，銀4，金4，角2，飛2．
const MAX_DISPROOF: [u8; HAND_KINDS_TT] =
    [18, 4, 4, 4, 4, 2, 2];

/// 持ち駒の要素ごと比較: a の全要素が b 以上なら true．
///
/// 証明駒の優越判定に使用: 持ち駒が多い方が有利(攻め方)．
#[inline(always)]
fn hand_gte(a: &[u8; HAND_KINDS_TT], b: &[u8; HAND_KINDS_TT]) -> bool {
    a[0] >= b[0]
        && a[1] >= b[1]
        && a[2] >= b[2]
        && a[3] >= b[3]
        && a[4] >= b[4]
        && a[5] >= b[5]
        && a[6] >= b[6]
}

/// 盤面のみのハッシュ(持ち駒を除外)を計算する．
///
/// board.hash から持ち駒の Zobrist ハッシュ成分を XOR で除去する．
/// 証明駒/反証駒による TT 参照で，同一盤面・異なる持ち駒の
/// エントリを同一スロットに集約するために使用する．
#[inline(always)]
fn position_key(board: &Board) -> u64 {
    let mut h = board.hash;
    // Black hand
    let bh = &board.hand[0];
    if bh[0] > 0 { h ^= ZOBRIST.hand_hash(Color::Black, 0, bh[0] as usize); }
    if bh[1] > 0 { h ^= ZOBRIST.hand_hash(Color::Black, 1, bh[1] as usize); }
    if bh[2] > 0 { h ^= ZOBRIST.hand_hash(Color::Black, 2, bh[2] as usize); }
    if bh[3] > 0 { h ^= ZOBRIST.hand_hash(Color::Black, 3, bh[3] as usize); }
    if bh[4] > 0 { h ^= ZOBRIST.hand_hash(Color::Black, 4, bh[4] as usize); }
    if bh[5] > 0 { h ^= ZOBRIST.hand_hash(Color::Black, 5, bh[5] as usize); }
    if bh[6] > 0 { h ^= ZOBRIST.hand_hash(Color::Black, 6, bh[6] as usize); }
    // White hand
    let wh = &board.hand[1];
    if wh[0] > 0 { h ^= ZOBRIST.hand_hash(Color::White, 0, wh[0] as usize); }
    if wh[1] > 0 { h ^= ZOBRIST.hand_hash(Color::White, 1, wh[1] as usize); }
    if wh[2] > 0 { h ^= ZOBRIST.hand_hash(Color::White, 2, wh[2] as usize); }
    if wh[3] > 0 { h ^= ZOBRIST.hand_hash(Color::White, 3, wh[3] as usize); }
    if wh[4] > 0 { h ^= ZOBRIST.hand_hash(Color::White, 4, wh[4] as usize); }
    if wh[5] > 0 { h ^= ZOBRIST.hand_hash(Color::White, 5, wh[5] as usize); }
    if wh[6] > 0 { h ^= ZOBRIST.hand_hash(Color::White, 6, wh[6] as usize); }
    h
}

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

/// 転置表のエントリ(証明駒/反証駒対応)．
///
/// - hand: 登録時の攻め方の持ち駒(証明駒/反証駒として使用)
/// - pn, dn: 証明数・反証数
#[derive(Debug, Clone, Copy)]
struct DfPnEntry {
    hand: [u8; HAND_KINDS_TT],
    pn: u32,
    dn: u32,
}

/// HashMap ベースの転置表(証明駒/反証駒対応)．
///
/// キーは盤面のみのハッシュ(持ち駒除外)を使用し，
/// 同一盤面・異なる持ち駒のエントリを同一クラスタに格納する．
/// cshogi と同様のアプローチで，証明駒/反証駒を正確に保持する．
///
/// 参照時に持ち駒の優越関係を利用して TT ヒット率を向上させる:
/// - 証明済み(pn=0): 現在の持ち駒 >= 登録時の持ち駒 → 再利用
/// - 反証済み(dn=0): 登録時の持ち駒 >= 現在の持ち駒 → 再利用
struct TranspositionTable {
    tt: FxHashMap<u64, Vec<DfPnEntry>>,
}

impl TranspositionTable {
    /// 転置表を生成する．
    fn new(_min_size: usize) -> Self {
        TranspositionTable {
            tt: FxHashMap::with_capacity_and_hasher(
                65536,
                Default::default(),
            ),
        }
    }

    /// 転置表を参照する(証明駒/反証駒の優越関係を利用)．
    ///
    /// 1. 証明済みエントリ: 現在の持ち駒 >= 登録時 → (0, dn) を返す
    /// 2. 反証済みエントリ: 登録時の持ち駒 >= 現在 → (pn, 0) を返す
    /// 3. 持ち駒完全一致: そのまま返す
    /// 4. 該当なし: (1, 1) を返す
    #[inline(always)]
    fn look_up(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS_TT],
    ) -> (u32, u32) {
        let entries = match self.tt.get(&pos_key) {
            Some(e) => e,
            None => return (1, 1),
        };

        let mut exact_match: Option<(u32, u32)> = None;

        for e in entries {
            // 証明済み: 持ち駒が多い(以上)なら再利用
            if e.pn == 0 && hand_gte(hand, &e.hand) {
                return (0, e.dn);
            }
            // 反証済み: 持ち駒が少ない(以下)なら再利用
            if e.dn == 0 && hand_gte(&e.hand, hand) {
                return (e.pn, 0);
            }
            // 完全一致
            if exact_match.is_none() && e.hand == *hand {
                exact_match = Some((e.pn, e.dn));
            }
        }

        exact_match.unwrap_or((1, 1))
    }

    /// 転置表を更新する．
    #[inline(always)]
    fn store(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS_TT],
        pn: u32,
        dn: u32,
    ) {
        let entries =
            self.tt.entry(pos_key).or_insert_with(Vec::new);

        // 同一持ち駒の既存エントリを更新
        for e in entries.iter_mut() {
            if e.hand == hand {
                e.pn = pn;
                e.dn = dn;
                return;
            }
        }

        // 新規エントリを追加
        entries.push(DfPnEntry { hand, pn, dn });
    }

    /// 証明済みエントリの証明駒(登録時の持ち駒)を返す．
    ///
    /// 持ち駒優越で一致する証明済みエントリの hand を返す．
    /// 見つからない場合は渡された hand をそのまま返す．
    #[inline(always)]
    fn get_proof_hand(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS_TT],
    ) -> [u8; HAND_KINDS_TT] {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.pn == 0 && hand_gte(hand, &e.hand) {
                    return e.hand;
                }
            }
        }
        *hand
    }

    /// 反証済みエントリの反証駒(登録時の持ち駒)を返す．
    ///
    /// 持ち駒劣越で一致する反証済みエントリの hand を返す．
    /// 見つからない場合は渡された hand をそのまま返す．
    #[inline(always)]
    fn get_disproof_hand(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS_TT],
    ) -> [u8; HAND_KINDS_TT] {
        if let Some(entries) = self.tt.get(&pos_key) {
            for e in entries {
                if e.dn == 0 && hand_gte(&e.hand, hand) {
                    return e.hand;
                }
            }
        }
        *hand
    }

    /// 全エントリをクリアする．
    fn clear(&mut self) {
        self.tt.clear();
    }
}

/// OR ノードの証明駒を子ノードの証明駒から計算する．
///
/// - 駒打ち: 子の証明駒 + 打った駒種1枚
/// - 駒取り: 子の証明駒 - 取った駒種1枚(0 以下にならない)
/// - それ以外: 子の証明駒をそのまま使用
#[inline]
fn adjust_or_proof(
    m: Move,
    child_proof: &[u8; HAND_KINDS_TT],
) -> [u8; HAND_KINDS_TT] {
    let mut ph = *child_proof;
    if m.is_drop() {
        if let Some(pt) = m.drop_piece_type() {
            if let Some(hi) = pt.hand_index() {
                ph[hi] = ph[hi].saturating_add(1);
            }
        }
    } else {
        let cap = m.captured_piece_raw();
        if cap > 0 {
            // captured_piece_raw() は Piece 値(色付き)を返す．
            // 白駒はオフセット 16 が加算されている．
            use crate::types::Piece;
            let piece = Piece::from_raw_u8(cap);
            if let Some(pt) = piece.piece_type() {
                let base_pt =
                    pt.unpromoted().unwrap_or(pt);
                if let Some(hi) = base_pt.hand_index() {
                    ph[hi] = ph[hi].saturating_sub(1);
                }
            }
        }
    }
    ph
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
    /// 転置表(固定サイズ，証明駒/反証駒対応)．
    table: TranspositionTable,
    /// 探索ノード数．
    nodes_searched: u64,
    /// 探索中の最大ply(デバッグ用)．
    max_ply: u32,
    /// 探索中のパス(ループ検出用，フルハッシュ)．
    path: FxHashSet<u64>,
    /// 探索開始時刻．
    start_time: Instant,
    /// タイムアウトしたかどうか．
    timed_out: bool,
    /// 攻め方の手番色(solve 時に設定)．
    attacker: Color,
}

impl DfPnSolver {
    /// 新しいソルバーを生成する(タイムアウト 300 秒)．
    pub fn new(depth: u32, max_nodes: u64, draw_ply: u32) -> Self {
        Self::with_timeout(depth, max_nodes, draw_ply, 300)
    }

    /// タイムアウト指定付きでソルバーを生成する．
    pub fn with_timeout(depth: u32, max_nodes: u64, draw_ply: u32, timeout_secs: u64) -> Self {
        let tt_size = 0; // HashMap ベースのため未使用
        DfPnSolver {
            depth,
            max_nodes,
            draw_ply,
            timeout: Duration::from_secs(timeout_secs),
            table: TranspositionTable::new(tt_size),
            nodes_searched: 0,
            max_ply: 0,
            path: FxHashSet::default(),
            start_time: Instant::now(),
            timed_out: false,
            attacker: Color::Black,
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

    /// 転置表を参照する(位置キー＋持ち駒指定)．
    #[inline]
    fn look_up_pn_dn(
        &self,
        pos_key: u64,
        hand: &[u8; HAND_KINDS_TT],
    ) -> (u32, u32) {
        self.table.look_up(pos_key, hand)
    }

    /// 転置表を更新する(位置キー＋持ち駒指定)．
    #[inline]
    fn store(
        &mut self,
        pos_key: u64,
        hand: [u8; HAND_KINDS_TT],
        pn: u32,
        dn: u32,
    ) {
        self.table.store(pos_key, hand, pn, dn);
    }

    /// 転置表を参照する(盤面から自動計算)．
    #[inline]
    fn look_up_board(&self, board: &Board) -> (u32, u32) {
        let pk = position_key(board);
        let hand = &board.hand[self.attacker.index()];
        self.table.look_up(pk, hand)
    }

    /// 転置表を更新する(盤面から自動計算)．
    #[inline]
    fn store_board(
        &mut self,
        board: &Board,
        pn: u32,
        dn: u32,
    ) {
        let pk = position_key(board);
        let hand = board.hand[self.attacker.index()];
        self.table.store(pk, hand, pn, dn);
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
        self.attacker = board.turn;

        // 単一パス Df-Pn: 反復深化なしで全深さを一度に探索．
        // 証明数・反証数が自然に探索を最も有望な手順に誘導する．
        self.mid(board, INF - 1, INF - 1, 0, true);

        let (root_pn, root_dn) = self.look_up_board(board);

        if root_pn == 0 {
            // PV 抽出を先に試行し，失敗時のみ追加証明を実行
            let moves = self.extract_pv(board);
            let moves = if moves.is_empty() {
                self.complete_or_proofs(board);
                self.extract_pv(board)
            } else {
                moves
            };
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

    /// OR/AND ノードの Df-Pn 探索(証明駒/反証駒対応)．
    ///
    /// `or_node` が true のとき OR ノード(攻め方)，false のとき AND ノード(玉方)．
    /// TT のキーには盤面のみのハッシュ(持ち駒除外)を使用し，
    /// 持ち駒の優越関係により TT ヒット率を向上させる．
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
        // 1024 ノードごとにタイマーをチェック
        if self.nodes_searched & 0x3FF == 0 && self.is_timed_out() {
            self.timed_out = true;
            return;
        }
        self.nodes_searched += 1;
        if ply > self.max_ply {
            self.max_ply = ply;
        }

        let full_hash = board.hash;
        let pos_key = position_key(board);
        let att_hand = board.hand[self.attacker.index()];

        // ループ検出: フルハッシュで判定(持ち駒込みの完全一致)
        if self.path.contains(&full_hash) {
            return;
        }

        // TT 参照: 既に閾値を超えている/証明済み/反証済みなら
        // 手生成をスキップして早期 return
        let (tt_pn, tt_dn) =
            self.look_up_pn_dn(pos_key, &att_hand);
        if tt_pn == 0 || tt_dn == 0 {
            return;
        }
        if tt_pn >= pn_threshold || tt_dn >= dn_threshold {
            return;
        }

        // 終端条件: 深さ制限・手数制限
        if ply >= self.depth || board.ply() as u32 >= self.draw_ply {
            // 反証駒 = MAX(任意の持ち駒で不詰)
            self.store(pos_key, MAX_DISPROOF, INF, 0);
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
                // 王手手段なし → 不詰(反証駒 = MAX)
                self.store(pos_key, MAX_DISPROOF, INF, 0);
            } else {
                // 応手なし → 詰み(証明駒 = 空)
                self.store(
                    pos_key,
                    [0; HAND_KINDS_TT],
                    0,
                    INF,
                );
            }
            return;
        }

        // 子ノード情報を事前計算:
        // (Move, full_hash, pos_key, attacker_hand)
        let mut children: ArrayVec<
            (Move, u64, u64, [u8; HAND_KINDS_TT]),
            MAX_MOVES,
        > = ArrayVec::new();
        // OR ノードの反証駒(init ループ中に蓄積)
        let mut init_or_disproof = MAX_DISPROOF;
        for m in &moves {
            let captured = board.do_move(*m);
            let child_full_hash = board.hash;
            let child_pk = position_key(board);
            let child_hand = board.hand[self.attacker.index()];

            let (cpn, cdn) =
                self.look_up_pn_dn(child_pk, &child_hand);
            if cpn == 1 && cdn == 1 {
                if or_node {
                    // 1手詰め判定: has_any_defense で高速判定
                    if !self.has_any_defense(board) {
                        // 応手なし → 詰み(証明駒 = 空)
                        self.store(
                            child_pk,
                            [0; HAND_KINDS_TT],
                            0,
                            INF,
                        );
                    } else if ply + 2 < self.depth {
                        // 3手詰めチェック: 応手を生成して確認
                        let defenses =
                            self.generate_defense_moves(board);
                        if defenses.len() <= 4 {
                            let mut all_mated = true;
                            for d in &defenses {
                                let cap_d = board.do_move(*d);
                                let mate =
                                    self.has_mate_in_1(board);
                                if mate {
                                    self.store_board(
                                        board, 0, INF,
                                    );
                                }
                                board.undo_move(*d, cap_d);
                                if !mate {
                                    all_mated = false;
                                    break;
                                }
                            }
                            if all_mated {
                                self.store(
                                    child_pk, child_hand, 0,
                                    INF,
                                );
                            } else {
                                let n = defenses.len() as u32;
                                self.store(
                                    child_pk, child_hand, n, n,
                                );
                            }
                        } else {
                            let n = defenses.len() as u32;
                            self.store(
                                child_pk, child_hand, n, n,
                            );
                        }
                    } else {
                        // 深い位置では応手数を初期値として設定
                        // 正確な数は不要なのでデフォルト値を使用
                        self.store(
                            child_pk, child_hand, 1, 1,
                        );
                    }
                } else {
                    let checks =
                        self.generate_check_moves(board);
                    if checks.is_empty() {
                        // 攻め方に王手がない → 反証駒 = MAX
                        self.store(
                            child_pk, MAX_DISPROOF, INF, 0,
                        );
                    } else if ply + 2 < self.depth
                        && self.has_mate_in_1_with(
                            board, &checks,
                        )
                    {
                        self.store(child_pk, child_hand, 0, INF);
                    } else {
                        self.store(
                            child_pk,
                            child_hand,
                            1,
                            checks.len() as u32,
                        );
                    }
                }
            }

            board.undo_move(*m, captured);

            // 即座に解決チェック(子ノード初期化時に証明/反証を検出)
            let (cpn_now, cdn_now) =
                self.look_up_pn_dn(child_pk, &child_hand);
            if or_node && cpn_now == 0 {
                // OR 証明: 子の証明駒から親の証明駒を計算
                let child_ph = self
                    .table
                    .get_proof_hand(child_pk, &child_hand);
                let mut proof =
                    adjust_or_proof(*m, &child_ph);
                // 証明駒を現在の持ち駒で上限クリップ
                for k in 0..HAND_KINDS_TT {
                    proof[k] = proof[k].min(att_hand[k]);
                }
                self.store(pos_key, proof, 0, INF);
                return;
            }
            if !or_node && cdn_now == 0 {
                // AND 反証: 子の反証駒を親に伝播
                let child_dp = self
                    .table
                    .get_disproof_hand(child_pk, &child_hand);
                self.store(pos_key, child_dp, INF, 0);
                return;
            }
            if or_node && cdn_now == 0 {
                // OR: この子は反証済み → 反証駒を蓄積
                let child_dp = self
                    .table
                    .get_disproof_hand(child_pk, &child_hand);
                let adj =
                    adjust_or_proof(*m, &child_dp);
                for k in 0..HAND_KINDS_TT {
                    init_or_disproof[k] =
                        init_or_disproof[k].min(adj[k]);
                }
                continue;
            }

            children.push((
                *m,
                child_full_hash,
                child_pk,
                child_hand,
            ));
        }

        // OR ノードで全子が反証済み(children が空)
        if or_node && children.is_empty() {
            self.store(
                pos_key, init_or_disproof, INF, 0,
            );
            return;
        }

        // パスに追加(フルハッシュ)
        self.path.insert(full_hash);

        // MID ループ(証明駒/反証駒の伝播を含む)
        loop {
            // 各子ノードの pn/dn を収集し，証明/反証を検出
            let mut current_pn: u32;
            let mut current_dn: u32;
            let mut best_idx: usize = 0;
            let mut second_best: u32;
            let mut best_pn_dn: (u32, u32) = (INF, 0);
            let mut proved_or_disproved = false;

            if or_node {
                // OR ノード: min(pn), sum(dn)
                current_pn = INF;
                current_dn = 0;
                second_best = INF; // 2番目に小さい pn
                // 反証駒の交差(全子の反証駒の min)
                let mut or_disproof = MAX_DISPROOF;

                for (i, &(ref _m, child_fh, child_pk, ref child_hand)) in
                    children.iter().enumerate()
                {
                    let (cpn, cdn) =
                        if self.path.contains(&child_fh) {
                            (INF, 0)
                        } else {
                            self.look_up_pn_dn(
                                child_pk, child_hand,
                            )
                        };

                    if cpn == 0 {
                        // 子が証明済み → OR ノード証明
                        let child_ph = self
                            .table
                            .get_proof_hand(
                                child_pk, child_hand,
                            );
                        let mut proof = adjust_or_proof(
                            children[i].0,
                            &child_ph,
                        );
                        // 証明駒を現在の持ち駒で上限クリップ
                        for k in 0..HAND_KINDS_TT {
                            proof[k] =
                                proof[k].min(att_hand[k]);
                        }
                        self.store(
                            pos_key, proof, 0, INF,
                        );
                        proved_or_disproved = true;
                        break;
                    }

                    // 反証済みの子: 反証駒を蓄積
                    if cdn == 0 {
                        let child_dp = self
                            .table
                            .get_disproof_hand(
                                child_pk, child_hand,
                            );
                        let adj = adjust_or_proof(
                            children[i].0,
                            &child_dp,
                        );
                        for k in 0..HAND_KINDS_TT {
                            or_disproof[k] =
                                or_disproof[k].min(adj[k]);
                        }
                    }

                    if cpn < current_pn {
                        second_best = current_pn;
                        current_pn = cpn;
                        best_idx = i;
                        best_pn_dn = (cpn, cdn);
                    } else if cpn < second_best {
                        second_best = cpn;
                    }
                    current_dn = (current_dn as u64)
                        .saturating_add(cdn as u64)
                        .min(INF as u64)
                        as u32;
                }

                if proved_or_disproved {
                    self.path.remove(&full_hash);
                    return;
                }

                // 全子が反証済み(dn=0) → OR ノード反証
                if current_dn == 0 {
                    self.store(
                        pos_key, or_disproof,
                        current_pn, current_dn,
                    );
                    self.path.remove(&full_hash);
                    return;
                }
            } else {
                // AND ノード: sum(pn), min(dn)
                current_pn = 0;
                current_dn = INF;
                second_best = INF; // 2番目に小さい dn
                let mut all_proved = true;
                let mut and_proof =
                    [0u8; HAND_KINDS_TT]; // 証明駒の和集合(max)

                for (i, &(ref _m, child_fh, child_pk, ref child_hand)) in
                    children.iter().enumerate()
                {
                    let (cpn, cdn) =
                        if self.path.contains(&child_fh) {
                            (INF, 0)
                        } else {
                            self.look_up_pn_dn(
                                child_pk, child_hand,
                            )
                        };

                    if cdn == 0 {
                        // 子が反証済み → AND ノード反証
                        // 反証駒 = 子の反証駒
                        let child_dp = self
                            .table
                            .get_disproof_hand(
                                child_pk, child_hand,
                            );
                        self.store(
                            pos_key, child_dp, INF, 0,
                        );
                        proved_or_disproved = true;
                        break;
                    }

                    if cpn == 0 {
                        // 子が証明済み → 証明駒を蓄積
                        let child_ph = self
                            .table
                            .get_proof_hand(
                                child_pk, child_hand,
                            );
                        for k in 0..HAND_KINDS_TT {
                            if child_ph[k] > and_proof[k]
                            {
                                and_proof[k] = child_ph[k];
                            }
                        }
                    } else {
                        all_proved = false;
                    }

                    current_pn = (current_pn as u64)
                        .saturating_add(cpn as u64)
                        .min(INF as u64)
                        as u32;
                    if cdn < current_dn {
                        second_best = current_dn;
                        current_dn = cdn;
                        best_idx = i;
                        best_pn_dn = (cpn, cdn);
                    } else if cdn < second_best {
                        second_best = cdn;
                    }
                }

                if proved_or_disproved {
                    self.path.remove(&full_hash);
                    return;
                }

                // 全子が証明済み → AND ノード証明
                if all_proved && current_pn == 0 {
                    // 証明駒を現在の持ち駒で制限
                    for k in 0..HAND_KINDS_TT {
                        and_proof[k] =
                            and_proof[k].min(att_hand[k]);
                    }
                    self.store(
                        pos_key, and_proof, 0, INF,
                    );
                    self.path.remove(&full_hash);
                    return;
                }
            }

            // 転置表を更新
            self.store(
                pos_key, att_hand, current_pn, current_dn,
            );

            // 閾値チェック
            if current_pn >= pn_threshold
                || current_dn >= dn_threshold
            {
                break;
            }

            // ノード制限・タイムアウトチェック
            if self.nodes_searched >= self.max_nodes
                || self.timed_out
            {
                break;
            }

            // 閾値計算
            let (child_pn_th, child_dn_th) = if or_node {
                let child_dn_th = dn_threshold
                    .saturating_sub(current_dn)
                    .saturating_add(best_pn_dn.1)
                    .min(INF - 1);
                let child_pn_th = pn_threshold
                    .min(second_best.saturating_add(1))
                    .min(INF - 1);
                (child_pn_th, child_dn_th)
            } else {
                let child_pn_th = pn_threshold
                    .saturating_sub(current_pn)
                    .saturating_add(best_pn_dn.0)
                    .min(INF - 1);
                let child_dn_th = dn_threshold
                    .min(second_best.saturating_add(1))
                    .min(INF - 1);
                (child_pn_th, child_dn_th)
            };

            // 子ノードを探索
            let (m, _, _, _) = children[best_idx];
            let captured = board.do_move(m);
            self.mid(
                board,
                child_pn_th,
                child_dn_th,
                ply + 1,
                !or_node,
            );
            board.undo_move(m, captured);
        }

        // パスから除去
        self.path.remove(&full_hash);
    }

    /// 現在の局面(攻め方の手番)に1手詰めがあるか判定する．
    ///
    /// 王手を生成し，応手が0の王手があれば1手詰め．
    /// 詰みの局面(応手なし AND ノード)を TT に記録する．
    fn has_mate_in_1(&mut self, board: &mut Board) -> bool {
        let checks = self.generate_check_moves(board);
        self.has_mate_in_1_with(board, &checks)
    }

    /// 既に生成済みの王手リストを使って1手詰め判定する．
    fn has_mate_in_1_with(
        &mut self,
        board: &mut Board,
        checks: &ArrayVec<Move, MAX_MOVES>,
    ) -> bool {
        for m in checks {
            let captured = board.do_move(*m);
            let has_defense = self.has_any_defense(board);
            if !has_defense {
                // 詰み局面を TT に記録(証明駒 = 空)
                let pk = position_key(board);
                self.store(pk, [0; HAND_KINDS_TT], 0, INF);
                board.undo_move(*m, captured);
                return true;
            }
            board.undo_move(*m, captured);
        }
        false
    }

    /// 玉方に1つでも王手回避手があるか高速判定する．
    ///
    /// generate_defense_moves と同じロジックだが，
    /// 最初の合法手が見つかった時点で即座に true を返す．
    fn has_any_defense(&self, board: &mut Board) -> bool {
        let defender = board.turn;
        let attacker = defender.opponent();

        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => return !movegen::generate_legal_moves(board).is_empty(),
        };

        let checkers =
            self.find_checkers(board, king_sq, attacker);
        if checkers.is_empty() {
            return true; // 王手なし → 回避不要
        }

        let all_occ = board.all_occupied();
        let our_occ = board.occupied[defender.index()];

        // 1. 玉の移動
        let king_attacks = attack::step_attacks(
            defender,
            PieceType::King,
            king_sq,
        );
        let king_targets = king_attacks & !our_occ;
        for to in king_targets {
            let captured_piece = board.squares[to.index()];
            let captured_raw = captured_piece.0;
            let m = Move::new_move(
                king_sq,
                to,
                false,
                captured_raw,
                PieceType::King as u8,
            );
            let captured = board.do_move(m);
            let safe = !board.is_in_check(defender);
            board.undo_move(m, captured);
            if safe {
                return true;
            }
        }

        // 両王手 → 玉移動のみ
        if checkers.count() > 1 {
            return false;
        }

        let checker_sq = checkers.lsb().unwrap();

        // 2. 王手駒の捕獲(玉以外)
        let mut our_bb = our_occ;
        while our_bb.is_not_empty() {
            let from = our_bb.pop_lsb();
            if from == king_sq {
                continue;
            }
            let piece = board.squares[from.index()];
            let pt = piece.piece_type().unwrap();
            let attacks =
                attack::piece_attacks(defender, pt, from, all_occ);
            if !attacks.contains(checker_sq) {
                continue;
            }
            let captured_raw =
                board.squares[checker_sq.index()].0;
            let in_promo_zone = checker_sq
                .is_promotion_zone(defender)
                || from.is_promotion_zone(defender);

            if pt.can_promote() && in_promo_zone {
                let m = Move::new_move(
                    from,
                    checker_sq,
                    true,
                    captured_raw,
                    pt as u8,
                );
                if self.is_evasion_legal(board, m, defender) {
                    return true;
                }
                if !movegen::must_promote(defender, pt, checker_sq)
                {
                    let m = Move::new_move(
                        from,
                        checker_sq,
                        false,
                        captured_raw,
                        pt as u8,
                    );
                    if self.is_evasion_legal(board, m, defender) {
                        return true;
                    }
                }
            } else if !movegen::must_promote(
                defender, pt, checker_sq,
            ) {
                let m = Move::new_move(
                    from,
                    checker_sq,
                    false,
                    captured_raw,
                    pt as u8,
                );
                if self.is_evasion_legal(board, m, defender) {
                    return true;
                }
            }
        }

        // 3. 合い駒(飛び駒王手の場合)
        let sliding_checker =
            self.find_sliding_checker(board, king_sq, attacker);
        if let Some(_) = sliding_checker {
            let between =
                attack::between_bb(checker_sq, king_sq);
            if between.is_not_empty() {
                let futile = self.compute_futile_squares(
                    board, &between, king_sq, checker_sq,
                    defender, attacker,
                );
                // 間のマスへの移動・打ち
                for to in between {
                    // 駒移動
                    let mut our_bb2 = our_occ;
                    while our_bb2.is_not_empty() {
                        let from = our_bb2.pop_lsb();
                        if from == king_sq {
                            continue;
                        }
                        let piece =
                            board.squares[from.index()];
                        let pt =
                            piece.piece_type().unwrap();
                        let attacks = attack::piece_attacks(
                            defender, pt, from, all_occ,
                        );
                        if !attacks.contains(to) {
                            continue;
                        }
                        let captured_raw =
                            board.squares[to.index()].0;
                        let in_promo_zone = to
                            .is_promotion_zone(defender)
                            || from
                                .is_promotion_zone(defender);
                        if pt.can_promote() && in_promo_zone {
                            let m = Move::new_move(
                                from,
                                to,
                                true,
                                captured_raw,
                                pt as u8,
                            );
                            if self.is_evasion_legal(
                                board, m, defender,
                            ) {
                                return true;
                            }
                        }
                        if !movegen::must_promote(
                            defender, pt, to,
                        ) {
                            let m = Move::new_move(
                                from,
                                to,
                                false,
                                captured_raw,
                                pt as u8,
                            );
                            if self.is_evasion_legal(
                                board, m, defender,
                            ) {
                                return true;
                            }
                        }
                    }
                    // 駒打ち(合い効かずでないマスのみ)
                    if !futile.contains(to) {
                        for (hand_idx, &pt) in PieceType::HAND_PIECES
                            .iter()
                            .enumerate()
                        {
                            if board.hand[defender.index()]
                                [hand_idx]
                                == 0
                            {
                                continue;
                            }
                            if movegen::must_promote(
                                defender, pt, to,
                            ) {
                                continue;
                            }
                            if pt == PieceType::Pawn {
                                let our_pawns = board.piece_bb
                                    [defender.index()]
                                    [PieceType::Pawn as usize];
                                let col = to.col();
                                if (our_pawns
                                    & Bitboard::file_mask(col))
                                .is_not_empty()
                                {
                                    continue;
                                }
                            }
                            let m = Move::new_drop(to, pt);
                            if pt == PieceType::Pawn
                                && movegen::is_pawn_drop_mate(
                                    board, m,
                                )
                            {
                                continue;
                            }
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// 玉方の王手回避手を生成する(合い効かずを除外)．
    ///
    /// 全合法手生成の代わりに回避手のみを直接生成する:
    /// 1. 玉の移動(攻め方に利かれていないマスへ)
    /// 2. 王手駒の捕獲(ピンされていない駒による)
    /// 3. 合い駒(飛び駒の王手の場合，間のマスへ移動または打つ)
    ///
    /// 合い効かず(futile interposition)もフィルタする．
    fn generate_defense_moves(
        &self,
        board: &mut Board,
    ) -> ArrayVec<Move, MAX_MOVES> {
        let defender = board.turn;
        let attacker = defender.opponent();

        let king_sq = match board.king_square(defender) {
            Some(sq) => sq,
            None => {
                let legal = movegen::generate_legal_moves(board);
                let mut out = ArrayVec::new();
                for m in legal {
                    if out.is_full() {
                        break;
                    }
                    out.push(m);
                }
                return out;
            }
        };

        // 王手している駒を特定
        let checkers = self.find_checkers(board, king_sq, attacker);
        if checkers.is_empty() {
            // 王手されていない(通常ありえないが安全策)
            let legal = movegen::generate_legal_moves(board);
            let mut out = ArrayVec::new();
            for m in legal {
                if out.is_full() {
                    break;
                }
                out.push(m);
            }
            return out;
        }

        let all_occ = board.all_occupied();
        let our_occ = board.occupied[defender.index()];
        let mut moves = ArrayVec::<Move, MAX_MOVES>::new();

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
        moves: &mut ArrayVec<Move, MAX_MOVES>,
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
        moves: &mut ArrayVec<Move, MAX_MOVES>,
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
    fn generate_check_moves(
        &self,
        board: &mut Board,
    ) -> ArrayVec<Move, MAX_MOVES> {
        let us = board.turn;
        let them = us.opponent();
        let has_own_king = board.king_square(us).is_some();

        let king_sq = match board.king_square(them) {
            Some(sq) => sq,
            None => return ArrayVec::new(),
        };

        let our_occ = board.occupied[us.index()];
        let all_occ = board.all_occupied();
        let empty = !all_occ;

        // 各駒種について「このマスに置くと玉に王手がかかる」ターゲットを事前計算
        // step_attacks(them, pt, king_sq) は「玉から見た逆利き」= 王手元になれるマス

        let mut moves = ArrayVec::<Move, MAX_MOVES>::new();

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
        // 証明完了フェーズは主探索の半分を上限とする．
        let mid_nodes = self.nodes_searched;
        self.max_nodes =
            self.nodes_searched.saturating_add(
                mid_nodes.min(8192),
            );

        // 反復: PV を抽出 → PV 上の OR ノードを完成 → 再抽出
        for _ in 0..2 {
            if self.is_timed_out()
                || self.nodes_searched >= self.max_nodes
            {
                break;
            }
            let pv = self.extract_pv(board);
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
    fn complete_pv_or_nodes(&mut self, board: &mut Board, pv: &[Move]) -> bool {
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
                        // 1子あたり 1024 ノード上限
                        self.max_nodes = self
                            .nodes_searched
                            .saturating_add(1024)
                            .min(saved);
                        self.mid(&mut board_clone, INF - 1, INF - 1, ply + 1, false);
                        self.max_nodes = saved;

                        if self.look_up_board(&board_clone).0 == 0 {
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
        self.extract_pv_recursive(&mut board_clone, true, &mut FxHashSet::default())
    }

    /// PV 復元の再帰実装．
    ///
    /// 各ノードで全候補手のサブPVを生成し，攻め方は最短，玉方は最長を選ぶ．
    /// ループ検出にはフルハッシュ，TT 参照には位置キー＋持ち駒を使用する．
    fn extract_pv_recursive(
        &self,
        board: &mut Board,
        or_node: bool,
        visited: &mut FxHashSet<u64>,
    ) -> Vec<Move> {
        let full_hash = board.hash;

        // ループ検出(フルハッシュ)
        if visited.contains(&full_hash) {
            return Vec::new();
        }

        let (node_pn, _node_dn) = self.look_up_board(board);

        if or_node {
            if node_pn != 0 {
                return Vec::new();
            }

            let moves = self.generate_check_moves(board);
            if moves.is_empty() {
                return Vec::new();
            }

            let mut best_pv: Option<Vec<Move>> = None;

            for m in &moves {
                let captured = board.do_move(*m);
                let (child_pn, _) = self.look_up_board(board);

                if child_pn == 0 {
                    visited.insert(full_hash);
                    let sub_pv =
                        self.extract_pv_recursive(
                            board, false, visited,
                        );
                    visited.remove(&full_hash);

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
            let moves = self.generate_defense_moves(board);
            if moves.is_empty() {
                return Vec::new();
            }

            let mut best_pv: Option<Vec<Move>> = None;
            let mut best_is_capture = false;

            for m in &moves {
                let captured = board.do_move(*m);
                let (child_pn, _) = self.look_up_board(board);

                if child_pn == 0 {
                    visited.insert(full_hash);
                    let sub_pv =
                        self.extract_pv_recursive(
                            board, true, visited,
                        );
                    visited.remove(&full_hash);

                    let total_len = 1 + sub_pv.len();
                    let is_capture = m.captured_piece_raw() > 0;
                    let is_better = match &best_pv {
                        None => true,
                        Some(prev) => {
                            if total_len > prev.len() {
                                true
                            } else if total_len == prev.len()
                                && is_capture
                                && !best_is_capture
                            {
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
    /// 11手詰め(合い効かずにより3二歩打を除外)．
    /// 最終2手は玉の逃げ方により3パターンの正解が存在:
    /// - 1一玉，2二桂成 / 1一玉，2二龍 / 1三玉，2二龍
    #[test]
    fn test_tsume_4() {
        let sfen = "7nk/9/5R3/8p/6P2/9/9/9/9 b SNPr2b4g3s2n4l15p 1";
        let result = solve_tsume(sfen, Some(31), Some(2_000_000), None).unwrap();

        match &result {
            TsumeResult::Checkmate { moves, .. } => {
                let usi_moves: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                assert_eq!(
                    usi_moves.len(), 11,
                    "expected 11 moves, got {}: {:?}", usi_moves.len(), usi_moves
                );
                // 最初の9手は共通(7手目は成/不成どちらも可)
                let common_prefix = [
                    "S*2b", "1a1b", "P*1c", "1b2b", "N*3d", "2b1a",
                ];
                assert_eq!(
                    &usi_moves[..6], &common_prefix,
                    "first 6 moves mismatch:\n  got:      {:?}\n  expected: {:?}",
                    &usi_moves[..6], &common_prefix,
                );
                // 7手目: 1c1b+ (成) or 1c1b (不成) どちらも正解
                assert!(
                    usi_moves[6] == "1c1b+" || usi_moves[6] == "1c1b",
                    "move 7 should be 1c1b+ or 1c1b, got: {}",
                    usi_moves[6],
                );
                let common_suffix = ["1a1b", "4c4b+"];
                assert_eq!(
                    &usi_moves[7..9], &common_suffix,
                    "moves 8-9 mismatch:\n  got:      {:?}\n  expected: {:?}",
                    &usi_moves[7..9], &common_suffix,
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

            let moves: ArrayVec<Move, MAX_MOVES> = if or_node {
                solver.generate_check_moves(board)
            } else {
                let legal = movegen::generate_legal_moves(board);
                let mut out = ArrayVec::new();
                for m in legal {
                    if out.is_full() { break; }
                    out.push(m);
                }
                out
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
