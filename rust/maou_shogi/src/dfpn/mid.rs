//! mid: df-pn (Nagai) ベースの反復深化探索エンジン (探索本体 + 子展開の配線)．
//!
//! component を結線して探索を構成する:
//! - [`super::local_expansion::LocalExpansion`] (resort 後の δ 再計算込の δ/閾値/選択)
//! - [`super::tt::TranspositionTable`] (len-aware + cross-hand 置換表)
//! - [`super::mate_len::MateLen`] threading (全ノードへ len-1 を伝播)
//!
//! ## single-thread / 近道排除
//! - tt は **local の `&mut TranspositionTable`** を再帰へ渡す (DfPnSolver field を増やさず，
//!   `&mut self` (movegen) と `&mut tt` を disjoint に保つ)．
//! - 並列化なし (single-thread)．
//!
//! ## 子展開ベースの探索構造
//! - **1 手詰先読み**: AND first-visit child を do_move し OR 子の 1 手詰/詰み無を先読みし
//!   proven/disproven を seed する ([`check_obvious_final_or_node`])．
//! - **DML deferral**: `DelayedMoveList` を build_expansion に配線．同マス合駒/成不成ペアの
//!   prev chain が未 final の手を idx から除外 (後回し) し，final 化で `update_best_child` が revival する．
//! - **position-only TT board_key**: TT を `position_key` (board_hash, 持駒除外) で索引し，hand は
//!   entry に別管理する．board_key を持駒除外にすることで cross-hand Superior/Inferior 再利用が有効化される．
//!
//! ## soundness 上の制約 / 未実装
//! - **proof hand 極小化は unsound**: cross-hand 有効化後に minimal proof hand を使うと
//!   過剰な Superior 再利用で **mate-39 偽証明**を生むため不使用 (full hand は sound)．
//!   (`before_hand`/proof hand 極小化/`add_if` 周りに soundness bug がある; 低優先)．
//! - **EliminateDoubleCount (DAG)**: ancestor 展開を再帰 stack で walk する．
//! - STRICT PV replay / 無意味中合いの cross-square DML．

use super::movegen::delayed_move_list::DelayedMoveList;
use super::local_expansion::{BranchRootEdge, LocalExpansion, K_ANCESTOR_SEARCH_THRESHOLD};
use super::mate_len::{MateLen, DEPTH_MAX_MATE_LEN, ZERO_MATE_LEN};
use super::path_key::path_key_after;
use super::search_result::{
    extend_search_threshold, BitSet64, Hand, PnDn, SearchResult, K_INFINITE_PN_DN,
};
use super::solver::{DfPnSolver, TsumeResult};
use super::tt::TranspositionTable;
use crate::board::Board;

/// init_pn_dn (unit-16) を pn/dn unit=2 へ縮約する除数．
const DIV: u64 = 8;

/// δ がこの値以上の子は build 時に sum→max 集約へ落とす (local_expansion.rs と同値)．
const K_FORCE_SUM_PN_DN: PnDn = K_INFINITE_PN_DN / 1024;

/// move のδ値を sum で計上すべきか判定する．
/// 基本は true (sum)．OR node で「2/3 段目 (受け方先) への香の成/不成 (玉が直前にいる)」のみ
/// false (max) を返す = 似た子局面で過小評価を避ける．`board` は親局面 (この手を指す側の手番)．
fn is_sum_delta_node(board: &Board, m: crate::moves::Move, or_node: bool) -> bool {
    use crate::types::{Color, PieceType};
    if m.is_drop() || !or_node {
        return true;
    }
    let from = m.from_sq();
    // 移動駒種 (raw piece type の下位 4bit)．香 = Lance のみ判定対象．
    if (board.piece_at(from) & 0x0F) != PieceType::Lance as u8 {
        return true;
    }
    let to = m.to_sq();
    let to_raw = to.raw_u8();
    let rank0 = (to_raw % 9) as i32; // 0-based rank (rank1=0 .. rank9=8)
    let king = match board.king_square(board.turn.opponent()) {
        Some(k) => k.raw_u8() as i32,
        None => return true,
    };
    let to_i = to_raw as i32;
    // 黒番は to が rank2/3 (rank0∈{1,2}) かつ玉が to の 1 つ上 (rank 減 = raw-1)．
    // 白番は to が rank7/8 (rank0∈{6,7}) かつ玉が to の 1 つ下 (rank 増 = raw+1)．
    let hit = if board.turn == Color::Black {
        (rank0 == 1 || rank0 == 2) && king == to_i - 1
    } else {
        (rank0 == 6 || rank0 == 7) && king == to_i + 1
    };
    !hit
}

/// `defender` が `to` に駒 `pt` を打つと `atk_king` (攻め方玉) に王手がかかるか．
/// drop は不成駒なので基本駒種の利きで判定する (DML 中合い対称性の逆王手判定用)．
fn drop_gives_check(
    board: &Board,
    to: crate::types::Square,
    pt: crate::types::PieceType,
    defender: crate::types::Color,
    atk_king: crate::types::Square,
) -> bool {
    use crate::types::PieceType;
    let occ = board.all_occupied();
    let atk = match pt {
        PieceType::Pawn | PieceType::Knight | PieceType::Silver | PieceType::Gold => {
            crate::attack::step_attacks(defender, pt, to)
        }
        PieceType::Lance => crate::attack::lance_attacks(defender, to, occ),
        PieceType::Bishop => crate::attack::bishop_attacks(to, occ),
        PieceType::Rook => crate::attack::rook_attacks(to, occ),
        _ => return false,
    };
    atk.contains(atk_king)
}

/// `SEL` env (process 内 1 回読み)．ply 0-7 初出ノードの sort 済子リスト (move/pn/dn) を
/// sfen 付きで dump する診断用．
fn sel_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("SEL").is_ok())
}

/// proof/disproof hand を hand-set 極小化する (cross-hand TT 再利用を効かせる)．常時 ON．
pub(super) fn handset_enabled() -> bool {
    true
}

/// `FULLPROOFHAND` 診断 gate (process 内 1 回読み)．proof (詰み) hand の hand-set 極小化のみを
/// 無効化し full `attacker_hand` で格納する (disproof 側は handset のまま)．proof 極小化を切ると
/// cross-hand proof 再利用が抑止され re-Emplace へ反転する (= proof generalization の影響を切り分ける
/// 制御実験用)．通常探索では使わない (node 退行する)．
pub(super) fn full_proof_hand_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("FULLPROOFHAND").is_ok())
}

/// look-ahead 不詰判定に `does_have_mate_possibility` (blocker 無視 over-approx) を使う．
/// **default ON** (`NODHMP` で opt-out)．exact `!has_checks` だと blocker で塞がれた王手候補を
/// 即 disproof してしまうが (例: 王手0 でも香の成り候補がある局面)，over-approx は詰みの可能性が
/// 残る局面を defer するため取りこぼしを避ける (sound)．
pub(super) fn dhmp_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("NODHMP").is_err())
}

/// look-ahead 1 手詰 scan を玉から距離 ≤2 候補に限定する (default ON; `NONEAR2` で opt-out)．
/// 距離 ≤2 は full scan と node 不変かつ sound (per-candidate verify)．遠方候補の検証・do_move
/// fallback を省き do_moves を削減する ([`super::solver::mate1ply_cached_near2`])．
pub(super) fn near2_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("NONEAR2").is_err())
}

/// `INTROSORT` 実験 (process 内 1 回読み)．idx ソートに introsort (`std_sort`) を使う．
/// default OFF = stable sort (movegen 順保持)．
pub(super) fn introsort_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("INTROSORT").is_ok())
}

/// `TRACE` env (process 内 1 回読み)．各 build の best 子 (idx[0]) の move/pn/dn を sfen 付きで
/// chronological dump する診断用．
fn trace_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("TRACE").is_ok())
}

/// `NODE` env: 指定 sfen prefix に一致するノードの sort 済子リストを dump する診断用 (process 内 1 回読み)．
fn node_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("NODE").ok().filter(|s| !s.is_empty()))
        .clone()
}

/// `TH` env: 指定 sfen prefix に一致するノードが受け取った thpn/thdn を dump する診断用．
fn th_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("TH").ok().filter(|s| !s.is_empty()))
        .clone()
}

/// `THX` env: 指定 sfen prefix に一致するノードの探索ループを per-iteration dump する診断用
/// (inc_flag/threshold/best/curr の追跡)．
fn thx_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("THX").ok().filter(|s| !s.is_empty()))
        .clone()
}

/// `HAND` env: 指定 sfen prefix のノードの final 結果 (proof/disproof hand) を dump する診断用．
pub(super) fn hand_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("HAND").ok().filter(|s| !s.is_empty()))
        .clone()
}

/// `SEED` env: 指定 sfen prefix の親ノードで各子の look_up 結果 (TT/cross-hand) と look-ahead
/// 前後を dump する診断．child loop の per-child hot path で参照するため，env::var ではなく
/// OnceLock で 1 回読みして借用を返す (clone しない = 20.2M children/39te での lock+alloc を回避)．
fn seed_prefix() -> &'static Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("SEED").ok().filter(|s| !s.is_empty()))
}

/// `INC` env: 指定 sfen prefix のノードを起点に active-window を開き，window 内の探索ループの
/// ENTER / INC(does_have_old_child) / DEC(first_visit) / EXIT(clamp=min(inc,orig_inc)) を逐次 dump する．
/// TCA `inc_flag` の累積収支を localize するための診断 (process 内 1 回読み)．
fn inc_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("INC").ok().filter(|s| !s.is_empty()))
        .clone()
}

#[inline]
fn inc_active() -> bool {
    INC_ACTIVE.with(|c| c.get())
}

/// window が未 active かつ board.sfen() が INC prefix 一致なら window を開く (このノードが opener)．
fn inc_open_window(board: &Board) -> bool {
    if let Some(p) = inc_prefix() {
        if !inc_active() && board.sfen().starts_with(p.as_str()) {
            INC_ACTIVE.with(|c| c.set(true));
            INC_CNT.with(|c| c.set(0));
            return true;
        }
    }
    false
}

/// opener のみ window を閉じる (nested 呼び出しは false で素通り)．
#[inline]
fn inc_close_window(opened: bool) {
    if opened {
        INC_ACTIVE.with(|c| c.set(false));
    }
}

/// window active 時のみ dump (暴走防止に 4000 行 cap)．
#[inline]
fn inc_log(s: &str) {
    if inc_active() {
        let n = INC_CNT.with(|c| {
            let v = c.get();
            c.set(v + 1);
            v
        });
        if n < 4000 {
            eprintln!("{}", s);
        }
    }
}

/// `NOSMRESET` env: build-time sum_mask reset を無効化 (Full のまま from_parts) して
/// 旧挙動 (update_best_child の reset のみ) と切り分ける診断用 (process 内 1 回読み)．
fn no_sum_mask_reset() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("NOSMRESET").is_ok())
}

/// `NODAG` env: EliminateDoubleCount (DAG 二重カウント抑止) を無効化する診断用 (process 内 1 回読み)．
fn no_dag() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("NODAG").is_ok())
}

/// 王手生成の生成順を再現する sort key．square index は (file-1)*9+(rank-1) なので raw square で
/// 昇順比較できる．順序: ① 盤上移動 (from-square 昇順 → to-square 昇順 → 不成→成)，② 駒打ち
/// (PAWN,LANCE,KNIGHT,SILVER,GOLD,BISHOP,ROOK 順 → to-square 昇順)．
/// 注: 開き王手/直接王手の細分順は未反映 (直接王手のみの局面では from-square 順で一致)．
fn drop_piece_order(pt: crate::types::PieceType) -> u64 {
    use crate::types::PieceType;
    // 駒打ちの drops[] 順: PAWN 先頭 + KNIGHT,LANCE,SILVER,GOLD,BISHOP,ROOK (桂が香より先)．
    match pt {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Lance => 2,
        PieceType::Silver => 3,
        PieceType::Gold => 4,
        PieceType::Bishop => 5,
        PieceType::Rook => 6,
        _ => 7,
    }
}

pub(super) fn check_order_key(
    m: crate::moves::Move,
    discoverers: crate::bitboard::Bitboard,
    def_king: Option<crate::types::Square>,
) -> u64 {
    use crate::types::PieceType;
    if let Some(pt) = m.drop_piece_type() {
        // group 2: 駒打ち王手．駒打ち王手の生成順
        // (PAWN,LANCE,KNIGHT,SILVER,GOLD,BISHOP,ROOK) → 各 check_square 昇順．
        // ⚠ evasion drops[] (KNIGHT,LANCE,..) とは Knight/Lance が逆順なので別 mapping．
        let pmo = match pt {
            PieceType::Pawn => 0u64,
            PieceType::Lance => 1,
            PieceType::Knight => 2,
            PieceType::Silver => 3,
            PieceType::Gold => 4,
            PieceType::Bishop => 5,
            PieceType::Rook => 6,
            _ => 7,
        };
        (2u64 << 42) | (pmo << 7) | (m.to_sq().raw_u8() as u64)
    } else {
        // 盤上移動王手．王手生成順:
        //   ① 開き王手候補 from (= discoverers = blockers_for_king(them) & pieces(us)) を square 昇順で，
        //      各 from は pin-line 外への移動 (純開き王手) → pin-line 上への移動 (直接王手) の順に生成．
        //   ② 非候補 from (直接王手のみ) を square 昇順で生成．
        let from = m.from_sq();
        let to = m.to_sq();
        let pf = if m.is_promotion() { 0u64 } else { 1u64 };
        let disc = discoverers.contains(from);
        let group = if disc { 0u64 } else { 1u64 };
        // 開き王手候補 from 内: pin-line 外 (sub 0) → pin-line 上 (sub 1)．
        let sub = if disc {
            match def_king {
                Some(k) if crate::attack::line_through(k, from).contains(to) => 1u64,
                _ => 0u64,
            }
        } else {
            0u64
        };
        (group << 42)
            | ((from.raw_u8() as u64) << 22)
            | (sub << 21)
            | (pf << 20)
            | ((to.raw_u8() as u64) << 11)
    }
}

/// 盤上移動の駒種を駒種別生成順へ写像する (Evasion 非玉手の group 番号)．
/// PAWN→0, LANCE→1, KNIGHT→2, SILVER→3, BISHOP→4, ROOK→5, GOLD/成駒(金移動)→6, HORSE→7, DRAGON→8．
fn piece_gen_order(raw_pt: u8) -> u64 {
    match raw_pt {
        1 => 0,                // Pawn
        2 => 1,                // Lance
        3 => 2,                // Knight
        4 => 3,                // Silver
        5 => 4,                // Bishop
        6 => 5,                // Rook
        7 => 6,                // Gold
        9 | 10 | 11 | 12 => 6, // と金/成香/成桂/成銀 (金の動き)
        13 => 7,               // Horse
        14 => 8,               // Dragon
        _ => 9,
    }
}

/// 王手回避手の生成順を再現する key．
/// ① 玉移動 (to-square 昇順) → ② 非玉の盤上移動 (駒種 group 順 → from-square 昇順 → 不成/成) →
/// ③ 駒打ち (PAWN..ROOK 順 → to-square 昇順)．`board` は親 (受け方手番) 局面．
pub(super) fn evasion_order_key(board: &crate::board::Board, m: crate::moves::Move) -> u64 {
    if let Some(pt) = m.drop_piece_type() {
        // group 2 (駒打ちは最後)．evasion drop 順は **歩を全マス先に並べ，続いて各 to_sq ごとに
        // drops[] 順 (香→桂→銀→金→角→飛)** で生成する (例: P*8c P*8d P*8e P*8f / L*8c G*8c R*8c
        // / L*8d ...)．歩優先 (bit39) → to_sq 主 (bits 11-) → 駒種 従 (低 bit) とすることで同マス
        // drop が連続し，DML の同マス chain (to1==to2) が成立する (中合い G*8c が deferred される)．
        let pawn_first = if pt == crate::types::PieceType::Pawn {
            0u64
        } else {
            1u64
        };
        (2u64 << 40)
            | (pawn_first << 39)
            | ((m.to_sq().raw_u8() as u64) << 11)
            | drop_piece_order(pt)
    } else {
        let from = m.from_sq().raw_u8() as u64;
        let to = m.to_sq().raw_u8() as u64;
        let pf = if m.is_promotion() { 0u64 } else { 1u64 };
        let own_king = board.king_square(board.turn);
        if own_king == Some(m.from_sq()) {
            // group 0: 玉移動 (to-square 昇順)．
            to << 11
        } else {
            // group 1: 非玉の盤上移動 (駒種 group → from 昇順 → 成/不成 → to 昇順)．
            let raw_pt = board.piece_at(m.from_sq()) & 0x0F;
            let go = piece_gen_order(raw_pt);
            (1u64 << 40) | (go << 30) | (from << 20) | (pf << 19) | (to << 11)
        }
    }
}

thread_local! {
    /// ply 0-7 の dump 済 bitmask (solve 毎に reset)．
    static SEL_DUMPED: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    /// RAW (sort/DML 前の raw movegen 順) の ply 0-7 dump 済 bitmask．
    static RAW_DUMPED: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    /// TRACE の chronological build カウンタ (solve 毎に reset)．
    static TRACE_CNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// INC: inc_flag origin window が active か (opener が set/reset)．
    static INC_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// INC: window 内 dump 行カウンタ (cap 用; window open 毎に reset)．
    static INC_CNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// do_moves per-site breakdown (DMBREAK 報告用; solve 毎に reset)．
    /// [0]=step_best_child(再帰) [1]=eliminate_double_count(DAG) [2]=look-ahead [3]=proof-hand(1手詰).
    static DM_SITE: std::cell::Cell<[u64; 4]> = const { std::cell::Cell::new([0; 4]) };
    /// PROF per-phase 累積時間(ns) / 呼び出し回数 (PROF 報告用; solve 毎に reset)．
    /// idx: 0=movegen 1=tt_lookup 2=lookahead(check_obvious) 3=dag(EliminateDoubleCount) 4=dml_sort(build 残).
    static PROF_NS: std::cell::Cell<[u64; 12]> = const { std::cell::Cell::new([0; 12]) };
    static PROF_CNT: std::cell::Cell<[u64; 12]> = const { std::cell::Cell::new([0; 12]) };
}

/// `PROF` env: mid 探索の per-phase 時間内訳を report する (process 内 1 回読み)．
/// off 時は `then(Instant::now)` の bool 評価のみで実質ゼロコスト．
#[inline]
fn prof_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("PROF").is_ok())
}
#[inline]
fn prof_add(idx: usize, ns: u64) {
    PROF_NS.with(|c| {
        let mut a = c.get();
        a[idx] += ns;
        c.set(a);
    });
    PROF_CNT.with(|c| {
        let mut a = c.get();
        a[idx] += 1;
        c.set(a);
    });
}

/// do_moves per-site カウンタを 1 増やす (idx: 0=step 1=dag 2=lookahead 3=proofhand)．
#[inline]
fn dm_bump(idx: usize) {
    DM_SITE.with(|c| {
        let mut a = c.get();
        a[idx] += 1;
        c.set(a);
    });
}

/// `build_expansion` が per-node に確保する 6 本の Vec を再利用する free-list pool．
///
/// `LocalExpansion` は moves / move_evals / queries / results / idx / dml_next の 6 Vec を所有する．
/// 従来は node 毎に `Vec::new` / `Vec::with_capacity` で確保し，node pop (`expansion_stack` truncate) で
/// drop していた (39te = 3.1M nodes × 6 = ~18.6M 回のヒープ alloc/free)．本 pool は node pop 時に
/// 容量保持したまま Vec を返却し，次の build で再取得することで alloc/free 回数を削減する．
/// 内容は build 毎に上書き (取得時 clear) されるため **探索完全不変**．`clear` で solve 間にリセット．
#[derive(Default)]
pub(super) struct BufPool {
    moves: Vec<Vec<crate::moves::Move>>,
    evals: Vec<Vec<i32>>,
    queries: Vec<Vec<super::tt::TtContext>>,
    results: Vec<Vec<SearchResult>>,
    idx: Vec<Vec<u32>>,
    dml: Vec<Vec<i32>>,
}

impl BufPool {
    fn take_moves(&mut self) -> Vec<crate::moves::Move> {
        let mut v = self.moves.pop().unwrap_or_default();
        v.clear();
        v
    }
    fn take_evals(&mut self) -> Vec<i32> {
        let mut v = self.evals.pop().unwrap_or_default();
        v.clear();
        v
    }
    fn take_queries(&mut self) -> Vec<super::tt::TtContext> {
        let mut v = self.queries.pop().unwrap_or_default();
        v.clear();
        v
    }
    fn take_results(&mut self) -> Vec<SearchResult> {
        let mut v = self.results.pop().unwrap_or_default();
        v.clear();
        v
    }
    fn take_idx(&mut self) -> Vec<u32> {
        let mut v = self.idx.pop().unwrap_or_default();
        v.clear();
        v
    }
    fn take_dml(&mut self) -> Vec<i32> {
        let mut v = self.dml.pop().unwrap_or_default();
        v.clear();
        v
    }
    /// `moves` のみ返却する (build 早期 Err = terminal node では他 5 本は未取得)．
    fn release_moves(&mut self, v: Vec<crate::moves::Move>) {
        self.moves.push(v);
    }
    /// node pop 時に 6 本まとめて返却する (容量保持; 次取得時に clear)．
    #[allow(clippy::too_many_arguments)]
    fn release(
        &mut self,
        moves: Vec<crate::moves::Move>,
        evals: Vec<i32>,
        queries: Vec<super::tt::TtContext>,
        results: Vec<SearchResult>,
        idx: Vec<u32>,
        dml: Vec<i32>,
    ) {
        self.moves.push(moves);
        self.evals.push(evals);
        self.queries.push(queries);
        self.results.push(results);
        self.idx.push(idx);
        self.dml.push(dml);
    }
    fn clear(&mut self) {
        self.moves.clear();
        self.evals.clear();
        self.queries.clear();
        self.results.clear();
        self.idx.clear();
        self.dml.clear();
    }
}

impl DfPnSolver {
    /// mid 探索の root (反復深化 + 診断)．production `solve()` の唯一の engine．
    pub(super) fn solve_impl(&mut self, board: &mut Board) -> TsumeResult {
        self.attacker = board.turn;
        self.nodes = 0;
        self.path_depths.clear();
        self.expansion_stack.clear();
        self.expansion_buf_pool.clear();
        self.dag_fires = 0;
        self.dom_path.clear();
        self.dom_fires = 0;
        SEL_DUMPED.with(|c| c.set(0));
        RAW_DUMPED.with(|c| c.set(0));
        TRACE_CNT.with(|c| c.set(0));
        DM_SITE.with(|c| c.set([0; 4]));
        PROF_NS.with(|c| c.set([0; 12]));
        PROF_CNT.with(|c| c.set([0; 12]));
        super::movegen::reset_legal_quick_dm();
        crate::movegen::reset_pawn_drop_mate_dm();
        crate::board::reset_do_move_count();
        super::movegen::mate1ply::reset_mate_cand_stats();
        super::movegen::mate1ply::reset_mate1ply_stats();
        // path dominance (劣位局面の刈り込み)．常時 ON．
        self.param_path_dominance = true;
        self.timed_out = false;
        self.start_time = std::time::Instant::now();

        // len-aware TT (local; 再帰へ &mut で渡す)．サイズは budget 比例で確保し，満杯時は
        // GC (maybe_collect_garbage) で低 amount entry を間引く．`TTSIZE` で entry 数を上書き可．
        let size = if let Ok(s) = std::env::var("TTSIZE") {
            s.parse::<usize>().unwrap_or(1 << 23).max(1 << 12)
        } else {
            ((self.max_nodes as usize).saturating_mul(2)).clamp(1 << 18, 1 << 23)
        };
        let mut tt = TranspositionTable::new(size);

        // do_move 数を計測開始 (verify は除外するため探索後に読む)．
        crate::board::reset_do_move_count();
        // root expansion を **一度だけ** 構築し，閾値反復 (反復深化) 間で `expansion_stack[0]` に
        // **持続**させる (root を持続させると子の results/idx/sum_mask/excluded が反復跨ぎで累積する)．
        let attacker_hand = board.hand[self.attacker.index()];
        let last = match self.emplace(
            &mut tt,
            board,
            DEPTH_MAX_MATE_LEN,
            0,
            0u64,
            true,
            BitSet64::full(),
        ) {
            // root が終局 (王手なし=不詰 / 受けなし=詰み)．即結果 (空 expansion の CurrentResult)．
            Err(terminal) => terminal,
            Ok(()) => {
                // root は expansion_stack[0] に push 済 (emplace が nodes++/dump/key_hand_pair も実施)．
                // thpn=thdn=0 (len==kDepthMaxMateLen)．初反復は no-op warmup．
                let mut thpn: PnDn = 0;
                let mut thdn: PnDn = 0;
                let cap = K_INFINITE_PN_DN - 1;
                let mut result = self.expansion_stack[0].current_result(board, 0);
                loop {
                    if self.nodes >= self.max_nodes || self.is_timed_out() {
                        self.timed_out = self.is_timed_out();
                        break;
                    }
                    result =
                        self.search_impl_root(&mut tt, board, thpn, thdn, DEPTH_MAX_MATE_LEN);
                    // ROOT: root 反復ごとの th と結果を dump する診断用．
                    if std::env::var("ROOT").is_ok() {
                        eprintln!(
                            "ROOT th=({},{}) -> pn={} dn={}",
                            thpn,
                            thdn,
                            result.pn(),
                            result.dn()
                        );
                    }
                    if result.pn() == 0 || result.dn() == 0 {
                        break;
                    }
                    if result.pn() >= K_INFINITE_PN_DN || result.dn() >= K_INFINITE_PN_DN {
                        break;
                    }
                    // 次閾値: th = max(curr_th, val*1.7+1) (INF clamp)．val>=1 かつ val≈th なので
                    // 必ず増加 = stall 無し (budget/timeout は loop 冒頭で break する)．
                    thpn = thpn.max((result.pn() as f64 * 1.7) as PnDn + 1).min(cap);
                    thdn = thdn.max((result.dn() as f64 * 1.7) as PnDn + 1).min(cap);
                    if thpn >= cap && thdn >= cap {
                        break;
                    }
                }
                // 末尾: root の結果を TT へ格納し expansion を pop する．
                let root_q = tt.build_query(0, super::position_key(board), attacker_hand, 0);
                tt.set_result(&root_q, result, (0u64, super::tt::entry::NULL_HAND));
                self.expansion_stack.clear();
                result
            }
        };

        // 2 指標を併記: nodes = 探索ノード訪問数 (Emplace 毎に 1),
        // do_moves = do_move 数 (nodes_searched)．混同しないこと．
        let search_do_moves = crate::board::do_move_count();
        eprintln!(
            "[dfpn] root pn={} dn={} mate_len={} nodes={} do_moves={} dm/node={:.2} tt_cap={} gc={} dag={} dom={}",
            last.pn(),
            last.dn(),
            last.len().len(),
            self.nodes,
            search_do_moves,
            search_do_moves as f64 / (self.nodes.max(1)) as f64,
            tt.capacity(),
            tt.gc_count(),
            self.dag_fires,
            self.dom_fires
        );
        super::movegen::mate1ply::report_mate_cand_stats();
        super::movegen::mate1ply::report_mate1ply_stats();
        if std::env::var("DMBREAK").is_ok() {
            let dm = DM_SITE.with(|c| c.get());
            let lq = super::movegen::legal_quick_dm();
            let m1n = crate::board::mate1ply_none_dm();
            let pdm = crate::movegen::pawn_drop_mate_dm();
            let known: u64 = dm.iter().sum::<u64>() + lq + m1n + pdm;
            eprintln!(
                "[dfpn] do_move breakdown: step(再帰)={} dag(DAG)={} lookahead={} proofhand={} is_legal_quick={} mate1ply_None={} pawn_drop_mate(打歩詰検証)={} | known_sum={} other={}",
                dm[0],
                dm[1],
                dm[2],
                dm[3],
                lq,
                m1n,
                pdm,
                known,
                search_do_moves.saturating_sub(known)
            );
        }
        if prof_enabled() {
            let wall = self.start_time.elapsed().as_nanos() as u64;
            let ns = PROF_NS.with(|c| c.get());
            let cnt = PROF_CNT.with(|c| c.get());
            // idx: 0=movegen 1=tt_lookup 2=lookahead 3=dag 4=build_total (umbrella ⊃ 0,1,2 + build_other)．
            let w = wall.max(1) as f64;
            let row = |name: &str, t: u64, c: u64| {
                let avg = if c > 0 { t / c } else { 0 };
                eprintln!(
                    "[dfpn] PROF  {:<32} {:>10.1}us cnt={:<9} avg={:>5}ns {:>5.1}%",
                    name,
                    t as f64 / 1000.0,
                    c,
                    avg,
                    t as f64 / w * 100.0
                );
            };
            // idx: 0=movegen 1=tt_lookup 2=lookahead 3=dag 4=build_total(umbrella)
            //      5=dml_build 6=child_loop(⊃1,2) 7=from_parts(sort)．
            let cl_other = ns[6].saturating_sub(ns[1] + ns[2]); // child-loop per-child overhead (seed/query/path)
                                                                // cl_other 内訳 (idx 8=seed, 9=querysetup, 10=pathcheck, 11=sm_dml); 残り cl_rest は簿記/分岐．
            let cl_rest = cl_other.saturating_sub(ns[8] + ns[9] + ns[10] + ns[11]);
            let glue = wall.saturating_sub(ns[4] + ns[3]); // search/step machinery (do_move/setresult/recursion)
            eprintln!(
                "[dfpn] PROF search_wall={:.1}us (探索のみ; verify/PV 除く)",
                wall as f64 / 1000.0
            );
            row("build_total(node 展開, umbrella)", ns[4], cnt[4]);
            row("  movegen", ns[0], cnt[0]);
            row("  dml_build", ns[5], cnt[5]);
            row("  child_loop(⊃lookup,lookahead)", ns[6], cnt[6]);
            row("    tt_lookup", ns[1], cnt[1]);
            row("    lookahead(mate1ply)", ns[2], cnt[2]);
            row("    cl_other(seed/query/pathcheck)", cl_other, 0);
            row("      seed(eval+init_pn_dn)", ns[8], cnt[8]);
            row("      querysetup(hash+build_query)", ns[9], cnt[9]);
            row("      pathcheck(dom+rep stack)", ns[10], cnt[10]);
            row("      sm_dml(sum_mask+DML skip)", ns[11], cnt[11]);
            row("      cl_rest(簿記/分岐)", cl_rest, 0);
            row("  from_parts(sort/recalc)", ns[7], cnt[7]);
            row("dag(EliminateDoubleCount)", ns[3], cnt[3]);
            row("glue(do_move/setresult/recursion)", glue, 0);
        }

        if last.pn() == 0 {
            // STRICT PV replay: proven 証明木を実際に replay し，OR は proven child を，
            // AND は **全合法防御** を列挙して詰みに帰着するか厳密検証する (TT pn/dn を信用しない)．
            // Some(d) = sound mate-d / None = 偽証明 or 不完全 (budget 枯渇は別表示)．
            let mut path: Vec<u64> = Vec::new();
            let mut memo: std::collections::HashMap<u64, Option<u16>> =
                std::collections::HashMap::new();
            let mut budget: u64 = 80_000_000;
            let verified = self.verify_proof(&mut tt, board, &mut path, &mut memo, &mut budget);
            match verified {
                Some(d) => eprintln!(
                    "[dfpn] STRICT VERIFY Some({}) (root mate_len={}, budget_left={})",
                    d,
                    last.len().len(),
                    budget
                ),
                None if budget == 0 => {
                    eprintln!("[dfpn] STRICT VERIFY INCONCLUSIVE (budget exhausted)")
                }
                None => eprintln!("[dfpn] STRICT VERIFY None — UNSOUND or incomplete proof tree"),
            }
            // PV は memo (検証済距離) を辿って復元 (OR=最短 proven, AND=max-resistance)．
            let pv = self.build_pv(board, &mut tt, &memo, last.len().len() as usize + 8);
            TsumeResult::Checkmate {
                moves: pv,
                nodes_searched: self.nodes,
            }
        } else if last.dn() == 0 {
            TsumeResult::NoCheckmate {
                nodes_searched: self.nodes,
            }
        } else {
            TsumeResult::Unknown {
                nodes_searched: self.nodes,
            }
        }
    }

    /// LocalExpansion の build + push (Emplace)．
    ///
    /// 子展開を「親のループ内」で Emplace し，その**集約済 `CurrentResult`** を first-visit-exceed
    /// 判定に使う (= 子の合法手を全列挙し 1 段深い δ 集約で評価)．flat な seed で評価すると exceed
    /// 判定が緩く over-explore するため，集約済結果で判定する．
    /// `Err(result)` = 終局/budget/深さ上限 (push しない; 親が即時結果に使う)．`Ok(())` = expansion を top へ push．
    fn emplace(
        &mut self,
        tt: &mut TranspositionTable,
        board: &mut Board,
        len: MateLen,
        depth: u32,
        path_key: u64,
        first_search: bool,
        sum_mask: BitSet64,
    ) -> Result<(), SearchResult> {
        self.nodes += 1; // 探索ノード数 (Emplace 毎に 1)．
        if self.nodes >= self.max_nodes || (self.nodes & 0x3FF == 0 && self.is_timed_out()) {
            self.timed_out = self.timed_out || self.is_timed_out();
            // budget 切れは非 final unknown で unwind (反復深化が次で break)．
            return Err(SearchResult::make_first_visit(1, 1, len, 1));
        }
        // 深さ上限到達は千日手扱いで打ち切る．
        if depth as usize >= super::mate_len::KDEPTH_MAX as usize {
            let attacker_hand = board.hand[self.attacker.index()];
            return Err(SearchResult::make_repetition(attacker_hand, len, 1, 0));
        }
        // GC: table-fill による look_up O(cap) 退化を防ぐ (4096 build 毎)．
        if self.nodes & 0xFFF == 0 {
            tt.maybe_collect_garbage();
        }
        let attacker_hand = board.hand[self.attacker.index()];
        let __bt = prof_enabled().then(std::time::Instant::now);
        let built =
            self.build_expansion(tt, board, len, depth, path_key, first_search, sum_mask);
        if let Some(t) = __bt {
            prof_add(4, t.elapsed().as_nanos() as u64);
        }
        let exp = match built {
            Ok(e) => e,
            Err(terminal) => return Err(terminal),
        };
        self.dump_node_diag(&exp, board, depth);
        self.expansion_stack.push(exp);
        let my = self.expansion_stack.len() - 1;
        self.expansion_stack[my].set_key_hand_pair((super::position_key(board), attacker_hand));
        Ok(())
    }

    /// 親ループの 1 子処理 (探索ループ本体)．`my` = 親 expansion の stack index．best child を
    /// DoMove → **Emplace (build)** → first-visit は **子の集約済 CurrentResult** で exceed 判定 →
    /// 非 exceed なら `search_impl` 再帰 → Pop → UndoMove → UpdateBestChild + TT 書込．
    #[allow(clippy::too_many_arguments)]
    fn step_best_child(
        &mut self,
        tt: &mut TranspositionTable,
        board: &mut Board,
        my: usize,
        depth: u32,
        len: MateLen,
        path_key: u64,
        thpn: PnDn,
        thdn: PnDn,
        inc_flag: &mut u32,
    ) {
        let attacker_hand = board.hand[self.attacker.index()];
        let best_move = self.expansion_stack[my].best_move();
        let is_first = self.expansion_stack[my].front_is_first_visit();
        let (cthpn, cthdn) = self.expansion_stack[my].front_pn_dn_thresholds(thpn, thdn);
        if inc_active() {
            inc_log(&format!(
                "INC STEP d={} best={} first={} inc={} cth=({},{})",
                depth,
                best_move.to_usi(),
                is_first as i32,
                *inc_flag,
                cthpn,
                cthdn
            ));
        }
        if thx_prefix()
            .map(|p| board.sfen().starts_with(p.as_str()))
            .unwrap_or(false)
        {
            eprintln!("THX {}", self.expansion_stack[my].thx_breakdown(thpn, thdn));
        }
        let best_raw = self.expansion_stack[my].front_raw();
        let child_query = self.expansion_stack[my].query_at(best_raw);
        // best child へ伝播する sum_mask (δ 集約方式の親→子伝播)．
        let child_sum_mask = self.expansion_stack[my].front_sum_mask();

        let captured = board.do_move(best_move);
        dm_bump(0); // step_best_child 再帰
        let child_pk = path_key_after(path_key, best_move, depth as usize);
        let child_result = match self.emplace(
            tt,
            board,
            len.sub(1),
            depth + 1,
            child_pk,
            is_first,
            child_sum_mask,
        ) {
            // 終局/budget/深さ上限: push されていない (Pop 不要)．
            // terminal を Err で返し push を省くが，**inc_flag 会計は Ok ブランチの DEC と対称に保つ**
            // 必要がある．これを欠くと terminal first-visit 子で inc_flag が減らず TCA threshold を
            // 過剰拡張し，OR node で次手を過剰展開してしまう．
            Err(terminal) => {
                if is_first && *inc_flag > 0 {
                    *inc_flag -= 1;
                    inc_log(&format!(
                        "INC DEC d={} best={} -> inc={} (first_visit/terminal)",
                        depth,
                        best_move.to_usi(),
                        *inc_flag
                    ));
                }
                terminal
            }
            Ok(()) => {
                let cidx = self.expansion_stack.len() - 1;
                let r = if is_first {
                    if *inc_flag > 0 {
                        *inc_flag -= 1;
                        inc_log(&format!(
                            "INC DEC d={} best={} -> inc={} (first_visit)",
                            depth,
                            best_move.to_usi(),
                            *inc_flag
                        ));
                    }
                    // 子 expansion の集約済 CurrentResult で exceed 判定 (flat seed でない)．
                    let cur = self.expansion_stack[cidx].current_result(board, (depth + 1) as i32);
                    if cur.is_final() || cur.pn() >= cthpn || cur.dn() >= cthdn {
                        cur
                    } else {
                        self.search_impl(
                            tt,
                            board,
                            cthpn,
                            cthdn,
                            len.sub(1),
                            depth + 1,
                            child_pk,
                            inc_flag,
                        )
                    }
                } else {
                    self.search_impl(
                        tt,
                        board,
                        cthpn,
                        cthdn,
                        len.sub(1),
                        depth + 1,
                        child_pk,
                        inc_flag,
                    )
                };
                // 子探索終了時に子 expansion を pop する．末尾 1 要素 (index cidx == len-1) を除去
                // し，pop した子 expansion の 6 Vec を pool へ返却して次 build で再利用する
                // (per-node アロケーション削減; 探索不変)．
                if let Some(child_exp) = self.expansion_stack.pop() {
                    debug_assert_eq!(self.expansion_stack.len(), cidx);
                    let (bm, be, bq, br, bi, bd) = child_exp.into_buffers();
                    self.expansion_buf_pool.release(bm, be, bq, br, bi, bd);
                }
                r
            }
        };

        board.undo_move(best_move, captured);
        self.expansion_stack[my].update_best_child(child_result);
        // 子結果を TT へ格納: 親 board_key は position-only (cross-hand のため hand 別管理)．
        tt.set_result(
            &child_query,
            child_result,
            (super::position_key(board), attacker_hand),
        );
    }

    /// 1 ノードの探索本体．**呼出し前に親が `emplace` で expansion を push 済** (top の expansion を
    /// 使う)．自分では build も Pop もしない (親が Pop する)．
    fn search_impl(
        &mut self,
        tt: &mut TranspositionTable,
        board: &mut Board,
        mut thpn: PnDn,
        mut thdn: PnDn,
        len: MateLen,
        depth: u32,
        path_key: u64,
        inc_flag: &mut u32,
    ) -> SearchResult {
        let my = self.expansion_stack.len() - 1;
        // INC: inc_flag origin window (opener が active 化; nested は素通り)．
        let inc_opened = inc_open_window(board);
        if inc_active() {
            inc_log(&format!(
                "INC ENTER d={} inc_in={} dhoc={} th=({},{}) sfen={}",
                depth,
                *inc_flag,
                self.expansion_stack[my].does_have_old_child() as i32,
                thpn,
                thdn,
                board.sfen()
            ));
        }
        // TH: 受領 thpn/thdn dump (threshold 追跡用診断)．
        if let Some(prefix) = th_prefix() {
            let sfen = board.sfen();
            if sfen.starts_with(prefix.as_str()) {
                eprintln!(
                    "TH depth={} thpn={} thdn={} sfen={}",
                    depth, thpn, thdn, sfen
                );
            }
        }
        // DAG 二重カウント抑止 (EliminateDoubleCount)．
        if !no_dag() {
            let __dag = prof_enabled().then(std::time::Instant::now);
            self.eliminate_double_count(tt, board, my, depth);
            if let Some(t) = __dag {
                prof_add(3, t.elapsed().as_nanos() as u64);
            }
        }

        // THX: 探索ループの per-iteration dump (inc_flag/threshold 追跡用診断)．
        let khthx_here = thx_prefix()
            .map(|p| board.sfen().starts_with(p.as_str()))
            .unwrap_or(false);
        if khthx_here {
            eprintln!("THX enter depth={} th=({},{})", depth, thpn, thdn);
        }

        let orig_thpn = thpn;
        let orig_thdn = thdn;
        let orig_inc = *inc_flag;

        let mut curr = self.expansion_stack[my].current_result(board, depth as i32);
        if curr.is_final() {
            if inc_active() {
                inc_log(&format!(
                    "INC EXIT-curfinal d={} inc={}",
                    depth, *inc_flag
                ));
            }
            inc_close_window(inc_opened);
            return curr; // 親 (step_best_child) が Pop する．
        }
        if self.expansion_stack[my].does_have_old_child() {
            *inc_flag += 1;
            inc_log(&format!(
                "INC INC d={} -> inc={} (dhoc)",
                depth, *inc_flag
            ));
        }
        if *inc_flag > 0 {
            extend_search_threshold(curr, &mut thpn, &mut thdn);
        }

        self.path_depths.insert(board.hash, depth);
        if self.param_path_dominance {
            let attacker_hand = board.hand[self.attacker.index()];
            self.dom_path
                .push(super::position_key(board), attacker_hand, depth);
        }

        while curr.pn() < thpn && curr.dn() < thdn {
            if self.nodes >= self.max_nodes || self.timed_out {
                break;
            }
            if khthx_here {
                eprintln!(
                    "THX best={} first={}",
                    self.expansion_stack[my].best_move().to_usi(),
                    self.expansion_stack[my].front_is_first_visit() as i32
                );
            }
            self.step_best_child(tt, board, my, depth, len, path_key, thpn, thdn, inc_flag);
            curr = self.expansion_stack[my].current_result(board, depth as i32);
            if khthx_here {
                eprintln!(
                    "THXR -> curr pn={} dn={} inc={} (th={},{})",
                    curr.pn(),
                    curr.dn(),
                    *inc_flag,
                    thpn,
                    thdn
                );
            }

            thpn = orig_thpn;
            thdn = orig_thdn;
            if *inc_flag > 0 {
                extend_search_threshold(curr, &mut thpn, &mut thdn);
            } else if *inc_flag == 0 && orig_inc > 0 {
                if khthx_here {
                    eprintln!("THX break (inc==0 && orig_inc>0)");
                }
                break;
            }
        }
        if khthx_here {
            eprintln!(
                "THX exit depth={} curr=(pn{},dn{}) inc={}",
                depth,
                curr.pn(),
                curr.dn(),
                *inc_flag
            );
        }

        self.path_depths.remove(&board.hash);
        if self.param_path_dominance {
            self.dom_path.pop(super::position_key(board));
        }
        let pre_clamp = *inc_flag;
        *inc_flag = (*inc_flag).min(orig_inc);
        if inc_active() {
            inc_log(&format!(
                "INC EXIT d={} pre={} orig={} -> inc={} curr=(pn{},dn{})",
                depth,
                pre_clamp,
                orig_inc,
                *inc_flag,
                curr.pn(),
                curr.dn()
            ));
        }
        inc_close_window(inc_opened);
        curr
    }

    /// root ノードの探索本体．
    ///
    /// root expansion は **solve_impl が一度だけ構築**し `expansion_stack[0]` に持続させる (Emplace を
    /// 1 回だけ呼び閾値反復間で再利用する)．本関数は毎反復その持続 expansion を再利用し，子だけを
    /// [`search_impl`] で展開する．[`search_impl`] との差: inc_flag は local (親 inc_flag 無し)，
    /// 末尾の `orig_inc break` / `inc_flag = min(...)` は無い．
    fn search_impl_root(
        &mut self,
        tt: &mut TranspositionTable,
        board: &mut Board,
        mut thpn: PnDn,
        mut thdn: PnDn,
        len: MateLen,
    ) -> SearchResult {
        let attacker_hand = board.hand[self.attacker.index()];
        let orig_thpn = thpn;
        let orig_thdn = thdn;
        let mut inc_flag = 0u32;

        if !no_dag() {
            self.eliminate_double_count(tt, board, 0, 0);
        }
        let mut curr = self.expansion_stack[0].current_result(board, 0);
        if self.expansion_stack[0].does_have_old_child() {
            inc_flag += 1;
        }
        if inc_flag > 0 {
            extend_search_threshold(curr, &mut thpn, &mut thdn);
        }

        // 子探索中 root を path に積む (子が repetition/inferior 判定で参照)．
        self.path_depths.insert(board.hash, 0);
        if self.param_path_dominance {
            self.dom_path
                .push(super::position_key(board), attacker_hand, 0);
        }

        while curr.pn() < thpn && curr.dn() < thdn {
            if self.nodes >= self.max_nodes || self.timed_out {
                break;
            }
            self.step_best_child(tt, board, 0, 0, len, 0u64, thpn, thdn, &mut inc_flag);
            curr = self.expansion_stack[0].current_result(board, 0);

            thpn = orig_thpn;
            thdn = orig_thdn;
            if inc_flag > 0 {
                extend_search_threshold(curr, &mut thpn, &mut thdn);
            }
            // NOTE: root には [`search_impl`] の `else if inc_flag==0 && orig_inc>0 break` は無い
            // (root は親 inc_flag を持たないため)．末尾の min も無い．
        }

        self.path_depths.remove(&board.hash);
        if self.param_path_dominance {
            self.dom_path.pop(super::position_key(board));
        }
        curr
    }

    /// TRACE / NODE / SEL の chronological / 子リスト診断 dump (env-gated)．search_impl と
    /// 持続 root build の双方から呼ぶ．
    fn dump_node_diag(&self, exp: &LocalExpansion, board: &Board, depth: u32) {
        let oc = if board.turn == self.attacker { 1 } else { 0 };
        if trace_enabled() {
            let cnt = TRACE_CNT.with(|c| {
                let v = c.get() + 1;
                c.set(v);
                v
            });
            if cnt <= 20000 {
                if let Some((m, pn, dn, _ev, _am)) = exp.trace_children().first() {
                    eprintln!(
                        "TRACE {} ply={} or={} best={} bpn={} bdn={} sfen={}",
                        cnt,
                        depth,
                        oc,
                        m.to_usi(),
                        pn,
                        dn,
                        board.sfen()
                    );
                }
            }
        }
        if let Some(prefix) = node_prefix() {
            let sfen = board.sfen();
            if sfen.starts_with(prefix.as_str()) {
                eprintln!("NODE ply={} or={} sfen={}", depth, oc, sfen);
                for (k, (m, pn, dn, ev, am)) in exp.trace_children().iter().enumerate() {
                    eprintln!(
                        "NODE   {} {} pn={} dn={} ev={} am={}",
                        k,
                        m.to_usi(),
                        pn,
                        dn,
                        ev,
                        am
                    );
                }
            }
        }
        if sel_enabled() && depth <= 7 {
            let bit = 1u8 << depth;
            let already = SEL_DUMPED.with(|c| {
                let v = c.get();
                c.set(v | bit);
                v & bit != 0
            });
            if !already {
                eprintln!("SEL ply={} or={} sfen={}", depth, oc, board.sfen());
                for (k, (m, pn, dn, ev, am)) in exp.trace_children().iter().enumerate() {
                    eprintln!(
                        "SEL   {} {} pn={} dn={} ev={} am={}",
                        k,
                        m.to_usi(),
                        pn,
                        dn,
                        ev,
                        am
                    );
                }
            }
        }
    }

    /// `LocalExpansion` の構築: movegen + 各子 seed (TT LookUp) + 千日手判定．
    /// `Err(result)` = 終局 (合法手なし) の即時結果．`Ok(exp)` = 展開済ノード．
    fn build_expansion(
        &mut self,
        tt: &mut TranspositionTable,
        board: &mut Board,
        len: MateLen,
        depth: u32,
        path_key: u64,
        first_search: bool,
        sum_mask: BitSet64,
    ) -> Result<LocalExpansion, SearchResult> {
        let attacker = self.attacker;
        let or_node = board.turn == attacker;

        let mut moves = self.expansion_buf_pool.take_moves();
        let __mg = prof_enabled().then(std::time::Instant::now);
        if or_node {
            self.check_moves_into(board, &mut moves);
        } else {
            let av = self.generate_defense_moves_inner(board, false);
            moves.extend_from_slice(av.as_slice());
        }
        if let Some(t) = __mg {
            prof_add(0, t.elapsed().as_nanos() as u64);
        }

        if moves.is_empty() {
            // OR (王手なし) = 不詰 disproven / AND (受けなし=詰み) = proven (詰み完了 len 0)．
            // 終端 hand: OR=disproof hand (remove_if(MAX)) / AND=proof hand (add_if(空))．
            let attacker_hand = board.hand[attacker.index()];
            let use_handset = handset_enabled();
            let r = if or_node {
                let hand = if use_handset {
                    super::proof_hand::disproof_hand_terminal_or(board)
                } else {
                    attacker_hand
                };
                SearchResult::make_final(false, hand, DEPTH_MAX_MATE_LEN, 1)
            } else {
                let hand = if use_handset {
                    super::proof_hand::proof_hand_terminal_and(board)
                } else {
                    attacker_hand
                };
                SearchResult::make_final(true, hand, ZERO_MATE_LEN, 1)
            };
            // terminal node では child loop の 5 本は未取得．moves のみ pool へ返却する．
            self.expansion_buf_pool.release_moves(moves);
            return Err(r);
        }

        // 生成手を生成順へ並べ替える (これがソート入力)．DML/results/idx すべてこの順で構築する．
        // promo は成→不成の順．
        if or_node {
            // OR-node 王手順 (開き王手候補 → 直接王手 → 駒打ち)．
            // discoverers (= blockers_for_king(them) & pieces(us)) は build 毎に 1 回算出して共有．
            let def_king = board.king_square(attacker.opponent());
            let discoverers = def_king
                .map(|k| board.compute_discoverers(attacker, k))
                .unwrap_or(crate::bitboard::Bitboard::EMPTY);
            moves.sort_by_key(|&m| check_order_key(m, discoverers, def_king));
        } else {
            moves.sort_by_key(|&m| evasion_order_key(board, m));
        }

        // RAW: build 時点の raw movegen 順 (sort/DML 前) を ply 0-7 初出で dump する診断用．
        if sel_enabled() && depth <= 7 {
            let bit = 1u8 << depth;
            let already = RAW_DUMPED.with(|c| {
                let v = c.get();
                c.set(v | bit);
                v & bit != 0
            });
            if !already {
                let oc = if or_node { 1 } else { 0 };
                let raw: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                eprintln!("RAW ply={} or={} raw={}", depth, oc, raw.join(" "));
            }
        }

        // DelayedMoveList: 同マス合駒/成不成ペアを双方向 chain 化し，prev が未 final の手は idx から
        // 除外 (= 後回し) する．is_delayable 判定のため移動元駒種 (raw ID) を渡す (駒打ちは 0)．
        let __dml_t = prof_enabled().then(std::time::Instant::now);
        let dml = {
            let raw_pts: Vec<u8> = moves
                .iter()
                .map(|&m| {
                    if m.is_drop() {
                        0
                    } else {
                        board.piece_at(m.from_sq()) & 0x0F
                    }
                })
                .collect();
            let us_black = board.turn == crate::types::Color::Black;
            if !or_node {
                // 中合い対称性: AND node の drop で「攻方支援なし & 逆王手でない」もの同士は別マスでも
                // 同一 chain とみなし後回しにする．
                let defender = board.turn;
                let atk_king = board.king_square(attacker);
                let interp_chain: Vec<bool> = moves
                    .iter()
                    .map(|&m| {
                        if !m.is_drop() {
                            return false;
                        }
                        let to = m.to_sq();
                        let unsupported = !board.is_attacked_by(to, defender);
                        let no_check = match (atk_king, m.drop_piece_type()) {
                            (Some(k), Some(pt)) => !drop_gives_check(board, to, pt, defender, k),
                            _ => true,
                        };
                        unsupported && no_check
                    })
                    .collect();
                DelayedMoveList::build_with_types_interp(
                    &moves,
                    or_node,
                    &raw_pts,
                    us_black,
                    &interp_chain,
                )
            } else {
                DelayedMoveList::build_with_types(&moves, or_node, &raw_pts, us_black)
            }
        };
        if let Some(t) = __dml_t {
            prof_add(5, t.elapsed().as_nanos() as u64);
        }

        let __cl_t = prof_enabled().then(std::time::Instant::now);
        let defender_king = board.king_square(attacker.opponent());
        let n = moves.len();
        // pool から取得 (clear 済; 容量は前 node から保持)．n 分予約して最初の数 node 以降は realloc 無し．
        let mut evals = self.expansion_buf_pool.take_evals();
        let mut results = self.expansion_buf_pool.take_results();
        let mut queries = self.expansion_buf_pool.take_queries();
        let mut idx = self.expansion_buf_pool.take_idx();
        evals.reserve(n);
        results.reserve(n);
        queries.reserve(n);
        idx.reserve(n);
        let mut does_have_old = false;
        // 親から渡された sum_mask を起点に，非 final 子のうち sum 集約に適さない手 (is_sum_delta_node が
        // false) / δ が `K_FORCE_SUM_PN_DN` 以上の手を max 集約へ落とす．
        let mut cur_sum_mask = sum_mask;

        for (i, &m) in moves.iter().enumerate() {
            // eval / seed は親局面で計算 (move_brief_eval / init_pn_dn)．
            // [PROF idx8=seed] cl_other 内訳: move_brief_eval + init_pn_dn．
            let __seed_t = prof_enabled().then(std::time::Instant::now);
            let eval = match defender_king {
                Some(k) => super::move_brief_eval(m, k, board),
                None => 0,
            };
            let (sp, sd) = if or_node {
                super::init_pn_dn_or(board, m, attacker)
            } else {
                super::init_pn_dn_and(board, m, attacker)
            };
            let seed = ((sp as u64 / DIV).max(1), (sd as u64 / DIV).max(1));
            if let Some(t) = __seed_t {
                prof_add(8, t.elapsed().as_nanos() as u64);
            }

            // 子 key/hand を **do_move せず** incremental に算出する (per-child do_move を回避し
            // per-node time を節約)．do_move は子局面が必須な look-ahead でのみ行う．**TT board_key は
            // position-only** (`board_hash_after`)．hand は entry に別管理し cross-hand を効かせる．
            // 千日手判定だけ full hash．
            // [PROF idx9=querysetup] cl_other 内訳: hashes_after(2 hash) + hand_after + path_key_after
            // + build_query + prefetch．
            let __qs_t = prof_enabled().then(std::time::Instant::now);
            let (ch_full, ch_pos) = board.hashes_after(m);
            let child_hand = board.hand_after(m, attacker);
            let child_pk = path_key_after(path_key, m, depth as usize);
            let q = tt.build_query(child_pk, ch_pos, child_hand, (depth + 1) as i32);
            // memory-bound な TT cluster fetch を投機 prefetch し，直後の dom_path/path_depths lookup と
            // DRAM latency を重ねる (search-invariant な hint; look_up が skip されても無害)．
            tt.prefetch(&q);
            if let Some(t) = __qs_t {
                prof_add(9, t.elapsed().as_nanos() as u64);
            }

            // path 上に同一 board_key かつ攻め方持駒が child 以上 (= child が劣位) の祖先があれば反復
            // として刈る．tsume では (position_key, attacker_hand) が全局面を一意決定するため，これは
            // exact 千日手 (`path_depths`) の持駒 superset 方向への一般化 (exact は hand 等値の特殊ケース)．
            // **TT LookUp より前に** 刈ることで高コストな劣位部分木の展開を未然に防ぐ (node 削減の本体)．
            // [PROF idx10=pathcheck] cl_other 内訳: dominance find_dominator + 千日手 path_depths.get
            // (両方 path stack の O(depth) 逆順走査)．path_depths.get は dom miss 時のみ評価で探索不変．
            let __pc_t = prof_enabled().then(std::time::Instant::now);
            let dom_depth = if self.param_path_dominance {
                self.dom_path.find_dominator(ch_pos, &child_hand)
            } else {
                None
            };
            let rep_ply = if dom_depth.is_none() {
                self.path_depths.get(&ch_full).copied()
            } else {
                None
            };
            if let Some(t) = __pc_t {
                prof_add(10, t.elapsed().as_nanos() as u64);
            }
            let mut r = if let Some(anc_depth) = dom_depth {
                self.dom_fires += 1;
                SearchResult::make_repetition(child_hand, len, 1, anc_depth as i32)
            } else if let Some(anc_ply) = rep_ply {
                // path 上の同一局面 = 千日手．
                SearchResult::make_repetition(child_hand, len, 1, anc_ply as i32)
            } else {
                let mut dhoc = false;
                let __lu = prof_enabled().then(std::time::Instant::now);
                let res = tt.look_up(&q, len.sub(1), &mut dhoc, || seed);
                if let Some(t) = __lu {
                    prof_add(1, t.elapsed().as_nanos() as u64);
                }
                does_have_old = does_have_old || dhoc;
                res
            };
            // 非 final 子の δ 集約方式を決める．似た子局面で過小評価を招く手 (`!is_sum_delta_node`) と
            // 既に δ が巨大な子 (`>= K_FORCE_SUM_PN_DN`) は max 集約へ落とす (sum_mask bit を reset)．
            // look-ahead で final 化する前の seed の δ で判定する．
            // [PROF idx11=sm_dml] cl_rest 内訳: sum_mask reset (is_sum_delta_node) + DML skip 判定．
            let __sd_t = prof_enabled().then(std::time::Instant::now);
            if !no_sum_mask_reset()
                && !r.is_final()
                && (!is_sum_delta_node(board, m, or_node) || r.delta(or_node) >= K_FORCE_SUM_PN_DN)
            {
                cur_sum_mask.reset(i);
            }
            // DML skip 判定: 非 final かつ prev chain に未 final の先行手があれば後回し (idx に積まない)．
            // prev は i より前に処理済なので results が揃っている．
            let is_skipped = !r.is_final() && dml.has_unresolved_prev(i, |j| results[j].is_final());
            if let Some(t) = __sd_t {
                prof_add(11, t.elapsed().as_nanos() as u64);
            }

            // 1 手詰先読み (check_obvious_final_or_node)．non-skipped のみ．
            // AND node の first-visit child は子が OR node (攻め方手番)．board は do_move 済 (= 子局面)
            // なので攻め方の 1 手詰／詰み無を先読みし，proven/disproven を seed する．これにより
            // 「詰む応手」を展開せず除外，「逃れる応手」を即 disproof し breadth を抑える．
            // [diag] SEED: 指定 sfen prefix の親ノードで各子の look_up 結果 (TT/cross-hand) と
            // look-ahead 前後を dump し，disproof の出所 (cross-hand TT vs fresh look-ahead) を特定する．
            // SEED は OnceLock で 1 回読み (per-child の env::var lock+alloc を回避)．未設定時は
            // sfen() も呼ばず false (production hot path で実質ゼロコスト)．
            let seed_diag = seed_prefix()
                .as_ref()
                .map(|p| board.sfen().starts_with(p.as_str()))
                .unwrap_or(false);
            let pre_lu = (r.pn(), r.dn(), r.is_final(), r.is_first_visit());

            // 子結果が final になったら TT へ格納する (PV/伝播の整合)．
            if !is_skipped && !or_node && first_search && r.is_first_visit() {
                // 子局面 (do_move 済) が必要なのはここだけ (look-ahead 内で DoMove/UndoMove する)．
                // [PROF idx2=lookahead] do_move+undo+set_result 込で計時．
                let __la = prof_enabled().then(std::time::Instant::now);
                let captured = board.do_move(m);
                dm_bump(2); // look-ahead (AND node の first-visit 子へ do_move)
                let __obv = self.check_obvious_final_or_node(board);
                if let Some(res) = __obv {
                    tt.set_result(&q, res, (ch_pos, child_hand));
                    r = res;
                }
                board.undo_move(m, captured);
                if let Some(t) = __la {
                    prof_add(2, t.elapsed().as_nanos() as u64);
                }
            }
            if seed_diag {
                eprintln!(
                    "SEED child={} skip={} seed=({},{}) lookup=(pn{} dn{} fin{} fv{}) final=(pn{} dn{} fin{})",
                    m.to_usi(),
                    is_skipped,
                    seed.0,
                    seed.1,
                    pre_lu.0,
                    pre_lu.1,
                    pre_lu.2,
                    pre_lu.3,
                    r.pn(),
                    r.dn(),
                    r.is_final()
                );
            }

            evals.push(eval);
            results.push(r);
            queries.push(q);
            if !is_skipped {
                idx.push(i as u32);
            }
        }
        if let Some(t) = __cl_t {
            prof_add(6, t.elapsed().as_nanos() as u64);
        }
        let __fp_t = prof_enabled().then(std::time::Instant::now);

        // revival 用 next chain (DelayedMoveList.next)．後回し手が final 化したら
        // update_best_child が dml_next を辿って idx へ復活させる．
        let mut dml_next = self.expansion_buf_pool.take_dml();
        dml_next.resize(n, -1);
        for (i, slot) in dml_next.iter_mut().enumerate() {
            if let Some(nx) = dml.next(i) {
                *slot = nx as i32;
            }
        }

        let __exp = LocalExpansion::from_parts(
            or_node,
            len,
            moves,
            evals,
            queries,
            results,
            idx,
            cur_sum_mask,
            does_have_old,
            dml_next,
            1,
        );
        if let Some(t) = __fp_t {
            prof_add(7, t.elapsed().as_nanos() as u64);
        }
        Ok(__exp)
    }

    /// `board` は子 OR node (攻め方手番) へ do_move 済前提．末端の固定深さ探索により
    /// 自明な詰み (1 手詰) / 不詰 (王手手段なし) を**展開せず**検知する:
    /// - 王手手段なし (`!does_have_mate_possibility`) → disproven (不詰確定)．
    /// - 1 手詰あり → proven (mate-1)．
    ///
    /// proof/disproof hand は **full attacker_hand** (= 子 OR node の手番側持駒) を使う．hand-set
    /// 極小化は cross-hand TT 有効時に偽証明 (mate-39) を生むため不使用 (sound 優先)．
    fn check_obvious_final_or_node(&self, board: &mut Board) -> Option<SearchResult> {
        let or_hand = board.hand[board.turn.index()];
        let use_handset = handset_enabled();
        // 不詰判定に `does_have_mate_possibility` (blocker 無視の over-approx) を使う．exact
        // `!has_checks` は blocker で塞がれた王手候補を即不詰断定してしまい，詰みの可能性が残る局面を
        // 早期に disproof してしまう．over-approx はそうした局面を defer する (sound)．
        //
        // **安価な不詰判定を先に評価**し，詰みの可能性がある場合のみ高コストな 1 手詰スキャン
        // (`mate1ply_with_cached_checks` = 王手生成 + 各王手の mate scan) を行う．
        // `does_have_mate_possibility` が false のとき王手手段が無く scan も必ず None を返すため
        // (no_mate 時の scan は無駄)，この遅延は探索不変 (search-invariant)．これにより不詰
        // (disproof) 経路で 1 手詰スキャンを完全に省略できる．
        let (no_mate, mm_opt) = if dhmp_enabled() {
            if !board.does_have_mate_possibility(board.turn) {
                (true, None)
            } else if super::movegen::mate1ply::mate1ply_enabled() {
                // mate_1ply 忠実列挙 (full movegen 回避)．
                (false, self.mate1ply(board))
            } else if near2_enabled() {
                // 距離 ≤2 候補のみ検査 (node 不変; 遠方候補の verify/do_move fallback を省く)．
                (false, self.mate1ply_cached_near2(board))
            } else {
                (false, self.mate1ply_with_cached_checks(board).0)
            }
        } else {
            let (mm, has_checks) = self.mate1ply_with_cached_checks(board);
            (!has_checks, mm)
        };
        if no_mate {
            // 攻め方に王手手段なし → 詰み不可能 → 不詰 (final<false>, kDepthMaxMateLen)．
            // disproof hand = remove_if(MAX)．
            let hand = if use_handset {
                super::proof_hand::disproof_hand_terminal_or(board)
            } else {
                or_hand
            };
            Some(SearchResult::make_final(false, hand, DEPTH_MAX_MATE_LEN, 1))
        } else if let Some(mate_move) = mm_opt {
            // 1 手詰 → 詰み proven (mate-1)．
            // proof hand = before_hand(mate_move, 詰み局面の proof hand)．
            let hand = if use_handset {
                let cap = board.do_move(mate_move);
                dm_bump(3); // 1手詰 proof-hand 計算 (handset 時のみ)
                let proof_after = super::proof_hand::proof_hand_terminal_and(board);
                board.undo_move(mate_move, cap);
                super::proof_hand::before_hand(board, mate_move, proof_after)
            } else {
                or_hand
            };
            // [diag] MATE: look-ahead 1 手詰の手と proof hand を dump (HAND prefix gate)．
            if let Some(prefix) = hand_prefix() {
                if board.sfen().starts_with(prefix.as_str()) {
                    eprintln!(
                        "MATE mate1ply={} proof P{} L{} N{} S{} G{} B{} R{} sfen={}",
                        mate_move.to_usi(),
                        hand[0],
                        hand[1],
                        hand[2],
                        hand[3],
                        hand[4],
                        hand[5],
                        hand[6],
                        board.sfen()
                    );
                }
            }
            Some(SearchResult::make_final(
                true,
                hand,
                MateLen::from_len(1),
                1,
            ))
        } else {
            None
        }
    }

    /// 本ノード `my` の best_move が TT 親鎖を遡って祖先へ合流する (DAG) なら，その分岐元の
    /// 合流子を sum→max 集約へ落とし pn/dn の二重カウントを抑止する．
    fn eliminate_double_count(
        &mut self,
        tt: &TranspositionTable,
        board: &mut Board,
        my: usize,
        depth: u32,
    ) {
        if self.expansion_stack[my].empty() {
            return;
        }
        let best_move = self.expansion_stack[my].best_move();
        let current_key_hand = self.expansion_stack[my].key_hand_pair();
        let or_node = self.expansion_stack[my].is_or_node();
        // best_move 後の子 (board_key, attacker_hand) を **incremental に算出** (do_move 不要)．
        // `board_hash_after`==do_move 後の `position_key`, `hand_after`==do_move 後の攻め方持駒 で
        // 完全一致するため探索は不変 (do_moves のみ削減)．
        let child_key_hand = (
            board.board_hash_after(best_move),
            board.hand_after(best_move, self.attacker),
        );

        let edge = match self.find_known_ancestor(tt, current_key_hand, child_key_hand, or_node, depth, my) {
            Some(e) => e,
            None => return,
        };
        self.dag_fires += 1;
        // 分岐元から下流側 (= my-1 .. 0) を辿り resolve する．
        let mut i = my;
        while i > 0 {
            i -= 1;
            if self.expansion_stack[i].resolve_double_count_if_branch_root(edge) {
                break;
            }
            if self.expansion_stack[i].should_stop_ancestor_search(edge.branch_root_is_or_node) {
                break;
            }
        }
    }

    /// `child_key_hand` を起点に TT 親鎖を遡り，path 上の祖先へ合流したら分岐元の辺を返す
    /// (exact-hand 簡約版)．
    fn find_known_ancestor(
        &self,
        tt: &TranspositionTable,
        current_key_hand: (u64, Hand),
        child_key_hand: (u64, Hand),
        or_node_current: bool,
        depth: u32,
        my: usize,
    ) -> Option<BranchRootEdge> {
        let mut key_hand = child_key_hand;
        let mut last_pn = K_INFINITE_PN_DN;
        let mut last_dn = K_INFINITE_PN_DN;
        let mut pn_flag = true;
        let mut dn_flag = true;
        let mut or_node = or_node_current;
        let mut i = 0u32;
        while i < depth && (pn_flag || dn_flag) {
            let (pbk, ph, pn, dn) = match tt.look_up_parent(key_hand.0, key_hand.1) {
                Some(v) => v,
                None => break,
            };
            let parent_key_hand = (pbk, ph);
            // 初回の親が現局面なら二重カウントの疑い無し．
            if i == 0 && parent_key_hand == current_key_hand {
                break;
            }
            if dn > last_dn.saturating_add(K_ANCESTOR_SEARCH_THRESHOLD) {
                dn_flag = false;
            }
            if pn > last_pn.saturating_add(K_ANCESTOR_SEARCH_THRESHOLD) {
                pn_flag = false;
            }
            if self.contains_in_path(parent_key_hand, my) {
                if (or_node && dn_flag) || (!or_node && pn_flag) {
                    return Some(BranchRootEdge {
                        branch_root: parent_key_hand,
                        child: key_hand,
                        branch_root_is_or_node: or_node,
                    });
                } else {
                    break;
                }
            }
            key_hand = parent_key_hand;
            last_pn = pn;
            last_dn = dn;
            or_node = !or_node;
            i += 1;
        }
        None
    }

    /// (board_key, hand) が現探索 path の祖先 (`expansion_stack[0..my]`) に在るか．
    fn contains_in_path(&self, kh: (u64, Hand), my: usize) -> bool {
        self.expansion_stack[..my].iter().any(|e| e.key_hand_pair() == kh)
    }

}
