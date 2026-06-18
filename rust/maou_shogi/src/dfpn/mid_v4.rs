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
use super::kh_local_expansion::{BranchRootEdge, LocalExpansion, K_ANCESTOR_SEARCH_THRESHOLD};
use super::mate_len::{MateLen, DEPTH_MAX_MATE_LEN, ZERO_MATE_LEN};
use super::path_key::path_key_after;
use super::search_result::{
    extend_search_threshold, BitSet64, Hand, PnDn, SearchResult, K_INFINITE_PN_DN,
};
use super::solver::{DfPnSolver, TsumeResult};
use super::tt_v4::TranspositionTable;
use crate::board::Board;

/// init_pn_dn (unit-16) を KH `kPnDnUnit=2` へ縮約する除数 (mid_v3 `PN_UNIT_SCALE` と同値)．
const DIV: u64 = 8;

/// KH `detail::kForceSumPnDn` (local_expansion.hpp:35)．δ がこの値以上の子は build 時に
/// sum→max 集約へ落とす (kh_local_expansion.rs と同値)．
const K_FORCE_SUM_PN_DN: PnDn = K_INFINITE_PN_DN / 1024;

/// KH `IsSumDeltaNode` (initial_estimation.hpp:227)．move のδ値を sum で計上すべきか判定する．
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
    // KH: 黒番は to が rank2/3 (rank0∈{1,2}) かつ玉が to の 1 つ上 (rank 減 = raw-1)．
    //     白番は to が rank7/8 (rank0∈{6,7}) かつ玉が to の 1 つ下 (rank 増 = raw+1)．
    let hit = if board.turn == Color::Black {
        (rank0 == 1 || rank0 == 2) && king == to_i - 1
    } else {
        (rank0 == 6 || rank0 == 7) && king == to_i + 1
    };
    !hit
}

/// `defender` が `to` に駒 `pt` を打つと `atk_king` (攻め方玉) に王手がかかるか (KH `gives_check`)．
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

/// `V4SEL` env (process 内 1 回読み)．KH `KHSEL` と同形式で ply 0-7 初出ノードの sort 済
/// 子リスト (move/pn/dn) を sfen 付きで dump し，KH との guidance 乖離を突合する．
fn v4sel_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V4SEL").is_ok())
}

/// `V4_KHORDER` 実験 (process 内 1 回読み)．完全同点手の tie-break を KH `generateMoves` 順にする．
pub(super) fn khorder_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V4_KHORDER").is_ok())
}

/// `V4_HANDSET` 実験 (process 内 1 回読み)．proof/disproof hand を KH `HandSet` で極小化する
/// (default OFF = full attacker_hand)．cross-hand TT 再利用が KH と一致し count 収束を狙う．
pub(super) fn handset_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V4_HANDSET").is_ok())
}

/// look-ahead 不詰判定を KH `DoesHaveMatePossibility` (blocker 無視 over-approx) に合わせる．
/// **default ON** (`V4_NODHMP` で opt-out)．exact `!has_checks` だと KH が defer する局面 (例
/// cnt=487 Kx7i: 白王手0 だが lance promote 候補で DHMP=true) を maou が即 disproof し探索経路が
/// 乖離する (487→941 まで KH と byte 一致, 29te bundle 13,296→8,587 sound)．[[feedback_kh_divergence_hunting]]．
pub(super) fn dhmp_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V4_NODHMP").is_err())
}

/// `V4_INTROSORT` 実験 (process 内 1 回読み)．idx ソートに libstdc++ introsort 移植 (`kh_std_sort`) を使う．
/// default OFF = stable sort (movegen 順保持)．KH を `std::stable_sort` にした診断ビルドと突合する際は
/// 両者 stable に揃えるため本フラグを **OFF** のままにする．
pub(super) fn introsort_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V4_INTROSORT").is_ok())
}

/// `V4TRACE` env (process 内 1 回読み)．KH `KHTRACE` と同形式で各 build の best 子 (idx[0]) の
/// move/pn/dn を sfen 付き chronological dump し，最初の探索乖離を突合する ([[feedback_kh_divergence_hunting]])．
fn v4trace_enabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V4TRACE").is_ok())
}

/// `V4NODE` env: 指定 sfen prefix に一致するノードの sort 済子リストを dump する診断用 (process 内 1 回読み)．
fn v4node_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("V4NODE").ok().filter(|s| !s.is_empty()))
        .clone()
}

/// `V4TH` env: 指定 sfen prefix に一致するノードが受け取った thpn/thdn を dump (KH `KHTH` と同形式)．
fn v4th_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("V4TH").ok().filter(|s| !s.is_empty()))
        .clone()
}

/// `V4THX` env: 指定 sfen prefix に一致するノードの SearchImpl ループを per-iteration dump
/// (KH `KHTHX`/`KHTHXR` と同形式; inc_flag/threshold/best/curr の乖離 hunting)．
fn v4thx_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("V4THX").ok().filter(|s| !s.is_empty()))
        .clone()
}

/// `V4HAND` env: 指定 sfen prefix のノードの final 結果 (proof/disproof hand) を dump (KH `KHHAND` と突合)．
pub(super) fn v4hand_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("V4HAND").ok().filter(|s| !s.is_empty()))
        .clone()
}

/// `V4INC` env: 指定 sfen prefix のノードを起点に active-window を開き，window 内の SearchImpl
/// ENTER / INC(does_have_old_child) / DEC(first_visit) / EXIT(clamp=min(inc,orig_inc)) を逐次 dump する．
/// TCA `inc_flag` 累積収支が KH (`KHINC`) と分岐するノードを localize するための診断 (process 内 1 回読み)．
fn v4inc_prefix() -> Option<String> {
    static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    C.get_or_init(|| std::env::var("V4INC").ok().filter(|s| !s.is_empty()))
        .clone()
}

#[inline]
fn v4inc_active() -> bool {
    V4INC_ACTIVE.with(|c| c.get())
}

/// window が未 active かつ board.sfen() が V4INC prefix 一致なら window を開く (このノードが opener)．
fn v4inc_open_window(board: &Board) -> bool {
    if let Some(p) = v4inc_prefix() {
        if !v4inc_active() && board.sfen().starts_with(p.as_str()) {
            V4INC_ACTIVE.with(|c| c.set(true));
            V4INC_CNT.with(|c| c.set(0));
            return true;
        }
    }
    false
}

/// opener のみ window を閉じる (nested 呼び出しは false で素通り)．
#[inline]
fn v4inc_close_window(opened: bool) {
    if opened {
        V4INC_ACTIVE.with(|c| c.set(false));
    }
}

/// window active 時のみ dump (暴走防止に 4000 行 cap)．
#[inline]
fn v4inc_log(s: &str) {
    if v4inc_active() {
        let n = V4INC_CNT.with(|c| {
            let v = c.get();
            c.set(v + 1);
            v
        });
        if n < 4000 {
            eprintln!("{}", s);
        }
    }
}

/// `V4_SMPROP` env: KH `FrontSumMask` の cross-expansion propagation を**有効化**する (process 内 1 回読み)．
/// **default OFF**．理由: propagation 単体は default 探索順 (movegen 順 ≠ KH) では DAG (EliminateDoubleCount)
/// の効果を打ち消し 11,286→18,470 と退行する (KHTRACE 突合で cnt=24 tie-break 乖離が真因と判明)．
/// movegen 順を KH に揃える `V4_KHORDER` + `V4_DOM` と**併用**して初めて co-adapt し有益になる
/// (V4_KHORDER+V4_DOM+V4_SMPROP = 11,544, dominance が -45% に転じ KH と同符号; 単独は退行)．
/// → 忠実 bundle は opt-in．default v4 (無印) は 11,286 を維持する ([[project_dfpn_incremental_unreachable_v2_9_0]])．
fn v4_smprop() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V4_SMPROP").is_ok())
}

/// `V4_NOSMRESET` env: build-time sum_mask reset を無効化 (Full のまま from_parts) して
/// 旧挙動 (update_best_child の reset のみ) と切り分ける診断用 (process 内 1 回読み)．
fn v4_nosmreset() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V4_NOSMRESET").is_ok())
}

/// `V4_NODAG` env: EliminateDoubleCount (DAG 二重カウント抑止) を無効化する診断用 (process 内 1 回読み)．
fn v4_nodag() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V4_NODAG").is_ok())
}

/// KH `generate_checks<CHECKS_ALL>` (movegen.cpp:818-865) の生成順を完全に再現する sort key．
/// maou の square index は YaneuraOu と同一 ((file-1)*9+(rank-1)) なので raw square で昇順比較できる．
/// 順序: ① 盤上移動 (from-square 昇順 → to-square 昇順 → 不成→成)，② 駒打ち
/// (PAWN,LANCE,KNIGHT,SILVER,GOLD,BISHOP,ROOK 順 → to-square 昇順)．
/// 注: 開き王手/直接王手の細分順は未反映 (直接王手のみの局面では from-square 順で一致)．
fn drop_piece_order(pt: crate::types::PieceType) -> u64 {
    use crate::types::PieceType;
    // KH/yaneuraou GenerateDropMoves の drops[] 順: PAWN 先頭 + KNIGHT,LANCE,SILVER,GOLD,BISHOP,ROOK．
    // (KHSEED 実測 cnt=3129: 同マス 7e の tie で KH は N*7e を chain head に選ぶ = 桂が香より先)．
    // 旧実装は LANCE=1<KNIGHT=2 で逆順だった (node_movegen DROP_ORDER_KH とも不整合)．
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

pub(super) fn kh_check_order_key(
    m: crate::moves::Move,
    discoverers: crate::bitboard::Bitboard,
    def_king: Option<crate::types::Square>,
) -> u64 {
    use crate::types::PieceType;
    if let Some(pt) = m.drop_piece_type() {
        // group 2: 駒打ち王手．KH `generate_checks` の GenerateCheckDropMoves 呼出順
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
        // 盤上移動王手．KH `generate_checks` (movegen.cpp:806-844):
        //   ① 開き王手候補 from (= discoverers = blockers_for_king(them) & pieces(us)) を square 昇順で，
        //      各 from は pin-line 外への移動 (純開き王手) → pin-line 上への移動 (直接王手) の順に生成．
        //   ② 非候補 from (直接王手のみ) を square 昇順で生成．
        // 旧 key は ①② を区別せず単純 from 昇順だったため開き王手が先頭に来ず KH と乖離していた．
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

/// 盤上移動の駒種を KH `GeneratePieceMoves` の呼出し順へ写像する (Evasion 非玉手の group 番号)．
/// PAWN→0, LANCE→1, KNIGHT→2, SILVER→3, BISHOP→4, ROOK→5, GOLD/成駒(金移動)→6, HORSE→7, DRAGON→8．
fn kh_piece_gen_order(raw_pt: u8) -> u64 {
    match raw_pt {
        1 => 0,                // Pawn
        2 => 1,                // Lance
        3 => 2,                // Knight
        4 => 3,                // Silver
        5 => 4,                // Bishop
        6 => 5,                // Rook
        7 => 6,                // Gold
        9 | 10 | 11 | 12 => 6, // と金/成香/成桂/成銀 (金の動き; GPM_GHD)
        13 => 7,               // Horse
        14 => 8,               // Dragon
        _ => 9,
    }
}

/// KH `generate_evasions` (movegen.cpp:454) の生成順を再現する key．
/// ① 玉移動 (to-square 昇順) → ② 非玉の盤上移動 (駒種 group 順 → from-square 昇順 → 不成/成) →
/// ③ 駒打ち (PAWN..ROOK 順 → to-square 昇順)．`board` は親 (受け方手番) 局面．
pub(super) fn kh_evasion_order_key(board: &crate::board::Board, m: crate::moves::Move) -> u64 {
    if let Some(pt) = m.drop_piece_type() {
        // group 2 (駒打ちは最後)．KH `MovePicker` の evasion drop 順は **歩を全マス先に並べ，
        // 続いて各 to_sq ごとに drops[] 順 (香→桂→銀→金→角→飛)** で生成する (yaneuraou
        // movegen.cpp; KHSEED 実測: P*8c P*8d P*8e P*8f / L*8c G*8c R*8c / L*8d ...)．
        // 旧 key は piece 主・square 従 (歩全→香全→…) で，同マス drop が連続せず DML の
        // 同マス chain (IsSame to1==to2) が壊れていた (中合い G*8c が deferred されない)．
        // → 歩優先 (bit39) → to_sq 主 (bits 11-) → 駒種 従 (低 bit) に修正し KH 順に一致させる．
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
            let go = kh_piece_gen_order(raw_pt);
            (1u64 << 40) | (go << 30) | (from << 20) | (pf << 19) | (to << 11)
        }
    }
}

thread_local! {
    /// ply 0-7 の dump 済 bitmask (KH `kh_sel_dumped[16]` 相当; solve 毎に reset)．
    static V4SEL_DUMPED: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    /// V4RAW (sort/DML 前の raw movegen 順) の ply 0-7 dump 済 bitmask．
    static V4RAW_DUMPED: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    /// V4TRACE の chronological build カウンタ (KH `kh_trace_cnt` 相当; solve 毎に reset)．
    static V4TRACE_CNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// V4INC: inc_flag origin window が active か (opener が set/reset)．
    static V4INC_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// V4INC: window 内 dump 行カウンタ (cap 用; window open 毎に reset)．
    static V4INC_CNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

impl DfPnSolver {
    /// mid_v4 探索の root (KH `SearchEntry` 相当の IDS + 診断)．`V3_V4ENG=1` で起動する．
    pub(super) fn solve_via_v4(&mut self, board: &mut Board) -> TsumeResult {
        self.attacker = board.turn;
        self.v3_nodes = 0;
        self.v3_path.clear();
        self.v4_stack.clear();
        self.v4_dag_fires = 0;
        self.v4_dom_path.clear();
        self.v4_dom_fires = 0;
        V4SEL_DUMPED.with(|c| c.set(0));
        V4RAW_DUMPED.with(|c| c.set(0));
        V4TRACE_CNT.with(|c| c.set(0));
        // KH VisitHistory path dominance (IsRepetitionOrInferiorAfter)．`V4_DOM` で opt-in (default OFF)．
        // 単位を揃えた計測で判明: KH (dominance 有) = 9,296 visits に対し，maou+dominance = 16,902
        // (1.82×) と KH から **遠ざかる** (DAG EliminateDoubleCount との二重カウント相互作用; fires
        // 97→167)．KH の dominance は node を減らすので，maou 実装は未だ非忠実 = reference から除外する．
        // 39te では node を削る (13.4M→10.88M) ため opt-in で残す．忠実化 (DAG 相互作用の修正) は別課題．
        self.param_v4_path_dominance = std::env::var("V4_DOM").is_ok();
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

        // do_move 数を計測開始 (KH `info nodes` = do_move と同単位; verify は除外するため探索後に読む)．
        crate::board::reset_do_move_count();
        // KH `SearchEntry` (komoring_heights.cpp:265): root expansion を **一度だけ** 構築し，閾値反復
        // (IDS) 間で `v4_stack[0]` に**持続**させる (旧実装は反復毎に root を rebuild = KH IDS 構造との乖離．
        // root を持続させると子の results/idx/sum_mask/excluded が反復跨ぎで KH と同様に累積する)．
        let attacker_hand = board.hand[self.attacker.index()];
        let last = match self.emplace_v4(
            &mut tt,
            board,
            DEPTH_MAX_MATE_LEN,
            0,
            0u64,
            true,
            BitSet64::full(),
        ) {
            // root が終局 (王手なし=不詰 / 受けなし=詰み)．即結果 (KH: 空 expansion の CurrentResult)．
            Err(terminal) => terminal,
            Ok(()) => {
                // root は v4_stack[0] に push 済 (emplace_v4 が v3_nodes++/dump/key_hand_pair も実施)．
                // KH: thpn=thdn=tl_thread_id(=0) (len==kDepthMaxMateLen)．初反復は no-op warmup．
                let mut thpn: PnDn = 0;
                let mut thdn: PnDn = 0;
                let cap = K_INFINITE_PN_DN - 1;
                let mut result = self.v4_stack[0].current_result(board, 0);
                loop {
                    if self.v3_nodes >= self.max_nodes || self.is_timed_out() {
                        self.timed_out = self.is_timed_out();
                        break;
                    }
                    result =
                        self.search_impl_v4_root(&mut tt, board, thpn, thdn, DEPTH_MAX_MATE_LEN);
                    // V4ROOT: root IDS 反復ごとの th と結果 (KH `KHROOT` と突合)．
                    if std::env::var("V4ROOT").is_ok() {
                        eprintln!(
                            "V4ROOT th=({},{}) -> pn={} dn={}",
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
                    // KH NextPnDnThresholds: th = max(curr_th, val*1.7+1) (INF clamp)．val>=1 かつ
                    // val≈th なので必ず増加 = stall 無し (budget/timeout は loop 冒頭で break するため
                    // 旧実装の *2 fallback は不要)．
                    thpn = thpn.max((result.pn() as f64 * 1.7) as PnDn + 1).min(cap);
                    thdn = thdn.max((result.dn() as f64 * 1.7) as PnDn + 1).min(cap);
                    if thpn >= cap && thdn >= cap {
                        break;
                    }
                }
                // KH SearchEntry 末尾: `query.SetResult(result); expansion_list_.Pop()`．
                let root_q = tt.build_query(0, super::position_key(board), attacker_hand, 0);
                tt.set_result(&root_q, result, (0u64, super::ttentry::NULL_HAND));
                self.v4_stack.clear();
                result
            }
        };

        // KH と単位を揃えた 2 指標を併記: nodes = SearchImpl 訪問数 (v3_nodes; KH KHEMP builds と同単位),
        // do_moves = do_move 数 (KH `info nodes`/`nodes_searched` と同単位)．混同しないこと．
        let search_do_moves = crate::board::do_move_count();
        eprintln!(
            "[v4] root pn={} dn={} mate_len={} nodes={} do_moves={} dm/node={:.2} tt_cap={} gc={} dag={} dom={}",
            last.pn(),
            last.dn(),
            last.len().len(),
            self.v3_nodes,
            search_do_moves,
            search_do_moves as f64 / (self.v3_nodes.max(1)) as f64,
            tt.capacity(),
            tt.gc_count(),
            self.v4_dag_fires,
            self.v4_dom_fires
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

    /// KH `tt::LocalExpansion` の Emplace に相当する build + push (komoring_heights.cpp:351/470)．
    ///
    /// **KH は子展開を「親のループ内」で Emplace** し，その**集約済 `CurrentResult`** を first-visit-exceed
    /// 判定に使う (= 子の合法手を全列挙し 1 段深い δ 集約で評価)．maou 旧実装は子を recurse 時まで build せず
    /// first-visit を **flat な InitialPnDn seed** で評価していたため exceed 判定が緩く over-explore していた
    /// (mid_v3 から続く構造的乖離の本丸; root IDS iter5 で dn 126 vs KH 136 として観測)．
    /// `Err(result)` = 終局/budget/深さ上限 (push しない; 親が即時結果に使う)．`Ok(())` = expansion を top へ push．
    fn emplace_v4(
        &mut self,
        tt: &mut TranspositionTable,
        board: &mut Board,
        len: MateLen,
        depth: u32,
        path_key: u64,
        first_search: bool,
        sum_mask: BitSet64,
    ) -> Result<(), SearchResult> {
        self.v3_nodes += 1; // KH `KHEMP` builds と同単位 (Emplace 毎に 1)．
        if self.v3_nodes >= self.max_nodes || (self.v3_nodes & 0x3FF == 0 && self.is_timed_out()) {
            self.timed_out = self.timed_out || self.is_timed_out();
            // budget 切れは非 final unknown で unwind (root IDS が次で break)．
            return Err(SearchResult::make_first_visit(1, 1, len, 1));
        }
        // KH `if (n.GetDepth() >= kDepthMax) return MakeRepetition(...)` (komoring_heights.cpp:428)．
        if depth as usize >= super::mate_len::KDEPTH_MAX as usize {
            let attacker_hand = board.hand[self.attacker.index()];
            return Err(SearchResult::make_repetition(attacker_hand, len, 1, 0));
        }
        // KH 流 GC (komoring_heights.cpp:446): table-fill による look_up O(cap) 退化を防ぐ (4096 build 毎)．
        if self.v3_nodes & 0xFFF == 0 {
            tt.maybe_collect_garbage();
        }
        let attacker_hand = board.hand[self.attacker.index()];
        let exp = match self.build_v4_expansion(
            tt,
            board,
            len,
            depth,
            path_key,
            first_search,
            sum_mask,
        ) {
            Ok(e) => e,
            Err(terminal) => return Err(terminal),
        };
        self.dump_node_diag(&exp, board, depth);
        self.v4_stack.push(exp);
        let my = self.v4_stack.len() - 1;
        self.v4_stack[my].set_key_hand_pair((super::position_key(board), attacker_hand));
        Ok(())
    }

    /// 親ループの 1 子処理 (KH `SearchImpl` ループ本体, komoring_heights.cpp:454-514)．`my` = 親 expansion の
    /// stack index．best child を DoMove → **Emplace (build)** → first-visit は **子の集約済 CurrentResult**
    /// で exceed 判定 → 非 exceed なら `search_impl_v4` 再帰 → Pop → UndoMove → UpdateBestChild + TT 書込．
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
        let best_move = self.v4_stack[my].best_move();
        let is_first = self.v4_stack[my].front_is_first_visit();
        let (cthpn, cthdn) = self.v4_stack[my].front_pn_dn_thresholds(thpn, thdn);
        if v4inc_active() {
            v4inc_log(&format!(
                "V4INC STEP d={} best={} first={} inc={} cth=({},{})",
                depth,
                best_move.to_usi(),
                is_first as i32,
                *inc_flag,
                cthpn,
                cthdn
            ));
        }
        if v4thx_prefix()
            .map(|p| board.sfen().starts_with(p.as_str()))
            .unwrap_or(false)
        {
            eprintln!("V4THX {}", self.v4_stack[my].thx_breakdown(thpn, thdn));
        }
        let best_raw = self.v4_stack[my].front_raw();
        let child_query = self.v4_stack[my].query_at(best_raw);
        // KH `sum_mask = local_expansion.FrontSumMask()` (komoring_heights.cpp:458)．V4_SMPROP opt-in．
        let child_sum_mask = if v4_smprop() {
            self.v4_stack[my].front_sum_mask()
        } else {
            BitSet64::full()
        };

        let captured = board.do_move(best_move);
        let child_pk = path_key_after(path_key, best_move, depth as usize);
        let child_result = match self.emplace_v4(
            tt,
            board,
            len.sub(1),
            depth + 1,
            child_pk,
            is_first,
            child_sum_mask,
        ) {
            // 終局/budget/深さ上限: push されていない (Pop 不要)．
            // KH は terminal 子も Emplace し first_search なら inc_flag-- する (komoring_heights.cpp:477-484
            // の Emplace→DEC は終局/exceed の goto CHILD_SEARCH_END 前に実行される)．maou は terminal を
            // Err で返し push を省くが，**inc_flag 会計は Ok ブランチの DEC と対称に保つ**必要がある．
            // これを欠くと terminal first-visit 子で inc_flag が減らず TCA threshold を過剰拡張し，KH が
            // pop する OR node で次手を過剰展開する (39te cnt=7537 の真因; KHINC/V4INC per-step 突合で確定)．
            Err(terminal) => {
                if is_first && *inc_flag > 0 {
                    *inc_flag -= 1;
                    v4inc_log(&format!(
                        "V4INC DEC d={} best={} -> inc={} (first_visit/terminal)",
                        depth,
                        best_move.to_usi(),
                        *inc_flag
                    ));
                }
                terminal
            }
            Ok(()) => {
                let cidx = self.v4_stack.len() - 1;
                let r = if is_first {
                    if *inc_flag > 0 {
                        *inc_flag -= 1;
                        v4inc_log(&format!(
                            "V4INC DEC d={} best={} -> inc={} (first_visit)",
                            depth,
                            best_move.to_usi(),
                            *inc_flag
                        ));
                    }
                    // KH: 子 expansion の集約済 CurrentResult で exceed 判定 (flat seed でない)．
                    let cur = self.v4_stack[cidx].current_result(board, (depth + 1) as i32);
                    if cur.is_final() || cur.pn() >= cthpn || cur.dn() >= cthdn {
                        cur
                    } else {
                        self.search_impl_v4(
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
                    self.search_impl_v4(
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
                self.v4_stack.truncate(cidx); // KH `CHILD_SEARCH_END: expansion_list_.Pop()`
                r
            }
        };

        board.undo_move(best_move, captured);
        self.v4_stack[my].update_best_child(child_result);
        // KH UpdateBestChild 内 query.SetResult: 親 board_key は position-only (cross-hand のため hand 別管理)．
        tt.set_result(
            &child_query,
            child_result,
            (super::position_key(board), attacker_hand),
        );
    }

    /// KH `SearchImpl` (komoring_heights.cpp:389)．**呼出し前に親が `emplace_v4` で expansion を push 済**
    /// (KH が `expansion_list_.Current()` を使うのと同型)．自分では build も Pop もしない (親が Pop する)．
    fn search_impl_v4(
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
        let my = self.v4_stack.len() - 1;
        // V4INC: inc_flag origin window (opener が active 化; nested は素通り)．
        let v4inc_opened = v4inc_open_window(board);
        if v4inc_active() {
            v4inc_log(&format!(
                "V4INC ENTER d={} inc_in={} dhoc={} th=({},{}) sfen={}",
                depth,
                *inc_flag,
                self.v4_stack[my].does_have_old_child() as i32,
                thpn,
                thdn,
                board.sfen()
            ));
        }
        // V4TH: 受領 thpn/thdn dump (KH `KHTH` と同形式; threshold 乖離 hunting)．
        if let Some(prefix) = v4th_prefix() {
            let sfen = board.sfen();
            if sfen.starts_with(prefix.as_str()) {
                eprintln!(
                    "V4TH depth={} thpn={} thdn={} sfen={}",
                    depth, thpn, thdn, sfen
                );
            }
        }
        // KH `EliminateDoubleCount` (komoring_heights.cpp:432)．
        if !v4_nodag() {
            self.eliminate_double_count_v4(tt, board, my, depth);
        }

        // V4THX: SearchImpl per-iteration dump (KH KHTHX/KHTHXR と突合; inc_flag/threshold 乖離 hunting)．
        let khthx_here = v4thx_prefix()
            .map(|p| board.sfen().starts_with(p.as_str()))
            .unwrap_or(false);
        if khthx_here {
            eprintln!("V4THX enter depth={} th=({},{})", depth, thpn, thdn);
        }

        let orig_thpn = thpn;
        let orig_thdn = thdn;
        let orig_inc = *inc_flag;

        let mut curr = self.v4_stack[my].current_result(board, depth as i32);
        if curr.is_final() {
            if v4inc_active() {
                v4inc_log(&format!(
                    "V4INC EXIT-curfinal d={} inc={}",
                    depth, *inc_flag
                ));
            }
            v4inc_close_window(v4inc_opened);
            return curr; // 親 (step_best_child) が Pop する．
        }
        if self.v4_stack[my].does_have_old_child() {
            *inc_flag += 1;
            v4inc_log(&format!(
                "V4INC INC d={} -> inc={} (dhoc)",
                depth, *inc_flag
            ));
        }
        if *inc_flag > 0 {
            extend_search_threshold(curr, &mut thpn, &mut thdn);
        }

        self.v3_path.insert(board.hash, depth);
        if self.param_v4_path_dominance {
            let attacker_hand = board.hand[self.attacker.index()];
            self.v4_dom_path
                .entry(super::position_key(board))
                .or_default()
                .push((attacker_hand, depth));
        }

        while curr.pn() < thpn && curr.dn() < thdn {
            if self.v3_nodes >= self.max_nodes || self.timed_out {
                break;
            }
            if khthx_here {
                eprintln!(
                    "V4THX best={} first={}",
                    self.v4_stack[my].best_move().to_usi(),
                    self.v4_stack[my].front_is_first_visit() as i32
                );
            }
            self.step_best_child(tt, board, my, depth, len, path_key, thpn, thdn, inc_flag);
            curr = self.v4_stack[my].current_result(board, depth as i32);
            if khthx_here {
                eprintln!(
                    "V4THXR -> curr pn={} dn={} inc={} (th={},{})",
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
                    eprintln!("V4THX break (inc==0 && orig_inc>0)");
                }
                break;
            }
        }
        if khthx_here {
            eprintln!(
                "V4THX exit depth={} curr=(pn{},dn{}) inc={}",
                depth,
                curr.pn(),
                curr.dn(),
                *inc_flag
            );
        }

        self.v3_path.remove(&board.hash);
        if self.param_v4_path_dominance {
            let pk = super::position_key(board);
            if let Some(v) = self.v4_dom_path.get_mut(&pk) {
                v.pop();
                if v.is_empty() {
                    self.v4_dom_path.remove(&pk);
                }
            }
        }
        let pre_clamp = *inc_flag;
        *inc_flag = (*inc_flag).min(orig_inc);
        if v4inc_active() {
            v4inc_log(&format!(
                "V4INC EXIT d={} pre={} orig={} -> inc={} curr=(pn{},dn{})",
                depth,
                pre_clamp,
                orig_inc,
                *inc_flag,
                curr.pn(),
                curr.dn()
            ));
        }
        v4inc_close_window(v4inc_opened);
        curr
    }

    /// KH `SearchImplForRoot` (komoring_heights.cpp:325) の忠実移植．
    ///
    /// root expansion は **solve_via_v4 が一度だけ構築**し `v4_stack[0]` に持続させる (KH `SearchEntry` が
    /// Emplace を 1 回だけ呼び閾値反復間で再利用するのと同型)．本関数は毎反復その持続 expansion を再利用し，
    /// 子だけを [`search_impl_v4`] で展開する．SearchImpl との差: inc_flag は local (親 inc_flag 無し)，
    /// 末尾の `orig_inc break` / `inc_flag = min(...)` は無い．
    fn search_impl_v4_root(
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

        if !v4_nodag() {
            self.eliminate_double_count_v4(tt, board, 0, 0);
        }
        let mut curr = self.v4_stack[0].current_result(board, 0);
        if self.v4_stack[0].does_have_old_child() {
            inc_flag += 1;
        }
        if inc_flag > 0 {
            extend_search_threshold(curr, &mut thpn, &mut thdn);
        }

        // KH VisitHistory `Visit`: 子探索中 root を path に積む (子が IsRepetition/Inferior で参照)．
        self.v3_path.insert(board.hash, 0);
        if self.param_v4_path_dominance {
            self.v4_dom_path
                .entry(super::position_key(board))
                .or_default()
                .push((attacker_hand, 0));
        }

        while curr.pn() < thpn && curr.dn() < thdn {
            if self.v3_nodes >= self.max_nodes || self.timed_out {
                break;
            }
            self.step_best_child(tt, board, 0, 0, len, 0u64, thpn, thdn, &mut inc_flag);
            curr = self.v4_stack[0].current_result(board, 0);

            thpn = orig_thpn;
            thdn = orig_thdn;
            if inc_flag > 0 {
                extend_search_threshold(curr, &mut thpn, &mut thdn);
            }
            // NOTE: KH SearchImplForRoot には SearchImpl の `else if inc_flag==0 && orig_inc>0 break`
            // は無い (root は親 inc_flag を持たないため)．末尾の min も無い．
        }

        self.v3_path.remove(&board.hash);
        if self.param_v4_path_dominance {
            let pk = super::position_key(board);
            if let Some(v) = self.v4_dom_path.get_mut(&pk) {
                v.pop();
                if v.is_empty() {
                    self.v4_dom_path.remove(&pk);
                }
            }
        }
        curr
    }

    /// V4TRACE / V4NODE / V4SEL の chronological / 子リスト診断 dump (env-gated)．search_impl_v4 と
    /// 持続 root build の双方から呼び，KH `KHTRACE`/`KHSEL` と 1:1 で突合する ([[feedback_kh_divergence_hunting]])．
    fn dump_node_diag(&self, exp: &LocalExpansion, board: &Board, depth: u32) {
        let oc = if board.turn == self.attacker { 1 } else { 0 };
        if v4trace_enabled() {
            let cnt = V4TRACE_CNT.with(|c| {
                let v = c.get() + 1;
                c.set(v);
                v
            });
            if cnt <= 20000 {
                if let Some((m, pn, dn, _ev, _am)) = exp.trace_children().first() {
                    eprintln!(
                        "V4TRACE {} ply={} or={} best={} bpn={} bdn={} sfen={}",
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
        if let Some(prefix) = v4node_prefix() {
            let sfen = board.sfen();
            if sfen.starts_with(prefix.as_str()) {
                eprintln!("V4NODE ply={} or={} sfen={}", depth, oc, sfen);
                for (k, (m, pn, dn, ev, am)) in exp.trace_children().iter().enumerate() {
                    eprintln!(
                        "V4NODE   {} {} pn={} dn={} ev={} am={}",
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
        if v4sel_enabled() && depth <= 7 {
            let bit = 1u8 << depth;
            let already = V4SEL_DUMPED.with(|c| {
                let v = c.get();
                c.set(v | bit);
                v & bit != 0
            });
            if !already {
                eprintln!("V4SEL ply={} or={} sfen={}", depth, oc, board.sfen());
                for (k, (m, pn, dn, ev, am)) in exp.trace_children().iter().enumerate() {
                    eprintln!(
                        "V4SEL   {} {} pn={} dn={} ev={} am={}",
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
        sum_mask: BitSet64,
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
            // KH 終端 hand: OR=DisproofHandSet.Get(子なし)=remove_if(MAX) / AND=ProofHandSet.Get(子なし)=add_if(空)．
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
            return Err(r);
        }

        // V4_KHORDER: 生成手を KH `generateMoves` 順へ並べ替える (これが KH のソート入力 = 同じ
        // スタート地点)．DML/results/idx すべてこの順で構築する．promo は成→不成の順 (KH 一致)．
        if khorder_enabled() {
            if or_node {
                // OR-node 王手順は KH `generate_checks` (開き王手候補 → 直接王手 → 駒打ち) に従う．
                // discoverers (= blockers_for_king(them) & pieces(us)) は build 毎に 1 回算出して共有．
                let def_king = board.king_square(attacker.opponent());
                let discoverers = def_king
                    .map(|k| board.compute_discoverers(attacker, k))
                    .unwrap_or(crate::bitboard::Bitboard::EMPTY);
                moves.sort_by_key(|&m| kh_check_order_key(m, discoverers, def_king));
            } else {
                moves.sort_by_key(|&m| kh_evasion_order_key(board, m));
            }
        }

        // V4RAW: build 時点の raw movegen 順 (sort/DML 前) を ply 0-7 初出で dump (KH MovePicker 順と突合)．
        if v4sel_enabled() && depth <= 7 {
            let bit = 1u8 << depth;
            let already = V4RAW_DUMPED.with(|c| {
                let v = c.get();
                c.set(v | bit);
                v & bit != 0
            });
            if !already {
                let oc = if or_node { 1 } else { 0 };
                let raw: Vec<String> = moves.iter().map(|m| m.to_usi()).collect();
                eprintln!("V4RAW ply={} or={} raw={}", depth, oc, raw.join(" "));
            }
        }

        // KH `delayed_move_list_{n, mp_}` (local_expansion.hpp:155)．同マス合駒/成不成ペアを
        // 双方向 chain 化し，prev が未 final の手は idx から除外 (= 後回し) する．
        // is_delayable を KH 忠実化するため移動元駒種 (raw ID) を渡す (駒打ちは 0)．
        // `V4_KHORDER` 実験時のみ忠実版 (default は co-adapt 済の従来 DML を維持)．
        let dml = if khorder_enabled() {
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
            if super::v4_kh_moves() && !or_node {
                // KH `IsSame` 中合い対称性 (delayed_move_list.hpp:152-156): AND node の drop で
                // 「攻方支援なし & 逆王手でない」もの同士は別マスでも同一 chain とみなし後回し．
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
        } else {
            DelayedMoveList::build(&moves, or_node)
        };

        let defender_king = board.king_square(attacker.opponent());
        let n = moves.len();
        let mut evals: Vec<i32> = Vec::with_capacity(n);
        let mut results: Vec<SearchResult> = Vec::with_capacity(n);
        let mut queries = Vec::with_capacity(n);
        let mut idx: Vec<u32> = Vec::with_capacity(n);
        let mut does_have_old = false;
        // KH ctor `sum_mask_{sum_mask}` (local_expansion.hpp:159): 親から渡された sum_mask を起点に，
        // 非 final 子のうち IsSumDeltaNode でない / δ が kForceSumPnDn 以上のものを max 集約へ落とす．
        let mut cur_sum_mask = sum_mask;

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

            // KH `BoardKeyAfter`/`OrHandAfter` (local_expansion.hpp:192): 子 key/hand を **do_move
            // せず** incremental に算出する．従来の per-child do_move は 7.5 do_moves/node (KH 2.07)
            // と乖離し per-node time を浪費していた (= incremental child key の欠落)．do_move は子局面
            // が必須な look-ahead でのみ行う．**TT board_key は position-only** (`board_hash_after`;
            // KH BoardKey)．hand は entry に別管理し cross-hand を効かせる．千日手判定だけ full hash．
            let ch_full = board.hash_after(m);
            let ch_pos = board.board_hash_after(m);
            let child_hand = board.hand_after(m, attacker);
            let child_pk = path_key_after(path_key, m, depth as usize);
            let q = tt.build_query(child_pk, ch_pos, child_hand, (depth + 1) as i32);

            // KH `IsRepetitionOrInferiorAfter` (node.hpp:160 + local_expansion.hpp:178): path 上に
            // 同一 board_key かつ攻め方持駒が child 以上 (= child が劣位) の祖先があれば反復として刈る．
            // tsume では (position_key, attacker_hand) が全局面を一意決定するため，これは exact 千日手
            // (`v3_path`) の持駒 superset 方向への一般化 (exact は hand 等値の特殊ケース)．KH 同様に
            // **TT LookUp より前に** 刈ることで高コストな劣位部分木の展開を未然に防ぐ (これが node 削減の本体)．
            let dom_depth = if self.param_v4_path_dominance {
                self.v4_dom_path.get(&ch_pos).and_then(|anc| {
                    anc.iter()
                        .rev()
                        .find(|(h, _)| super::hand_gte(h, &child_hand))
                        .map(|&(_, d)| d)
                })
            } else {
                None
            };
            let mut r = if let Some(anc_depth) = dom_depth {
                self.v4_dom_fires += 1;
                SearchResult::make_repetition(child_hand, len, 1, anc_depth as i32)
            } else if let Some(&anc_ply) = self.v3_path.get(&ch_full) {
                // path 上の同一局面 = 千日手 (KH IsRepetitionAfter)．
                SearchResult::make_repetition(child_hand, len, 1, anc_ply as i32)
            } else {
                let mut dhoc = false;
                let res = tt.look_up(&q, len.sub(1), &mut dhoc, || seed);
                does_have_old = does_have_old || dhoc;
                res
            };
            // KH ctor (local_expansion.hpp:194-197): 非 final 子の δ 集約方式を決める．似た子局面で
            // 過小評価を招く手 (`!IsSumDeltaNode`) と既に δ が巨大な子 (`>= kForceSumPnDn`) は max 集約へ
            // 落とす (sum_mask bit を reset)．look-ahead で final 化する前の seed の δ で判定する．
            if !v4_nosmreset()
                && !r.is_final()
                && (!is_sum_delta_node(board, m, or_node) || r.delta(or_node) >= K_FORCE_SUM_PN_DN)
            {
                cur_sum_mask.reset(i);
            }
            // KH DML skip 判定 (local_expansion.hpp:194-203): 非 final かつ prev chain に未 final の
            // 先行手があれば後回し (idx に積まない)．prev は i より前に処理済なので results が揃っている．
            let is_skipped = !r.is_final() && dml.has_unresolved_prev(i, |j| results[j].is_final());

            // KH `CheckObviousFinalOrNode` 先読み (local_expansion.hpp:217-221)．non-skipped のみ．
            // AND node の first-visit child は子が OR node (攻め方手番)．board は do_move 済 (= 子局面)
            // なので攻め方の 1 手詰／詰み無を先読みし，proven/disproven を seed する．これにより
            // 「詰む応手」を展開せず除外，「逃れる応手」を即 disproof し breadth を抑える．
            // [diag] V4SEED: 指定 sfen prefix の親ノードで各子の look_up 結果 (TT/cross-hand) と
            // look-ahead 前後を dump し，disproof の出所 (cross-hand TT vs fresh look-ahead) を特定する．
            let seed_diag = std::env::var("V4SEED")
                .ok()
                .filter(|p| !p.is_empty())
                .map(|p| board.sfen().starts_with(p.as_str()))
                .unwrap_or(false);
            let pre_lu = (r.pn(), r.dn(), r.is_final(), r.is_first_visit());

            // 子結果が final になったら KH `query.SetResult` 相当で TT へ格納する (PV/伝播の整合)．
            if !is_skipped && !or_node && first_search && r.is_first_visit() {
                // 子局面 (do_move 済) が必要なのはここだけ．KH も CheckObviousFinalOrNode 内で
                // DoMove/UndoMove する (= この do_move は KH と同単位; seeding 分のみ削減した)．
                let captured = board.do_move(m);
                if let Some(res) = self.check_obvious_final_or_node_v4(board) {
                    tt.set_result(&q, res, (ch_pos, child_hand));
                    r = res;
                }
                board.undo_move(m, captured);
            }
            if seed_diag {
                eprintln!(
                    "V4SEED child={} skip={} seed=({},{}) lookup=(pn{} dn{} fin{} fv{}) final=(pn{} dn{} fin{})",
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
            cur_sum_mask,
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
        let use_handset = handset_enabled();
        let (mm_opt, has_checks) = self.mate1ply_with_cached_checks(board);
        // KH `CheckObviousFinalOrNode` は不詰判定に `!DoesHaveMatePossibility` (blocker 無視の
        // over-approx) を使う．maou の exact `!has_checks` は blocker で塞がれた王手候補を即不詰断定
        // して KH より早く disproof し探索経路が乖離する (例: cnt=487 の Kx7i)．V4_DHMP で KH と一致．
        let no_mate = if dhmp_enabled() {
            !board.does_have_mate_possibility(board.turn)
        } else {
            !has_checks
        };
        if no_mate {
            // 攻め方に王手手段なし → 詰み不可能 → 不詰 (KH MakeFinal<false>, kDepthMaxMateLen)．
            // KH: HandSet{DisproofHandTag}.Get(pos) = remove_if(MAX)．
            let hand = if use_handset {
                super::proof_hand::disproof_hand_terminal_or(board)
            } else {
                or_hand
            };
            Some(SearchResult::make_final(false, hand, DEPTH_MAX_MATE_LEN, 1))
        } else if let Some(mate_move) = mm_opt {
            // 1 手詰 → 詰み proven (mate-1)．KH CheckMate1Ply:
            // BeforeHand(mate_move, ProofHandSet.Get(詰み局面))．
            let hand = if use_handset {
                let cap = board.do_move(mate_move);
                let proof_after = super::proof_hand::proof_hand_terminal_and(board);
                board.undo_move(mate_move, cap);
                super::proof_hand::before_hand(board, mate_move, proof_after)
            } else {
                or_hand
            };
            // [diag] V4MATE: look-ahead 1 手詰の手と proof hand を dump (V4HAND prefix gate)．
            if let Some(prefix) = v4hand_prefix() {
                if board.sfen().starts_with(prefix.as_str()) {
                    eprintln!(
                        "V4MATE mate1ply={} proof P{} L{} N{} S{} G{} B{} R{} sfen={}",
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

    /// KH `EliminateDoubleCount` (komoring_heights.cpp:432 + expansion_stack.hpp:70)．
    /// 本ノード `my` の best_move が TT 親鎖を遡って祖先へ合流する (DAG) なら，その分岐元の
    /// 合流子を sum→max 集約へ落とし pn/dn の二重カウントを抑止する．
    fn eliminate_double_count_v4(
        &mut self,
        tt: &TranspositionTable,
        board: &mut Board,
        my: usize,
        depth: u32,
    ) {
        if self.v4_stack[my].empty() {
            return;
        }
        let best_move = self.v4_stack[my].best_move();
        let current_kh = self.v4_stack[my].key_hand_pair();
        let or_node = self.v4_stack[my].is_or_node();
        // best_move 後の子 (board_key, attacker_hand) を取得 (KH `BoardKeyHandPairAfter`)．
        let captured = board.do_move(best_move);
        let child_kh = (
            super::position_key(board),
            board.hand[self.attacker.index()],
        );
        board.undo_move(best_move, captured);

        let edge = match self.find_known_ancestor_v4(tt, current_kh, child_kh, or_node, depth, my) {
            Some(e) => e,
            None => return,
        };
        self.v4_dag_fires += 1;
        // 分岐元から下流側 (= my-1 .. 0) を辿り resolve する (KH `list_.rbegin()+1 .. rend()`)．
        let mut i = my;
        while i > 0 {
            i -= 1;
            if self.v4_stack[i].resolve_double_count_if_branch_root(edge) {
                break;
            }
            if self.v4_stack[i].should_stop_ancestor_search(edge.branch_root_is_or_node) {
                break;
            }
        }
    }

    /// KH `FindKnownAncestor` (double_count_elimination.hpp:102) の exact-hand 簡約版．
    /// `child_kh` を起点に TT 親鎖を遡り，path 上の祖先へ合流したら分岐元の辺を返す．
    fn find_known_ancestor_v4(
        &self,
        tt: &TranspositionTable,
        current_kh: (u64, Hand),
        child_kh: (u64, Hand),
        or_node_current: bool,
        depth: u32,
        my: usize,
    ) -> Option<BranchRootEdge> {
        let mut key_hand = child_kh;
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
            let parent_kh = (pbk, ph);
            // 初回の親が現局面なら二重カウントの疑い無し (KH :123)．
            if i == 0 && parent_kh == current_kh {
                break;
            }
            if dn > last_dn.saturating_add(K_ANCESTOR_SEARCH_THRESHOLD) {
                dn_flag = false;
            }
            if pn > last_pn.saturating_add(K_ANCESTOR_SEARCH_THRESHOLD) {
                pn_flag = false;
            }
            if self.contains_in_path_v4(parent_kh, my) {
                if (or_node && dn_flag) || (!or_node && pn_flag) {
                    return Some(BranchRootEdge {
                        branch_root: parent_kh,
                        child: key_hand,
                        branch_root_is_or_node: or_node,
                    });
                } else {
                    break;
                }
            }
            key_hand = parent_kh;
            last_pn = pn;
            last_dn = dn;
            or_node = !or_node;
            i += 1;
        }
        None
    }

    /// (board_key, hand) が現探索 path の祖先 (`v4_stack[0..my]`) に在るか (KH `Node::ContainsInPath`)．
    fn contains_in_path_v4(&self, kh: (u64, Hand), my: usize) -> bool {
        self.v4_stack[..my].iter().any(|e| e.key_hand_pair() == kh)
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
