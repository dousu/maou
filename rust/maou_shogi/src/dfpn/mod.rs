//! Df-Pn (Depth-First Proof-Number Search) による詰将棋ソルバー．
//!
//! cshogi と同じ Df-Pn アルゴリズムを採用し，攻め方が玉方を
//! 詰ませる最善手順を求める．先手・後手どちらが攻め方でも動作する．
//!
//! # 引数
//!
//! - `depth`: 最大探索手数(デフォルト 31)．無限ループ防止用．
//! - `nodes`: 最大ノード数(デフォルト 1,048,576 = 2^20)．計算時間・メモリ制限用．
//! - `draw_ply`: 引き分け手数(デフォルト 32767)．

use arrayvec::ArrayVec;

use crate::board::Board;
use crate::moves::Move;
use crate::types::{Color, Square, HAND_KINDS};

/// `eprintln!` の `verbose` feature ガード版．
///
/// `verbose` feature が無効の場合はコンパイル時に完全に除去される．
/// デバッグ・分析用の進捗表示やノード情報出力に使用する．
macro_rules! verbose_eprintln {
    ($($arg:tt)*) => {
        #[cfg(feature = "verbose")]
        eprintln!($($arg)*)
    };
}

// profile_timed! マクロは profile.rs で定義．
// Rust のマクロは定義順で解決されるため，profile モジュールを先に宣言する．
#[macro_use]
mod profile;
mod entry;
mod tt;
mod solver;
mod pns;
mod node_movegen;
mod delayed_move_list;
mod local_expansion;
mod mid_v3;
mod proof_hand;
mod path_key;
mod repetition_memo;
#[cfg(test)]
mod tests;

pub use solver::{DfPnSolver, TsumeResult};
pub use pns::{solve_tsume, solve_tsume_with_timeout, solve_tsume_and_collect_pn_dn_dist};

/// 王手手/応手の最大数．
/// 将棋の合法手上限は593であり，長手数の詰将棋では
/// 持ち駒が多い局面で320を超えるケースが存在するため，
/// 合法手上限に合わせる．
const MAX_MOVES: usize = 593;

/// V3_KHPAR: KH/yaneuraou parity の手生成順 (王手 = 盤上移動→駒打ち raw 順,
/// AND 合駒 drop = 歩→桂→香→… 順) を有効化する experiment gate．
/// default off = committed 挙動 (29te 18,539 canonical を維持)．
/// 39te campaign では V3_KHPAR=1 V3_CHUAI=1 で計測する (worklog 参照)．
/// process 内で 1 回だけ読む (hot path の env::var 回避)．
pub(super) fn kh_parity_order() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("V3_KHPAR").is_ok())
}

/// `ArrayVec::try_push` のラッパー．
/// デバッグビルドでは容量超過時にパニックし，リリースビルドでは無音で破棄する．
#[inline(always)]
fn push_move<T, const N: usize>(buf: &mut ArrayVec<T, N>, val: T) {
    let result = buf.try_push(val);
    debug_assert!(result.is_ok(), "move buffer overflow: capacity {N} exceeded");
}

/// pn/dn の 1 単位を表す定数．
///
/// 全ての pn/dn 初期値・加算定数・フロア値はこの定数の倍数で表現する．
/// PN_UNIT=1 が従来動作と等価であり，PN_UNIT を拡大することで
/// 1+ε 閾値の余裕を確保し閾値飢餓(§10.2)を緩和できる．
///
/// 完全なスケーリング要件: pn/dn の「量」を表す全ての定数に PN_UNIT を
/// 適用する必要がある．スケーリング対象は以下の通り:
/// - 初期値: TT ミス時の pn=1/dn=1，heuristic_or_pn/heuristic_and_pn 返り値
/// - 加算値: edge_cost_or/and，sacrifice_check_boost，epsilon の +1，
///   progress_floor の +1，TCA の +1
/// - フロア/バイアス: DN_FLOOR，INTERPOSE_DN_BIAS，.max(N) のリテラル
///
/// スケーリング不要: 終端値(INF, 0)，相対比率(/4, /2, *2/3)，
/// 盤面状態の比較(safe_escapes >= 4 等)，ループカウンタ．
const PN_UNIT: u32 = 16;

// MID ループの dn 閾値フロア(スラッシング防止)は v0.24.41 で
// `DfPnSolver::param_dn_floor_mult` (デフォルト 100) に移行した．
// 旧 `const DN_FLOOR: u32 = 100 * PN_UNIT;` は削除．
// 子ノードの dn が小さすぎると MID ループが閾値超過で即座に返り，
// 進捗のない空転が発生するため，dn_threshold を最低
// `param_dn_floor_mult * PN_UNIT` まで引き上げる．

/// Deep df-pn (Song Zhang et al. 2017) の深さ係数 R．
///
/// 深い位置ほど初期 pn を高く設定し，浅い分岐の探索を優先させる．
/// `look_up_pn_dn` で TT ミス時に `ply > depth/2` の場合に適用される．
const DEEP_DFPN_R: u32 = 4;

/// `param_epsilon_denom` の depth-adaptive モードを示すセンチネル値．
///
/// この値が設定されている場合，epsilon 除数は `saved_depth_for_epsilon` に基づいて
/// 自動決定される(depth ≥ 19 → 2，それ以外 → 3)．
const EPSILON_DENOM_ADAPTIVE: u32 = 0;

/// `param_disproof_remaining_threshold` の depth-adaptive モードを示すセンチネル値 (v0.25.1)．
///
/// この値が設定されている場合，depth-limited disproof 格納閾値は
/// `outer_solve_depth` に基づいて自動決定される．
/// 詳細は `DfPnSolver::effective_disproof_remaining_threshold` 参照．
pub(super) const DISPROOF_THRESHOLD_ADAPTIVE: u16 = u16::MAX;

/// 深さ制限なし(真の証明/反証)を示す定数．
///
/// remaining_flags のビット 0-14 に格納されるため 15 ビットの最大値(0x7FFF)．
/// 実用上の depth は 31〜127 であり 32767 は十分に大きい．
const REMAINING_INFINITE: u16 = 0x7FFF;


/// DFPN-E (Kishimoto et al., NeurIPS 2019) エッジコスト型ヒューリスティック．
///
/// 標準 df-pn+ はノード(局面)の特徴で初期 pn/dn を設定するが，
/// DFPN-E は**エッジ(手)**の特徴に基づくコストを加算する．
/// 展開済みノードではエッジコストをゼロに戻すため，
/// 実質的には初期 pn への加算として機能する．
///
/// 詰将棋での手の質:
/// - OR ノードの王手: 成+取 > 取/成 > 近い静か手 > 遠い静か手
/// - AND ノードの応手: 合駒(攻め方有利) < 駒取り < 玉の逃げ

/// KH `InitialPnDnPlusOrNode` 移植 (v0.89.0, Phase 10).
///
/// OR ノード(攻め方の王手)の per-move (pn, dn) tuple を返す．
/// KH `initial_estimation.hpp:87-119` と等価．
///
/// KH `attackers_to(to)` は玉を含むため，玉が `to` に隣接していれば support を +1 する
/// (compute_checkers_at は玉を除外するので，InitialPnDn を KH と一致させる補正)．
#[inline]
pub(super) fn king_supports(board: &Board, to: Square, color: crate::types::Color) -> u32 {
    match board.king_square(color) {
        Some(k) => {
            let dc = (k.col() as i32 - to.col() as i32).abs();
            let dr = (k.row() as i32 - to.row() as i32).abs();
            u32::from(dc <= 1 && dr <= 1 && (dc != 0 || dr != 0))
        }
        None => 0,
    }
}

/// パラメータは KH デフォルト (all = PN_UNIT) を使用．
/// - 受け駒 ≥ 2: pn += PN_UNIT (後回し)
/// - 攻め支援 + drop_bonus > 受け支援: dn += PN_UNIT (探索優先)
/// - 金/銀取り: dn += PN_UNIT
/// - その他の駒取り: pn += PN_UNIT
/// - 静か手: pn += PN_UNIT
pub(super) fn init_pn_dn_or_kh(
    board: &crate::board::Board,
    m: Move,
    attacker: crate::types::Color,
) -> (u32, u32) {
    let mut pn = PN_UNIT;
    let mut dn = PN_UNIT;

    let to = m.to_sq();
    let att_bb = board.compute_checkers_at(to, attacker);
    let def_bb = board.compute_checkers_at(to, attacker.opponent());
    // KH `attackers_to(to)` は玉も含む (compute_checkers_at は玉を除外する) ので，
    // KH InitialPnDn と一致させるため玉が `to` に隣接していれば support に +1 する．
    let attack_support = att_bb.count() + king_supports(board, to, attacker);
    let defense_support = def_bb.count() + king_supports(board, to, attacker.opponent());
    let drop_bonus: u32 = if m.is_drop() { 1 } else { 0 };

    if defense_support >= 2 {
        pn += PN_UNIT;
    }

    if attack_support + drop_bonus > defense_support {
        dn += PN_UNIT;
    } else {
        let captured = m.captured_piece_raw();
        if captured > 0 {
            let cap_pt = crate::types::PieceType::from_u8(captured);
            if matches!(
                cap_pt,
                Some(crate::types::PieceType::Gold) | Some(crate::types::PieceType::Silver)
            ) {
                dn += PN_UNIT;
            } else {
                pn += PN_UNIT;
            }
        } else {
            pn += PN_UNIT;
        }
    }

    (pn, dn)
}

/// KH `InitialPnDnPlusAndNode` 移植 (v0.89.0, Phase 10).
///
/// AND ノード(防御側の応手)の per-move (pn, dn) tuple を返す．
/// KH `initial_estimation.hpp:127-151` と等価．
///
/// - 駒取り応手: (2U, U)
/// - 玉移動: (U, U)
/// - 攻め支援 < 受け支援 + drop_bonus (good escape): (2U, U)
/// - その他 (bad escape): (U, 2U)
pub(super) fn init_pn_dn_and_kh(
    board: &crate::board::Board,
    m: Move,
    attacker: crate::types::Color,
) -> (u32, u32) {
    let defender = attacker.opponent();

    if m.captured_piece_raw() > 0 {
        return (2 * PN_UNIT, PN_UNIT);
    }

    let king_sq = board.king_square(defender);
    if !m.is_drop() {
        if let Some(ksq) = king_sq {
            if m.from_sq() == ksq {
                return (PN_UNIT, PN_UNIT);
            }
        }
    }

    let to = m.to_sq();
    let att_bb = board.compute_checkers_at(to, attacker);
    let def_bb = board.compute_checkers_at(to, defender);
    // KH `attackers_to` は玉も含むので，玉が `to` 隣接なら support に +1 (KH InitialPnDn 一致)．
    let attack_support = att_bb.count() + king_supports(board, to, attacker);
    let defense_support = def_bb.count() + king_supports(board, to, defender);
    let drop_bonus: u32 = if m.is_drop() { 1 } else { 0 };

    if attack_support < defense_support + drop_bonus {
        // good escape
        (2 * PN_UNIT, PN_UNIT)
    } else {
        // bad escape
        (PN_UNIT, 2 * PN_UNIT)
    }
}

/// KH `MoveBriefEvaluation` 準拠の move ordering key (Phase 20)．
///
/// 値が小さいほど「良い手」．KH と同じ基準:
/// - 成れるのに成らない歩/角/飛: +1000
/// - 移動後の駒価値が高いほど優先 (−pt_value)
/// - 玉に近いほど優先 (+10 × distance)
pub(super) fn move_brief_eval(m: Move, king_sq: Square, board: &Board) -> i32 {
    let to = m.to_sq();
    let mut value: i32 = 0;

    // 移動後の駒種 raw ID (1=Pawn .. 14=Dragon)
    let raw_pt: u8 = if m.is_drop() {
        m.drop_piece_type().map(|pt| pt as u8).unwrap_or(0)
    } else {
        // board.piece_at returns raw piece byte; strip color (& 0x0F)
        board.piece_at(m.from_sq()) & 0x0F
    };

    // 成れるのに成らない歩/角/飛: +1000 penalty
    if !m.is_drop() && !m.is_promotion() {
        if matches!(raw_pt, 1 | 5 | 6) {
            let from = m.from_sq();
            let us = board.turn;
            let in_enemy = |sq: Square| -> bool {
                let r = sq.row() as u8;
                match us {
                    Color::Black => r <= 2,
                    Color::White => r >= 6,
                }
            };
            if in_enemy(from) || in_enemy(to) {
                value += 1000;
            }
        }
    }

    let after_raw = if m.is_promotion() { raw_pt + 8 } else { raw_pt };
    let pt_value: i32 = match after_raw {
        1 => 10,                   // Pawn
        2 => 20,                   // Lance
        3 => 20,                   // Knight
        4 => 30,                   // Silver
        5 => 50,                   // Bishop
        6 => 50,                   // Rook
        7 => 50,                   // Gold
        8 => 80,                   // King (KH tl_pt_values[KING]=80; 欠落で AND 玉捕獲手の tie-break が乖離していた)
        9 | 10 | 11 | 12 => 50,   // ProPawn..ProSilver
        13 => 80,                  // Horse
        14 => 80,                  // Dragon
        _ => 0,
    };
    value -= pt_value;

    let dc = (to.col() as i32 - king_sq.col() as i32).abs();
    let dr = (to.row() as i32 - king_sq.row() as i32).abs();
    value += 10 * dc.max(dr);

    value
}

/// KH `initial_estimation.hpp:227 IsSumDeltaNode` 移植 (Phase 27)．
///
/// δ値を **和 (sum)** で計上すべき子なら `true`，**最大値 (max)** で計上すべきなら `false`．
/// KH 同様，ほぼ全ての手で `true` を返し，OR ノードの香成/不成 near-duplicate のみ `false`：
///
/// > 似た子局面になる手が複数あると，δ値を定義通り sum すると局面を過小評価
/// > (実値より大きく出る) ことがある．→ そのような手は max で計上する．
///
/// 具体条件 (`false` を返す = max 集約)：
/// - OR ノード (攻め方手番)
/// - 駒打ちでない香車の移動 (移動元の駒種が Lance)
/// - 行き先 `to` が敵陣の rank 2/3 (先手) または rank 7/8 (後手)
///   (maou 行座標: 先手 row∈{1,2}，後手 row∈{6,7}; row 0 = rank 1)
/// - 敵玉が `to` の真正面 (先手は 1 つ上=row-1，後手は 1 つ下=row+1) の同一筋にいる
///
/// この配置では香成と香不成が同一マスへ向かう near-duplicate child となり，両者を
/// sum 計上すると delta が二重に膨れて breadth が発散する (KH の本質的な breadth 抑制)．
///
/// `defender_king` は OR ノードの受け方 (敵) 玉位置．`None` なら `true` (安全側)．
pub(super) fn is_sum_delta_node(
    m: Move,
    or_node: bool,
    us_is_black: bool,
    defender_king: Option<Square>,
    board: &Board,
) -> bool {
    if m.is_drop() || !or_node {
        return true;
    }
    // 移動元が (不成の) 香車か．成香 (raw 10) は除外され true を返す．
    if (board.piece_at(m.from_sq()) & 0x0F) != 2 {
        return true;
    }
    let to = m.to_sq();
    let king = match defender_king {
        Some(k) => k,
        None => return true,
    };
    if king.col() != to.col() {
        return true;
    }
    let r = to.row();
    if us_is_black {
        // 先手香は row 0 方向 (敵陣) へ進む．玉は to の 1 つ手前 (row-1)．
        if (r == 1 || r == 2) && king.row() + 1 == r {
            return false;
        }
    } else {
        // 後手香は row 8 方向 (敵陣) へ進む．玉は to の 1 つ手前 (row+1)．
        if (r == 6 || r == 7) && r + 1 == king.row() {
            return false;
        }
    }
    true
}

/// 持ち駒の要素ごと比較: a の全要素が b 以上なら true．
///
/// 証明駒の優越判定に使用: 持ち駒が多い方が有利(攻め方)．
///
/// SWAR (SIMD Within A Register): 7 バイトを u64 にパックし分岐なしで一括比較する．
/// 各バイトに MSB(0x80) をセットして引き算し，MSB が全て残れば全要素 a[i] >= b[i]．
/// 持ち駒値は 0-18 の範囲なので各バイトの MSB は常に 0 であり，
/// (a[i] + 128) - b[i] >= 110 となるためバイト間の桁借りは発生しない．
// SWAR パッキングは HAND_KINDS == 7 を前提とする(u64 の 7 バイトに収める)．
const _: () = assert!(HAND_KINDS == 7, "hand_gte SWAR assumes HAND_KINDS == 7");

// SWAR 比較はエンディアン非依存(両オペランドが同一バイト順序でパックされるため
// バイト単位の加減算とマスク演算は正しく動作する)が，明示性のため LE を使用する．
#[cfg(not(target_endian = "little"))]
compile_error!("hand_gte SWAR is tested only on little-endian targets");

#[inline(always)]
fn hand_gte(a: &[u8; HAND_KINDS], b: &[u8; HAND_KINDS]) -> bool {
    let a_packed = u64::from_le_bytes([a[0], a[1], a[2], a[3], a[4], a[5], a[6], 0]);
    let b_packed = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], 0]);
    const H: u64 = 0x8080_8080_8080_8080;
    ((a_packed | H) - b_packed) & H == H
}

/// 前方利き駒チェーン(歩 ≤ 香 ≤ 飛)を考慮した持ち駒の半順序比較．
///
/// 標準の `hand_gte`(駒種ごと独立比較)に加えて，
/// 前方利き駒の上位互換関係を利用した代替判定を行う:
///
/// - 飛車は香車の上位互換(飛車は前方を含む4方向に利く)
/// - 香車は歩の上位互換(香車は前方に複数マス利く)
///
/// これにより「歩で詰む → 香でも詰む → 飛でも詰む」が成立し，
/// TT の証明/反証エントリの再利用範囲が拡大する．
///
/// # アルゴリズム
///
/// 1. 高速パス: 標準の `hand_gte` で判定(大半はここで確定)
/// 2. 低速パス: 桂・銀・金・角は標準比較，歩・香・飛は
///    弱い駒種から順に不足分を上位駒種で補填するカスケード判定
///
/// # 正当性の根拠
///
/// 詰み証明で歩を使う手(打ち・移動)は，香で代替可能:
/// - 歩打ち → 香打ち: 前方の利きは香 ⊇ 歩
/// - 相手に歩を渡す → 香を渡す: 守備側の持ち駒が強くなるが，
///   これは攻め方にとって不利方向のため証明は依然有効
/// - 成り後: と金 = 成香(ともに金の動き)
///
/// 同様に香 → 飛も成立(飛の利き ⊇ 香の利き，竜 ⊇ 成香)．
#[inline(always)]
pub(super) fn hand_gte_forward_chain(a: &[u8; HAND_KINDS], b: &[u8; HAND_KINDS]) -> bool {
    // 高速パス: 全駒種で a[i] >= b[i] なら即 true
    if hand_gte(a, b) {
        return true;
    }
    // 桂(2)・銀(3)・金(4)・角(5): 標準比較(代替関係なし)
    if a[2] < b[2] || a[3] < b[3] || a[4] < b[4] || a[5] < b[5] {
        return false;
    }
    // 歩(0) ≤ 香(1) ≤ 飛(6) チェーン: 不足分をカスケード補填
    // deficit: 弱い駒種の不足を上位駒種で吸収していく
    let deficit = (b[0] as i16 - a[0] as i16).max(0);
    let deficit = (b[1] as i16 + deficit - a[1] as i16).max(0);
    b[6] as i16 + deficit <= a[6] as i16
}

/// 盤面のみのハッシュ(持ち駒を除外)を返す．
///
/// Board が `board_hash` をインクリメンタルに維持しているため O(1)．
/// 証明駒/反証駒による TT 参照で，同一盤面・異なる持ち駒の
/// エントリを同一スロットに集約するために使用する．
#[inline(always)]
fn position_key(board: &Board) -> u64 {
    board.board_hash
}

// --- 王手生成キャッシュ (E2 最適化) ---

/// 王手生成キャッシュのサイズ(2^13 = 8192 エントリ，direct-mapped)．
///
/// NOTE (2026-06-12, 拡大は非レバー): 2^18 (36MB) へ拡大しても 39te full solve は
/// 107.5s で不変 (gen_checks 20% は hit 率でなく，CHECK_CACHE_CAPACITY=32 を超える
/// 王手数の重い局面が cache 対象外なこと + memory-bound 下で memcpy ≈ 再生成のため)．
const CHECK_CACHE_SIZE: usize = 8192;

/// 1エントリあたりのキャッシュ容量(典型的な王手数は 3-15)．
const CHECK_CACHE_CAPACITY: usize = 32;

/// 王手生成キャッシュの1エントリ．
struct CheckCacheEntry {
    hash: u64,
    moves: ArrayVec<Move, CHECK_CACHE_CAPACITY>,
}

impl Default for CheckCacheEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            moves: ArrayVec::new(),
        }
    }
}

/// 局面ハッシュをキーとする王手リストのキャッシュ．
///
/// `generate_check_moves` の結果を direct-mapped テーブルに保存し，
/// 同一局面への再計算を回避する．MID ループで同一局面が繰り返し
/// 出現するため，キャッシュヒット率が高い．
///
/// 内部可変性(UnsafeCell)を使用して `&self` でアクセス可能にする．
/// これにより `generate_check_moves_cached` を `&self` で呼び出せ，
/// mid() のスタックフレーム最適化を阻害しない．
pub(super) struct CheckCache {
    table: std::cell::UnsafeCell<Vec<CheckCacheEntry>>,
}

impl CheckCache {
    pub(super) fn new() -> Self {
        let mut table = Vec::with_capacity(CHECK_CACHE_SIZE);
        for _ in 0..CHECK_CACHE_SIZE {
            table.push(CheckCacheEntry::default());
        }
        Self { table: std::cell::UnsafeCell::new(table) }
    }

    /// キャッシュから王手リストを取得し，ヒットした場合はコピーを返す．
    #[inline(always)]
    pub(super) fn get(&self, hash: u64) -> Option<ArrayVec<Move, CHECK_CACHE_CAPACITY>> {
        let table = unsafe { &*self.table.get() };
        let idx = (hash as usize) & (CHECK_CACHE_SIZE - 1);
        let entry = &table[idx];
        if entry.hash == hash {
            Some(entry.moves.clone())
        } else {
            None
        }
    }

    /// キャッシュ内の王手リストをコピーせず slice で借用する (zero-copy 経路)．
    ///
    /// 返り値の slice は次の `insert` まで有効．呼び出し側は借用中に本 cache へ
    /// 再挿入しうる処理 (`generate_check_moves_cached` 等) を呼ばないこと．
    #[inline(always)]
    pub(super) fn get_slice(&self, hash: u64) -> Option<&[Move]> {
        let table = unsafe { &*self.table.get() };
        let idx = (hash as usize) & (CHECK_CACHE_SIZE - 1);
        let entry = &table[idx];
        if entry.hash == hash {
            Some(entry.moves.as_slice())
        } else {
            None
        }
    }

    /// 王手リストをキャッシュに格納する．
    #[inline(always)]
    pub(super) fn insert(&self, hash: u64, moves: &ArrayVec<Move, MAX_MOVES>) {
        if moves.len() <= CHECK_CACHE_CAPACITY {
            let table = unsafe { &mut *self.table.get() };
            let idx = (hash as usize) & (CHECK_CACHE_SIZE - 1);
            let entry = &mut table[idx];
            entry.hash = hash;
            entry.moves.clear();
            for &m in moves.iter() {
                entry.moves.push(m);
            }
        }
    }
}

