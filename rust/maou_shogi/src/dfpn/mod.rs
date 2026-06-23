//! Df-Pn (Depth-First Proof-Number Search) による詰将棋ソルバー．
//!
//! Df-Pn (Nagai) アルゴリズムを採用し，攻め方が玉方を
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
/// 現在の唯一の使用箇所は `#[cfg(test)] mod tests` 内の診断テストであるため，
/// lib build での unused_macros 警告を避けるべく test build のみで定義する．
#[cfg(test)]
macro_rules! verbose_eprintln {
    ($($arg:tt)*) => {
        #[cfg(feature = "verbose")]
        eprintln!($($arg)*)
    };
}

mod delayed_move_list;
mod entry;
mod local_expansion;
mod mate_len;
mod mid;
mod node_movegen;
mod path_key;
mod path_stack;
mod pns;
mod proof_hand;
mod search_result;
mod solver;
#[cfg(test)]
mod tests;
mod tt;
mod ttentry;

pub use pns::{solve_tsume, solve_tsume_and_collect_pn_dn_dist, solve_tsume_with_timeout};
pub use solver::{DfPnSolver, TsumeResult};

/// 王手手/応手の最大数．
/// 将棋の合法手上限は593であり，長手数の詰将棋では
/// 持ち駒が多い局面で320を超えるケースが存在するため，
/// 合法手上限に合わせる．
const MAX_MOVES: usize = 593;

/// `ArrayVec::try_push` のラッパー．
/// デバッグビルドでは容量超過時にパニックし，リリースビルドでは無音で破棄する．
#[inline(always)]
fn push_move<T, const N: usize>(buf: &mut ArrayVec<T, N>, val: T) {
    let result = buf.try_push(val);
    debug_assert!(
        result.is_ok(),
        "move buffer overflow: capacity {N} exceeded"
    );
}

/// pn/dn の 1 単位を表す定数．
///
/// 全ての pn/dn 初期値・加算定数・フロア値はこの定数の倍数で表現する．
/// PN_UNIT=1 が単位スケールに相当し，PN_UNIT を拡大することで
/// 1+ε 閾値の余裕を確保し閾値飢餓を緩和できる．
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

// MID ループの dn 閾値フロア(スラッシング防止)は
// `DfPnSolver::param_dn_floor_mult` (デフォルト 100) で管理する．
// 子ノードの dn が小さすぎると MID ループが閾値超過で即座に返り，
// 進捗のない空転が発生するため，dn_threshold を最低
// `param_dn_floor_mult * PN_UNIT` まで引き上げる．

/// `param_disproof_remaining_threshold` の depth-adaptive モードを示すセンチネル値．
///
/// この値が設定されている場合，depth-limited disproof 格納閾値は
/// `outer_solve_depth` に基づいて自動決定される．
/// 詳細は `DfPnSolver::effective_disproof_remaining_threshold` 参照．
pub(super) const DISPROOF_THRESHOLD_ADAPTIVE: u16 = u16::MAX;

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

/// `to` への利き数に玉の隣接分を補正する．
///
/// `compute_checkers_at` は玉を除外して利き数を数えるため，玉が `to` に
/// 隣接していれば +1 して玉込みの利き数にする (initial pn/dn 推定用)．
#[cfg_attr(feature = "effect_table", allow(dead_code))]
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

/// OR ノード(攻め方の王手)の per-move (pn, dn) tuple を返す．
///
/// 加算は全て PN_UNIT 単位:
/// - 受け駒 ≥ 2: pn += PN_UNIT (後回し)
/// - 攻め支援 + drop_bonus > 受け支援: dn += PN_UNIT (探索優先)
/// - 金/銀取り: dn += PN_UNIT
/// - その他の駒取り: pn += PN_UNIT
/// - 静か手: pn += PN_UNIT
pub(super) fn init_pn_dn_or(
    board: &crate::board::Board,
    m: Move,
    attacker: crate::types::Color,
) -> (u32, u32) {
    let mut pn = PN_UNIT;
    let mut dn = PN_UNIT;

    let to = m.to_sq();
    // attack/defense support = `to` への玉込み利き数．
    // effect テーブル有効時は `effect_count` (玉込みの全利き数) が
    // `compute_checkers_at(玉除外).count() + king_supports(玉補正)` と完全一致するため，
    // per-child の compute_checkers_at × 2 を参照に置換する (探索不変)．
    #[cfg(feature = "effect_table")]
    let (attack_support, defense_support) = (
        u32::from(board.effect_count(attacker, to)),
        u32::from(board.effect_count(attacker.opponent(), to)),
    );
    #[cfg(not(feature = "effect_table"))]
    let (attack_support, defense_support) = {
        let att_bb = board.compute_checkers_at(to, attacker);
        let def_bb = board.compute_checkers_at(to, attacker.opponent());
        // compute_checkers_at は玉を除外するので，玉が `to` に隣接していれば
        // support に +1 して玉込みの利き数にする．
        (
            att_bb.count() + king_supports(board, to, attacker),
            def_bb.count() + king_supports(board, to, attacker.opponent()),
        )
    };
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

/// AND ノード(防御側の応手)の per-move (pn, dn) tuple を返す (U = PN_UNIT)．
///
/// - 駒取り応手: (2U, U)
/// - 玉移動: (U, U)
/// - 攻め支援 < 受け支援 + drop_bonus (good escape): (2U, U)
/// - その他 (bad escape): (U, 2U)
pub(super) fn init_pn_dn_and(
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
    // attack/defense support = `to` への玉込み利き数 (effect 有効時は参照に置換; 探索不変)．
    #[cfg(feature = "effect_table")]
    let (attack_support, defense_support) = (
        u32::from(board.effect_count(attacker, to)),
        u32::from(board.effect_count(defender, to)),
    );
    #[cfg(not(feature = "effect_table"))]
    let (attack_support, defense_support) = {
        let att_bb = board.compute_checkers_at(to, attacker);
        let def_bb = board.compute_checkers_at(to, defender);
        // compute_checkers_at は玉を除外するので，玉が `to` 隣接なら support に +1 (玉込みの利き数)．
        (
            att_bb.count() + king_supports(board, to, attacker),
            def_bb.count() + king_supports(board, to, defender),
        )
    };
    let drop_bonus: u32 = if m.is_drop() { 1 } else { 0 };

    if attack_support < defense_support + drop_bonus {
        // good escape
        (2 * PN_UNIT, PN_UNIT)
    } else {
        // bad escape
        (PN_UNIT, 2 * PN_UNIT)
    }
}

/// move ordering key を返す (値が小さいほど「良い手」)．
///
/// 基準:
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
        1 => 10,                // Pawn
        2 => 20,                // Lance
        3 => 20,                // Knight
        4 => 30,                // Silver
        5 => 50,                // Bishop
        6 => 50,                // Rook
        7 => 50,                // Gold
        8 => 80, // King (AND 玉捕獲手の tie-break に必要)
        9 | 10 | 11 | 12 => 50, // ProPawn..ProSilver
        13 => 80, // Horse
        14 => 80, // Dragon
        _ => 0,
    };
    value -= pt_value;

    let dc = (to.col() as i32 - king_sq.col() as i32).abs();
    let dr = (to.row() as i32 - king_sq.row() as i32).abs();
    value += 10 * dc.max(dr);

    value
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

/// 盤面のみのハッシュ(持ち駒を除外)を返す．
///
/// Board が `board_hash` をインクリメンタルに維持しているため O(1)．
/// 証明駒/反証駒による TT 参照で，同一盤面・異なる持ち駒の
/// エントリを同一スロットに集約するために使用する．
#[inline(always)]
fn position_key(board: &Board) -> u64 {
    board.board_hash
}

// --- 王手生成キャッシュ ---

/// 王手生成キャッシュのサイズ(2^13 = 8192 エントリ，direct-mapped)．
const CHECK_CACHE_SIZE: usize = 8192;

/// 1エントリあたりのキャッシュ容量(典型的な王手数は 3-15)．
/// この容量を超える王手数の局面は cache 対象外となる．
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
        Self {
            table: std::cell::UnsafeCell::new(table),
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
