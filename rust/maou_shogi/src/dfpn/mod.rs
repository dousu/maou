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

mod api;
mod check_cache;
mod delayed_move_list;
mod heuristics;
mod local_expansion;
mod mate1ply;
mod mate_len;
mod mid;
mod node_movegen;
mod path_key;
mod path_stack;
mod proof_hand;
mod pv;
mod search_result;
mod solver;
#[cfg(test)]
mod tests;
mod tt;

pub use api::{solve_tsume, solve_tsume_and_collect_pn_dn_dist, solve_tsume_with_timeout};
pub use solver::{DfPnSolver, TsumeResult};

// heuristics / check_cache / proof_hand へ移設した item を root に再エクスポートし，
// sibling module から従来どおり `super::<name>` で参照できるようにする．
use check_cache::CheckCache;
use heuristics::{init_pn_dn_and, init_pn_dn_or, move_brief_eval};
use proof_hand::hand_gte;

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

/// 盤面のみのハッシュ(持ち駒を除外)を返す．
///
/// Board が `board_hash` をインクリメンタルに維持しているため O(1)．
/// 証明駒/反証駒による TT 参照で，同一盤面・異なる持ち駒の
/// エントリを同一スロットに集約するために使用する．
#[inline(always)]
fn position_key(board: &Board) -> u64 {
    board.board_hash
}

