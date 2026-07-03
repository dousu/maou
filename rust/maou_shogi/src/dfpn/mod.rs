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

mod api;
mod heuristics;
mod mate_len;
mod movegen;
mod path_key;
mod path_stack;
mod proof_hand;
mod search;
mod search_result;
mod solver;
#[cfg(test)]
mod tests;
mod tt;

pub use api::{solve_tsume, solve_tsume_with_timeout};
pub use solver::{DfPnSolver, TsumeResult};

// sibling module から `super::<name>` で参照するための再エクスポート．
use heuristics::{init_pn_dn_and, init_pn_dn_or, move_brief_eval};
use movegen::check_cache::CheckCache;
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
/// PN_UNIT=1 が素の df-pn に相当し，PN_UNIT を拡大することで中間的な
/// 初期値 (例 1.5 単位相当) を表現でき，heuristic と閾値配分の解像度が上がる．
///
/// スケーリング契約: pn/dn の「量」を表す全ての定数
/// (`init_pn_dn_*` の初期値・エッジコスト等の加算値・フロア値) に
/// PN_UNIT を適用する．終端値 (INF, 0)・相対比率・盤面状態の比較・
/// ループカウンタはスケール対象外．
const PN_UNIT: u32 = 16;

/// 盤面のみのハッシュ(持ち駒を除外)を返す．
///
/// Board が `board_hash` をインクリメンタルに維持しているため O(1)．
/// 証明駒/反証駒による TT 参照で，同一盤面・異なる持ち駒の
/// エントリを同一スロットに集約するために使用する．
#[inline(always)]
fn position_key(board: &Board) -> u64 {
    board.board_hash
}
