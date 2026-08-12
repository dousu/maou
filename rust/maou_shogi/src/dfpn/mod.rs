//! Df-Pn (Depth-First Proof-Number Search) による詰将棋ソルバー．
//!
//! Df-Pn (Nagai) アルゴリズムを採用し，攻め方が玉方を
//! 詰ませる最善手順を求める．先手・後手どちらが攻め方でも動作する．
//!
//! # 引数
//!
//! - `depth`: 最大探索手数(デフォルト 31，上限 2047)．無限ループ防止用．
//! - `nodes`: 最大ノード数(デフォルト 1,048,576 = 2^20)．計算時間・メモリ制限用．
//! - `timeout_secs`: タイムアウト秒数(デフォルト 300)．
//!
//! その他のオプション (`find_shortest` / `pv_nodes_per_child` / `tt_gc_threshold` /
//! `collect_progress`) は [`api`] の関数シグネチャを参照．

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

pub use api::{
    solve_tsume, solve_tsume_defense, solve_tsume_report_with_timeout, solve_tsume_with_timeout,
};
pub use solver::{DfPnSolver, ProgressSample, SearchReport, StopReason, TsumeResult};

/// 探索より前に TT バッファプールを温める (USI `isready` から呼ぶ想定)．
///
/// **なぜ要るか**: TT は `solve` ごとに確保され，`Entry::null()` が全ゼロで
/// ないため `alloc_zeroed` に落ちず**確保時に全バイトを書く**．そのため
/// **プロセス最初の探索だけ**が数百 MB 分の first-touch を負担し，
/// `byoyomi 1000` の初手が予算を超えることがある (実測 0.5〜3.5 秒)．
/// プール ([`tt`] の `BufferPool`) は 2 回目以降を安くするが，1 回目は救えない．
///
/// TensorRT のエンジンビルドを `isready` で前払いするのと同じ理屈で，
/// ここを `readyok` の前に寄せると初手が定常時と同じ速さになる
/// (実測: warm なし 1.889s → warm あり 0.502s = 2 手目以降と同値)．
/// USI は `isready` が長引くことを想定しており，maou_usi 側には
/// `--keep-alive-ms` の生存通知もある．
///
/// # 使い方
///
/// **実際に使うノード予算ごとに呼ぶこと．** プールはサイズ別バケットなので，
/// 予算が違えば別バケットになり温めた分は効かない．USI では攻め方向 root dfpn
/// (`root_dfpn_nodes`) と，**並行に走る**受け方向 dfpn
/// (`root_defensive_mate_nodes`) が別サイズで同時に生存するため両方要る．
///
/// 冪等 — 2 回目以降はプールから借りて返すだけなので安価．
pub fn warm_tt_pool(budget_nodes: u64) {
    // production 既定の floor (DfPnSolver::with_timeout と同じ)．leaf-mate の
    // 小 TT は per-leaf に作り捨てる規模なので温める価値がない．
    const DEFAULT_MIN_TT_ENTRIES: usize = 1 << 18;
    let size = search::tt_entries_for(budget_nodes, DEFAULT_MIN_TT_ENTRIES);
    // 確保して即 drop する — drop でプールへ戻るので，次の solve は pool hit
    // になり first-touch を払わない．
    drop(tt::TranspositionTable::new(size, size));
}

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
