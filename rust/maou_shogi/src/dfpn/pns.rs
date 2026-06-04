//! 手生成，PNS (Proof Number Search)，PV 復元，公開 API．


#[cfg(feature = "profile")]
use std::time::Instant;

use crate::board::Board;

use super::solver::{DfPnSolver, TsumeResult};

impl DfPnSolver {

    /// 探索ノード数を返す．
    pub fn nodes_searched(&self) -> u64 {
        self.nodes_searched
    }
}


/// デフォルトタイムアウト(300秒)で詰将棋を解く便利関数．
///
/// [`solve_tsume_with_timeout`] のラッパーで，タイムアウト・PV ノード予算・
/// TT GC 閾値にはデフォルト値を使用する．
///
/// # 引数
///
/// - `sfen`: 局面の SFEN 文字列．
/// - `depth`: 最大探索手数(`None` でデフォルト 31)．
/// - `nodes`: 最大ノード数(`None` でデフォルト 1,048,576)．
/// - `draw_ply`: 引き分け手数(`None` でデフォルト 32767)．
///
/// # 戻り値
///
/// [`TsumeResult`] を返す．SFEN パースエラー時は `Err` を返す．
pub fn solve_tsume(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
) -> Result<TsumeResult, crate::board::SfenError> {
    solve_tsume_with_timeout(sfen, depth, nodes, draw_ply, None, None, None, None)
}

/// タイムアウト指定付きで詰将棋を解く便利関数．
///
/// # 戻り値
///
/// 詰みが証明された場合でも，PV 復元フェーズ(`complete_pv_or_nodes`)の
/// ノード予算が不足すると [`TsumeResult::CheckmateNoPv`] が返ることがある．
/// 特に長手数(17手以上)の詰将棋では，PV 沿いの各未証明子に対する
/// 追加証明の1子あたり予算(デフォルト 1024 ノード)が不足しやすい．
/// `pv_nodes_per_child` を増やすことで改善できる．
///
/// # 引数
///
/// - `find_shortest`: 最短手数探索を行うか(None でデフォルト true)．
///   false にすると `complete_or_proofs()` による追加探索をスキップし，
///   最初に見つかった詰み手順をそのまま返す．ノード数は削減されるが，
///   返される手順が最短とは限らない．
/// - `pv_nodes_per_child`: PV 復元時の1子あたりノード予算(None でデフォルト 1024)．
///   長手数の詰将棋で `CheckmateNoPv` が返る場合に増やすと効果的．
pub fn solve_tsume_with_timeout(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
    timeout_secs: Option<u64>,
    find_shortest: Option<bool>,
    pv_nodes_per_child: Option<u64>,
    tt_gc_threshold: Option<usize>,
) -> Result<TsumeResult, crate::board::SfenError> {
    let mut board = Board::empty();
    board.set_sfen(sfen)?;

    let mut solver = DfPnSolver::with_timeout(
        depth.unwrap_or(31),
        nodes.unwrap_or(1_048_576),
        draw_ply.unwrap_or(32767),
        timeout_secs.unwrap_or(300),
    );
    solver.set_find_shortest(find_shortest.unwrap_or(true));
    if let Some(budget) = pv_nodes_per_child {
        solver.set_pv_nodes_per_child(budget);
    }
    if let Some(gc) = tt_gc_threshold {
        solver.set_tt_gc_threshold(gc);
    }

    Ok(solver.solve(&mut board))
}

/// 詰将棋を解き，探索終了時の WorkingTT pn/dn 分布を返す (分析用)．
///
/// 返り値: `(TsumeResult, pn_hist, dn_hist, joint_hist)`
/// - pn_hist: pn 値の log2 ヒストグラム (32 バケット)
/// - dn_hist: dn 値の log2 ヒストグラム (32 バケット)
/// - joint_hist: (pn バケット × dn バケット) の 2D ヒストグラム (32×32 = 1024 要素)
/// - per_depth: IDS 各 depth の `(ids_depth, nodes, elapsed_secs, pn_hist, dn_hist, joint)`
pub fn solve_tsume_and_collect_pn_dn_dist(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    draw_ply: Option<u32>,
    timeout_secs: Option<u64>,
    find_shortest: Option<bool>,
    pv_nodes_per_child: Option<u64>,
    tt_gc_threshold: Option<usize>,
) -> Result<
    (
        TsumeResult,
        [u64; 32],
        [u64; 32],
        Vec<u64>,
        Vec<(u32, u64, f64, [u64; 32], [u64; 32], Vec<u64>)>,
    ),
    crate::board::SfenError,
> {
    let mut board = Board::empty();
    board.set_sfen(sfen)?;

    let mut solver = DfPnSolver::with_timeout(
        depth.unwrap_or(31),
        nodes.unwrap_or(1_048_576),
        draw_ply.unwrap_or(32767),
        timeout_secs.unwrap_or(300),
    );
    solver.set_find_shortest(find_shortest.unwrap_or(true));
    if let Some(budget) = pv_nodes_per_child {
        solver.set_pv_nodes_per_child(budget);
    }
    if let Some(gc) = tt_gc_threshold {
        solver.set_tt_gc_threshold(gc);
    }

    let result = solver.solve(&mut board);
    let (pn_hist, dn_hist, joint_hist) = solver.collect_pn_dn_dist();
    let per_depth = solver.collect_pn_dn_dist_per_depth().to_vec();
    Ok((result, pn_hist, dn_hist, joint_hist, per_depth))
}
