//! 詰将棋ソルバーの公開 API (`solve_tsume*` 便利関数)．

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
///
/// # 例
///
/// ```
/// use maou_shogi::dfpn::{solve_tsume, TsumeResult};
///
/// // 後手玉 1一，先手金 2三，先手持ち駒: 金．G*1b (または G*2b) の 1 手詰．
/// let result = solve_tsume("8k/9/7G1/9/9/9/9/9/9 b G 1", Some(3), Some(100_000), None).unwrap();
/// match result {
///     TsumeResult::Checkmate { moves, .. } => assert_eq!(moves.len(), 1),
///     other => panic!("expected Checkmate, got {other:?}"),
/// }
/// ```
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
/// 詰みが証明された場合でも，PV 復元フェーズのノード予算が不足すると
/// [`TsumeResult::CheckmateNoPv`] が返ることがある．特に長手数(17手以上)の
/// 詰将棋では1子あたり予算(デフォルト 1024 ノード)が不足しやすい．
/// `pv_nodes_per_child` を増やすことで改善できる．
///
/// # 引数
///
/// - `find_shortest`: 最短手数探索を行うか(None でデフォルト true)．
///   false にすると最短手数を確定させる再探索をスキップし，
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
