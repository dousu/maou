//! 詰将棋ソルバーの公開 API (`solve_tsume*` 便利関数)．

use crate::board::Board;

use super::solver::{DfPnSolver, SearchReport, TsumeResult};

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
/// - `depth`: 最大探索手数(`None` でデフォルト 31; 上限 2047)．
/// - `nodes`: 最大ノード数(`None` でデフォルト 1,048,576)．
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
/// let result = solve_tsume("8k/9/7G1/9/9/9/9/9/9 b G 1", Some(3), Some(100_000)).unwrap();
/// match result {
///     TsumeResult::Checkmate { moves, .. } => assert_eq!(moves.len(), 1),
///     other => panic!("expected Checkmate, got {other:?}"),
/// }
/// ```
pub fn solve_tsume(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
) -> Result<TsumeResult, crate::board::SfenError> {
    solve_tsume_with_timeout(sfen, depth, nodes, None, None, None, None)
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
///   - `true`: **最短手数を確定できた場合のみ** [`TsumeResult::Checkmate`] を返す．
///     budget/timeout で最小性を証明しきれなかった場合は，非最小の詰みを返さず
///     [`TsumeResult::Unknown`] を返す (予算を増やせば解ける)．
///   - `false`: 最短保証を諦め，**最初に見つかった詰み手順を発見時点で返す**
///     (予算を使い切らず速い; 返される手順は最短とは限らない)．
/// - `pv_nodes_per_child`: PV 復元時の1子あたりノード予算(None でデフォルト 1024)．
///   長手数の詰将棋で `CheckmateNoPv` が返る場合に増やすと効果的．
pub fn solve_tsume_with_timeout(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    timeout_secs: Option<u64>,
    find_shortest: Option<bool>,
    pv_nodes_per_child: Option<u64>,
    tt_gc_threshold: Option<usize>,
) -> Result<TsumeResult, crate::board::SfenError> {
    solve_tsume_report_with_timeout(
        sfen,
        depth,
        nodes,
        timeout_secs,
        find_shortest,
        pv_nodes_per_child,
        tt_gc_threshold,
        None,
    )
    .map(|report| report.result)
}

/// [`solve_tsume_with_timeout`] と同一の探索を行い，結果に加えて診断メタデータ
/// ([`SearchReport`]) を返す．
///
/// `unknown` の内訳判断 (最小性未確定 / ノード切れ / タイムアウト / 偽証明) や
/// 「予算/時間を追加すれば現実的に解けるか」の見積り (root pn/dn の推移，検出済み
/// 詰み手数) に使う．
///
/// # 引数 (追加分)
///
/// - `collect_progress`: `Some(true)` で root 反復深化の各段階の pn/dn/nodes/経過時間を
///   [`SearchReport::progress`] に記録する (既定 `None` = 記録せず，Vec 確保もしないため
///   探索性能に影響しない)．
#[allow(clippy::too_many_arguments)]
pub fn solve_tsume_report_with_timeout(
    sfen: &str,
    depth: Option<u32>,
    nodes: Option<u64>,
    timeout_secs: Option<u64>,
    find_shortest: Option<bool>,
    pv_nodes_per_child: Option<u64>,
    tt_gc_threshold: Option<usize>,
    collect_progress: Option<bool>,
) -> Result<SearchReport, crate::board::SfenError> {
    let mut board = Board::empty();
    board.set_sfen(sfen)?;

    let mut solver = DfPnSolver::with_timeout(
        depth.unwrap_or(31),
        nodes.unwrap_or(1_048_576),
        timeout_secs.unwrap_or(300),
    );
    solver.set_find_shortest(find_shortest.unwrap_or(true));
    if let Some(budget) = pv_nodes_per_child {
        solver.set_pv_nodes_per_child(budget);
    }
    if let Some(gc) = tt_gc_threshold {
        solver.set_tt_gc_threshold(gc);
    }
    solver.set_collect_progress(collect_progress.unwrap_or(false));

    Ok(solver.solve_report(&mut board))
}

#[cfg(test)]
mod report_tests {
    use super::*;
    use crate::dfpn::StopReason;

    // 後手玉 1一，先手金 2三，先手持ち駒: 金．1 手詰 (自明に最短)．
    const MATE_1TE: &str = "8k/9/7G1/9/9/9/9/9/9 b G 1";
    // 29 手詰 canonical．first-mate(31 手) は ~7K node で見つかるが，最短 29 手の確定には
    // ~396K node 必要．予算 50K では最小性を証明できず MinimalityUnconfirmed の unknown．
    const MATE_29TE: &str =
        "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";

    #[test]
    fn report_solved_1te_carries_metadata() {
        let report = solve_tsume_report_with_timeout(
            MATE_1TE,
            Some(3),
            Some(100_000),
            None,
            Some(true),
            None,
            None,
            Some(true),
        )
        .unwrap();
        assert!(matches!(report.result, TsumeResult::Checkmate { .. }));
        assert_eq!(report.stop_reason, StopReason::Solved);
        assert_eq!(report.mate_len_found, Some(1));
        assert!(report.shortest_confirmed);
        // 詰み証明ゆえ root_pn=0．
        assert_eq!(report.root_pn, 0);
        // collect_progress=true → root 反復が最低 1 点は記録される．
        assert!(!report.progress.is_empty());
        assert!(report.progress[0].nodes > 0);
    }

    #[test]
    fn report_progress_empty_when_not_collected() {
        // collect_progress 未指定 (既定 false) では progress は空 (アロケーション無し)．
        let report = solve_tsume_report_with_timeout(
            MATE_1TE,
            Some(3),
            Some(100_000),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(report.progress.is_empty());
        assert_eq!(report.stop_reason, StopReason::Solved);
    }

    #[test]
    fn report_minimality_unconfirmed_carries_best_mate() {
        // find_shortest=true・予算 50K では最小性を証明できず MinimalityUnconfirmed の unknown．
        // このとき，そこまでに STRICT verify 済みの詰み手順を best_mate に残す (大きなリソースを
        // 費した探索の成果を最低限保持する)．
        let report = solve_tsume_report_with_timeout(
            MATE_29TE,
            Some(31),
            Some(50_000),
            None,
            Some(true), // find_shortest
            None,
            None,
            None,
        )
        .unwrap();
        assert!(matches!(report.result, TsumeResult::Unknown { .. }));
        assert_eq!(report.stop_reason, StopReason::MinimalityUnconfirmed);
        assert!(!report.shortest_confirmed);
        // 詰み自体は検証済 → mate_len_found=Some(d) かつ best_mate.len()==d の詰み手順が残る．
        let d = report
            .mate_len_found
            .expect("mate_len_found should be Some");
        assert_eq!(d % 2, 1, "詰将棋の手数は奇数");
        assert!(d >= 29);
        assert_eq!(
            report.best_mate.len(),
            d as usize,
            "best_mate は mate_len_found と同じ手数の詰み手順"
        );
        // 攻め方 (root 手番) から始まり，全手が合法な非 NONE 手であること．
        assert!(report.best_mate.iter().all(|m| m.raw_u32() != 0));
    }

    #[test]
    fn report_no_mate_is_disproven() {
        // 玉のみ (攻め方に王手手段なし) → 不詰 (dn=0)．
        let report = solve_tsume_report_with_timeout(
            "4k4/9/9/9/9/9/9/9/4K4 b - 1",
            Some(31),
            Some(100_000),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(matches!(report.result, TsumeResult::NoCheckmate { .. }));
        assert_eq!(report.stop_reason, StopReason::Disproven);
        assert_eq!(report.root_dn, 0);
        assert_eq!(report.mate_len_found, None);
    }
}
