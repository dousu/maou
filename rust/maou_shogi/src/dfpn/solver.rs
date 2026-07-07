//! DfPnSolver 構造体と探索エントリポイント．

use std::time::{Duration, Instant};

use crate::board::Board;
use crate::moves::Move;
use crate::types::Color;

use super::CheckCache;

/// path 配列の容量 (= 探索深さの上限 + 1)．
///
/// **長手数詰将棋 (ミクロコスモス級 1525 手など) に対応するため 2048**．
/// `depth < PATH_CAPACITY` を assert し，`path_key` の per-ply Zobrist テーブルも
/// この幅で確保する ([`super::path_key`])．path stack (`path_depths`/`dom_path`) や
/// `expansion_stack` は実深さで動的成長するため PATH_CAPACITY には依存しない．
pub(super) const PATH_CAPACITY: usize = 2048;

/// 詰将棋の探索結果．
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TsumeResult {
    /// 詰みが見つかった場合．手順を含む．
    Checkmate {
        moves: Vec<Move>,
        nodes_searched: u64,
    },
    /// 詰みは証明済みだが PV (手順) の復元に失敗した場合．
    ///
    /// PV 復元 (STRICT verify) の予算不足等により手順を構築できない
    /// ケースで返される．`pv_nodes_per_child` を増やすと改善する．
    CheckmateNoPv { nodes_searched: u64 },
    /// 不詰の場合．
    NoCheckmate { nodes_searched: u64 },
    /// 探索制限に達した場合(nodes上限 or depth上限)．
    Unknown { nodes_searched: u64 },
}

/// 探索の停止理由．[`SearchReport::stop_reason`] で返され，`unknown` の内訳
/// (予算/時間切れ・最小性未確定・偽証明) を機械可読に区別する．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// 詰みを返した (find_shortest=true なら最短確定，false なら最初の詰み)．
    Solved,
    /// 不詰を証明した (dn==0)．
    Disproven,
    /// 詰みは検証済 (Some(d)) だが，find_shortest の最小性 (len=d-2 の不詰) を
    /// 予算/時間内に証明しきれなかった → 非最小の詰みを返さず unknown．
    /// 予算を増やせば [`StopReason::Solved`] になる可能性が高い．
    MinimalityUnconfirmed,
    /// ノード上限に達し未解決 (pn≠0 かつ dn≠0)．予算追加の余地あり．
    NodesExhausted,
    /// タイムアウトで未解決 (pn≠0 かつ dn≠0)．時間追加の余地あり．
    Timeout,
    /// STRICT verify が偽証明/不完全と判定 → 偽詰み回避のため unknown．
    FalseProof,
    /// 上記以外の未解決 (閾値 cap 到達等，稀)．
    Inconclusive,
}

impl StopReason {
    /// Python バインディング等向けの snake_case 文字列表現．
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::Solved => "solved",
            StopReason::Disproven => "disproven",
            StopReason::MinimalityUnconfirmed => "minimality_unconfirmed",
            StopReason::NodesExhausted => "nodes_exhausted",
            StopReason::Timeout => "timeout",
            StopReason::FalseProof => "false_proof",
            StopReason::Inconclusive => "inconclusive",
        }
    }
}

/// 探索進捗のスナップショット．
///
/// `collect_progress=true` のとき，root 反復深化ループの各イテレーションで
/// 1 点ずつ記録される (初回 DEPTH_MAX 探索および find_shortest の各短縮探索を含む)．
/// pn が 0 へ下降しているか (詰みに接近)／dn が 0 へ下降しているか (不詰に接近) で，
/// 予算追加の有効性を外挿判断できる．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressSample {
    /// この時点の累積探索ノード数 (emplace 呼び出し回数)．
    pub nodes: u64,
    /// 探索開始からの経過時間 (ミリ秒)．
    pub elapsed_ms: u64,
    /// root の証明数 (0 = 詰み証明)．K_INFINITE_PN_DN (≈9.2e18) は ∞ を表す．
    pub pn: u64,
    /// root の反証数 (0 = 不詰証明)．
    pub dn: u64,
    /// この反復で探索している mate-length 上限 (初回は DEPTH_MAX，短縮探索では d-2)．
    pub mate_len: u32,
}

/// 詰将棋探索の結果 + 診断メタデータ．
///
/// [`TsumeResult`] (結果本体) に加え，`unknown` の内訳判断や「予算/時間を追加すれば
/// 現実的に解けるか」の見積りに必要な情報 (root pn/dn・経過時間・検出済み詰み手数・
/// 停止理由・進捗トラジェクトリ) を持つ．探索終了時に 1 回だけ構築されるため
/// per-node コストは無い ([`DfPnSolver::solve_report`] で取得)．
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchReport {
    /// 結果本体 (詰み手順を含む)．
    pub result: TsumeResult,
    /// 停止時の root 証明数 (詰み時は 0)．
    pub root_pn: u64,
    /// 停止時の root 反証数 (不詰時は 0)．
    pub root_dn: u64,
    /// 探索開始から結果確定までの経過時間 (ミリ秒，verify フェーズ込み)．
    pub elapsed_ms: u64,
    /// 詰みが検証された場合の手数 Some(d)．`unknown` (最小性未確定) でも詰みが
    /// 見つかっていれば Some(d) が入り「≤d 手の詰みが存在する」ことを意味する．
    pub mate_len_found: Option<u32>,
    /// find_shortest で最短手数を確定できたか (find_shortest=false では常に false)．
    pub shortest_confirmed: bool,
    /// 停止理由 (unknown の内訳を区別する)．
    pub stop_reason: StopReason,
    /// 進捗トラジェクトリ (`collect_progress=true` 時のみ記録; 既定は空)．
    pub progress: Vec<ProgressSample>,
    /// find_shortest=true が最小性未確定 (`StopReason::MinimalityUnconfirmed` の unknown) で
    /// 終わったが，探索中に STRICT verify 済みの詰み手順が得られている場合の「暫定最短」手順．
    ///
    /// 予算/時間切れで最短だけ確定できなくても，そこまでに見つけた最短の詰み手順を残すことで
    /// 大きなリソースを費した探索の成果を最低限保持する (`mate_len_found = Some(d)` に対応し
    /// `best_mate.len() == d`; 「≤d 手の詰みが存在する」ことの constructive proof)．**最短である
    /// ことは保証しない**．`result` が [`TsumeResult::Checkmate`] の場合 (最短確定手順は
    /// `result` 側の `moves` に入る) や，詰みが得られなかった場合は空．
    pub best_mate: Vec<Move>,
}

/// Df-Pn ソルバー．
///
/// 統一探索本体 (`search/mod.rs` の `solve_impl`) を駆動する．
pub struct DfPnSolver {
    /// 最大ノード数．
    pub(super) max_nodes: u64,
    /// 実行時間制限．
    pub(super) timeout: Duration,
    /// 探索ノード数．
    pub(super) nodes_searched: u64,
    /// 探索開始時刻．
    pub(super) start_time: Instant,
    /// タイムアウトしたかどうか．
    pub(super) timed_out: bool,
    /// 攻め方の手番色(solve 時に設定)．
    pub(super) attacker: Color,
    /// 最短手数探索を行うかどうか(デフォルト: true)．
    ///
    /// - true: 最初の詰み発見後に mate-length パラメータ化再探索 (`len = d-2` の反復) で
    ///   最短手数を確定させる．**最小性を証明できた場合のみ [`TsumeResult::Checkmate`]**，
    ///   budget/timeout で証明しきれなければ [`TsumeResult::Unknown`] を返す (非最小の詰みは返さない)．
    /// - false: 短縮再探索をせず，最初に見つかった詰み手順を発見時点で返す (予算を使い切らず速い;
    ///   最短保証なし)．
    pub(super) find_shortest: bool,
    /// PV 復元フェーズで未証明子1つあたりに割り当てるノード予算(デフォルト: 1024)．
    ///
    /// 長手数の詰将棋で [`TsumeResult::CheckmateNoPv`] が返る場合，
    /// この値を増やすことで PV 復元の成功率が向上する．
    pub(super) pv_nodes_per_child: u64,
    /// TT GC 閾値: TT のエントリ数がこの値を超えると GC を実行する．
    ///
    /// 0 にすると GC を無効化する．
    /// デフォルトは 0(無効)．超長手数問題で OOM を防ぐ場合に設定する．
    /// 推奨値: 探索ノード数の 1/5〜1/2 程度(例: 100M ノードなら 20M〜50M)．
    pub(super) tt_gc_threshold: usize,
    /// 王手生成キャッシュ．
    pub(super) check_cache: CheckCache,
    /// mid 探索パス上の board.hash → ply (千日手検出 + 参照祖先 ply 特定用)．
    /// 連続パス保持と同型の flat スタック ([`super::path_stack::PathStack`])．
    /// `FxHashMap` の散在 DRAM アクセスを排して memory-bound を解消する (探索不変)．
    pub(super) path_depths: super::path_stack::PathStack,
    /// mid 探索ノード数 (= emplace 呼び出し回数)．
    pub(super) nodes: u64,
    /// mid double-count elimination 用の明示的 expansion stack．
    /// 各 search_impl frame が自 LocalExpansion を push/truncate し，祖先を辿れるようにする．
    pub(super) expansion_stack: Vec<super::search::expansion::LocalExpansion>,
    /// mid build_expansion の per-node Vec 再利用 pool (node pop で返却・再取得)．
    /// ヒープ alloc/free を削減する (探索不変; [`super::search::BufPool`])．
    pub(super) expansion_buf_pool: super::search::BufPool,
    /// mid double-count elimination 発火数 (診断用)．
    pub(super) dag_fires: u64,
    /// mid 探索 path 上の hand-dominance 反復検出．
    /// position_key → 現探索 path 上の祖先 `(attacker_hand, depth)` のスタック．
    /// 子局面が同一 board_key かつ攻め方持駒が祖先以下 (= 劣位) なら反復として刈る．
    /// `board.hash` だけの exact 千日手 (`path_depths`) を持駒 superset 方向へ一般化する．
    /// flat スタック実装 ([`super::path_stack::DomPathStack`])．
    pub(super) dom_path: super::path_stack::DomPathStack,
    /// mid dominance pruning 発火数 (診断用)．
    pub(super) dom_fires: u64,
    /// 進捗トラジェクトリを記録するか (既定: false)．
    pub(super) collect_progress: bool,
    /// 進捗トラジェクトリ (collect_progress=true 時に root 反復ごとに push)．
    pub(super) progress: Vec<ProgressSample>,
}

impl DfPnSolver {
    /// 新しいソルバーを生成する(タイムアウト 300 秒)．
    pub fn new(depth: u32, max_nodes: u64) -> Self {
        Self::with_timeout(depth, max_nodes, 300)
    }

    /// デフォルトパラメータでソルバーを生成する．
    pub fn default_solver() -> Self {
        Self::new(31, 1_048_576)
    }

    /// タイムアウト指定付きでソルバーを生成する．
    ///
    /// # Panics
    ///
    /// `depth >= PATH_CAPACITY`(2048)の場合パニックする．
    pub fn with_timeout(depth: u32, max_nodes: u64, timeout_secs: u64) -> Self {
        assert!(
            (depth as usize) < PATH_CAPACITY,
            "depth {} exceeds path capacity {}",
            depth,
            PATH_CAPACITY,
        );
        DfPnSolver {
            max_nodes,
            timeout: Duration::from_secs(timeout_secs),
            find_shortest: true,
            pv_nodes_per_child: 1024,
            check_cache: CheckCache::new(),
            tt_gc_threshold: 0,
            path_depths: super::path_stack::PathStack::new(),
            nodes: 0,
            expansion_stack: Vec::new(),
            expansion_buf_pool: super::search::BufPool::default(),
            dag_fires: 0,
            dom_path: super::path_stack::DomPathStack::new(),
            dom_fires: 0,
            nodes_searched: 0,
            start_time: Instant::now(),
            timed_out: false,
            attacker: Color::Black,
            collect_progress: false,
            progress: Vec::new(),
        }
    }

    /// 最短手数探索の有無を設定する．
    ///
    /// `false` にすると最初に見つかった詰み手順をそのまま返す
    /// (最短保証なしのノード数削減)．デフォルトは true．
    pub fn set_find_shortest(&mut self, v: bool) -> &mut Self {
        self.find_shortest = v;
        self
    }

    /// PV 復元フェーズの1子あたりノード予算を設定する．
    ///
    /// デフォルトは 1024．長手数(17手以上)の詰将棋で
    /// `CheckmateNoPv` が返る場合に増やすと効果的．
    pub fn set_pv_nodes_per_child(&mut self, v: u64) -> &mut Self {
        self.pv_nodes_per_child = v;
        self
    }

    /// TT GC 閾値を設定する．
    ///
    /// TT のエントリ数がこの値を超えると GC を実行する．
    /// 0 にすると GC を無効化する．デフォルトは 0(GC 無効)．
    pub fn set_tt_gc_threshold(&mut self, v: usize) -> &mut Self {
        self.tt_gc_threshold = v;
        self
    }

    /// 進捗トラジェクトリ収集の有無を設定する．
    ///
    /// `true` にすると root 反復深化ループの各イテレーションで pn/dn/nodes/経過時間を
    /// [`SearchReport::progress`] に記録する ([`DfPnSolver::solve_report`] で取得)．
    /// per-node ではなく反復単位ゆえ低コスト．既定は false (記録せず，Vec の確保も
    /// 行わないため探索性能に影響しない)．
    pub fn set_collect_progress(&mut self, v: bool) -> &mut Self {
        self.collect_progress = v;
        self
    }

    /// タイムアウトしたかどうかを返す．
    #[inline]
    pub(super) fn is_timed_out(&self) -> bool {
        self.timed_out || self.start_time.elapsed() >= self.timeout
    }

    /// 詰将棋を解く．
    ///
    /// `board` は攻め方の手番から開始する局面．
    /// 片玉局面(攻め方に玉がない)を想定するが，両玉でも動作する．
    ///
    /// 実体は df-pn 統一探索 `solve_impl` (`search/mod.rs`)．`max_nodes` /
    /// `timeout` を尊重し，未解決 (budget/timeout) は [`TsumeResult::Unknown`]
    /// を返す (false NoMate を出さない)．
    pub fn solve(&mut self, board: &mut Board) -> TsumeResult {
        self.solve_impl(board)
    }

    /// 詰将棋を解き，結果 + 診断メタデータ ([`SearchReport`]) を返す．
    ///
    /// [`solve`](Self::solve) と同一の探索を行うが，`unknown` の内訳判断や
    /// 「予算/時間を追加すれば解けるか」の見積りに使う情報 (root pn/dn・経過時間・
    /// 検出済み詰み手数・停止理由・進捗トラジェクトリ) を付随して返す．追加コストは
    /// 探索終了時の 1 回だけで per-node 性能には影響しない．
    pub fn solve_report(&mut self, board: &mut Board) -> SearchReport {
        self.solve_report_impl(board)
    }
}
