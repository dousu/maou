//! DfPnSolver 構造体と探索エントリポイント．

use std::time::{Duration, Instant};

use crate::board::Board;
use crate::moves::Move;
use crate::types::Color;

use super::CheckCache;

/// path 配列の容量．depth の最大値(41) + マージン．
pub(super) const PATH_CAPACITY: usize = 48;

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
    /// `depth >= PATH_CAPACITY`(48)の場合パニックする．
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
}
