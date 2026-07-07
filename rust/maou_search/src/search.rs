//! PUCT ベースの MCTS 探索本体．
//!
//! # バッチ収集の仕組み
//!
//! 各探索スレッドは以下のループを回す:
//!
//! 1. ルートから PUCT で降下し，未展開の葉を最大 `batch_size` 個収集する．
//!    降下中は経路上の全ノードの visits を前置インクリメントする
//!    (virtual loss — 評価結果が返るまでその経路の Q を押し下げ，
//!    同一バッチ内・スレッド間の選択を別の枝へ分散させる)．
//! 2. 収集した葉を [`Evaluator::evaluate_batch`] でまとめて評価する．
//! 3. 各葉を展開し (priors を辺に設定)，value を経路に沿って
//!    バックプロパゲーションする．
//!
//! 他スレッドが評価中 (`EXPANDING`) の葉に到達した場合は衝突として
//! 前置 visits をロールバックし，収集を打ち切ってバッチを即時 flush する．
//! 衝突率とバッチ充填率は [`SearchStats`] で観測できる．

use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use maou_shogi::board::{Board, SfenError};
use maou_shogi::movegen::generate_legal_moves;
use maou_shogi::moves::Move;

use crate::evaluator::{EvalItem, Evaluator};
use crate::tree::{node_state, Edge, NodePool, NULL_NODE};

/// ルートノードの pool index (最初の alloc で必ず 0 になる)．
const ROOT_IDX: u32 = 0;

/// 上限指定なし ([`SearchLimits`] が両方 `None`) のときの playout 上限．
const DEFAULT_MAX_PLAYOUTS: u64 = 1 << 20;

/// PV 復元の最大長．
const MAX_PV_LEN: usize = 64;

/// 探索の設定．
#[derive(Clone, Debug)]
pub struct SearchOptions {
    /// 探索スレッド数．
    pub threads: usize,
    /// 1 スレッドが一度に収集・評価する葉の最大数 (評価バッチサイズ)．
    pub batch_size: usize,
    /// PUCT の探索定数．
    pub c_puct: f32,
    /// 未訪問の子に与える親視点 Q の初期値 (first play urgency)．
    pub fpu: f32,
    /// ノードプール容量 (メモリ上限)．到達すると探索を停止する．
    pub node_capacity: u32,
    /// 最大探索深さ．到達した経路は引き分け (0.5) として打ち切る
    /// (千日手未検出の現状で無限降下を防ぐガード)．
    pub max_ply: u16,
}

impl Default for SearchOptions {
    fn default() -> SearchOptions {
        SearchOptions {
            threads: 1,
            batch_size: 8,
            c_puct: 1.5,
            fpu: 0.5,
            node_capacity: 1 << 20,
            max_ply: 512,
        }
    }
}

/// 探索の停止条件 (予算)．
///
/// 両方 `None` の場合は playout 上限 [`DEFAULT_MAX_PLAYOUTS`] が適用される．
#[derive(Clone, Debug, Default)]
pub struct SearchLimits {
    /// playout 数の上限．バッチ処理の粒度により最大
    /// `threads × batch_size` だけ超過し得る．
    pub max_playouts: Option<u64>,
    /// 時間の上限 (ミリ秒)．
    pub time_ms: Option<u64>,
}

/// 探索が停止した理由．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopCause {
    /// playout 上限に到達した．
    PlayoutLimit,
    /// 時間上限に到達した．
    TimeLimit,
    /// ノードプールが枯渇した．
    PoolExhausted,
    /// ルート局面に合法手がない (探索不能)．
    RootTerminal,
}

impl StopCause {
    fn to_u8(self) -> u8 {
        match self {
            StopCause::PlayoutLimit => 1,
            StopCause::TimeLimit => 2,
            StopCause::PoolExhausted => 3,
            StopCause::RootTerminal => 4,
        }
    }

    fn from_u8(v: u8) -> Option<StopCause> {
        match v {
            1 => Some(StopCause::PlayoutLimit),
            2 => Some(StopCause::TimeLimit),
            3 => Some(StopCause::PoolExhausted),
            4 => Some(StopCause::RootTerminal),
            _ => None,
        }
    }
}

/// stop_cause 未設定を表す番兵値．
const CAUSE_NONE: u8 = 0;

/// ルート直下の子ごとの統計．
#[derive(Clone, Debug)]
pub struct RootChildStat {
    /// 指し手．
    pub mv: Move,
    /// 訪問回数．
    pub visits: u32,
    /// ルート手番側から見た勝率 (未訪問なら 0)．
    pub q: f64,
    /// policy 事前確率．
    pub prior: f32,
}

/// 探索の実行統計．
#[derive(Clone, Debug)]
pub struct SearchStats {
    /// 完了した playout 数 (評価 + 終端到達)．
    pub playouts: u64,
    /// 経過時間 (ミリ秒)．
    pub elapsed_ms: u64,
    /// playout 毎秒 (mock/実 NN いずれも「葉評価スループット」の指標)．
    pub nps: f64,
    /// 衝突数 (他スレッド評価中の葉に到達してロールバックした回数)．
    pub collisions: u64,
    /// evaluate_batch の呼び出し回数．
    pub eval_batches: u64,
    /// 評価した葉の総数．
    pub eval_items: u64,
    /// バッチあたり平均葉数 (バッチ充填率 = avg_batch / batch_size)．
    pub avg_batch: f64,
    /// 到達した最大深さ．
    pub max_depth: u16,
    /// 使用したノード数．
    pub nodes_used: u32,
    /// 子ノード生成の CAS 競合で捨てられたノード数．
    pub leaked_nodes: u64,
}

/// 探索結果．
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// 最有力手 (ルートに合法手がなければ `None`)．
    /// 選択基準: 訪問回数最大 → 同数なら Q 最大 → 同率なら合法手生成順で先頭．
    pub best_move: Option<Move>,
    /// ルート手番側から見た best_move の勝率 (Q)．
    pub winrate: f64,
    /// 訪問回数最大の経路 (PV)．
    pub pv: Vec<Move>,
    /// ルート直下の全子の統計 (合法手生成順)．
    pub root_children: Vec<RootChildStat>,
    /// 停止理由．
    pub stop: StopCause,
    /// 実行統計．
    pub stats: SearchStats,
}

/// スレッド間共有の探索状態．
struct Shared<'a, E: Evaluator> {
    pool: NodePool,
    root_board: Board,
    evaluator: &'a E,
    opts: &'a SearchOptions,
    deadline: Option<Instant>,
    max_playouts: u64,
    stop: AtomicBool,
    stop_cause: AtomicU8,
    playouts: AtomicU64,
    collisions: AtomicU64,
    eval_batches: AtomicU64,
    eval_items: AtomicU64,
    leaked_nodes: AtomicU64,
    max_depth: AtomicU16,
}

impl<E: Evaluator> Shared<'_, E> {
    /// 停止フラグを立てる．最初に立てたスレッドの cause が記録される．
    fn set_stop(&self, cause: StopCause) {
        let _ = self.stop_cause.compare_exchange(
            CAUSE_NONE,
            cause.to_u8(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        self.stop.store(true, Ordering::Release);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// playout 完了を n 件計上し，上限到達なら停止フラグを立てる．
    fn complete_playouts(&self, n: u64) {
        let prev = self.playouts.fetch_add(n, Ordering::Relaxed);
        if prev + n >= self.max_playouts {
            self.set_stop(StopCause::PlayoutLimit);
        }
    }

    /// 時間上限を過ぎていたら停止フラグを立てる．
    fn check_deadline(&self) {
        if let Some(d) = self.deadline {
            if Instant::now() >= d {
                self.set_stop(StopCause::TimeLimit);
            }
        }
    }

    fn note_depth(&self, depth: usize) {
        let d = depth.min(u16::MAX as usize) as u16;
        self.max_depth.fetch_max(d, Ordering::Relaxed);
    }
}

/// 葉選択の結果．
enum Selection {
    /// 評価すべき葉を確保した (path 末尾が葉ノード)．
    /// EvalItem は Board を含み大きいため Box で持つ．
    Leaf { path: Vec<u32>, item: Box<EvalItem> },
    /// 終端 (詰み/最大深さ) に到達し，その場でバックプロパゲーション済み．
    Backpropped,
    /// 他スレッド評価中の葉に到達した (ロールバック済み)．
    Collision,
    /// ノードプール枯渇 (ロールバック済み)．
    PoolExhausted,
}

/// path 上の前置 visits を取り消す．
fn rollback<E: Evaluator>(shared: &Shared<'_, E>, path: &[u32]) {
    for &idx in path {
        shared.pool.get(idx).revert_visit();
    }
}

/// 葉の評価値を経路に沿って伝播する．
///
/// `leaf_value` は path 末尾ノードの手番側から見た勝率．各ノードの wins は
/// 親手番視点なので，1 手さかのぼるごとに視点を反転しながら加算する．
fn backprop<E: Evaluator>(shared: &Shared<'_, E>, path: &[u32], leaf_value: f64) {
    let mut v = leaf_value;
    for &idx in path.iter().rev() {
        shared.pool.get(idx).add_win(1.0 - v);
        v = 1.0 - v;
    }
}

/// ルートから PUCT で降下して評価対象の葉を 1 つ選ぶ．
fn select_leaf<E: Evaluator>(shared: &Shared<'_, E>) -> Selection {
    let pool = &shared.pool;
    let opts = shared.opts;
    let mut board = shared.root_board.clone();
    let mut path: Vec<u32> = Vec::with_capacity(64);
    path.push(ROOT_IDX);
    pool.get(ROOT_IDX).add_visit();
    let mut idx = ROOT_IDX;

    loop {
        let node = pool.get(idx);
        match node.state() {
            node_state::EXPANDED => {
                let edges = node.edges();
                let sqrt_pn = (node.visits() as f32).sqrt();
                let mut best_i = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (i, e) in edges.iter().enumerate() {
                    let (q, n) = match e.child.load(Ordering::Acquire) {
                        NULL_NODE => (opts.fpu, 0u32),
                        c => {
                            let child = pool.get(c);
                            let v = child.visits();
                            if v == 0 {
                                (opts.fpu, 0)
                            } else {
                                ((child.wins() / f64::from(v)) as f32, v)
                            }
                        }
                    };
                    let u = opts.c_puct * e.prior * sqrt_pn / (1.0 + n as f32);
                    let score = q + u;
                    if score > best_score {
                        best_score = score;
                        best_i = i;
                    }
                }
                let edge = &edges[best_i];
                board.do_move(edge.mv);

                let mut child_idx = edge.child.load(Ordering::Acquire);
                if child_idx == NULL_NODE {
                    match pool.alloc() {
                        None => {
                            rollback(shared, &path);
                            return Selection::PoolExhausted;
                        }
                        Some(new_idx) => {
                            match edge.child.compare_exchange(
                                NULL_NODE,
                                new_idx,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => child_idx = new_idx,
                                Err(existing) => {
                                    // 競合に負けた: 確保済みノードは捨てる (プール容量を 1 消費)
                                    shared.leaked_nodes.fetch_add(1, Ordering::Relaxed);
                                    child_idx = existing;
                                }
                            }
                        }
                    }
                }
                pool.get(child_idx).add_visit();
                path.push(child_idx);
                idx = child_idx;

                if path.len() > opts.max_ply as usize {
                    shared.note_depth(path.len());
                    backprop(shared, &path, 0.5);
                    return Selection::Backpropped;
                }
            }
            node_state::UNEXPANDED => {
                if node.try_begin_expansion() {
                    let moves = generate_legal_moves(&mut board);
                    shared.note_depth(path.len());
                    if moves.is_empty() {
                        node.mark_terminal_loss();
                        backprop(shared, &path, 0.0);
                        return Selection::Backpropped;
                    }
                    return Selection::Leaf {
                        path,
                        item: Box::new(EvalItem { board, moves }),
                    };
                }
                rollback(shared, &path);
                return Selection::Collision;
            }
            node_state::EXPANDING => {
                rollback(shared, &path);
                return Selection::Collision;
            }
            node_state::TERMINAL_LOSS => {
                shared.note_depth(path.len());
                backprop(shared, &path, 0.0);
                return Selection::Backpropped;
            }
            other => unreachable!("未知のノード状態: {other}"),
        }
    }
}

/// 探索スレッドのメインループ．
fn worker<E: Evaluator>(shared: &Shared<'_, E>) {
    let batch_size = shared.opts.batch_size.max(1);
    let mut paths: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut items: Vec<EvalItem> = Vec::with_capacity(batch_size);

    while !shared.stopped() {
        shared.check_deadline();
        paths.clear();
        items.clear();
        let mut had_collision = false;

        while items.len() < batch_size && !shared.stopped() {
            match select_leaf(shared) {
                Selection::Leaf { path, item } => {
                    paths.push(path);
                    items.push(*item);
                }
                Selection::Backpropped => {
                    shared.complete_playouts(1);
                }
                Selection::Collision => {
                    shared.collisions.fetch_add(1, Ordering::Relaxed);
                    had_collision = true;
                    // 収集を打ち切って手持ちのバッチを即時評価する
                    break;
                }
                Selection::PoolExhausted => {
                    shared.set_stop(StopCause::PoolExhausted);
                    break;
                }
            }
        }

        if items.is_empty() {
            if had_collision {
                // 手ぶら衝突: 他スレッドの評価完了を待つ
                std::thread::yield_now();
            }
            continue;
        }

        let results = shared.evaluator.evaluate_batch(&items);
        assert_eq!(
            results.len(),
            items.len(),
            "evaluator は items と同数の結果を返すこと"
        );
        shared.eval_batches.fetch_add(1, Ordering::Relaxed);
        shared
            .eval_items
            .fetch_add(items.len() as u64, Ordering::Relaxed);

        let n = items.len() as u64;
        for ((path, item), result) in paths.drain(..).zip(items.drain(..)).zip(results) {
            assert_eq!(
                result.priors.len(),
                item.moves.len(),
                "evaluator は moves と同数の priors を返すこと"
            );
            let leaf = shared.pool.get(*path.last().expect("path は空にならない"));
            let edges: Box<[Edge]> = item
                .moves
                .iter()
                .zip(result.priors.iter())
                .map(|(&mv, &p)| Edge::new(mv, p))
                .collect();
            leaf.finish_expansion(edges);
            backprop(shared, &path, f64::from(result.value.clamp(0.0, 1.0)));
        }
        shared.complete_playouts(n);
        shared.check_deadline();
    }
}

/// MCTS 探索エンジン．
///
/// evaluator と設定を保持し，[`Searcher::search`] で 1 局面を探索する．
pub struct Searcher<'e, E: Evaluator> {
    evaluator: &'e E,
    options: SearchOptions,
}

impl<'e, E: Evaluator> Searcher<'e, E> {
    /// evaluator と設定から探索エンジンを作る．
    pub fn new(evaluator: &'e E, options: SearchOptions) -> Searcher<'e, E> {
        Searcher { evaluator, options }
    }

    /// SFEN 文字列で与えた局面を探索する．
    pub fn search_sfen(
        &self,
        sfen: &str,
        limits: &SearchLimits,
    ) -> Result<SearchResult, SfenError> {
        let mut board = Board::empty();
        board.set_sfen(sfen)?;
        Ok(self.search(&board, limits))
    }

    /// 局面を探索して最有力手・評価値・統計を返す．
    pub fn search(&self, root_board: &Board, limits: &SearchLimits) -> SearchResult {
        let start = Instant::now();
        let opts = &self.options;

        // ルートの合法手を確認する
        let mut probe = root_board.clone();
        let root_moves = generate_legal_moves(&mut probe);
        if root_moves.is_empty() {
            return SearchResult {
                best_move: None,
                winrate: 0.0,
                pv: Vec::new(),
                root_children: Vec::new(),
                stop: StopCause::RootTerminal,
                stats: SearchStats {
                    playouts: 0,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    nps: 0.0,
                    collisions: 0,
                    eval_batches: 0,
                    eval_items: 0,
                    avg_batch: 0.0,
                    max_depth: 0,
                    nodes_used: 0,
                    leaked_nodes: 0,
                },
            };
        }

        // ルートを同期的に評価して展開する
        let pool = NodePool::new(opts.node_capacity);
        let root_idx = pool.alloc().expect("capacity >= 1 なので必ず確保できる");
        debug_assert_eq!(root_idx, ROOT_IDX);
        let root_item = [EvalItem {
            board: root_board.clone(),
            moves: root_moves,
        }];
        let mut root_results = self.evaluator.evaluate_batch(&root_item);
        let root_result = root_results.pop().expect("バッチ 1 件の結果");
        let [root_item] = root_item;
        assert_eq!(
            root_result.priors.len(),
            root_item.moves.len(),
            "evaluator は moves と同数の priors を返すこと"
        );
        let edges: Box<[Edge]> = root_item
            .moves
            .iter()
            .zip(root_result.priors.iter())
            .map(|(&mv, &p)| Edge::new(mv, p))
            .collect();
        {
            let root_node = pool.get(ROOT_IDX);
            root_node.finish_expansion(edges);
            root_node.add_visit();
        }

        let shared = Shared {
            pool,
            root_board: root_board.clone(),
            evaluator: self.evaluator,
            opts,
            deadline: limits.time_ms.map(|ms| start + Duration::from_millis(ms)),
            max_playouts: limits.max_playouts.unwrap_or(if limits.time_ms.is_some() {
                u64::MAX
            } else {
                DEFAULT_MAX_PLAYOUTS
            }),
            stop: AtomicBool::new(false),
            stop_cause: AtomicU8::new(CAUSE_NONE),
            playouts: AtomicU64::new(0),
            collisions: AtomicU64::new(0),
            eval_batches: AtomicU64::new(0),
            eval_items: AtomicU64::new(0),
            leaked_nodes: AtomicU64::new(0),
            max_depth: AtomicU16::new(0),
        };

        std::thread::scope(|s| {
            for _ in 0..opts.threads.max(1) {
                s.spawn(|| worker(&shared));
            }
        });

        self.collect_result(&shared, start)
    }

    /// 探索終了後の木からベストムーブ・PV・統計を集計する．
    fn collect_result(&self, shared: &Shared<'_, E>, start: Instant) -> SearchResult {
        let pool = &shared.pool;
        let root_node = pool.get(ROOT_IDX);
        let root_children: Vec<RootChildStat> = root_node
            .edges()
            .iter()
            .map(|e| {
                let (visits, q) = match e.child.load(Ordering::Acquire) {
                    NULL_NODE => (0, 0.0),
                    c => {
                        let child = pool.get(c);
                        let v = child.visits();
                        if v == 0 {
                            (0, 0.0)
                        } else {
                            (v, child.wins() / f64::from(v))
                        }
                    }
                };
                RootChildStat {
                    mv: e.mv,
                    visits,
                    q,
                    prior: e.prior,
                }
            })
            .collect();

        // 訪問回数最大 → 同数なら Q 最大 → 同率なら合法手生成順で先頭
        let mut best_i = 0usize;
        for (i, c) in root_children.iter().enumerate().skip(1) {
            let b = &root_children[best_i];
            if c.visits > b.visits || (c.visits == b.visits && c.q > b.q) {
                best_i = i;
            }
        }
        let best = &root_children[best_i];

        // PV: 訪問回数最大の辺を辿る
        let mut pv = Vec::new();
        let mut idx = ROOT_IDX;
        while pv.len() < MAX_PV_LEN {
            let node = pool.get(idx);
            if node.state() != node_state::EXPANDED {
                break;
            }
            let edges = node.edges();
            let mut pv_best: Option<(&Edge, u32)> = None;
            for e in edges {
                let c = e.child.load(Ordering::Acquire);
                if c == NULL_NODE {
                    continue;
                }
                let v = pool.get(c).visits();
                if v == 0 {
                    continue;
                }
                if pv_best.is_none_or(|(_, bv)| v > bv) {
                    pv_best = Some((e, v));
                }
            }
            let Some((e, _)) = pv_best else { break };
            pv.push(e.mv);
            idx = e.child.load(Ordering::Acquire);
        }

        let elapsed = start.elapsed();
        let playouts = shared.playouts.load(Ordering::Relaxed);
        let eval_batches = shared.eval_batches.load(Ordering::Relaxed);
        let eval_items = shared.eval_items.load(Ordering::Relaxed);
        let stats = SearchStats {
            playouts,
            elapsed_ms: elapsed.as_millis() as u64,
            nps: playouts as f64 / elapsed.as_secs_f64().max(1e-9),
            collisions: shared.collisions.load(Ordering::Relaxed),
            eval_batches,
            eval_items,
            avg_batch: if eval_batches > 0 {
                eval_items as f64 / eval_batches as f64
            } else {
                0.0
            },
            max_depth: shared.max_depth.load(Ordering::Relaxed),
            nodes_used: shared.pool.used(),
            leaked_nodes: shared.leaked_nodes.load(Ordering::Relaxed),
        };

        SearchResult {
            best_move: Some(best.mv),
            winrate: best.q,
            pv,
            root_children,
            stop: StopCause::from_u8(shared.stop_cause.load(Ordering::Acquire))
                .unwrap_or(StopCause::PlayoutLimit),
            stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::MockEvaluator;

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    /// 先手: 5三歩・金 1 枚 (持駒)，後手: 5一玉のみ．G*5b (5二金打) の 1 手詰め．
    const MATE_IN_1: &str = "4k4/9/4P4/9/9/9/9/9/9 b G 1";

    fn run(sfen: &str, opts: SearchOptions, limits: SearchLimits, seed: u64) -> SearchResult {
        let evaluator = MockEvaluator::new(seed);
        let searcher = Searcher::new(&evaluator, opts);
        searcher
            .search_sfen(sfen, &limits)
            .expect("テスト SFEN は正当")
    }

    #[test]
    fn test_finds_mate_in_1() {
        let result = run(
            MATE_IN_1,
            SearchOptions {
                threads: 1,
                batch_size: 4,
                node_capacity: 1 << 14,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(3000),
                ..SearchLimits::default()
            },
            42,
        );
        let best = result.best_move.expect("合法手がある");
        assert_eq!(best.to_usi(), "G*5b", "詰み手 G*5b が最多訪問になるはず");
        assert!(
            result.winrate > 0.9,
            "詰み手の勝率が高いはず: {}",
            result.winrate
        );
        assert_eq!(
            result.pv.first().map(|m| m.to_usi()).as_deref(),
            Some("G*5b")
        );
    }

    #[test]
    fn test_playout_limit_respected() {
        let opts = SearchOptions {
            threads: 2,
            batch_size: 8,
            ..SearchOptions::default()
        };
        let overshoot = (opts.threads * opts.batch_size) as u64;
        let result = run(
            STARTPOS,
            opts,
            SearchLimits {
                max_playouts: Some(500),
                ..SearchLimits::default()
            },
            0,
        );
        assert_eq!(result.stop, StopCause::PlayoutLimit);
        assert!(result.stats.playouts >= 500);
        assert!(
            result.stats.playouts <= 500 + overshoot,
            "playouts = {} (上限 500 + 許容超過 {overshoot})",
            result.stats.playouts
        );
    }

    #[test]
    fn test_multithread_smoke() {
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 4,
                batch_size: 16,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(5000),
                ..SearchLimits::default()
            },
            0,
        );
        let best = result.best_move.expect("合法手がある");
        // best_move はルートの合法手のいずれかであること
        assert!(result.root_children.iter().any(|c| c.mv == best));
        // 全子の訪問回数の総和 + ルート初期評価 1 = ルート playout 相当
        let total_child_visits: u64 = result
            .root_children
            .iter()
            .map(|c| u64::from(c.visits))
            .sum();
        assert!(total_child_visits > 0);
        assert!(result.stats.max_depth >= 2, "複数手先まで読んでいるはず");
    }

    #[test]
    fn test_single_thread_deterministic() {
        let opts = SearchOptions {
            threads: 1,
            batch_size: 8,
            ..SearchOptions::default()
        };
        let limits = SearchLimits {
            max_playouts: Some(2000),
            ..SearchLimits::default()
        };
        let a = run(STARTPOS, opts.clone(), limits.clone(), 123);
        let b = run(STARTPOS, opts, limits, 123);
        assert_eq!(
            a.best_move.map(|m| m.to_usi()),
            b.best_move.map(|m| m.to_usi())
        );
        let visits_a: Vec<u32> = a.root_children.iter().map(|c| c.visits).collect();
        let visits_b: Vec<u32> = b.root_children.iter().map(|c| c.visits).collect();
        assert_eq!(visits_a, visits_b, "単一スレッドでは訪問分布まで再現される");
    }

    #[test]
    fn test_pool_exhaustion_stops_search() {
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 2,
                batch_size: 8,
                node_capacity: 128,
                ..SearchOptions::default()
            },
            SearchLimits {
                max_playouts: Some(u64::MAX >> 1),
                ..SearchLimits::default()
            },
            0,
        );
        assert_eq!(result.stop, StopCause::PoolExhausted);
        assert!(result.stats.nodes_used <= 128);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn test_time_limit_terminates() {
        let result = run(
            STARTPOS,
            SearchOptions {
                threads: 2,
                batch_size: 8,
                ..SearchOptions::default()
            },
            SearchLimits {
                time_ms: Some(100),
                ..SearchLimits::default()
            },
            0,
        );
        assert_eq!(result.stop, StopCause::TimeLimit);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn test_root_terminal() {
        // 後手玉が既に詰んでいる局面で後手番 (合法手なし): MATE_IN_1 の G*5b 後
        let evaluator = MockEvaluator::new(0);
        let searcher = Searcher::new(&evaluator, SearchOptions::default());
        let mut board = Board::empty();
        board.set_sfen(MATE_IN_1).expect("正当な SFEN");
        let mut probe = board.clone();
        let mate = generate_legal_moves(&mut probe)
            .into_iter()
            .find(|m| m.to_usi() == "G*5b")
            .expect("G*5b は合法手");
        board.do_move(mate);
        let result = searcher.search(&board, &SearchLimits::default());
        assert_eq!(result.stop, StopCause::RootTerminal);
        assert!(result.best_move.is_none());
    }
}
