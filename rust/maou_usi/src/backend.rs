//! [`SearchBackend`] の実装 — maou_search (MCTS + root-dfpn/leaf-mate) を使う．
//!
//! 評価器 (mock / ONNX) は [`MaouSearchBackend::build`] で 1 回だけ構築する
//! (USI `isready` のタイミング．TensorRT のエンジンビルドも warmup として
//! ここで済ませ，初手の `go` を遅らせない)．

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use maou_search::{
    build_board_and_history, MockEvaluator, ReusableTree, RootSnapshot, SearchLimits,
    SearchOptions, SearchResult, Searcher, StopCause,
};
use maou_shogi::dfpn::{DfPnSolver, TsumeResult};

use crate::agent::{
    EngineConfig, GoRules, ProgressSnapshot, SearchBackend, SearchBudget, SearchObserver,
    SearchOutcome,
};
use crate::protocol::CheckmateResult;

/// 進捗スナップショットを observer へ渡すポーリング間隔．
pub(crate) const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// 保持する評価器 (mock または ONNX)．自己対局 driver はこれを `Arc` で全対局
/// に共有する (モデルロード/warmup をプロセス内 1 回に — 設計 §9)．
pub(crate) enum EngineEvaluator {
    Mock(MockEvaluator),
    #[cfg(feature = "onnx")]
    Onnx(maou_search::OnnxEvaluator),
}

/// 設定から評価器 (mock または ONNX) を構築する (warmup は別途
/// [`warmup_evaluator`])．
pub(crate) fn build_evaluator(config: &EngineConfig) -> Result<EngineEvaluator, String> {
    match &config.model_path {
        None => Ok(EngineEvaluator::Mock(MockEvaluator::new(0))),
        #[cfg(feature = "onnx")]
        Some(path) => {
            let onnx_options = maou_search::onnx::OnnxOptions {
                intra_threads: 1,
                use_cuda: config.use_cuda,
                use_tensorrt: config.use_tensorrt,
                trt_engine_cache_dir: config.trt_cache_dir.clone(),
                // TensorRT は shape ごとにエンジンをビルドするため batch_size に固定する
                // (pad_buckets 有効時は batch_size を上限に 2 冪バケットへ切り上げ)
                pad_to: if config.use_tensorrt {
                    Some(config.batch_size)
                } else {
                    None
                },
                pad_buckets: config.pad_buckets,
            };
            Ok(EngineEvaluator::Onnx(
                maou_search::OnnxEvaluator::from_file(path, &onnx_options)
                    .map_err(|e| format!("ONNX model load failed: {e}"))?,
            ))
        }
        #[cfg(not(feature = "onnx"))]
        Some(_) => Err("this build has no onnx feature; ModelPath is unavailable \
             (build with `maturin develop --features onnx`)"
            .to_string()),
    }
}

/// 平手初期局面を 1 回評価して初回推論の固定費 (TensorRT エンジンビルド/
/// CUDA 初期化) を前払いする (USI では `isready` 中，自己対局では起動時)．
pub(crate) fn warmup_evaluator(evaluator: &EngineEvaluator) -> Result<(), String> {
    match evaluator {
        EngineEvaluator::Mock(e) => maou_search::warmup(e),
        #[cfg(feature = "onnx")]
        EngineEvaluator::Onnx(e) => maou_search::warmup(e),
    }
    Ok(())
}

/// [`EngineConfig`] → 探索オプション ([`SearchOptions`]) の写像
/// (build / 自己対局 driver で共有する単一実装)．
pub(crate) fn search_options(config: &EngineConfig) -> SearchOptions {
    let mut options = SearchOptions {
        threads: config.threads,
        batch_size: config.batch_size,
        ..SearchOptions::default()
    };
    if let Some(v) = config.effective_node_capacity() {
        options.node_capacity = v;
    }
    if let Some(v) = config.root_dfpn {
        options.root_dfpn = v;
    }
    if let Some(v) = config.root_dfpn_nodes {
        options.root_dfpn_nodes = v;
    }
    if let Some(v) = config.root_dfpn_depth {
        options.root_dfpn_depth = v;
    }
    if let Some(v) = config.leaf_mate {
        options.leaf_mate = v;
    }
    if let Some(v) = config.leaf_mate_nodes {
        options.leaf_mate_nodes = v;
    }
    if let Some(v) = config.leaf_mate_threads {
        options.leaf_mate_threads = v;
    }
    options.spin_budget_relief = config.spin_budget_relief;
    options.skip_proven_children = config.skip_proven_children;
    options
}

/// maou_search を使う実バックエンド．
pub struct MaouSearchBackend {
    /// 評価器 (`Arc` 共有: USI では単独所有と等価，自己対局では全対局共有)．
    evaluator: Arc<EngineEvaluator>,
    options: SearchOptions,
    /// 対局手番間で保持する探索木 (subtree 再利用)．手番進行で局面が前進した
    /// ときに reroot して warm start する．`reset` (usinewgame/gameover) で破棄．
    retained: Option<ReusableTree>,
    /// subtree 再利用を行うか ([`EngineConfig::subtree_reuse`]，計測用
    /// トグル)．false なら毎手 fresh 探索 (保持もしない)．
    reuse_tree: bool,
}

impl MaouSearchBackend {
    /// 設定から評価器を構築し，warmup (初回推論 = TensorRT エンジンビルド等)
    /// まで済ませる．
    pub fn build(config: &EngineConfig) -> Result<MaouSearchBackend, String> {
        let evaluator = build_evaluator(config)?;
        warmup_evaluator(&evaluator)?;
        Ok(MaouSearchBackend::from_shared(
            Arc::new(evaluator),
            search_options(config),
            config.subtree_reuse,
        ))
    }

    /// 構築・warmup 済みの共有評価器からバックエンドを作る (自己対局 driver
    /// 用 — 評価器の再ロード/warmup なしで対局ごとに安価に構築できる)．
    pub(crate) fn from_shared(
        evaluator: Arc<EngineEvaluator>,
        options: SearchOptions,
        reuse_tree: bool,
    ) -> MaouSearchBackend {
        MaouSearchBackend {
            evaluator,
            options,
            retained: None,
            reuse_tree,
        }
    }
}

impl SearchBackend for MaouSearchBackend {
    fn search(
        &mut self,
        sfen: &str,
        moves: &[String],
        budget: &SearchBudget,
        rules: &GoRules,
        stop: &Arc<AtomicBool>,
        observer: &mut dyn SearchObserver,
    ) -> Result<SearchOutcome, String> {
        // 対局ルール由来の per-go パラメータ: 手番視点の引き分け価値 (千日手
        // 戦略) と最大手数 (in-search 引き分け終端化) を探索へ渡す
        let mut options = self.options.clone();
        options.draw_value = rules.draw_value;
        options.max_moves_to_draw = rules.max_moves_to_draw;
        // 進捗スナップショットの発行先 (monitor がポーリングして observer へ渡す)
        let progress: Arc<Mutex<Option<RootSnapshot>>> = Arc::new(Mutex::new(None));
        let limits = SearchLimits {
            // 無期限 (go ponder / go infinite) は playout 上限 u64::MAX + stop
            // token で表現する (SearchLimits の規約)
            max_playouts: if budget.unbounded {
                Some(u64::MAX)
            } else {
                budget.max_playouts
            },
            // hard_ms を探索の絶対上限 (backstop) に．soft 到達時の延長判断は
            // monitor が observer 経由で行い stop フラグを立てる
            time_ms: if budget.unbounded {
                None
            } else {
                budget.time.map(|t| t.hard_ms)
            },
            stop: Some(Arc::clone(stop)),
            progress: Some(Arc::clone(&progress)),
        };
        // 前回の探索木を取り出す — 手番進行で局面が前進していれば search_reusing
        // が reroot して warm start する (前進していなければ fresh)．
        // reuse_tree が false (計測用トグル off) なら常に fresh
        let retained = if self.reuse_tree {
            self.retained.take()
        } else {
            None
        };
        // 探索を専用スレッドで走らせ，呼び出しスレッド (dispatcher) が monitor
        // ループを回す (progress をポーリング → observer 駆動 → 早期停止)．
        // GIL/GC を挟まない Rust 内で完結する (設計 §5)．
        let evaluator: &EngineEvaluator = &self.evaluator;
        let outcome = std::thread::scope(|s| {
            // 完了検知を sleep 越しに行うと，探索が終わってから dispatcher が
            // 気付くまで平均 POLL_INTERVAL/2 の死に時間が 1 手ごとに乗る
            // (探索は既に終わっているのに wall clock だけ進む)．完了は
            // channel の切断で即時に検知し，POLL_INTERVAL は進捗ポーリングの
            // 間隔としてのみ使う — observer の駆動間隔は従来と同じ
            let (done_tx, done_rx) = mpsc::channel::<()>();
            let handle = s.spawn(move || {
                // 探索から抜けた時点で drop され，recv 側が Disconnected を得る
                let _done = done_tx;
                match evaluator {
                    EngineEvaluator::Mock(e) => Searcher::new(e, options.clone())
                        .search_reusing(sfen, moves, &limits, retained),
                    #[cfg(feature = "onnx")]
                    EngineEvaluator::Onnx(e) => Searcher::new(e, options.clone())
                        .search_reusing(sfen, moves, &limits, retained),
                }
            });
            let start = Instant::now();
            // Timeout = 探索継続中 (進捗を観測する) / Disconnected = 探索終了
            while matches!(
                done_rx.recv_timeout(POLL_INTERVAL),
                Err(RecvTimeoutError::Timeout)
            ) {
                let latest = progress.lock().ok().and_then(|g| g.clone());
                if let Some(snap) = latest {
                    let elapsed = start.elapsed().as_millis() as u64;
                    if observer.on_progress(&to_progress_snapshot(&snap), elapsed) {
                        stop.store(true, Ordering::Release);
                    }
                }
            }
            handle.join().expect("探索スレッドは panic しない")
        });
        // 更新後の木を保持して次回の subtree 再利用に備える (fresh でも保持する．
        // 再利用 off なら保持もしない — メモリを残さない)
        let (result, tree) = outcome.map_err(|e| e.to_string())?;
        if self.reuse_tree {
            self.retained = Some(tree);
        }
        Ok(to_outcome(&result))
    }

    fn nyugyoku_declarable(&self, sfen: &str, moves: &[String]) -> Result<bool, String> {
        let (board, _) = build_board_and_history(sfen, moves).map_err(|e| e.to_string())?;
        Ok(board.nyugyoku_declarable())
    }

    fn solve_mate(
        &self,
        sfen: &str,
        moves: &[String],
        time_ms: Option<u64>,
        stop: &Arc<AtomicBool>,
    ) -> Result<CheckmateResult, String> {
        let (mut board, _) = build_board_and_history(sfen, moves).map_err(|e| e.to_string())?;
        // 予算: 時間指定は秒へ切り上げ (dfpn の timeout 粒度)．無制限は
        // 十分大きな値を置き，停止は stop トークンに委ねる (`go mate
        // infinite` は GUI の stop まで走る規約)
        let timeout_secs = time_ms.map_or(u64::MAX, |ms| ms.div_ceil(1000).max(1));
        let mut solver = DfPnSolver::with_timeout(
            self.options.root_dfpn_depth,
            // ノード予算は無制限側に倒し，実際の打ち切りは時間と stop で行う
            // (GUI は「この時間だけ考えて」と言っているため)
            u64::MAX,
            timeout_secs,
        );
        // 詰将棋としての最短手順を返す (検討機能の用途に合う)
        solver.set_find_shortest(true);
        solver.set_stop_flag(Arc::clone(stop));
        // 停止理由まで見る: `nomate` は**不詰を証明できたときだけ**返す．
        // 打ち切り (stop/時間切れ) の未解決を nomate と報告すると GUI に
        // 「詰みは無い」と嘘をつくことになる
        let report = solver.solve_report(&mut board);
        Ok(match report.result {
            TsumeResult::Checkmate { ref moves, .. } if !moves.is_empty() => {
                CheckmateResult::Mate(moves.iter().map(|m| m.to_usi()).collect())
            }
            // 詰みだが手順を復元できない場合は手順を示せないので timeout 扱い
            // (誤った `checkmate` 行を出さない)
            TsumeResult::Checkmate { .. } | TsumeResult::CheckmateNoPv { .. } => {
                CheckmateResult::Timeout
            }
            TsumeResult::NoCheckmate { .. }
                if report.stop_reason == maou_shogi::dfpn::StopReason::Disproven =>
            {
                CheckmateResult::NoMate
            }
            _ => CheckmateResult::Timeout,
        })
    }

    fn is_mock(&self) -> bool {
        matches!(*self.evaluator, EngineEvaluator::Mock(_))
    }

    fn reset(&mut self) {
        // 対局リセット: 保持木を破棄する (次の探索は fresh)
        self.retained = None;
    }
}

/// maou_search の [`RootSnapshot`] → transport 非依存の [`ProgressSnapshot`]．
fn to_progress_snapshot(snap: &RootSnapshot) -> ProgressSnapshot {
    ProgressSnapshot {
        playouts: snap.playouts,
        nps: snap.nps as u64,
        max_depth: snap.max_depth,
        best_usi: snap.best_move.map(|m| m.to_usi()),
        best_visits: snap.best_visits,
        second_visits: snap.second_visits,
        winrate: snap.winrate,
        pv: snap.pv.iter().map(|m| m.to_usi()).collect(),
        proven: snap.proven,
    }
}

/// [`SearchResult`] → transport 非依存の [`SearchOutcome`]．
fn to_outcome(r: &SearchResult) -> SearchOutcome {
    // subtree 再利用の実効量: root 直下の訪問数合計のうち今回の backprop を
    // 超える分が前回木から引き継いだ訪問 (fresh 探索では 0 になる)．
    // root 訪問は葉評価・空回りの**どちらでも**増えるので，両方を引く
    // (playouts だけを引くと空回りの分が引き継ぎとして二重計上される)
    let root_visits: u64 = r.root_children.iter().map(|c| c.visits).sum();
    let carried_visits = root_visits.saturating_sub(r.stats.playouts + r.stats.terminal_backprops);
    SearchOutcome {
        carried_visits,
        best_usi: r.best_move.map(|m| m.to_usi()),
        winrate: r.winrate,
        pv: r.pv.iter().map(|m| m.to_usi()).collect(),
        playouts: r.stats.playouts,
        terminal_backprops: r.stats.terminal_backprops,
        // GUI へ報告する消費時間は warmup (root 評価) 込みの壁時計
        elapsed_ms: r.stats.warmup_ms + r.stats.elapsed_ms,
        nps: r.stats.nps as u64,
        max_depth: r.stats.max_depth,
        proven: if r.stop == StopCause::RootProven {
            Some(r.winrate)
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::STARTPOS_SFEN;
    use maou_shogi::movegen::generate_legal_moves;
    use std::sync::atomic::Ordering;

    fn config() -> EngineConfig {
        EngineConfig {
            // テストは軽量に: 詰み探索 off + 小さい木
            root_dfpn: Some(false),
            leaf_mate: Some(false),
            node_capacity: Some(1 << 14),
            ..EngineConfig::default()
        }
    }

    /// 進捗を無視する観測者 (backend 単体テスト用)．
    struct NoopObserver;
    impl SearchObserver for NoopObserver {
        fn on_progress(&mut self, _snapshot: &ProgressSnapshot, _elapsed_ms: u64) -> bool {
            false
        }
    }

    #[test]
    fn test_build_and_search_with_mock() {
        let mut backend = MaouSearchBackend::build(&config()).expect("mock 構築は成功する");
        assert!(backend.is_mock());
        let stop = Arc::new(AtomicBool::new(false));
        let outcome = backend
            .search(
                STARTPOS_SFEN,
                &["7g7f".to_string()],
                &SearchBudget {
                    time: None,
                    max_playouts: Some(200),
                    unbounded: false,
                },
                &GoRules {
                    draw_value: 0.5,
                    max_moves_to_draw: 0,
                },
                &stop,
                &mut NoopObserver,
            )
            .expect("mock 探索は成功する");
        let best = outcome.best_usi.expect("平手 1 手目後に合法手はある");
        // bestmove が現局面の合法手であること
        let (board, _) =
            build_board_and_history(STARTPOS_SFEN, &["7g7f".to_string()]).expect("正当");
        let legal: Vec<String> = generate_legal_moves(&mut board.clone())
            .into_iter()
            .map(|m| m.to_usi())
            .collect();
        assert!(legal.contains(&best), "{best} は合法手であるべき");
        assert!(outcome.playouts >= 200);
    }

    #[test]
    fn test_unbounded_search_stops_via_token() {
        let mut backend = MaouSearchBackend::build(&config()).expect("mock 構築は成功する");
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let setter = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            flag.store(true, Ordering::Release);
        });
        let outcome = backend
            .search(
                STARTPOS_SFEN,
                &[],
                &SearchBudget {
                    time: None,
                    max_playouts: None,
                    unbounded: true,
                },
                &GoRules {
                    draw_value: 0.5,
                    max_moves_to_draw: 0,
                },
                &stop,
                &mut NoopObserver,
            )
            .expect("mock 探索は成功する");
        setter.join().expect("setter 正常終了");
        assert!(outcome.best_usi.is_some());
    }

    #[test]
    fn test_solve_mate_finds_mate_in_1() {
        // 先手 5三歩 + 持駒金，後手 5一玉のみ: G*5b の 1 手詰め
        let backend = MaouSearchBackend::build(&config()).expect("mock 構築は成功する");
        let stop = Arc::new(AtomicBool::new(false));
        let result = backend
            .solve_mate("4k4/9/4P4/9/9/9/9/9/9 b G 1", &[], Some(5_000), &stop)
            .expect("詰み探索は成功する");
        match result {
            CheckmateResult::Mate(moves) => {
                assert_eq!(moves.first().map(String::as_str), Some("G*5b"), "{moves:?}");
            }
            other => panic!("詰みを期待: {other:?}"),
        }
    }

    #[test]
    fn test_solve_mate_reports_nomate() {
        // 平手初期局面は当然不詰 (dfpn が即 NoCheckmate を返す)
        let backend = MaouSearchBackend::build(&config()).expect("mock 構築は成功する");
        let stop = Arc::new(AtomicBool::new(false));
        let result = backend
            .solve_mate(STARTPOS_SFEN, &[], Some(5_000), &stop)
            .expect("詰み探索は成功する");
        assert_eq!(result, CheckmateResult::NoMate);
    }

    #[test]
    fn test_solve_mate_honors_stop_token() {
        // 29 手詰め (canonical: 396,516 ノード) を stop 済みで走らせる．
        // 打ち切りは「不詰」ではなく timeout として報告しなければならない
        // (nomate は Disproven のときだけ — GUI に嘘をつかないため)
        const TSUME_29: &str =
            "l2+P5/2k4+L1/2n1p2B1/p1pp1spN1/4Ps3/PlPP2P2/1P1Sb4/1KG2+p3/LN7 w R2GPrgsn4p 1";
        let backend = MaouSearchBackend::build(&config()).expect("mock 構築は成功する");
        let stop = Arc::new(AtomicBool::new(true));
        let started = Instant::now();
        let result = backend
            .solve_mate(TSUME_29, &[], None, &stop)
            .expect("詰み探索は成功する");
        assert_eq!(result, CheckmateResult::Timeout);
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "stop 済みなら即座に戻る: {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn test_solve_mate_nomate_only_when_disproven() {
        // 平手初期局面は「王手できる手が無い」ので 1 ノードで不詰が証明される
        // (打ち切りではなく Disproven ゆえ nomate が正しい)
        let backend = MaouSearchBackend::build(&config()).expect("mock 構築は成功する");
        let stop = Arc::new(AtomicBool::new(false));
        assert_eq!(
            backend
                .solve_mate(STARTPOS_SFEN, &[], Some(5_000), &stop)
                .expect("詰み探索は成功する"),
            CheckmateResult::NoMate
        );
    }

    #[test]
    fn test_illegal_position_is_error() {
        let mut backend = MaouSearchBackend::build(&config()).expect("mock 構築は成功する");
        let stop = Arc::new(AtomicBool::new(false));
        let err = backend
            .search(
                STARTPOS_SFEN,
                &["7g7e".to_string()],
                &SearchBudget {
                    time: None,
                    max_playouts: Some(10),
                    unbounded: false,
                },
                &GoRules {
                    draw_value: 0.5,
                    max_moves_to_draw: 0,
                },
                &stop,
                &mut NoopObserver,
            )
            .expect_err("非合法手はエラー");
        assert!(err.contains("7g7e"));
    }

    #[test]
    fn test_reuse_across_moves_stays_sound_and_resets() {
        let mut backend = MaouSearchBackend::build(&config()).expect("mock 構築は成功する");
        let stop = Arc::new(AtomicBool::new(false));
        let budget = SearchBudget {
            time: None,
            max_playouts: Some(500),
            unbounded: false,
        };
        // 1 手目後を探索して木を保持する
        let o1 = backend
            .search(
                STARTPOS_SFEN,
                &["7g7f".to_string()],
                &budget,
                &GoRules {
                    draw_value: 0.5,
                    max_moves_to_draw: 0,
                },
                &stop,
                &mut NoopObserver,
            )
            .expect("探索成功");
        let best1 = o1.best_usi.expect("合法手がある");
        // best1 と PV の続きで前進 = 探索済みの筋 → reroot して再利用する経路
        let mut moves = vec!["7g7f".to_string(), best1];
        if let Some(reply) = o1.pv.get(1) {
            moves.push(reply.clone());
        }
        let o2 = backend
            .search(
                STARTPOS_SFEN,
                &moves,
                &budget,
                &GoRules {
                    draw_value: 0.5,
                    max_moves_to_draw: 0,
                },
                &stop,
                &mut NoopObserver,
            )
            .expect("再利用探索成功");
        // soundness: 新局面の合法手を返す
        let best2 = o2.best_usi.expect("合法手がある");
        let (board, _) = build_board_and_history(STARTPOS_SFEN, &moves).expect("正当");
        let legal: Vec<String> = generate_legal_moves(&mut board.clone())
            .into_iter()
            .map(|m| m.to_usi())
            .collect();
        assert!(legal.contains(&best2), "{best2} は合法手であるべき");

        // reset (usinewgame 相当) 後も fresh 探索が正当に動く
        backend.reset();
        let o3 = backend
            .search(
                STARTPOS_SFEN,
                &["2g2f".to_string()],
                &budget,
                &GoRules {
                    draw_value: 0.5,
                    max_moves_to_draw: 0,
                },
                &stop,
                &mut NoopObserver,
            )
            .expect("reset 後の探索成功");
        assert!(o3.best_usi.is_some());
    }
}
