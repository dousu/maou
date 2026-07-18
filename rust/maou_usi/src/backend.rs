//! [`SearchBackend`] の実装 — maou_search (MCTS + root-dfpn/leaf-mate) を使う．
//!
//! 評価器 (mock / ONNX) は [`MaouSearchBackend::build`] で 1 回だけ構築する
//! (USI `isready` のタイミング．TensorRT のエンジンビルドも warmup として
//! ここで済ませ，初手の `go` を遅らせない)．

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use maou_search::{
    build_board_and_history, EvalItem, Evaluator, MockEvaluator, ReusableTree, RootSnapshot,
    SearchLimits, SearchOptions, SearchResult, Searcher, StopCause,
};
use maou_shogi::board::Board;
use maou_shogi::movegen::generate_legal_moves;

use crate::agent::{
    EngineConfig, ProgressSnapshot, SearchBackend, SearchBudget, SearchObserver, SearchOutcome,
    STARTPOS_SFEN,
};

/// 進捗スナップショットを observer へ渡すポーリング間隔．
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// 保持する評価器 (mock または ONNX)．
enum EngineEvaluator {
    Mock(MockEvaluator),
    #[cfg(feature = "onnx")]
    Onnx(maou_search::OnnxEvaluator),
}

/// maou_search を使う実バックエンド．
pub struct MaouSearchBackend {
    evaluator: EngineEvaluator,
    options: SearchOptions,
    /// 対局手番間で保持する探索木 (subtree 再利用)．手番進行で局面が前進した
    /// ときに reroot して warm start する．`reset` (usinewgame/gameover) で破棄．
    retained: Option<ReusableTree>,
}

impl MaouSearchBackend {
    /// 設定から評価器を構築し，warmup (初回推論 = TensorRT エンジンビルド等)
    /// まで済ませる．
    pub fn build(config: &EngineConfig) -> Result<MaouSearchBackend, String> {
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

        let evaluator = match &config.model_path {
            None => EngineEvaluator::Mock(MockEvaluator::new(0)),
            #[cfg(feature = "onnx")]
            Some(path) => {
                let onnx_options = maou_search::onnx::OnnxOptions {
                    intra_threads: 1,
                    use_cuda: config.use_cuda,
                    use_tensorrt: config.use_tensorrt,
                    trt_engine_cache_dir: config.trt_cache_dir.clone(),
                    // TensorRT は shape ごとにエンジンをビルドするため batch_size に固定する
                    pad_to: if config.use_tensorrt {
                        Some(config.batch_size)
                    } else {
                        None
                    },
                };
                EngineEvaluator::Onnx(
                    maou_search::OnnxEvaluator::from_file(path, &onnx_options)
                        .map_err(|e| format!("ONNX model load failed: {e}"))?,
                )
            }
            #[cfg(not(feature = "onnx"))]
            Some(_) => {
                return Err("this build has no onnx feature; ModelPath is unavailable \
                     (build with `maturin develop --features onnx`)"
                    .to_string())
            }
        };

        let backend = MaouSearchBackend {
            evaluator,
            options,
            retained: None,
        };
        backend.warmup()?;
        Ok(backend)
    }

    /// 平手初期局面を 1 回評価して初回推論の固定費 (TensorRT エンジンビルド/
    /// CUDA 初期化) を isready 中に支払う．
    fn warmup(&self) -> Result<(), String> {
        let mut board = Board::empty();
        board
            .set_sfen(STARTPOS_SFEN)
            .map_err(|e| format!("startpos SFEN must parse: {e:?}"))?;
        let moves = generate_legal_moves(&mut board.clone());
        let item = [EvalItem { board, moves }];
        match &self.evaluator {
            EngineEvaluator::Mock(e) => {
                let _ = e.evaluate_batch(&item);
            }
            #[cfg(feature = "onnx")]
            EngineEvaluator::Onnx(e) => {
                let _ = e.evaluate_batch(&item);
            }
        }
        Ok(())
    }
}

impl SearchBackend for MaouSearchBackend {
    fn search(
        &mut self,
        sfen: &str,
        moves: &[String],
        budget: &SearchBudget,
        draw_value: f64,
        stop: &Arc<AtomicBool>,
        observer: &mut dyn SearchObserver,
    ) -> Result<SearchOutcome, String> {
        // 千日手戦略: 手番視点の引き分け価値を探索へ渡す
        let mut options = self.options.clone();
        options.draw_value = draw_value;
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
        let retained = self.retained.take();
        // 探索を専用スレッドで走らせ，呼び出しスレッド (dispatcher) が monitor
        // ループを回す (progress をポーリング → observer 駆動 → 早期停止)．
        // GIL/GC を挟まない Rust 内で完結する (設計 §5)．
        let evaluator = &self.evaluator;
        let outcome = std::thread::scope(|s| {
            let handle =
                s.spawn(move || match evaluator {
                    EngineEvaluator::Mock(e) => Searcher::new(e, options.clone())
                        .search_reusing(sfen, moves, &limits, retained),
                    #[cfg(feature = "onnx")]
                    EngineEvaluator::Onnx(e) => Searcher::new(e, options.clone())
                        .search_reusing(sfen, moves, &limits, retained),
                });
            let start = Instant::now();
            while !handle.is_finished() {
                std::thread::sleep(POLL_INTERVAL);
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
        // 更新後の木を保持して次回の subtree 再利用に備える (fresh でも保持する)
        let (result, tree) = outcome.map_err(|e| e.to_string())?;
        self.retained = Some(tree);
        Ok(to_outcome(&result))
    }

    fn nyugyoku_declarable(&self, sfen: &str, moves: &[String]) -> Result<bool, String> {
        let (board, _) = build_board_and_history(sfen, moves).map_err(|e| e.to_string())?;
        Ok(board.nyugyoku_declarable())
    }

    fn is_mock(&self) -> bool {
        matches!(self.evaluator, EngineEvaluator::Mock(_))
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
    SearchOutcome {
        best_usi: r.best_move.map(|m| m.to_usi()),
        winrate: r.winrate,
        pv: r.pv.iter().map(|m| m.to_usi()).collect(),
        playouts: r.stats.playouts,
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
                0.5,
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
                0.5,
                &stop,
                &mut NoopObserver,
            )
            .expect("mock 探索は成功する");
        setter.join().expect("setter 正常終了");
        assert!(outcome.best_usi.is_some());
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
                0.5,
                &stop,
                &mut NoopObserver,
            )
            .err()
            .expect("非合法手はエラー");
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
                0.5,
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
                0.5,
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
                0.5,
                &stop,
                &mut NoopObserver,
            )
            .expect("reset 後の探索成功");
        assert!(o3.best_usi.is_some());
    }
}
