//! [`SearchBackend`] の実装 — maou_search (MCTS + root-dfpn/leaf-mate) を使う．
//!
//! 評価器 (mock / ONNX) は [`MaouSearchBackend::build`] で 1 回だけ構築する
//! (USI `isready` のタイミング．TensorRT のエンジンビルドも warmup として
//! ここで済ませ，初手の `go` を遅らせない)．

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use maou_search::{
    build_board_and_history, EvalItem, Evaluator, MockEvaluator, SearchLimits, SearchOptions,
    SearchResult, Searcher, StopCause,
};
use maou_shogi::board::Board;
use maou_shogi::movegen::generate_legal_moves;

use crate::agent::{EngineConfig, SearchBackend, SearchBudget, SearchOutcome, STARTPOS_SFEN};

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

        let backend = MaouSearchBackend { evaluator, options };
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
        stop: &Arc<AtomicBool>,
    ) -> Result<SearchOutcome, String> {
        let (board, history) = build_board_and_history(sfen, moves).map_err(|e| e.to_string())?;
        let limits = SearchLimits {
            // 無期限 (go ponder / go infinite) は playout 上限 u64::MAX + stop
            // token で表現する (SearchLimits の規約)
            max_playouts: if budget.unbounded {
                Some(u64::MAX)
            } else {
                budget.max_playouts
            },
            time_ms: if budget.unbounded {
                None
            } else {
                budget.time_ms
            },
            stop: Some(Arc::clone(stop)),
        };
        let result = match &self.evaluator {
            EngineEvaluator::Mock(e) => Searcher::new(e, self.options.clone())
                .search_with_history(&board, &history, &limits),
            #[cfg(feature = "onnx")]
            EngineEvaluator::Onnx(e) => Searcher::new(e, self.options.clone())
                .search_with_history(&board, &history, &limits),
        };
        Ok(to_outcome(&result))
    }

    fn is_mock(&self) -> bool {
        matches!(self.evaluator, EngineEvaluator::Mock(_))
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
                    time_ms: None,
                    max_playouts: Some(200),
                    unbounded: false,
                },
                &stop,
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
                    time_ms: None,
                    max_playouts: None,
                    unbounded: true,
                },
                &stop,
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
                    time_ms: None,
                    max_playouts: Some(10),
                    unbounded: false,
                },
                &stop,
            )
            .err()
            .expect("非合法手はエラー");
        assert!(err.contains("7g7e"));
    }
}
