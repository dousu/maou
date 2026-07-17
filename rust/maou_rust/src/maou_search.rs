//! maou_search (MCTS 1 局面探索エンジン) の Python バインディング．
//!
//! デフォルトビルドは pure Rust (MockEvaluator のみ — API 検証/開発用)．
//! 実 NN (ONNX) 探索は cargo feature `onnx` (CUDA は `onnx-cuda`，TensorRT は
//! `onnx-tensorrt`) を付けてビルドした場合のみ有効 (wheel 可搬性の維持)．

use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use maou_search::preprocess;
use maou_search::{
    Evaluator, HistoryEntry, SearchLimits, SearchOptions, SearchResult, Searcher, StopCause,
};
use maou_shogi::board::Board;
use maou_shogi::moves::Move;
use maou_shogi::types::Color;

/// ルート直下の候補手 1 つの統計．
#[pyclass(frozen, skip_from_py_object)]
#[derive(Clone)]
struct SearchRootChild {
    /// 指し手 (USI 形式)．
    #[pyo3(get)]
    usi: String,
    /// 訪問回数．
    #[pyo3(get)]
    visits: u64,
    /// ルート手番側から見た勝率 (0-1，未訪問なら 0)．
    #[pyo3(get)]
    winrate: f64,
    /// policy 事前確率．
    #[pyo3(get)]
    prior: f32,
    /// ルート手番側から見た確定値 (0 = この手を指すと負け確定，0.5 = 引き分け
    /// 確定，1 = 勝ち確定)．未確定なら `None`．
    #[pyo3(get)]
    proven: Option<f64>,
}

#[pymethods]
impl SearchRootChild {
    fn __repr__(&self) -> String {
        let proven = self
            .proven
            .map_or_else(|| "None".to_string(), |v| format!("{v}"));
        format!(
            "SearchRootChild(usi='{}', visits={}, winrate={:.4}, prior={:.4}, proven={})",
            self.usi, self.visits, self.winrate, self.prior, proven
        )
    }
}

/// 1 局面探索 ([`search`]) の結果．
///
/// - `best_move`: 最有力手 (USI 形式; ルートに合法手がなければ `None`)．
/// - `winrate`: ルート手番側から見た best_move の勝率 (0-1)．
///   勝敗確定時 (`stop == "root_proven"`) は確定値 (0 / 0.5 / 1)．
/// - `pv`: 読み筋 (USI 形式のリスト)．
/// - `root_children`: ルート直下の全候補手の統計 (合法手生成順)．
/// - `stop`: 停止理由 (`"playout_limit"` / `"time_limit"` / `"pool_exhausted"` /
///   `"root_terminal"` / `"root_proven"`)．
/// - 統計: `playouts` / `warmup_ms` (ルート評価/エンジンビルドの所要; 計測
///   区間外) / `elapsed_ms` / `nps` / `max_depth` / `repetitions`
///   (千日手検出数) / `proven_nodes` (AND-OR 確定ノード数) / `leaf_mates`
///   (leaf-mate が葉で詰みを証明した回数) / `nodes_used` / `collisions` /
///   `eval_batches` / `avg_batch` / `gc_runs`．
#[pyclass(frozen, name = "SearchResult")]
struct PySearchResult {
    #[pyo3(get)]
    best_move: Option<String>,
    #[pyo3(get)]
    winrate: f64,
    #[pyo3(get)]
    pv: Vec<String>,
    #[pyo3(get)]
    root_children: Vec<SearchRootChild>,
    #[pyo3(get)]
    stop: String,
    #[pyo3(get)]
    playouts: u64,
    #[pyo3(get)]
    warmup_ms: u64,
    #[pyo3(get)]
    elapsed_ms: u64,
    #[pyo3(get)]
    nps: f64,
    #[pyo3(get)]
    max_depth: u16,
    #[pyo3(get)]
    repetitions: u64,
    #[pyo3(get)]
    proven_nodes: u64,
    #[pyo3(get)]
    leaf_mates: u64,
    #[pyo3(get)]
    nodes_used: u32,
    #[pyo3(get)]
    collisions: u64,
    #[pyo3(get)]
    eval_batches: u64,
    #[pyo3(get)]
    avg_batch: f64,
    #[pyo3(get)]
    gc_runs: u64,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(best_move={:?}, winrate={:.4}, stop='{}', playouts={}, nps={:.0}, pv={:?})",
            self.best_move, self.winrate, self.stop, self.playouts, self.nps, self.pv
        )
    }
}

/// [`StopCause`] を Python 向けの snake_case 文字列にする．
fn stop_cause_str(stop: StopCause) -> &'static str {
    match stop {
        StopCause::PlayoutLimit => "playout_limit",
        StopCause::TimeLimit => "time_limit",
        StopCause::PoolExhausted => "pool_exhausted",
        StopCause::RootTerminal => "root_terminal",
        StopCause::RootProven => "root_proven",
        StopCause::External => "external",
    }
}

fn to_py_result(r: SearchResult) -> PySearchResult {
    PySearchResult {
        best_move: r.best_move.map(|m| m.to_usi()),
        winrate: r.winrate,
        pv: r.pv.iter().map(|m| m.to_usi()).collect(),
        root_children: r
            .root_children
            .iter()
            .map(|c| SearchRootChild {
                usi: c.mv.to_usi(),
                visits: c.visits,
                winrate: c.q,
                prior: c.prior,
                proven: c.proven,
            })
            .collect(),
        stop: stop_cause_str(r.stop).to_string(),
        playouts: r.stats.playouts,
        warmup_ms: r.stats.warmup_ms,
        elapsed_ms: r.stats.elapsed_ms,
        nps: r.stats.nps,
        max_depth: r.stats.max_depth,
        repetitions: r.stats.repetitions,
        proven_nodes: r.stats.proven_nodes,
        leaf_mates: r.stats.leaf_mates,
        nodes_used: r.stats.nodes_used,
        collisions: r.stats.collisions,
        eval_batches: r.stats.eval_batches,
        avg_batch: r.stats.avg_batch,
        gc_runs: r.stats.gc_runs,
    }
}

/// GIL を解放して探索を実行する (solve_tsume と同じパターン)．
fn run_search<E: Evaluator>(
    py: Python<'_>,
    evaluator: E,
    options: SearchOptions,
    board: Board,
    history: Vec<HistoryEntry>,
    limits: SearchLimits,
) -> PySearchResult {
    let result = py.detach(move || {
        let searcher = Searcher::new(&evaluator, options);
        searcher.search_with_history(&board, &history, &limits)
    });
    to_py_result(result)
}

/// 基準局面 SFEN + USI 指し手列から root 局面と対局履歴 (千日手判定用) を構築する．
///
/// 実装は [`maou_search::build_board_and_history`] (単一実装) への委譲．
/// エラーメッセージは従来どおり `ValueError` に載せる．
fn build_board_and_history(
    sfen: &str,
    moves: Option<&[String]>,
) -> PyResult<(Board, Vec<HistoryEntry>)> {
    maou_search::build_board_and_history(sfen, moves.unwrap_or(&[]))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// dfpn は depth >= 2048 で panic するため ValueError に変換する．
fn validate_root_dfpn_depth(root_dfpn_depth: Option<u32>) -> PyResult<()> {
    if let Some(d) = root_dfpn_depth {
        if !(1..=2047).contains(&d) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "root_dfpn_depth must be in 1..=2047, got {d}"
            )));
        }
    }
    Ok(())
}

/// 1 局面を MCTS で探索して最有力手・勝率・読み筋を返す．
///
/// 返り値: [`PySearchResult`] (`SearchResult`) オブジェクト．
///
/// # 引数
///
/// - `sfen` (str): 基準局面の SFEN 文字列．不正な場合は `ValueError`．
/// - `moves` (list[str], optional): `sfen` から root 局面までの指し手列 (USI 形式)．
///   USI の `position ... moves ...` 相当．途中局面は千日手判定の対局履歴として
///   使われる．非合法手を含む場合は `ValueError`．
/// - `model_path` (str, optional): ONNX モデルのパス．未指定なら決定論的な
///   mock 評価器で探索する (API 検証/開発用 — 指し手の品質は無意味)．
///   指定時は `onnx` feature 付きでビルドされた wheel が必要 (無ければ
///   `RuntimeError`)．
/// - `threads` (int, optional): 探索スレッド数 (デフォルト 1)．
/// - `batch_size` (int, optional): 評価バッチサイズ (デフォルト 8．GPU では
///   256 程度を推奨)．
/// - `max_playouts` (int, optional): playout 数上限．
/// - `time_ms` (int, optional): 時間上限 (ミリ秒)．`max_playouts` と両方未指定
///   なら playout 上限 2^20 が適用される．
/// - `node_capacity` (int, optional): ノードプール容量 (デフォルト 2^20)．
/// - `c_puct` (float, optional) / `fpu` (float, optional): PUCT パラメータ．
/// - `max_ply` (int, optional): 最大探索深さ (デフォルト 512)．
/// - `gc_keep_ratio` (float, optional): GC 後に残すノード比 (デフォルト 0.5)．
/// - `root_dfpn` (bool, optional): ルート並行 dfpn 詰み探索 (デフォルト True)．
///   詰みが証明されると `stop == "root_proven"` で即返り，`pv` は詰み手順になる．
/// - `root_dfpn_nodes` (int, optional) / `root_dfpn_depth` (int, optional):
///   ルート dfpn の予算 (デフォルト 2,000,000 ノード / 深さ 2047)．
/// - `leaf_mate` (bool, optional): MCTS の葉の短手詰み探索を有効にする
///   (デフォルト True)．探索スレッドは王手手段を持つ葉を専用 mate スレッドへ
///   非同期依頼するだけで solve せず，mate スレッド (余剰 CPU) が詰みを証明
///   したら当該葉を勝ち確定にして AND-OR 伝播する (探索 NPS に影響しない)．
/// - `leaf_mate_nodes` (int, optional): leaf-mate 1 回あたりのノード予算
///   (デフォルト 50)．小さいほど cheap かつ短手のみ検出する．
/// - `leaf_mate_threads` (int, optional): leaf-mate 専用スレッド数
///   (デフォルト 1)．余る CPU スレッド数に合わせて増やす．
/// - `use_cuda` (bool, optional): CUDA Execution Provider (`onnx-cuda` feature 必要)．
/// - `use_tensorrt` (bool, optional): TensorRT Execution Provider
///   (`onnx-tensorrt` feature 必要)．有効時は `pad_to` 未指定なら `batch_size`
///   に固定して shape 別エンジンビルドを防ぐ．
/// - `trt_engine_cache_dir` (str, optional): TensorRT エンジンキャッシュ保存先．
/// - `pad_to` (int, optional): 評価バッチをこのサイズへゼロ局面 padding する．
/// - `intra_threads` (int, optional): ONNX Runtime の intra-op スレッド数 (デフォルト 1)．
///
/// # 注意
///
/// - 探索中は GIL を解放するが Python シグナルは処理しない — Ctrl-C は探索が
///   返るまで効かない．中断制御は `max_playouts` / `time_ms` で行うこと．
#[pyfunction]
#[pyo3(signature = (sfen, *, moves=None, model_path=None, threads=None, batch_size=None, max_playouts=None, time_ms=None, node_capacity=None, c_puct=None, fpu=None, max_ply=None, gc_keep_ratio=None, root_dfpn=None, root_dfpn_nodes=None, root_dfpn_depth=None, leaf_mate=None, leaf_mate_nodes=None, leaf_mate_threads=None, use_cuda=None, use_tensorrt=None, trt_engine_cache_dir=None, pad_to=None, intra_threads=None))]
#[allow(clippy::too_many_arguments)]
fn search(
    py: Python<'_>,
    sfen: &str,
    moves: Option<Vec<String>>,
    model_path: Option<String>,
    threads: Option<usize>,
    batch_size: Option<usize>,
    max_playouts: Option<u64>,
    time_ms: Option<u64>,
    node_capacity: Option<u32>,
    c_puct: Option<f32>,
    fpu: Option<f32>,
    max_ply: Option<u16>,
    gc_keep_ratio: Option<f32>,
    root_dfpn: Option<bool>,
    root_dfpn_nodes: Option<u64>,
    root_dfpn_depth: Option<u32>,
    leaf_mate: Option<bool>,
    leaf_mate_nodes: Option<u64>,
    leaf_mate_threads: Option<usize>,
    use_cuda: Option<bool>,
    use_tensorrt: Option<bool>,
    trt_engine_cache_dir: Option<String>,
    pad_to: Option<usize>,
    intra_threads: Option<usize>,
) -> PyResult<PySearchResult> {
    // 基準局面 + 対局履歴 (千日手判定用) を構築する
    let (board, history) = build_board_and_history(sfen, moves.as_deref())?;

    validate_root_dfpn_depth(root_dfpn_depth)?;

    let mut options = SearchOptions::default();
    if let Some(v) = threads {
        options.threads = v;
    }
    if let Some(v) = batch_size {
        options.batch_size = v;
    }
    if let Some(v) = node_capacity {
        options.node_capacity = v;
    }
    if let Some(v) = c_puct {
        options.c_puct = v;
    }
    if let Some(v) = fpu {
        options.fpu = v;
    }
    if let Some(v) = max_ply {
        options.max_ply = v;
    }
    if let Some(v) = gc_keep_ratio {
        options.gc_keep_ratio = v;
    }
    if let Some(v) = root_dfpn {
        options.root_dfpn = v;
    }
    if let Some(v) = root_dfpn_nodes {
        options.root_dfpn_nodes = v;
    }
    if let Some(v) = root_dfpn_depth {
        options.root_dfpn_depth = v;
    }
    if let Some(v) = leaf_mate {
        options.leaf_mate = v;
    }
    if let Some(v) = leaf_mate_nodes {
        options.leaf_mate_nodes = v;
    }
    if let Some(v) = leaf_mate_threads {
        options.leaf_mate_threads = v;
    }
    let limits = SearchLimits {
        max_playouts,
        time_ms,
        ..SearchLimits::default()
    };

    match model_path {
        None => {
            // mock 評価器 (決定論的擬似乱数)．API 検証/開発用
            let _ = (
                use_cuda,
                use_tensorrt,
                trt_engine_cache_dir,
                pad_to,
                intra_threads,
            );
            let evaluator = maou_search::MockEvaluator::new(0);
            Ok(run_search(py, evaluator, options, board, history, limits))
        }
        #[cfg(feature = "onnx")]
        Some(path) => {
            let onnx_options = maou_search::onnx::OnnxOptions {
                intra_threads: intra_threads.unwrap_or(1),
                use_cuda: use_cuda.unwrap_or(false),
                use_tensorrt: use_tensorrt.unwrap_or(false),
                trt_engine_cache_dir,
                // TensorRT は shape ごとにエンジンをビルドするため batch_size に固定する
                pad_to: pad_to.or(if use_tensorrt.unwrap_or(false) {
                    Some(options.batch_size)
                } else {
                    None
                }),
            };
            let evaluator =
                maou_search::OnnxEvaluator::from_file(&path, &onnx_options).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "ONNX モデルの読み込みに失敗: {e}"
                    ))
                })?;
            Ok(run_search(py, evaluator, options, board, history, limits))
        }
        #[cfg(not(feature = "onnx"))]
        Some(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "この wheel は onnx feature なしでビルドされているため model_path を使えません．\
             `maturin develop --features onnx` (CUDA: onnx-cuda / TensorRT: onnx-tensorrt) \
             でビルドしてください",
        )),
    }
}

/// [`SearchEngine`] が保持する評価器 (mock または ONNX)．
enum EngineEvaluator {
    Mock(maou_search::MockEvaluator),
    #[cfg(feature = "onnx")]
    Onnx(maou_search::OnnxEvaluator),
}

/// 評価器を 1 回だけ構築・保持して複数局面を連続探索する永続エンジン．
///
/// 関数 [`search`] は呼び出しごとに ONNX モデルをロードするため，棋譜解析の
/// ように N 局面を連続探索する用途では本クラスを使う (モデルロードは
/// `__init__` の 1 回のみ．TensorRT はプロセス内でエンジンを使い回せる)．
///
/// 予算 (時間/ノード数) は `search` の引数として毎回受け取る — 時間配分の
/// 計画は上位レイヤーの責務であり本クラスは持たない
/// (docs/design/game-analysis/index.md §4)．
///
/// # コンストラクタ引数
///
/// - `model_path` (str, optional): ONNX モデルのパス．未指定なら決定論的な
///   mock 評価器 (API 検証/開発用 — 指し手の品質は無意味)．指定時は `onnx`
///   feature 付きの wheel が必要 (無ければ `RuntimeError`)．
/// - `threads` (int, optional): 探索スレッド数．
/// - `batch_size` (int, optional): 評価バッチサイズ (TensorRT の padding
///   サイズもこれに固定される)．
/// - `use_cuda` (bool, optional): CUDA Execution Provider (`onnx-cuda` feature 必要)．
/// - `use_tensorrt` (bool, optional): TensorRT Execution Provider
///   (`onnx-tensorrt` feature 必要)．
/// - `trt_engine_cache_dir` (str, optional): TensorRT エンジンキャッシュ保存先．
#[pyclass(frozen)]
struct SearchEngine {
    evaluator: EngineEvaluator,
    threads: usize,
    batch_size: usize,
}

#[pymethods]
impl SearchEngine {
    #[new]
    #[pyo3(signature = (*, model_path=None, threads=None, batch_size=None, use_cuda=None, use_tensorrt=None, trt_engine_cache_dir=None))]
    fn new(
        model_path: Option<String>,
        threads: Option<usize>,
        batch_size: Option<usize>,
        use_cuda: Option<bool>,
        use_tensorrt: Option<bool>,
        trt_engine_cache_dir: Option<String>,
    ) -> PyResult<Self> {
        let defaults = SearchOptions::default();
        let threads = threads.unwrap_or(defaults.threads);
        let batch_size = batch_size.unwrap_or(defaults.batch_size);
        let evaluator = match model_path {
            None => {
                // mock 評価器 (決定論的擬似乱数)．API 検証/開発用
                let _ = (use_cuda, use_tensorrt, trt_engine_cache_dir);
                EngineEvaluator::Mock(maou_search::MockEvaluator::new(0))
            }
            #[cfg(feature = "onnx")]
            Some(path) => {
                let onnx_options = maou_search::onnx::OnnxOptions {
                    intra_threads: 1,
                    use_cuda: use_cuda.unwrap_or(false),
                    use_tensorrt: use_tensorrt.unwrap_or(false),
                    trt_engine_cache_dir,
                    // TensorRT は shape ごとにエンジンをビルドするため batch_size に固定する
                    pad_to: if use_tensorrt.unwrap_or(false) {
                        Some(batch_size)
                    } else {
                        None
                    },
                };
                EngineEvaluator::Onnx(
                    maou_search::OnnxEvaluator::from_file(&path, &onnx_options).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "ONNX モデルの読み込みに失敗: {e}"
                        ))
                    })?,
                )
            }
            #[cfg(not(feature = "onnx"))]
            Some(_) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "この wheel は onnx feature なしでビルドされているため model_path を使えません．\
                     `maturin develop --features onnx` (CUDA: onnx-cuda / TensorRT: onnx-tensorrt) \
                     でビルドしてください",
                ));
            }
        };
        Ok(SearchEngine {
            evaluator,
            threads,
            batch_size,
        })
    }

    /// 保持している評価器で 1 局面を探索する．
    ///
    /// 引数・返り値の意味は関数 [`search`] と同じ (評価器系オプションは
    /// コンストラクタで固定済みのため受け取らない)．探索中は GIL を解放する．
    #[pyo3(signature = (sfen, *, moves=None, max_playouts=None, time_ms=None, root_dfpn=None, root_dfpn_nodes=None, root_dfpn_depth=None, leaf_mate=None, leaf_mate_nodes=None, leaf_mate_threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        py: Python<'_>,
        sfen: &str,
        moves: Option<Vec<String>>,
        max_playouts: Option<u64>,
        time_ms: Option<u64>,
        root_dfpn: Option<bool>,
        root_dfpn_nodes: Option<u64>,
        root_dfpn_depth: Option<u32>,
        leaf_mate: Option<bool>,
        leaf_mate_nodes: Option<u64>,
        leaf_mate_threads: Option<usize>,
    ) -> PyResult<PySearchResult> {
        let (board, history) = build_board_and_history(sfen, moves.as_deref())?;
        validate_root_dfpn_depth(root_dfpn_depth)?;

        let mut options = SearchOptions {
            threads: self.threads,
            batch_size: self.batch_size,
            ..SearchOptions::default()
        };
        if let Some(v) = root_dfpn {
            options.root_dfpn = v;
        }
        if let Some(v) = root_dfpn_nodes {
            options.root_dfpn_nodes = v;
        }
        if let Some(v) = root_dfpn_depth {
            options.root_dfpn_depth = v;
        }
        if let Some(v) = leaf_mate {
            options.leaf_mate = v;
        }
        if let Some(v) = leaf_mate_nodes {
            options.leaf_mate_nodes = v;
        }
        if let Some(v) = leaf_mate_threads {
            options.leaf_mate_threads = v;
        }
        let limits = SearchLimits {
            max_playouts,
            time_ms,
            ..SearchLimits::default()
        };

        let result = match &self.evaluator {
            EngineEvaluator::Mock(ev) => py.detach(move || {
                Searcher::new(ev, options).search_with_history(&board, &history, &limits)
            }),
            #[cfg(feature = "onnx")]
            EngineEvaluator::Onnx(ev) => py.detach(move || {
                Searcher::new(ev, options).search_with_history(&board, &history, &limits)
            }),
        };
        Ok(to_py_result(result))
    }
}

/// (N, 32) の HCP 配列を検証し，(N, フラットな Vec) に変換する．
fn hcp_array_to_vec(hcp: &PyReadonlyArray2<'_, u8>) -> PyResult<(usize, Vec<u8>)> {
    let shape = hcp.shape();
    if shape[1] != 32 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hcp array must have shape (N, 32), got (N, {})",
            shape[1]
        )));
    }
    let n = shape[0];
    // 非 contiguous でも論理順 (row-major) で吸い出す
    let data: Vec<u8> = hcp.as_array().iter().copied().collect();
    Ok((n, data))
}

/// PreprocessError を Python ValueError に変換する．
fn preprocess_err(e: preprocess::PreprocessError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// `preprocess_hcpes` の返り値 (hashes, move_labels, result_values)．
type ProcessedArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u16>>,
    Bound<'py, PyArray1<f32>>,
);

/// `encode_hcp_features` の返り値 (board_id_positions, pieces_in_hand)．
type FeatureArrays<'py> = (Bound<'py, PyArray3<u8>>, Bound<'py, PyArray2<u8>>);

/// N 局面の (zobrist hash, move label, result value) を一括計算する．
///
/// HCPE 前処理 (`_process_single_array`) の per-position ループの置き換え．
/// GIL を解放して計算する．
///
/// # 引数
///
/// - `hcp`: HCP 配列 (N, 32) uint8
/// - `move16`: 16-bit move 配列 (N,) int16 (HCPE の bestMove16)
/// - `game_result`: 勝敗配列 (N,) int8 (cshogi 規約: 0=引き分け, 1=先手勝ち,
///   2=後手勝ち)
///
/// # 返り値
///
/// `(hashes (N,) uint64, move_labels (N,) uint16, result_values (N,) float32)`
///
/// 不正な HCP や move label に変換できない指し手が含まれる場合は ValueError．
#[pyfunction]
fn preprocess_hcpes<'py>(
    py: Python<'py>,
    hcp: PyReadonlyArray2<'py, u8>,
    move16: PyReadonlyArray1<'py, i16>,
    game_result: PyReadonlyArray1<'py, i8>,
) -> PyResult<ProcessedArrays<'py>> {
    let (_, hcp_vec) = hcp_array_to_vec(&hcp)?;
    let move16_vec = move16.as_array().to_vec();
    let game_result_vec = game_result.as_array().to_vec();
    let (hashes, labels, results) = py
        .detach(move || preprocess::process_hcpes(&hcp_vec, &move16_vec, &game_result_vec))
        .map_err(preprocess_err)?;
    Ok((
        hashes.into_pyarray(py),
        labels.into_pyarray(py),
        results.into_pyarray(py),
    ))
}

/// N 局面の NN 入力特徴量を一括エンコードする．
///
/// Python `Board.get_normalized_board_id_positions` /
/// `get_normalized_pieces_in_hand` の一括版
/// (手番視点正規化込み)．GIL を解放して計算する．
///
/// # 引数
///
/// - `hcp`: HCP 配列 (N, 32) uint8
///
/// # 返り値
///
/// `(board_id_positions (N, 9, 9) uint8, pieces_in_hand (N, 14) uint8)`
#[pyfunction]
fn encode_hcp_features<'py>(
    py: Python<'py>,
    hcp: PyReadonlyArray2<'py, u8>,
) -> PyResult<FeatureArrays<'py>> {
    let (n, hcp_vec) = hcp_array_to_vec(&hcp)?;
    let (boards, hands) = py
        .detach(move || preprocess::encode_hcp_features(&hcp_vec))
        .map_err(preprocess_err)?;
    let boards =
        Array3::from_shape_vec((n, 9, 9), boards).expect("encode_hcp_features は N*81 要素を返す");
    let hands =
        Array2::from_shape_vec((n, 14), hands).expect("encode_hcp_features は N*14 要素を返す");
    Ok((boards.into_pyarray(py), hands.into_pyarray(py)))
}

/// N 局面の合法手ラベルマスクを一括生成する．
///
/// stage2 データ生成の合法手ラベル計算の置き換え (手番視点正規化込み)．
/// GIL を解放して計算する．
///
/// # 引数
///
/// - `hcp`: HCP 配列 (N, 32) uint8
///
/// # 返り値
///
/// `legal_move_mask (N, 1496) uint8` (1=合法手ラベル)
#[pyfunction]
fn legal_move_masks<'py>(
    py: Python<'py>,
    hcp: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (n, hcp_vec) = hcp_array_to_vec(&hcp)?;
    let masks = py
        .detach(move || preprocess::legal_move_masks(&hcp_vec))
        .map_err(preprocess_err)?;
    let masks = Array2::from_shape_vec((n, maou_search::label::MOVE_LABELS_NUM), masks)
        .expect("legal_move_masks は N*1496 要素を返す");
    Ok(masks.into_pyarray(py))
}

/// N 局面の zobrist hash を一括計算する．
///
/// stage2 データ生成の重複排除 (unique HCP 収集) の置き換え．
/// GIL を解放して計算する．
///
/// # 引数
///
/// - `hcp`: HCP 配列 (N, 32) uint8
///
/// # 返り値
///
/// `hashes (N,) uint64`
#[pyfunction]
fn hcp_hashes<'py>(
    py: Python<'py>,
    hcp: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let (_, hcp_vec) = hcp_array_to_vec(&hcp)?;
    let hashes = py
        .detach(move || preprocess::hcp_hashes(&hcp_vec))
        .map_err(preprocess_err)?;
    Ok(hashes.into_pyarray(py))
}

/// 指し手 1 つを policy ラベル (0..1496) に変換する．
///
/// Python `make_move_label` の Rust 委譲先 (手番視点正規化込み)．
///
/// # 引数
///
/// - `turn`: 手番 (0=先手, 1=後手)
/// - `m`: 指し手 (32-bit move / 16-bit move のどちらでも可 —
///   ラベルは下位 15 bit のみから決まる)
///
/// ラベルに変換できない指し手 (盤の端への行き止まり成らず等) は ValueError．
#[pyfunction]
fn move_label(turn: u8, m: u32) -> PyResult<u16> {
    let color = Color::from_u8(turn)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("turn must be 0 or 1"))?;
    maou_search::label::try_move_label(color, Move::from_raw_u32(m)).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Can not transform illegal move to move label.")
    })
}

/// Create maou_search submodule
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "maou_search")?;

    m.add_class::<PySearchResult>()?;
    m.add_class::<SearchRootChild>()?;
    m.add_class::<SearchEngine>()?;
    m.add_function(wrap_pyfunction!(search, &m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_hcpes, &m)?)?;
    m.add_function(wrap_pyfunction!(encode_hcp_features, &m)?)?;
    m.add_function(wrap_pyfunction!(legal_move_masks, &m)?)?;
    m.add_function(wrap_pyfunction!(hcp_hashes, &m)?)?;
    m.add_function(wrap_pyfunction!(move_label, &m)?)?;
    m.add("MOVE_LABELS_NUM", maou_search::label::MOVE_LABELS_NUM)?;

    Ok(m)
}
