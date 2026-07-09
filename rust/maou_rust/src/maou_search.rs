//! maou_search (MCTS 1 局面探索エンジン) の Python バインディング．
//!
//! デフォルトビルドは pure Rust (MockEvaluator のみ — API 検証/開発用)．
//! 実 NN (ONNX) 探索は cargo feature `onnx` (CUDA は `onnx-cuda`，TensorRT は
//! `onnx-tensorrt`) を付けてビルドした場合のみ有効 (wheel 可搬性の維持)．

use pyo3::prelude::*;

use maou_search::{
    Evaluator, HistoryEntry, SearchLimits, SearchOptions, SearchResult, Searcher, StopCause,
};
use maou_shogi::board::Board;
use maou_shogi::movegen::generate_legal_moves;

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
}

#[pymethods]
impl SearchRootChild {
    fn __repr__(&self) -> String {
        format!(
            "SearchRootChild(usi='{}', visits={}, winrate={:.4}, prior={:.4})",
            self.usi, self.visits, self.winrate, self.prior
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
/// - 統計: `playouts` / `elapsed_ms` / `nps` / `max_depth` / `repetitions`
///   (千日手検出数) / `proven_nodes` (AND-OR 確定ノード数) / `nodes_used` /
///   `collisions` / `eval_batches` / `avg_batch` / `gc_runs`．
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
            })
            .collect(),
        stop: stop_cause_str(r.stop).to_string(),
        playouts: r.stats.playouts,
        elapsed_ms: r.stats.elapsed_ms,
        nps: r.stats.nps,
        max_depth: r.stats.max_depth,
        repetitions: r.stats.repetitions,
        proven_nodes: r.stats.proven_nodes,
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
/// - `root_dfpn` (bool, optional): ルート並行 dfpn 詰み探索 (デフォルト False)．
///   詰みが証明されると `stop == "root_proven"` で即返り，`pv` は詰み手順になる．
/// - `root_dfpn_nodes` (int, optional) / `root_dfpn_depth` (int, optional):
///   ルート dfpn の予算 (デフォルト 2^20 ノード / 深さ 2047)．
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
#[pyo3(signature = (sfen, *, moves=None, model_path=None, threads=None, batch_size=None, max_playouts=None, time_ms=None, node_capacity=None, c_puct=None, fpu=None, max_ply=None, gc_keep_ratio=None, root_dfpn=None, root_dfpn_nodes=None, root_dfpn_depth=None, use_cuda=None, use_tensorrt=None, trt_engine_cache_dir=None, pad_to=None, intra_threads=None))]
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
    use_cuda: Option<bool>,
    use_tensorrt: Option<bool>,
    trt_engine_cache_dir: Option<String>,
    pad_to: Option<usize>,
    intra_threads: Option<usize>,
) -> PyResult<PySearchResult> {
    // 基準局面 + 対局履歴 (千日手判定用) を構築する
    let mut board = Board::empty();
    board
        .set_sfen(sfen)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("不正な SFEN: {e:?}")))?;
    let mut history: Vec<HistoryEntry> = Vec::new();
    if let Some(moves) = &moves {
        for usi in moves {
            let mut probe = board.clone();
            let mv = generate_legal_moves(&mut probe)
                .into_iter()
                .find(|m| m.to_usi() == *usi)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "非合法または不正な指し手: {usi}"
                    ))
                })?;
            history.push(HistoryEntry::from_board(&board));
            board.do_move(mv);
        }
    }

    if let Some(d) = root_dfpn_depth {
        // dfpn は depth >= 2048 で panic するため ValueError に変換する
        if !(1..=2047).contains(&d) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "root_dfpn_depth must be in 1..=2047, got {d}"
            )));
        }
    }

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
    let limits = SearchLimits {
        max_playouts,
        time_ms,
    };

    match model_path {
        None => {
            // mock 評価器 (決定論的擬似乱数)．API 検証/開発用
            let _ = (use_cuda, use_tensorrt, trt_engine_cache_dir, pad_to, intra_threads);
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
            let evaluator = maou_search::OnnxEvaluator::from_file(&path, &onnx_options)
                .map_err(|e| {
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

/// Create maou_search submodule
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "maou_search")?;

    m.add_class::<PySearchResult>()?;
    m.add_class::<SearchRootChild>()?;
    m.add_function(wrap_pyfunction!(search, &m)?)?;

    Ok(m)
}
