//! maou_usi (USI 対局エージェント) の PyO3 バインディング．

use pyo3::prelude::*;

use maou_usi::{EngineConfig, TimeStrategyConfig};

/// USI エンジンを標準入出力で実行する (`quit`/EOF まで戻らない)．
///
/// GIL を解放して Rust の USI ループ (reader スレッド + dispatcher) が
/// stdin/stdout を専有する — Python 側は起動前に logging を stderr へ向けて
/// おくこと (stdout はプロトコル専用)．
///
/// # 引数 (全て keyword-only．CLI フラグ = 初期値で，USI `setoption` が上書きする)
///
/// - `engine_name` (str, optional): `id name` に出す名前 (デフォルト
///   "maou <maou_usi crate version>")．
/// - `engine_author` (str, optional): `id author` に出す作者名．
/// - `model_path` (str, optional): ONNX モデルパス．未指定なら mock 評価器
///   (開発検証用 — `isready` 時に `info string` で明示される)．
/// - `threads` / `batch_size` (int, optional): 探索スレッド数/評価バッチサイズ．
/// - `node_capacity` (int, optional): ノードプール容量．
/// - `use_cuda` / `use_tensorrt` (bool, optional): Execution Provider
///   (対応 feature 付き wheel が必要)．
/// - `trt_engine_cache_dir` (str, optional): TensorRT エンジンキャッシュ保存先．
/// - `network_delay_ms` (int, optional): 通信マージン (デフォルト 1000)．
/// - `min_think_ms` (int, optional): 最低思考時間 (デフォルト 100)．
/// - `root_dfpn` / `root_dfpn_nodes` / `root_dfpn_depth`: ルート並行 dfpn．
/// - `leaf_mate` / `leaf_mate_nodes` / `leaf_mate_threads`: leaf-mate．
///
/// # エラー
///
/// モデルロード失敗・不正局面などの致命的エラーは `RuntimeError`．
#[pyfunction]
#[pyo3(signature = (*, engine_name=None, engine_author=None, model_path=None, threads=None, batch_size=None, node_capacity=None, use_cuda=None, use_tensorrt=None, trt_engine_cache_dir=None, network_delay_ms=None, min_think_ms=None, root_dfpn=None, root_dfpn_nodes=None, root_dfpn_depth=None, leaf_mate=None, leaf_mate_nodes=None, leaf_mate_threads=None))]
#[allow(clippy::too_many_arguments)]
fn run_usi(
    py: Python<'_>,
    engine_name: Option<String>,
    engine_author: Option<String>,
    model_path: Option<String>,
    threads: Option<usize>,
    batch_size: Option<usize>,
    node_capacity: Option<u32>,
    use_cuda: Option<bool>,
    use_tensorrt: Option<bool>,
    trt_engine_cache_dir: Option<String>,
    network_delay_ms: Option<u64>,
    min_think_ms: Option<u64>,
    root_dfpn: Option<bool>,
    root_dfpn_nodes: Option<u64>,
    root_dfpn_depth: Option<u32>,
    leaf_mate: Option<bool>,
    leaf_mate_nodes: Option<u64>,
    leaf_mate_threads: Option<usize>,
) -> PyResult<()> {
    let mut config = EngineConfig::default();
    if let Some(v) = engine_name {
        config.engine_name = v;
    }
    if let Some(v) = engine_author {
        config.engine_author = v;
    }
    config.model_path = model_path.filter(|p| !p.is_empty());
    if let Some(v) = threads {
        config.threads = v;
    }
    if let Some(v) = batch_size {
        config.batch_size = v;
    }
    config.node_capacity = node_capacity;
    config.use_cuda = use_cuda.unwrap_or(false);
    config.use_tensorrt = use_tensorrt.unwrap_or(false);
    config.trt_cache_dir = trt_engine_cache_dir;
    let time_defaults = TimeStrategyConfig::default();
    config.time = TimeStrategyConfig {
        network_delay_ms: network_delay_ms.unwrap_or(time_defaults.network_delay_ms),
        min_think_ms: min_think_ms.unwrap_or(time_defaults.min_think_ms),
        horizon_moves: time_defaults.horizon_moves,
    };
    config.root_dfpn = root_dfpn;
    config.root_dfpn_nodes = root_dfpn_nodes;
    config.root_dfpn_depth = root_dfpn_depth;
    config.leaf_mate = leaf_mate;
    config.leaf_mate_nodes = leaf_mate_nodes;
    config.leaf_mate_threads = leaf_mate_threads;

    py.detach(move || maou_usi::run_stdio(config))
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Create maou_usi submodule
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "maou_usi")?;
    m.add_function(wrap_pyfunction!(run_usi, &m)?)?;
    Ok(m)
}
