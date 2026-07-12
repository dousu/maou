//! maou_convert submodule — 棋譜 (CSA/KIF) → HCPE 一括変換の PyO3 バインディング．
//!
//! Python `hcpe_converter.py` の per-move ループ + ProcessPoolExecutor を置換する．
//! ファイル直読み (UTF-8→cp932 fallback)・複数局 CSA 全変換・rayon 並列・
//! Arrow 直出力を Rust 側で完結し，GIL を解放して実行する．

use std::path::PathBuf;

use arrow_pyarrow::ToPyArrow;
use pyo3::prelude::*;

use maou_convert::{
    build_record_batch, convert_content, convert_files, ConvertOptions, ConvertOutcome,
    InputFormat, PipelineError,
};

/// Python 引数から `ConvertOptions` を組み立てる．
#[allow(clippy::too_many_arguments)]
fn build_options(
    min_rating: Option<i32>,
    min_moves: Option<usize>,
    max_moves: Option<usize>,
    allowed_endgame_status: Option<Vec<String>>,
    exclude_moves: Option<Vec<u32>>,
) -> ConvertOptions {
    ConvertOptions {
        min_rating,
        min_moves,
        max_moves,
        allowed_endgame_status,
        exclude_moves: exclude_moves.unwrap_or_default(),
    }
}

/// `--input-format` 文字列を `InputFormat` にパースする (未知は ValueError)．
fn parse_format(input_format: &str) -> PyResult<InputFormat> {
    InputFormat::parse(input_format).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("undefined format {input_format}"))
    })
}

/// 棋譜ファイル群を HCPE feather に一括変換する．
///
/// GIL を解放し rayon で並列にファイルを処理する．各ファイルの成功時は
/// `{output_dir}/{stem}.feather` に書き出し，戻り値として (path, status) の
/// リストを返す (status: `"success N rows"` / `"skipped"` /
/// `"skipped (no moves)"` / `"error: ..."`)．
///
/// # 引数
///
/// - `paths`: 入力ファイルパスのリスト
/// - `input_format`: `"csa"` または `"kif"`
/// - `output_dir`: feather 出力先ディレクトリ (呼び出し側で作成済みのこと)
/// - `min_rating` / `min_moves` / `max_moves` / `allowed_endgame_status`:
///   品質フィルタ (Python `game_filter` 相当)
/// - `exclude_moves`: 変換から除外する指し手 (cshogi 互換 32-bit int)
/// - `threads`: rayon スレッド数 (`None` はグローバルプール)
#[pyfunction]
#[pyo3(signature = (
    paths, input_format, output_dir, *,
    min_rating=None, min_moves=None, max_moves=None,
    allowed_endgame_status=None, exclude_moves=None, threads=None
))]
#[allow(clippy::too_many_arguments)]
fn convert_hcpe_files(
    py: Python<'_>,
    paths: Vec<String>,
    input_format: &str,
    output_dir: &str,
    min_rating: Option<i32>,
    min_moves: Option<usize>,
    max_moves: Option<usize>,
    allowed_endgame_status: Option<Vec<String>>,
    exclude_moves: Option<Vec<u32>>,
    threads: Option<usize>,
) -> PyResult<Vec<(String, String)>> {
    let format = parse_format(input_format)?;
    let opts = build_options(
        min_rating,
        min_moves,
        max_moves,
        allowed_endgame_status,
        exclude_moves,
    );
    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    let out_dir = PathBuf::from(output_dir);
    let results = py.detach(move || convert_files(&path_bufs, format, &out_dir, &opts, threads));
    Ok(results)
}

/// 棋譜文字列を HCPE の pyarrow RecordBatch に変換する (parity テスト用)．
///
/// 1 行も生成されない (フィルタ / 指し手なし) 場合や変換エラーは ValueError．
///
/// # 引数
///
/// - `content`: 棋譜文字列 (デコード済み)
/// - `input_format`: `"csa"` または `"kif"`
/// - `id_prefix`: id の接頭辞 (`{id_prefix}_{ply}`; Python は
///   `"{stem}.hcpe"` を渡す)
/// - フィルタ引数は `convert_hcpe_files` と同じ
#[pyfunction]
#[pyo3(signature = (
    content, input_format, id_prefix, *,
    min_rating=None, min_moves=None, max_moves=None,
    allowed_endgame_status=None, exclude_moves=None
))]
#[allow(clippy::too_many_arguments)]
fn convert_hcpe_str<'py>(
    py: Python<'py>,
    content: &str,
    input_format: &str,
    id_prefix: &str,
    min_rating: Option<i32>,
    min_moves: Option<usize>,
    max_moves: Option<usize>,
    allowed_endgame_status: Option<Vec<String>>,
    exclude_moves: Option<Vec<u32>>,
) -> PyResult<Bound<'py, PyAny>> {
    let format = parse_format(input_format)?;
    let opts = build_options(
        min_rating,
        min_moves,
        max_moves,
        allowed_endgame_status,
        exclude_moves,
    );
    let outcome = convert_content(content, format, id_prefix, &opts).map_err(pipeline_err)?;
    let rows = match outcome {
        ConvertOutcome::Success(rows) => rows,
        ConvertOutcome::Skipped => return Err(pyo3::exceptions::PyValueError::new_err("skipped")),
        ConvertOutcome::SkippedNoMoves => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "skipped (no moves)",
            ))
        }
    };
    let batch = build_record_batch(&rows)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    batch.to_pyarrow(py)
}

/// PipelineError を Python ValueError に変換する．
fn pipeline_err(e: PipelineError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// maou_convert サブモジュールを作成する．
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "maou_convert")?;
    m.add_function(wrap_pyfunction!(convert_hcpe_files, &m)?)?;
    m.add_function(wrap_pyfunction!(convert_hcpe_str, &m)?)?;
    Ok(m)
}
