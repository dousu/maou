//! maou_io submodule - Arrow IPC I/O and sparse array operations

use pyo3::prelude::*;
use arrow::array::RecordBatch;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use ::maou_io as maou_io_core;
use maou_io_core::MaouIOError;

#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Maou I/O Rust backend initialized".to_string())
}

#[pyfunction]
fn save_hcpe_feather(py: Python, batch: &Bound<'_, PyAny>, file_path: String) -> PyResult<()> {
    let batch = RecordBatch::from_pyarrow_bound(batch)?;
    py.allow_threads(|| maou_io_core::arrow_io::save_feather(&batch, &file_path))
        .map_err(|e: MaouIOError| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })?;
    Ok(())
}

#[pyfunction]
fn load_hcpe_feather(py: Python, file_path: String) -> PyResult<PyObject> {
    let batch = py
        .allow_threads(|| maou_io_core::arrow_io::load_feather(&file_path))
        .map_err(|e: MaouIOError| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })?;
    batch.to_pyarrow(py)
}

#[pyfunction]
fn save_preprocessing_feather(py: Python, batch: &Bound<'_, PyAny>, file_path: String) -> PyResult<()> {
    let batch = RecordBatch::from_pyarrow_bound(batch)?;
    py.allow_threads(|| maou_io_core::arrow_io::save_feather(&batch, &file_path))
        .map_err(|e: MaouIOError| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })?;
    Ok(())
}

#[pyfunction]
fn load_preprocessing_feather(py: Python, file_path: String) -> PyResult<PyObject> {
    let batch = py
        .allow_threads(|| maou_io_core::arrow_io::load_feather(&file_path))
        .map_err(|e: MaouIOError| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })?;
    batch.to_pyarrow(py)
}

// Generic feather I/O functions (for Stage1/Stage2 data)

#[pyfunction]
fn save_feather_file(py: Python, batch: &Bound<'_, PyAny>, file_path: String) -> PyResult<()> {
    let batch = RecordBatch::from_pyarrow_bound(batch)?;
    py.allow_threads(|| maou_io_core::arrow_io::save_feather(&batch, &file_path))
        .map_err(|e: MaouIOError| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })?;
    Ok(())
}

#[pyfunction]
fn load_feather_file(py: Python, file_path: String) -> PyResult<PyObject> {
    let batch = py
        .allow_threads(|| maou_io_core::arrow_io::load_feather(&file_path))
        .map_err(|e: MaouIOError| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })?;
    batch.to_pyarrow(py)
}

// Sparse array compression functions

#[pyfunction]
fn compress_sparse_array_rust(dense: Vec<i32>) -> PyResult<(Vec<u16>, Vec<i32>)> {
    maou_io_core::sparse_array::compress_sparse_array(&dense)
        .map_err(|e: MaouIOError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
fn expand_sparse_array_rust(indices: Vec<u16>, values: Vec<i32>, size: usize) -> PyResult<Vec<i32>> {
    maou_io_core::sparse_array::expand_sparse_array(&indices, &values, size)
        .map_err(|e: MaouIOError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
fn add_sparse_arrays_rust(
    indices1: Vec<u16>,
    values1: Vec<i32>,
    indices2: Vec<u16>,
    values2: Vec<i32>,
) -> PyResult<(Vec<u16>, Vec<i32>)> {
    maou_io_core::sparse_array::add_sparse_arrays(&indices1, &values1, &indices2, &values2)
        .map_err(|e: MaouIOError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
fn add_sparse_arrays_dual_rust(
    indices1: Vec<u16>,
    label_values1: Vec<i32>,
    win_values1: Vec<f32>,
    indices2: Vec<u16>,
    label_values2: Vec<i32>,
    win_values2: Vec<f32>,
) -> PyResult<(Vec<u16>, Vec<i32>, Vec<f32>)> {
    maou_io_core::sparse_array::add_sparse_arrays_dual(
        &indices1, &label_values1, &win_values1,
        &indices2, &label_values2, &win_values2,
    )
    .map_err(|e: MaouIOError| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

#[pyfunction]
fn split_feather_file(
    py: Python,
    file_path: String,
    output_dir: String,
    rows_per_file: usize,
) -> PyResult<Vec<String>> {
    py.allow_threads(|| {
        maou_io_core::arrow_io::split_feather(&file_path, &output_dir, rows_per_file)
    })
    .map_err(|e: MaouIOError| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
    })
}

#[pyfunction]
fn merge_feather_files(
    py: Python,
    file_paths: Vec<String>,
    output_dir: String,
    rows_per_chunk: usize,
    output_prefix: String,
) -> PyResult<Vec<String>> {
    py.allow_threads(|| {
        maou_io_core::arrow_io::merge_feather_files(
            &file_paths,
            &output_dir,
            rows_per_chunk,
            &output_prefix,
        )
    })
    .map_err(|e: MaouIOError| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
    })
}

/// Create maou_io submodule
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "maou_io")?;

    m.add_function(wrap_pyfunction!(hello, &m)?)?;
    m.add_function(wrap_pyfunction!(save_hcpe_feather, &m)?)?;
    m.add_function(wrap_pyfunction!(load_hcpe_feather, &m)?)?;
    m.add_function(wrap_pyfunction!(save_preprocessing_feather, &m)?)?;
    m.add_function(wrap_pyfunction!(load_preprocessing_feather, &m)?)?;
    m.add_function(wrap_pyfunction!(save_feather_file, &m)?)?;
    m.add_function(wrap_pyfunction!(load_feather_file, &m)?)?;
    m.add_function(wrap_pyfunction!(compress_sparse_array_rust, &m)?)?;
    m.add_function(wrap_pyfunction!(expand_sparse_array_rust, &m)?)?;
    m.add_function(wrap_pyfunction!(add_sparse_arrays_rust, &m)?)?;
    m.add_function(wrap_pyfunction!(add_sparse_arrays_dual_rust, &m)?)?;
    m.add_function(wrap_pyfunction!(split_feather_file, &m)?)?;
    m.add_function(wrap_pyfunction!(merge_feather_files, &m)?)?;

    Ok(m)
}
