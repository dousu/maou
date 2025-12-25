use pyo3::prelude::*;
use arrow::array::RecordBatch;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};

mod arrow_io;
mod error;
mod schema;

#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("Maou I/O Rust backend initialized".to_string())
}

#[pyfunction]
fn save_hcpe_feather(_py: Python, batch: &Bound<'_, PyAny>, file_path: String) -> PyResult<()> {
    let batch = RecordBatch::from_pyarrow_bound(batch)?;
    arrow_io::save_feather(&batch, &file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    Ok(())
}

#[pyfunction]
fn load_hcpe_feather(py: Python, file_path: String) -> PyResult<PyObject> {
    let batch = arrow_io::load_feather(&file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    batch.to_pyarrow(py)
}

#[pyfunction]
fn save_preprocessing_feather(_py: Python, batch: &Bound<'_, PyAny>, file_path: String) -> PyResult<()> {
    let batch = RecordBatch::from_pyarrow_bound(batch)?;
    arrow_io::save_feather(&batch, &file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    Ok(())
}

#[pyfunction]
fn load_preprocessing_feather(py: Python, file_path: String) -> PyResult<PyObject> {
    let batch = arrow_io::load_feather(&file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    batch.to_pyarrow(py)
}

#[pymodule]
fn maou_io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(save_hcpe_feather, m)?)?;
    m.add_function(wrap_pyfunction!(load_hcpe_feather, m)?)?;
    m.add_function(wrap_pyfunction!(save_preprocessing_feather, m)?)?;
    m.add_function(wrap_pyfunction!(load_preprocessing_feather, m)?)?;
    Ok(())
}
