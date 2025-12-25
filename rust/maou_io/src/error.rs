use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MaouIOError {
    #[error("Arrow error: {0}")]
    ArrowError(#[from] arrow::error::ArrowError),

    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Schema validation error: {0}")]
    SchemaError(String),

    #[error("Compression error: {0}")]
    CompressionError(String),
}

impl From<MaouIOError> for PyErr {
    fn from(err: MaouIOError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}
