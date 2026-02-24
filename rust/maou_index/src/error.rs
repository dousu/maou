//! エラー型定義．

use pyo3::exceptions::PyException;
use pyo3::PyErr;
use thiserror::Error;

/// インデックス操作のエラー型．
#[derive(Error, Debug)]
pub enum IndexError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid data format: {0}")]
    InvalidFormat(String),

    #[error("Record not found: {0}")]
    NotFound(String),

    #[error("Index build failed: {0}")]
    BuildFailed(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// PyO3用のエラー変換．
impl From<IndexError> for PyErr {
    fn from(err: IndexError) -> PyErr {
        PyException::new_err(err.to_string())
    }
}
