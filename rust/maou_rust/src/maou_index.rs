//! maou_index submodule - Search index for data visualization

use pyo3::prelude::*;

/// Create maou_index submodule
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "maou_index")?;

    // Add SearchIndex class from maou_index crate
    m.add_class::<::maou_index::SearchIndex>()?;

    // Add PathScanner class from maou_index crate
    m.add_class::<::maou_index::PathScanner>()?;

    Ok(m)
}
