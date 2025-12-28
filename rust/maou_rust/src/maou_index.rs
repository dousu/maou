//! maou_index submodule - Search index for data visualization

use pyo3::prelude::*;

/// Create maou_index submodule
pub fn create_module(py: Python) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new_bound(py, "maou_index")?;

    // Add SearchIndex class from maou_index crate
    m.add_class::<::maou_index::SearchIndex>()?;

    Ok(m)
}
