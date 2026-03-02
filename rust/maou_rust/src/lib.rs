//! Maou Rust - Python bindings for Maou's Rust components
//!
//! This crate provides PyO3 bindings for all Rust functionality in the Maou project.
//! Each Rust crate has a corresponding Python submodule:
//!
//! - `maou._rust.maou_io` - Arrow IPC I/O and sparse array operations
//! - `maou._rust.maou_index` - Search index for data visualization

use pyo3::prelude::*;

mod maou_io;
mod maou_index;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register submodules
    let py = m.py();

    m.add_submodule(&maou_io::create_module(py)?)?;
    m.add_submodule(&maou_index::create_module(py)?)?;

    // Register submodules in sys.modules for proper importing
    py.import("sys")?
        .getattr("modules")?
        .set_item("maou._rust.maou_io", m.getattr("maou_io")?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("maou._rust.maou_index", m.getattr("maou_index")?)?;

    Ok(())
}
