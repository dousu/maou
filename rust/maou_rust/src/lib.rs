//! Maou Rust - Python bindings for Maou's Rust components
//!
//! This crate provides PyO3 bindings for all Rust functionality in the Maou project.
//! Each Rust crate has a corresponding Python submodule:
//!
//! - `maou._rust.maou_io` - Arrow IPC I/O and sparse array operations
//! - `maou._rust.maou_index` - Search index for data visualization
//! - `maou._rust.maou_shogi` - Shogi board, move generation, and feature extraction
//! - `maou._rust.maou_search` - MCTS-based single-position search engine
//! - `maou._rust.maou_convert` - 棋譜 (CSA/KIF) → HCPE 一括変換パイプライン
//! - `maou._rust.maou_usi` - USI 対局エージェント (標準入出力ループ)

use pyo3::prelude::*;

mod maou_convert;
mod maou_index;
mod maou_io;
mod maou_search;
mod maou_shogi;
mod maou_usi;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register submodules
    let py = m.py();

    m.add_submodule(&maou_io::create_module(py)?)?;
    m.add_submodule(&maou_index::create_module(py)?)?;
    m.add_submodule(&maou_shogi::create_module(py)?)?;
    m.add_submodule(&maou_search::create_module(py)?)?;
    m.add_submodule(&maou_convert::create_module(py)?)?;
    m.add_submodule(&maou_usi::create_module(py)?)?;

    // Register submodules in sys.modules for proper importing
    py.import("sys")?
        .getattr("modules")?
        .set_item("maou._rust.maou_io", m.getattr("maou_io")?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("maou._rust.maou_index", m.getattr("maou_index")?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("maou._rust.maou_shogi", m.getattr("maou_shogi")?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("maou._rust.maou_search", m.getattr("maou_search")?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("maou._rust.maou_convert", m.getattr("maou_convert")?)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("maou._rust.maou_usi", m.getattr("maou_usi")?)?;

    Ok(())
}
