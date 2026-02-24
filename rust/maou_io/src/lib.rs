//! Maou I/O - High-performance data I/O library for Shogi AI
//!
//! This crate provides pure Rust implementations for:
//! - Arrow IPC file I/O with LZ4 compression
//! - Schema definitions for HCPE and preprocessing data
//! - Sparse array compression utilities

pub mod arrow_io;
pub mod error;
pub mod schema;
pub mod sparse_array;

// Re-export commonly used types
pub use error::MaouIOError;
