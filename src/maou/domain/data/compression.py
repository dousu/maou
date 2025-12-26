"""Sparse array compression utilities for intermediate data storage.

This module provides sparse array helpers for move label count arrays
which tend to be mostly zeros，reducing storage requirements in DuckDB．

Rust-accelerated functions are used when available，falling back to
Python implementations．
"""

import logging
from typing import Tuple

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

# Try to import Rust-accelerated compression functions
try:
    from maou._rust.maou_io import (
        compress_sparse_array_rust,
        expand_sparse_array_rust,
    )

    _USE_RUST_COMPRESSION = True
    logger.debug(
        "Using Rust-accelerated sparse array compression"
    )
except ImportError:
    _USE_RUST_COMPRESSION = False
    logger.debug(
        "Rust compression not available，using Python implementation"
    )


class CompressionError(Exception):
    """Raised when compression/decompression operations fail."""

    pass


def compress_sparse_int_array(
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compress sparse integer array by storing only non-zero elements.

    Uses Rust-accelerated compression when available，
    falls back to Python implementation．

    Args:
        arr: 1D numpy array with sparse data (mostly zeros)

    Returns:
        Tuple of (indices, values) where indices are uint16 and values are int32
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D array, got {arr.ndim}D array"
        )

    if _USE_RUST_COMPRESSION:
        # Use Rust-accelerated compression (5-10x faster)
        indices, values = compress_sparse_array_rust(
            arr.tolist()
        )
        return np.array(indices, dtype=np.uint16), np.array(
            values, dtype=np.int32
        )
    else:
        # Python fallback
        nonzero_indices = np.nonzero(arr)[0].astype(np.uint16)
        nonzero_values = arr[nonzero_indices].astype(np.int32)
        return nonzero_indices, nonzero_values


def decompress_sparse_int_array(
    indices: np.ndarray, values: np.ndarray, size: int
) -> np.ndarray:
    """Decompress sparse integer array back to full array.

    Uses Rust-accelerated decompression when available，
    falls back to Python implementation．

    Args:
        indices: Non-zero element positions (uint16)
        values: Corresponding values (int32)
        size: Target array size

    Returns:
        Dense numpy array with zeros and filled values
    """
    if _USE_RUST_COMPRESSION:
        # Use Rust-accelerated decompression (5-10x faster)
        dense = expand_sparse_array_rust(
            indices.tolist(), values.tolist(), size
        )
        return np.array(dense, dtype=np.int32)
    else:
        # Python fallback
        result = np.zeros(size, dtype=np.int32)
        result[indices] = values
        return result
