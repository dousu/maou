"""Sparse array compression utilities for intermediate data storage.

This module provides sparse array helpers for move label count arrays
which tend to be mostly zeros, reducing storage requirements in SQLite.
"""

import logging
from typing import Tuple

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


class CompressionError(Exception):
    """Raised when compression/decompression operations fail."""

    pass


def compress_sparse_int_array(
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compress sparse integer array by storing only non-zero elements."""

    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D array, got {arr.ndim}D array"
        )

    nonzero_indices = np.nonzero(arr)[0].astype(np.uint16)
    nonzero_values = arr[nonzero_indices].astype(np.int32)
    return nonzero_indices, nonzero_values


def decompress_sparse_int_array(
    indices: np.ndarray, values: np.ndarray, size: int
) -> np.ndarray:
    """Decompress sparse integer array back to full array."""

    result = np.zeros(size, dtype=np.int32)
    result[indices] = values
    return result
