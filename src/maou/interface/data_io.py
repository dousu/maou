"""Data I/O interface for infrastructure layer.

This module provides a unified interface for Polars DataFrame I/O operations
that can be used by infrastructure layer components while maintaining
Clean Architecture dependency rules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from maou.domain.data.columnar_batch import (
    ColumnarBatch as ColumnarBatch,  # noqa: F401
    convert_preprocessing_df_to_columnar as convert_preprocessing_df_to_columnar,  # noqa: F401
    convert_stage1_df_to_columnar as convert_stage1_df_to_columnar,  # noqa: F401
    convert_stage2_df_to_columnar as convert_stage2_df_to_columnar,  # noqa: F401
)

if TYPE_CHECKING:
    import polars as pl

logger: logging.Logger = logging.getLogger(__name__)


# --- rust_io re-exports (lazy: Rust backend may not be built) ---


def load_hcpe_df(path: Path) -> "pl.DataFrame":
    """Load HCPE DataFrame from Arrow IPC file."""
    import maou.domain.data.rust_io as _rust_io

    return _rust_io.load_hcpe_df(path)


def load_preprocessing_df(path: Path) -> "pl.DataFrame":
    """Load preprocessing DataFrame from Arrow IPC file."""
    import maou.domain.data.rust_io as _rust_io

    return _rust_io.load_preprocessing_df(path)


def load_stage1_df(path: Path) -> "pl.DataFrame":
    """Load stage1 DataFrame from Arrow IPC file."""
    import maou.domain.data.rust_io as _rust_io

    return _rust_io.load_stage1_df(path)


def load_stage2_df(path: Path) -> "pl.DataFrame":
    """Load stage2 DataFrame from Arrow IPC file."""
    import maou.domain.data.rust_io as _rust_io

    return _rust_io.load_stage2_df(path)


def save_hcpe_df(df: "pl.DataFrame", path: Path) -> None:
    """Save HCPE DataFrame to Arrow IPC file."""
    import maou.domain.data.rust_io as _rust_io

    _rust_io.save_hcpe_df(df, path)


def save_preprocessing_df(
    df: "pl.DataFrame", path: Path
) -> None:
    """Save preprocessing DataFrame to Arrow IPC file."""
    import maou.domain.data.rust_io as _rust_io

    _rust_io.save_preprocessing_df(df, path)


# --- dataframe_io re-exports (lazy: depends on optional polars) ---


def save_hcpe_df_to_bytes(df: "pl.DataFrame") -> bytes:
    """Save HCPE DataFrame to bytes (Arrow IPC format)."""
    import maou.domain.data.dataframe_io as _dataframe_io

    return _dataframe_io.save_hcpe_df_to_bytes(df)


def save_preprocessing_df_to_bytes(df: "pl.DataFrame") -> bytes:
    """Save preprocessing DataFrame to bytes (Arrow IPC format)."""
    import maou.domain.data.dataframe_io as _dataframe_io

    return _dataframe_io.save_preprocessing_df_to_bytes(df)


def load_df_from_bytes(
    data: bytes,
    *,
    array_type: Literal[
        "hcpe", "preprocessing", "stage1", "stage2"
    ],
) -> "pl.DataFrame":
    """Load Polars DataFrame from bytes (Arrow IPC format)．

    Args:
        data: Compressed Arrow IPC stream bytes
        array_type: Type of data ("hcpe", "preprocessing", "stage1", "stage2")

    Returns:
        pl.DataFrame: Deserialized DataFrame

    Raises:
        ValueError: If array_type is not supported

    Example:
        >>> # Download bytes_data from S3/GCS
        >>> df = load_df_from_bytes(bytes_data, array_type="hcpe")
    """
    import maou.domain.data.dataframe_io as _dataframe_io

    return _dataframe_io.load_df_from_bytes(
        data, array_type=array_type
    )


def save_df_to_bytes(
    df: "pl.DataFrame",
    *,
    array_type: Literal[
        "hcpe", "preprocessing", "stage1", "stage2"
    ],
) -> bytes:
    """Save Polars DataFrame to bytes (Arrow IPC format)．

    Args:
        df: Polars DataFrame
        array_type: Type of data ("hcpe", "preprocessing", "stage1", "stage2")

    Returns:
        bytes: Compressed Arrow IPC stream

    Raises:
        ValueError: If array_type is not supported

    Example:
        >>> df = pl.read_ipc("hcpe_data.feather")
        >>> bytes_data = save_df_to_bytes(df, array_type="hcpe")
        >>> # Upload bytes_data to S3/GCS
    """
    import maou.domain.data.dataframe_io as _dataframe_io

    return _dataframe_io.save_df_to_bytes(
        df, array_type=array_type
    )
