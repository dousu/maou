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


# --- rust_io re-exports ---


def load_hcpe_df(path: Path) -> "pl.DataFrame":
    """Load HCPE DataFrame from Arrow IPC file."""
    from maou.domain.data.rust_io import load_hcpe_df

    return load_hcpe_df(path)


def load_preprocessing_df(path: Path) -> "pl.DataFrame":
    """Load preprocessing DataFrame from Arrow IPC file."""
    from maou.domain.data.rust_io import load_preprocessing_df

    return load_preprocessing_df(path)


def load_stage1_df(path: Path) -> "pl.DataFrame":
    """Load stage1 DataFrame from Arrow IPC file."""
    from maou.domain.data.rust_io import load_stage1_df

    return load_stage1_df(path)


def load_stage2_df(path: Path) -> "pl.DataFrame":
    """Load stage2 DataFrame from Arrow IPC file."""
    from maou.domain.data.rust_io import load_stage2_df

    return load_stage2_df(path)


def save_hcpe_df(df: "pl.DataFrame", path: Path) -> None:
    """Save HCPE DataFrame to Arrow IPC file."""
    from maou.domain.data.rust_io import save_hcpe_df

    save_hcpe_df(df, path)


def save_preprocessing_df(
    df: "pl.DataFrame", path: Path
) -> None:
    """Save preprocessing DataFrame to Arrow IPC file."""
    from maou.domain.data.rust_io import save_preprocessing_df

    save_preprocessing_df(df, path)


# --- dataframe_io re-exports ---


def save_hcpe_df_to_bytes(df: "pl.DataFrame") -> bytes:
    """Save HCPE DataFrame to bytes (Arrow IPC format)."""
    from maou.domain.data.dataframe_io import (
        save_hcpe_df_to_bytes,
    )

    return save_hcpe_df_to_bytes(df)


def save_preprocessing_df_to_bytes(df: "pl.DataFrame") -> bytes:
    """Save preprocessing DataFrame to bytes (Arrow IPC format)."""
    from maou.domain.data.dataframe_io import (
        save_preprocessing_df_to_bytes,
    )

    return save_preprocessing_df_to_bytes(df)


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
    from maou.domain.data.dataframe_io import load_df_from_bytes

    return load_df_from_bytes(data, array_type=array_type)


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
    from maou.domain.data.dataframe_io import save_df_to_bytes

    return save_df_to_bytes(df, array_type=array_type)
