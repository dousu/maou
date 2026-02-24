"""Data I/O interface for infrastructure layer.

This module provides a unified interface for Polars DataFrame I/O operations
that can be used by infrastructure layer components while maintaining
Clean Architecture dependency rules.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import polars as pl

logger: logging.Logger = logging.getLogger(__name__)


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
