"""DataFrame byte serialization for cloud storage．

このモジュールは，Polars DataFrameをバイト列に変換してクラウドストレージに
アップロード/ダウンロードするための関数を提供する．Arrow IPC Stream形式を
使用し，LZ4圧縮により高速かつ効率的なシリアライゼーションを実現する．
"""

import io
import logging
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)


def save_hcpe_df_to_bytes(df: pl.DataFrame) -> bytes:
    """Serialize HCPE DataFrame to bytes using Arrow IPC Stream format．

    Args:
        df: HCPE Polars DataFrame

    Returns:
        bytes: Compressed Arrow IPC stream

    Example:
        >>> df = pl.read_ipc("hcpe_data.feather")
        >>> bytes_data = save_hcpe_df_to_bytes(df)
        >>> # Upload bytes_data to S3/GCS
    """
    buffer = io.BytesIO()
    df.write_ipc_stream(buffer, compression="lz4")
    return buffer.getvalue()


def load_hcpe_df_from_bytes(data: bytes) -> pl.DataFrame:
    """Deserialize HCPE DataFrame from bytes．

    Args:
        data: Compressed Arrow IPC stream bytes

    Returns:
        pl.DataFrame: HCPE DataFrame

    Example:
        >>> # Download bytes_data from S3/GCS
        >>> df = load_hcpe_df_from_bytes(bytes_data)
    """
    buffer = io.BytesIO(data)
    return pl.read_ipc_stream(buffer)


def save_preprocessing_df_to_bytes(df: pl.DataFrame) -> bytes:
    """Serialize preprocessing DataFrame to bytes using Arrow IPC Stream format．

    Args:
        df: Preprocessing Polars DataFrame

    Returns:
        bytes: Compressed Arrow IPC stream

    Example:
        >>> df = pl.read_ipc("preprocessing_data.feather")
        >>> bytes_data = save_preprocessing_df_to_bytes(df)
        >>> # Upload bytes_data to S3/GCS
    """
    buffer = io.BytesIO()
    df.write_ipc_stream(buffer, compression="lz4")
    return buffer.getvalue()


def load_preprocessing_df_from_bytes(
    data: bytes,
) -> pl.DataFrame:
    """Deserialize preprocessing DataFrame from bytes．

    Args:
        data: Compressed Arrow IPC stream bytes

    Returns:
        pl.DataFrame: Preprocessing DataFrame

    Example:
        >>> # Download bytes_data from S3/GCS
        >>> df = load_preprocessing_df_from_bytes(bytes_data)
    """
    buffer = io.BytesIO(data)
    return pl.read_ipc_stream(buffer)


def save_df_to_bytes(
    df: pl.DataFrame,
    *,
    array_type: Literal["hcpe", "preprocessing"],
) -> bytes:
    """Generic DataFrame to bytes serialization with array_type dispatch．

    Args:
        df: Polars DataFrame
        array_type: Type of data ("hcpe" or "preprocessing")

    Returns:
        bytes: Compressed Arrow IPC stream

    Raises:
        ValueError: If array_type is not supported
    """
    if array_type == "hcpe":
        return save_hcpe_df_to_bytes(df)
    elif array_type == "preprocessing":
        return save_preprocessing_df_to_bytes(df)
    else:
        raise ValueError(
            f"Unsupported array_type: {array_type}"
        )


def load_df_from_bytes(
    data: bytes,
    *,
    array_type: Literal["hcpe", "preprocessing"],
) -> pl.DataFrame:
    """Generic bytes to DataFrame deserialization with array_type dispatch．

    Args:
        data: Compressed Arrow IPC stream bytes
        array_type: Type of data ("hcpe" or "preprocessing")

    Returns:
        pl.DataFrame: Deserialized DataFrame

    Raises:
        ValueError: If array_type is not supported
    """
    if array_type == "hcpe":
        return load_hcpe_df_from_bytes(data)
    elif array_type == "preprocessing":
        return load_preprocessing_df_from_bytes(data)
    else:
        raise ValueError(
            f"Unsupported array_type: {array_type}"
        )
