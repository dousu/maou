"""Data I/O service for application layer.

This module provides a service layer for data I/O operations,
bridging between the interface and domain layers while maintaining
Clean Architecture dependency rules.

Uses Polars DataFrames (.feather) for efficient data processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

from maou.domain.data.array_io import (
    DataIOArrayTypeError,
    DataIOError,
)

if TYPE_CHECKING:
    import polars as pl

logger: logging.Logger = logging.getLogger(__name__)


class DataIOService:
    """Service class for data I/O operations.

    Provides a unified interface for loading and saving Polars DataFrames
    in efficient .feather format with the following benefits:
    - 30x storage reduction with LZ4 compression
    - 3-8x faster data loading
    - Zero-copy conversions with Arrow IPC format
    - Direct integration with PyTorch via polars_tensor module
    """

    # ========================================================================
    # DataFrame-based I/O methods
    # ========================================================================

    @staticmethod
    def load_dataframe(
        file_path: Union[str, Path],
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ],
    ) -> pl.DataFrame:
        """Load Polars DataFrame from .feather file．

        Modern replacement for load_array() with better performance．
        Uses Rust-backed Arrow IPC I/O for HCPE/preprocessing，
        Polars standard I/O for stage1/stage2．

        Args:
            file_path: Path to .feather file
            array_type: Type of data ("hcpe", "preprocessing", "stage1", "stage2")

        Returns:
            pl.DataFrame: Loaded DataFrame with appropriate schema

        Raises:
            DataIOError: If loading fails
            ImportError: If Rust backend is not available (HCPE/preprocessing only)

        Example:
            >>> df = DataIOService.load_dataframe("data.feather", "hcpe")
            >>> print(len(df))  # Fast DataFrame operations
        """
        try:
            from maou.domain.data.rust_io import (
                load_hcpe_df,
                load_preprocessing_df,
                load_stage1_df,
                load_stage2_df,
            )

            file_path = Path(file_path)

            if array_type == "hcpe":
                return load_hcpe_df(file_path)
            elif array_type == "preprocessing":
                return load_preprocessing_df(file_path)
            elif array_type == "stage1":
                return load_stage1_df(file_path)
            elif array_type == "stage2":
                return load_stage2_df(file_path)
            else:
                logger.error(
                    f"Unknown array type '{array_type}'"
                )
                raise DataIOArrayTypeError(
                    f"Unknown array type '{array_type}'"
                )

        except ImportError as e:
            logger.error(
                f"Rust backend not available for {file_path}: {e}"
            )
            raise DataIOError(
                f"Rust backend required for DataFrame I/O: {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"Failed to load DataFrame from {file_path}: {e}"
            )
            raise DataIOError(
                f"Failed to load DataFrame from {file_path}: {e}"
            ) from e

    @staticmethod
    def save_dataframe(
        df: pl.DataFrame,
        file_path: Union[str, Path],
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ],
    ) -> None:
        """Save Polars DataFrame to .feather file．

        Modern replacement for save_array() with better compression．
        Uses Rust-backed Arrow IPC I/O with LZ4 compression for HCPE/preprocessing，
        Polars standard I/O for stage1/stage2．

        Args:
            df: DataFrame to save
            file_path: Output .feather file path
            array_type: Type of data ("hcpe", "preprocessing", "stage1", "stage2")

        Raises:
            DataIOError: If saving fails
            ImportError: If Rust backend is not available (HCPE/preprocessing only)

        Example:
            >>> DataIOService.save_dataframe(df, "output.feather", "hcpe")
        """
        try:
            from maou.domain.data.rust_io import (
                save_hcpe_df,
                save_preprocessing_df,
                save_stage1_df,
                save_stage2_df,
            )

            file_path = Path(file_path)

            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if array_type == "hcpe":
                save_hcpe_df(df, file_path)
            elif array_type == "preprocessing":
                save_preprocessing_df(df, file_path)
            elif array_type == "stage1":
                save_stage1_df(df, file_path)
            elif array_type == "stage2":
                save_stage2_df(df, file_path)
            else:
                logger.error(
                    f"Unknown array type '{array_type}'"
                )
                raise DataIOArrayTypeError(
                    f"Unknown array type '{array_type}'"
                )

        except ImportError as e:
            logger.error(
                f"Rust backend not available for {file_path}: {e}"
            )
            raise DataIOError(
                f"Rust backend required for DataFrame I/O: {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"Failed to save DataFrame to {file_path}: {e}"
            )
            raise DataIOError(
                f"Failed to save DataFrame to {file_path}: {e}"
            ) from e

    @staticmethod
    def load_dataframe_from_bytes(
        data: bytes,
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ],
    ) -> pl.DataFrame:
        """Load Polars DataFrame from bytes (Arrow IPC format)．

        Modern replacement for load_array_from_bytes() for cloud storage．
        Uses LZ4-compressed Arrow IPC format for efficient network transfer．

        Args:
            data: Bytes containing Arrow IPC stream
            array_type: Type of data ("hcpe", "preprocessing", "stage1", "stage2")

        Returns:
            pl.DataFrame: Loaded DataFrame

        Raises:
            DataIOError: If loading fails

        Example:
            >>> bytes_data = cloud_storage.download("data.feather")
            >>> df = DataIOService.load_dataframe_from_bytes(
            ...     bytes_data, "hcpe"
            ... )
        """
        try:
            from maou.domain.data.dataframe_io import (
                load_df_from_bytes,
            )

            return load_df_from_bytes(
                data, array_type=array_type
            )

        except Exception as e:
            logger.error(
                f"Failed to load DataFrame from bytes: {e}"
            )
            raise DataIOError(
                f"Failed to load DataFrame from bytes: {e}"
            ) from e

    @staticmethod
    def save_dataframe_to_bytes(
        df: pl.DataFrame,
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ],
    ) -> bytes:
        """Save Polars DataFrame to bytes (Arrow IPC format)．

        Modern replacement for save_array_to_bytes() for cloud storage．
        Provides 30x compression compared to numpy format．

        Args:
            df: DataFrame to save
            array_type: Type of data ("hcpe", "preprocessing", "stage1", "stage2")

        Returns:
            bytes: LZ4-compressed Arrow IPC stream

        Raises:
            DataIOError: If saving fails

        Example:
            >>> bytes_data = DataIOService.save_dataframe_to_bytes(
            ...     df, "hcpe"
            ... )
            >>> cloud_storage.upload("data.feather", bytes_data)
        """
        try:
            from maou.domain.data.dataframe_io import (
                save_df_to_bytes,
            )

            return save_df_to_bytes(df, array_type=array_type)

        except Exception as e:
            logger.error(
                f"Failed to save DataFrame to bytes: {e}"
            )
            raise DataIOError(
                f"Failed to save DataFrame to bytes: {e}"
            ) from e
