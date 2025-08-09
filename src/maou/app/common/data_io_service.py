"""Data I/O service for application layer.

This module provides a service layer for data I/O operations,
bridging between the interface and domain layers while maintaining
Clean Architecture dependency rules.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from maou.domain.data.io import (
    DataIOArrayTypeError,
    DataIOError,
    load_hcpe_array,
    load_hcpe_array_from_buffer,
    load_preprocessing_array,
    load_preprocessing_array_from_buffer,
    save_hcpe_array,
    save_hcpe_array_to_buffer,
    save_preprocessing_array,
    save_preprocessing_array_to_buffer,
)

logger: logging.Logger = logging.getLogger(__name__)


class DataIOService:
    """Service class for data I/O operations.

    Provides a unified interface for loading and saving numpy arrays
    with automatic schema detection and validation.
    """

    @staticmethod
    def load_array(
        file_path: Union[str, Path],
        array_type: Literal["hcpe", "preprocessing"],
        *,
        mmap_mode: Optional[
            Literal["r", "r+", "w+", "c"]
        ] = None,
        bit_pack: bool = True,
    ) -> np.ndarray:
        """Load numpy array with automatic schema handling.

        Args:
            file_path: Path to numpy file (.npy)
            array_type: Type of array to load ("hcpe", "preprocessing")
            mmap_mode: Memory mapping mode for .npy files
            bit_pack: Whether to use bit packing compression for binary fields

        Returns:
            numpy.ndarray: Loaded array

        Raises:
            DataIOError: If loading fails
        """
        try:
            file_path = Path(file_path)

            if array_type == "hcpe":
                return load_hcpe_array(
                    file_path, mmap_mode=mmap_mode
                )
            elif array_type == "preprocessing":
                return load_preprocessing_array(
                    file_path,
                    mmap_mode=mmap_mode,
                    bit_pack=bit_pack,
                )
            else:
                logger.error(
                    f"Unknown array type '{array_type}', "
                    f"Failed to load array from {file_path}"
                )
                raise DataIOArrayTypeError(
                    f"Unknown array type '{array_type}'"
                )

        except Exception as e:
            logger.error(
                f"Failed to load array from {file_path}: {e}"
            )
            raise DataIOError(
                f"Failed to load array from {file_path}: {e}"
            ) from e

    @staticmethod
    def save_array(
        array: np.ndarray,
        file_path: Union[str, Path],
        array_type: Literal["hcpe", "preprocessing"],
        *,
        bit_pack: bool = True,
    ) -> None:
        """Save numpy array with automatic schema handling.

        Args:
            array: Array to save
            file_path: Output file path
            array_type: Type of array ("hcpe", "preprocessing")
            bit_pack: Whether to use bit-packing before saving

        Raises:
            DataIOError: If saving fails
        """
        try:
            file_path = Path(file_path)

            if array_type == "hcpe":
                save_hcpe_array(
                    array,
                    file_path,
                )
            elif array_type == "preprocessing":
                save_preprocessing_array(
                    array,
                    file_path,
                    bit_pack=bit_pack,
                )
            else:
                logger.error(
                    f"Unknown array type '{array_type}', "
                    f"Failed to save array to {file_path}"
                )
                raise DataIOArrayTypeError(
                    f"Unknown array type '{array_type}'"
                )

        except Exception as e:
            logger.error(
                f"Failed to save array to {file_path}: {e}"
            )
            raise DataIOError(
                f"Failed to save array to {file_path}: {e}"
            ) from e

    @staticmethod
    def load_array_from_bytes(
        data: bytes,
        *,
        array_type: Literal["hcpe", "preprocessing"],
        mmap_mode: Optional[
            Literal["r", "r+", "w+", "c"]
        ] = None,
        bit_pack: bool = True,
    ) -> np.ndarray:
        """Load numpy array from bytes data with automatic schema handling.

        Args:
            data: Bytes containing the numpy array
            array_type: Type of array to load ("hcpe", "preprocessing")
            mmap_mode: Memory mapping mode for .npy files
            bit_pack: Whether to use bit packing compression for binary fields

        Returns:
            numpy.ndarray: Loaded array

        Raises:
            DataIOError: If loading fails
        """
        buffer = BytesIO()
        buffer.write(data)
        buffer.seek(0)
        try:
            if array_type == "hcpe":
                return load_hcpe_array_from_buffer(
                    buffer, mmap_mode=mmap_mode
                )
            elif array_type == "preprocessing":
                return load_preprocessing_array_from_buffer(
                    buffer,
                    mmap_mode=mmap_mode,
                    bit_pack=bit_pack,
                )
            else:
                logger.error(
                    f"Unknown array type '{array_type}'"
                )
                raise DataIOArrayTypeError(
                    f"Unknown array type '{array_type}'"
                )

        except Exception as e:
            logger.error(
                f"Failed to load array from buffer: {e}"
            )
            raise DataIOError(
                f"Failed to load array from buffer: {e}"
            ) from e

    @staticmethod
    def save_array_to_bytes(
        array: np.ndarray,
        array_type: Literal["hcpe", "preprocessing"],
        bit_pack: bool = True,
    ) -> bytes:
        """Save numpy array with automatic schema handling.

        Args:
            array: Array to save
            array_type: Type of array ("hcpe", "preprocessing")
            bit_pack: Whether to use bit-packing before saving

        Returns:
            bytes: Bytes containing the numpy array

        Raises:
            DataIOError: If saving fails
        """
        try:
            if array_type == "hcpe":
                buffer = save_hcpe_array_to_buffer(
                    array,
                )
            elif array_type == "preprocessing":
                buffer = save_preprocessing_array_to_buffer(
                    array,
                    bit_pack=bit_pack,
                )
            else:
                logger.error(
                    f"Unknown array type '{array_type}'"
                )
                raise DataIOArrayTypeError(
                    f"Unknown array type '{array_type}'"
                )

        except Exception as e:
            logger.error(f"Failed to save array to buffer: {e}")
            raise DataIOError(
                f"Failed to save array to buffer: {e}"
            ) from e
        buffer.seek(0)
        return buffer.getvalue()
