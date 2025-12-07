"""Data I/O interface for infrastructure layer.

This module provides a unified interface for data I/O operations
that can be used by infrastructure layer components while maintaining
Clean Architecture dependency rules.
"""

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from maou.app.common.data_io_service import (
    DataIOService,
)

logger: logging.Logger = logging.getLogger(__name__)


def load_array(
    file_path: Union[str, Path],
    *,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
    preprocessing_mmap_mode: Optional[
        Literal["r", "r+", "w+", "c"]
    ] = "c",
    array_type: Literal[
        "hcpe", "preprocessing", "stage1", "stage2"
    ],
    bit_pack: bool = True,
) -> np.ndarray:
    """Load numpy array from file with automatic schema handling.

    This function provides a unified interface for loading numpy arrays
    from various file formats while maintaining compatibility with
    existing domain code.

    Args:
        file_path: Path to numpy file (.npy)
        mmap_mode: Memory mapping mode for .npy files ('r', 'r+', 'w+', 'c')
        preprocessing_mmap_mode: Default mmap mode for preprocessing arrays
        array_type: Expected array type ("hcpe", "preprocessing", "stage1", "stage2")

    Returns:
        numpy.ndarray: Loaded array

    Raises:
        DataIOError: If loading fails

    Example:
        >>> data = load_array("data.npy", mmap_mode="r", array_type="hcpe")
    """
    if array_type not in [
        "hcpe",
        "preprocessing",
        "stage1",
        "stage2",
    ]:
        raise ValueError(f"Unknown array type: {array_type}")

    return DataIOService.load_array(
        file_path,
        array_type=array_type,
        mmap_mode=mmap_mode,
        preprocessing_mmap_mode=preprocessing_mmap_mode,
        bit_pack=bit_pack,
    )


def save_array(
    array: np.ndarray,
    file_path: Union[str, Path],
    *,
    array_type: Literal["hcpe", "preprocessing"],
    bit_pack: bool = True,
) -> None:
    """Save numpy array to file with automatic schema handling.

    This function provides a unified interface for saving numpy arrays
    while maintaining compatibility with existing domain code.

    Args:
        array: Array to save
        file_path: Output file path
        array_type: Array type hint ("hcpe", "preprocessing")

    Raises:
        DataIOError: If saving fails

    Example:
        >>> # Basic usage (backward compatible)
        >>> save_array(data, "output.npy")
    """
    if array_type not in ["hcpe", "preprocessing"]:
        raise ValueError(f"Unknown array type: {array_type}")

    DataIOService.save_array(
        array,
        file_path,
        array_type=array_type,
        bit_pack=bit_pack,
    )


def load_array_from_bytes(
    data: bytes,
    *,
    array_type: Literal["hcpe", "preprocessing"],
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
    bit_pack: bool = True,
) -> np.ndarray:
    """Load numpy array from byte data with automatic schema handling.

    This function provides a unified interface for loading numpy arrays
    from bytes buffers while maintaining compatibility with existing domain code.

    Args:
        data: Bytes containing the numpy array
        array_type: Expected array type ("hcpe", "preprocessing")
        mmap_mode: Memory mapping mode for .npy files ('r', 'r+', 'w+', 'c')

    Returns:
        numpy.ndarray: Loaded array

    Raises:
        DataIOError: If loading fails
    """
    if array_type not in ["hcpe", "preprocessing"]:
        raise ValueError(f"Unknown array type: {array_type}")

    return DataIOService.load_array_from_bytes(
        data,
        array_type=array_type,
        mmap_mode=mmap_mode,
        bit_pack=bit_pack,
    )


def save_array_to_bytes(
    array: np.ndarray,
    *,
    array_type: Literal["hcpe", "preprocessing"],
    bit_pack: bool = True,
) -> bytes:
    """Save numpy array to bytes with automatic schema handling.

    This function provides a unified interface for saving numpy arrays
    to bytes while maintaining compatibility with existing domain code.

    Args:
        array: Array to save
        array_type: Array type hint ("hcpe", "preprocessing")

    Returns:
        bytes: Bytes containing the numpy array

    Raises:
        DataIOError: If saving fails
    """
    if array_type not in ["hcpe", "preprocessing"]:
        raise ValueError(f"Unknown array type: {array_type}")

    return DataIOService.save_array_to_bytes(
        array,
        array_type=array_type,
        bit_pack=bit_pack,
    )


def load_packed_array(
    file_path: Union[str, Path],
    *,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
    array_type: Literal["preprocessing"],
) -> np.ndarray:
    """Load numpy array from file with packed schema.

    Args:
        file_path: Path to numpy file (.npy)
        mmap_mode: Memory mapping mode for .npy files ('r', 'r+', 'w+', 'c')
        array_type: Expected array type ("preprocessing")

    Returns:
        numpy.ndarray: Loaded array

    Raises:
        DataIOError: If loading fails

    Example:
        >>> data = load_array("data.npy", mmap_mode="r", array_type="preprocessing")
    """
    if array_type not in ["preprocessing"]:
        raise ValueError(f"Unknown array type: {array_type}")

    return DataIOService.load_packed_array(
        file_path,
        array_type=array_type,
        mmap_mode=mmap_mode,
    )
