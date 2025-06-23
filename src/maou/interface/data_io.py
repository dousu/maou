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
    load_numpy_array,
    save_numpy_array,
)
from maou.app.common.bundling_service import BundlingService

logger: logging.Logger = logging.getLogger(__name__)


def load_array(
    file_path: Union[str, Path],
    *,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
    validate: bool = False,
    array_type: Literal["auto", "hcpe", "preprocessing"] = "auto",
) -> np.ndarray:
    """Load numpy array from file with automatic schema handling.

    This function provides a unified interface for loading numpy arrays
    from various file formats while maintaining compatibility with
    existing infrastructure code.

    Args:
        file_path: Path to numpy file (.npy or .npz)
        mmap_mode: Memory mapping mode for .npy files ('r', 'r+', 'w+', 'c')
        validate: Whether to validate schema after loading (default: False)
        array_type: Expected array type ("auto", "hcpe", "preprocessing")

    Returns:
        numpy.ndarray: Loaded array

    Raises:
        DataIOError: If loading fails
        SchemaValidationError: If validation fails and validate=True

    Example:
        >>> # Basic usage (backward compatible)
        >>> data = load_array("data.npy", mmap_mode="r")
        >>>
        >>> # With validation
        >>> data = load_array("hcpe_data.npy", validate=True, array_type="hcpe")
    """
    return DataIOService.load_array(
        file_path, array_type=array_type, validate=validate, mmap_mode=mmap_mode
    )


def save_array(
    array: np.ndarray,
    file_path: Union[str, Path],
    *,
    compress: bool = False,
    validate: bool = False,
    array_type: Literal["auto", "hcpe", "preprocessing"] = "auto",
) -> None:
    """Save numpy array to file with automatic schema handling.

    This function provides a unified interface for saving numpy arrays
    while maintaining compatibility with existing infrastructure code.

    Args:
        array: Array to save
        file_path: Output file path
        compress: Whether to use compression (saves as .npz)
        validate: Whether to validate schema before saving (default: False)
        array_type: Array type hint ("auto", "hcpe", "preprocessing")

    Raises:
        DataIOError: If saving fails
        SchemaValidationError: If validation fails and validate=True

    Example:
        >>> # Basic usage (backward compatible)
        >>> save_array(data, "output.npy")
        >>>
        >>> # With compression and validation
        >>> save_array(data, "output.npz", compress=True, validate=True)
    """
    DataIOService.save_array(
        array, file_path, array_type=array_type, validate=validate, compress=compress
    )


def get_array_info(file_path: Union[str, Path]) -> dict:
    """Get information about an array file.

    Args:
        file_path: Path to the file

    Returns:
        dict: File information including detected array type, shape, dtype

    Example:
        >>> info = get_array_info("data.npy")
        >>> print(f"Shape: {info['shape']}, Type: {info.get('detected_array_type')}")
    """
    return DataIOService.get_array_info(file_path)


# Backward compatibility aliases
def load_structured_array(
    file_path: Union[str, Path],
    *,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
) -> np.ndarray:
    """Load structured numpy array (backward compatibility alias).

    Args:
        file_path: Path to numpy file
        mmap_mode: Memory mapping mode

    Returns:
        numpy.ndarray: Loaded structured array
    """
    # For backward compatibility, try to load with auto-detection first
    try:
        return load_array(
            file_path, array_type="auto", mmap_mode=mmap_mode, validate=False
        )
    except Exception:
        # Fallback to the old behavior
        return load_numpy_array(file_path, mmap_mode=mmap_mode, validate=False)


def save_structured_array(array: np.ndarray, file_path: Union[str, Path]) -> None:
    """Save structured numpy array (backward compatibility alias).

    Args:
        array: Structured array to save
        file_path: Output file path
    """
    save_numpy_array(array, file_path, compress=False, validate=False)


def create_bundling_service(
    cache_dir: Union[str, Path],
    target_size_gb: float = 1.0,
    array_type: str = "hcpe",
) -> BundlingService:
    """Create bundling service instance.

    This function provides an interface layer abstraction for creating
    bundling services while maintaining Clean Architecture principles.

    Args:
        cache_dir: Directory for storing bundles and metadata
        target_size_gb: Target bundle size in GB
        array_type: Primary array type ("hcpe" or "preprocessing")

    Returns:
        BundlingService: Configured bundling service instance

    Example:
        >>> bundler = create_bundling_service("./cache", target_size_gb=1.5)
        >>> bundles = bundler.bundle_files(file_paths)
    """
    return BundlingService(
        cache_dir=Path(cache_dir),
        target_size_gb=target_size_gb,
        array_type=array_type,
    )
