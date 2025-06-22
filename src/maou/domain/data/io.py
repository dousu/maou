"""Centralized numpy I/O operations for Maou project.

This module provides standardized save and load functions for numpy arrays
with proper error handling, validation, and compression support.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np

from .schema import (
    SchemaValidationError,
    validate_hcpe_array,
    validate_preprocessing_array,
)


class DataIOError(Exception):
    """Raised when data I/O operations fail."""

    pass


logger: logging.Logger = logging.getLogger(__name__)


def save_hcpe_array(
    array: np.ndarray,
    file_path: Union[str, Path],
    *,
    validate: bool = True,
    compress: bool = False,
    compression_level: int = 6,
) -> None:
    """Save HCPE array to file with validation and optional compression.

    Args:
        array: HCPE numpy array to save
        file_path: Output file path
        validate: Whether to validate schema before saving
        compress: Whether to use compression (saves as .npz)
        compression_level: Compression level (0-9, higher is more compressed)

    Raises:
        DataIOError: If save operation fails
        SchemaValidationError: If validation fails
    """
    try:
        file_path = Path(file_path)

        if validate:
            validate_hcpe_array(array)
            logger.debug(f"HCPE array validation passed for {file_path}")

        if compress:
            # Change extension to .npz for compressed format
            if file_path.suffix == ".npy":
                file_path = file_path.with_suffix(".npz")

            np.savez_compressed(
                file_path, data=array, compression_level=compression_level
            )
            logger.debug(f"Saved compressed HCPE array to {file_path}")
        else:
            # Ensure .npy extension for uncompressed
            if file_path.suffix != ".npy":
                file_path = file_path.with_suffix(".npy")

            np.save(file_path, array)
            logger.debug(f"Saved HCPE array to {file_path}")

    except Exception as e:
        raise DataIOError(f"Failed to save HCPE array to {file_path}: {e}") from e


def load_hcpe_array(
    file_path: Union[str, Path],
    *,
    validate: bool = True,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
) -> np.ndarray:
    """Load HCPE array from file with validation.

    Args:
        file_path: Input file path (.npy or .npz)
        validate: Whether to validate schema after loading
        mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')

    Returns:
        numpy.ndarray: Loaded HCPE array

    Raises:
        DataIOError: If load operation fails
        SchemaValidationError: If validation fails
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataIOError(f"File not found: {file_path}")

        if file_path.suffix == ".npz":
            # Load compressed format
            with np.load(file_path) as data:
                if "data" not in data:
                    raise DataIOError(
                        f"Expected 'data' key in compressed file {file_path}"
                    )
                array = data["data"]
        else:
            # Load uncompressed format
            array = np.load(file_path, mmap_mode=mmap_mode)

        if validate:
            validate_hcpe_array(array)
            logger.debug(f"HCPE array validation passed for {file_path}")

        logger.debug(f"Loaded HCPE array from {file_path}, shape: {array.shape}")
        return array

    except Exception as e:
        raise DataIOError(f"Failed to load HCPE array from {file_path}: {e}") from e


def save_preprocessing_array(
    array: np.ndarray,
    file_path: Union[str, Path],
    *,
    validate: bool = True,
    compress: bool = False,
    compression_level: int = 6,
) -> None:
    """Save preprocessing array to file with validation and optional compression.

    Args:
        array: Preprocessing numpy array to save
        file_path: Output file path
        validate: Whether to validate schema before saving
        compress: Whether to use compression (saves as .npz)
        compression_level: Compression level (0-9, higher is more compressed)

    Raises:
        DataIOError: If save operation fails
        SchemaValidationError: If validation fails
    """
    try:
        file_path = Path(file_path)

        if validate:
            validate_preprocessing_array(array)
            logger.debug(f"Preprocessing array validation passed for {file_path}")

        if compress:
            # Change extension to .npz for compressed format
            if file_path.suffix == ".npy":
                file_path = file_path.with_suffix(".npz")

            np.savez_compressed(
                file_path, data=array, compression_level=compression_level
            )
            logger.debug(f"Saved compressed preprocessing array to {file_path}")
        else:
            # Ensure .npy extension for uncompressed
            if file_path.suffix != ".npy":
                file_path = file_path.with_suffix(".npy")

            np.save(file_path, array)
            logger.debug(f"Saved preprocessing array to {file_path}")

    except Exception as e:
        raise DataIOError(
            f"Failed to save preprocessing array to {file_path}: {e}"
        ) from e


def load_preprocessing_array(
    file_path: Union[str, Path],
    *,
    validate: bool = True,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
) -> np.ndarray:
    """Load preprocessing array from file with validation.

    Args:
        file_path: Input file path (.npy or .npz)
        validate: Whether to validate schema after loading
        mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')

    Returns:
        numpy.ndarray: Loaded preprocessing array

    Raises:
        DataIOError: If load operation fails
        SchemaValidationError: If validation fails
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataIOError(f"File not found: {file_path}")

        if file_path.suffix == ".npz":
            # Load compressed format
            with np.load(file_path) as data:
                if "data" not in data:
                    raise DataIOError(
                        f"Expected 'data' key in compressed file {file_path}"
                    )
                array = data["data"]
        else:
            # Load uncompressed format
            array = np.load(file_path, mmap_mode=mmap_mode)

        if validate:
            validate_preprocessing_array(array)
            logger.debug(f"Preprocessing array validation passed for {file_path}")

        logger.debug(
            f"Loaded preprocessing array from {file_path}, shape: {array.shape}"
        )
        return array

    except Exception as e:
        raise DataIOError(
            f"Failed to load preprocessing array from {file_path}: {e}"
        ) from e


def save_array_with_metadata(
    array: np.ndarray,
    file_path: Union[str, Path],
    metadata: Dict[str, Any],
    *,
    validate: bool = True,
    compression_level: int = 6,
) -> None:
    """Save array with additional metadata to compressed format.

    Args:
        array: numpy array to save
        file_path: Output file path (will be saved as .npz)
        metadata: Additional metadata to store
        validate: Whether to validate schema
        compression_level: Compression level (0-9)

    Raises:
        DataIOError: If save operation fails
        SchemaValidationError: If validation fails
    """
    try:
        file_path = Path(file_path).with_suffix(".npz")

        # Determine array type and validate if requested
        if validate:
            try:
                validate_hcpe_array(array)
                array_type = "hcpe"
            except SchemaValidationError:
                try:
                    validate_preprocessing_array(array)
                    array_type = "preprocessing"
                except SchemaValidationError:
                    raise SchemaValidationError("Array does not match any known schema")
        else:
            array_type = "unknown"

        # Combine data and metadata
        save_data = {
            "data": array,
            "metadata": metadata,
            "array_type": array_type,
            "dtype_info": str(array.dtype),
        }

        np.savez_compressed(file_path, **save_data)  # type: ignore[arg-type]
        logger.debug(f"Saved array with metadata to {file_path}")

    except Exception as e:
        raise DataIOError(
            f"Failed to save array with metadata to {file_path}: {e}"
        ) from e


def load_array_with_metadata(
    file_path: Union[str, Path], *, validate: bool = True
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Load array with metadata from compressed format.

    Args:
        file_path: Input file path (.npz)
        validate: Whether to validate schema

    Returns:
        Tuple of (array, metadata)

    Raises:
        DataIOError: If load operation fails
        SchemaValidationError: If validation fails
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataIOError(f"File not found: {file_path}")

        with np.load(file_path, allow_pickle=True) as data:
            if "data" not in data:
                raise DataIOError(f"Expected 'data' key in file {file_path}")

            array = data["data"]
            metadata = data.get("metadata", {})
            array_type = data.get("array_type", "unknown")

            # Convert metadata back to dict if it was saved as numpy array
            if isinstance(metadata, np.ndarray):
                metadata = metadata.item()

        if validate and array_type != "unknown":
            if array_type == "hcpe":
                validate_hcpe_array(array)
            elif array_type == "preprocessing":
                validate_preprocessing_array(array)

            logger.debug(f"Array validation passed for {file_path}")

        logger.debug(
            f"Loaded array with metadata from {file_path}, shape: {array.shape}"
        )
        return array, metadata

    except Exception as e:
        raise DataIOError(
            f"Failed to load array with metadata from {file_path}: {e}"
        ) from e


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a numpy data file without loading the full array.

    Args:
        file_path: Input file path

    Returns:
        Dict containing file information

    Raises:
        DataIOError: If file cannot be read
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataIOError(f"File not found: {file_path}")

        info = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "format": "compressed" if file_path.suffix == ".npz" else "uncompressed",
        }

        if file_path.suffix == ".npz":
            with np.load(file_path) as data:
                info["keys"] = list(data.keys())
                if "data" in data:
                    array = data["data"]
                    info["shape"] = array.shape
                    info["dtype"] = str(array.dtype)
                    info["size"] = array.size
                if "array_type" in data:
                    info["array_type"] = str(data["array_type"])
                if "metadata" in data:
                    info["has_metadata"] = True
        else:
            # For .npy files, we need to load the header only
            with open(file_path, "rb") as f:
                np.lib.format.read_magic(f)  # Read magic but don't store
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
                info["shape"] = shape
                info["dtype"] = str(dtype)
                info["size"] = np.prod(shape) if shape else 0
                info["fortran_order"] = fortran_order

        return info

    except Exception as e:
        raise DataIOError(f"Failed to get file info for {file_path}: {e}") from e
