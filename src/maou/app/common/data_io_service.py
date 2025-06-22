"""Data I/O service for application layer.

This module provides a service layer for data I/O operations,
bridging between the interface and domain layers while maintaining
Clean Architecture dependency rules.
"""

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from maou.domain.data.io import (
    DataIOError,
    get_file_info,
    load_hcpe_array,
    load_preprocessing_array,
    save_hcpe_array,
    save_preprocessing_array,
)
from maou.domain.data.schema import (
    SchemaValidationError,
    validate_hcpe_array,
    validate_preprocessing_array,
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
        *,
        array_type: Literal["auto", "hcpe", "preprocessing"] = "auto",
        validate: bool = False,
        mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
    ) -> np.ndarray:
        """Load numpy array with automatic schema detection.

        Args:
            file_path: Path to numpy file (.npy or .npz)
            array_type: Type of array to load ("auto", "hcpe", "preprocessing")
            validate: Whether to validate schema after loading
            mmap_mode: Memory mapping mode for .npy files

        Returns:
            numpy.ndarray: Loaded array

        Raises:
            DataIOError: If loading fails
            SchemaValidationError: If validation fails
        """
        try:
            file_path = Path(file_path)

            if array_type == "auto":
                # Try to determine array type from file info or content
                detected_type = DataIOService._detect_array_type(file_path)
                if detected_type in ("hcpe", "preprocessing"):
                    array_type = detected_type  # type: ignore[assignment]
                else:
                    array_type = "unknown"  # type: ignore[assignment]

            if array_type == "hcpe":
                return load_hcpe_array(
                    file_path, validate=validate, mmap_mode=mmap_mode
                )
            elif array_type == "preprocessing":
                return load_preprocessing_array(
                    file_path, validate=validate, mmap_mode=mmap_mode
                )
            else:
                # Fallback: load as generic numpy array for backward compatibility
                logger.warning(
                    f"Unknown array type '{array_type}', loading as generic numpy array"
                )
                if file_path.suffix == ".npz":
                    with np.load(file_path) as data:
                        if "data" in data:
                            return data["data"]
                        else:
                            # Get the first array in the npz file
                            first_key = list(data.keys())[0]
                            return data[first_key]
                else:
                    # Check if this is a .raw file that we can't handle as generic numpy
                    if file_path.suffix == ".raw":
                        raise DataIOError(f"Cannot load .raw file {file_path} without known schema. Use array_type='hcpe' or 'preprocessing'.")
                    
                    # Check if .raw file exists instead
                    if file_path.suffix in (".npy", ".npz"):
                        raw_path = file_path.with_suffix(".raw")
                        if raw_path.exists():
                            raise DataIOError(f"Found .raw file {raw_path} but cannot load without known schema. Use array_type='hcpe' or 'preprocessing'.")
                    
                    return np.load(file_path, mmap_mode=mmap_mode)

        except Exception as e:
            logger.error(f"Failed to load array from {file_path}: {e}")
            raise DataIOError(f"Failed to load array from {file_path}: {e}") from e

    @staticmethod
    def save_array(
        array: np.ndarray,
        file_path: Union[str, Path],
        *,
        array_type: Literal["auto", "hcpe", "preprocessing"] = "auto",
        validate: bool = True,
        compress: bool = False,
        compression_level: int = 6,
    ) -> None:
        """Save numpy array with automatic schema detection.

        Args:
            array: Array to save
            file_path: Output file path
            array_type: Type of array ("auto", "hcpe", "preprocessing")
            validate: Whether to validate schema before saving
            compress: Whether to use compression
            compression_level: Compression level (0-9)

        Raises:
            DataIOError: If saving fails
            SchemaValidationError: If validation fails
        """
        try:
            if array_type == "auto":
                detected_type = DataIOService._detect_array_type_from_data(array)
                if detected_type in ("hcpe", "preprocessing"):
                    array_type = detected_type  # type: ignore[assignment]
                else:
                    array_type = "unknown"  # type: ignore[assignment]

            if array_type == "hcpe":
                save_hcpe_array(
                    array,
                    file_path,
                    validate=validate,
                    compress=compress,
                    compression_level=compression_level,
                )
            elif array_type == "preprocessing":
                save_preprocessing_array(
                    array,
                    file_path,
                    validate=validate,
                    compress=compress,
                    compression_level=compression_level,
                )
            else:
                # Fallback: save as generic numpy array
                logger.warning(
                    f"Unknown array type '{array_type}', saving as generic numpy array"
                )
                if compress:
                    file_path = Path(file_path).with_suffix(".npz")
                    np.savez_compressed(file_path, data=array)
                else:
                    file_path = Path(file_path).with_suffix(".npy")
                    np.save(file_path, array)

        except Exception as e:
            logger.error(f"Failed to save array to {file_path}: {e}")
            raise DataIOError(f"Failed to save array to {file_path}: {e}") from e

    @staticmethod
    def _detect_array_type(file_path: Path) -> str:
        """Detect array type from file path and metadata.

        Args:
            file_path: Path to the file

        Returns:
            str: Detected array type ("hcpe", "preprocessing", or "unknown")
        """
        try:
            # Try to get file info first
            info = get_file_info(file_path)

            # Check if metadata contains array type
            if "array_type" in info:
                return info["array_type"]

            # Fallback: try to infer from file name patterns
            file_name = file_path.name.lower()
            logger.debug(f"Detecting array type for file: {file_name}")
            if (
                "hcpe" in file_name
                or file_name.endswith(".hcpe.npy")
                or file_name.endswith(".hcpe.npz")
                or file_name.endswith(".hcpe.raw")
            ):
                logger.debug(f"Detected HCPE type for: {file_name}")
                return "hcpe"
            elif (
                "pre" in file_name
                or "processing" in file_name
                or file_name.endswith(".pre.npy")
                or file_name.endswith(".pre.npz")
                or file_name.endswith(".pre.raw")
            ):
                return "preprocessing"

            # Last resort: try to load a small sample and validate
            try:
                if file_path.suffix == ".npz":
                    with np.load(file_path) as data:
                        if "data" in data:
                            sample = data["data"]
                        else:
                            first_key = list(data.keys())[0]
                            sample = data[first_key]
                else:
                    # Check if this is a .raw file that we need to handle differently
                    if file_path.suffix == ".raw":
                        # For .raw files, try to detect from filename patterns
                        file_name = file_path.name.lower()
                        if "hcpe" in file_name or "game" in file_name:
                            return "hcpe"
                        elif "pre" in file_name or "processing" in file_name or "train" in file_name:
                            return "preprocessing"
                        else:
                            return "unknown"
                    elif file_path.suffix in (".npy", ".npz"):
                        # Check if .raw file exists instead
                        raw_path = file_path.with_suffix(".raw")
                        if raw_path.exists():
                            # Recursively detect on the .raw file
                            return DataIOService._detect_array_type(raw_path)
                    
                    # Load just the first few records for type detection
                    full_array = np.load(file_path, mmap_mode="r")
                    sample = (
                        full_array[: min(1, len(full_array))]
                        if len(full_array) > 0
                        else full_array
                    )

                return DataIOService._detect_array_type_from_data(sample)

            except Exception:
                logger.debug(f"Could not determine array type for {file_path}")
                return "unknown"

        except Exception:
            logger.debug(f"Could not get file info for {file_path}")
            return "unknown"

    @staticmethod
    def _detect_array_type_from_data(array: np.ndarray) -> str:
        """Detect array type from array data structure.

        Args:
            array: Numpy array to analyze

        Returns:
            str: Detected array type ("hcpe", "preprocessing", or "unknown")
        """
        try:
            # Try HCPE validation first
            validate_hcpe_array(array)
            return "hcpe"
        except (SchemaValidationError, Exception):
            pass

        try:
            # Try preprocessing validation
            validate_preprocessing_array(array)
            return "preprocessing"
        except (SchemaValidationError, Exception):
            pass

        return "unknown"

    @staticmethod
    def get_array_info(file_path: Union[str, Path]) -> dict:
        """Get information about an array file.

        Args:
            file_path: Path to the file

        Returns:
            dict: File information including detected array type
        """
        try:
            info = get_file_info(file_path)

            # Add detected array type if not present
            if "array_type" not in info:
                info["detected_array_type"] = DataIOService._detect_array_type(
                    Path(file_path)
                )

            return info

        except Exception as e:
            logger.error(f"Failed to get array info for {file_path}: {e}")
            raise DataIOError(f"Failed to get array info for {file_path}: {e}") from e


# Convenience functions for backward compatibility
def load_numpy_array(
    file_path: Union[str, Path],
    *,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
    validate: bool = False,
) -> np.ndarray:
    """Load numpy array with automatic type detection.

    Convenience function for Infrastructure Layer compatibility.

    Args:
        file_path: Path to numpy file
        mmap_mode: Memory mapping mode
        validate: Whether to validate schema

    Returns:
        numpy.ndarray: Loaded array
    """
    return DataIOService.load_array(
        file_path, array_type="auto", validate=validate, mmap_mode=mmap_mode
    )


def save_numpy_array(
    array: np.ndarray,
    file_path: Union[str, Path],
    *,
    compress: bool = False,
    validate: bool = True,
) -> None:
    """Save numpy array with automatic type detection.

    Convenience function for Infrastructure Layer compatibility.

    Args:
        array: Array to save
        file_path: Output file path
        compress: Whether to compress
        validate: Whether to validate schema
    """
    DataIOService.save_array(
        array, file_path, array_type="auto", validate=validate, compress=compress
    )
