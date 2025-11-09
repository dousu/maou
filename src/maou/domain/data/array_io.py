"""Centralized numpy I/O operations for Maou project.

This module provides standardized save and load functions for numpy arrays
with proper error handling, validation, and compression support.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from maou.domain.data.compression import (
    pack_preprocessing_record,
)
from maou.domain.data.schema import (
    convert_array_from_packed_format,
    get_hcpe_dtype,
    get_packed_preprocessing_dtype,
    get_preprocessing_dtype,
)


class DataIOError(Exception):
    """Raised when data I/O operations fail."""

    pass


class DataIOArrayTypeError(Exception):
    """Raised when data I/O operations fail."""

    pass


logger: logging.Logger = logging.getLogger(__name__)


def _load_numpy_memmap(
    *,
    file_path: Path,
    dtype: np.dtype,
    mmap_mode: Literal["r", "r+", "w+", "c"],
) -> Optional[Union[np.ndarray, np.memmap]]:
    """Try loading `.npy` files using numpy's memmap support.

    Args:
        file_path: Path to the target file.
        dtype: Expected structured dtype for validation.
        mmap_mode: Memory mapping mode to request from numpy.

    Returns:
        Optional[Union[np.ndarray, np.memmap]]: Loaded memmap array when
        successful, otherwise ``None`` to signal fallback usage.
    """

    if file_path.suffix.lower() != ".npy":
        return None

    try:
        array = np.load(
            file_path,
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
    except ValueError as exc:
        logger.debug(
            "numpy.load could not interpret %s as an .npy file: %s",
            file_path,
            exc,
        )
        return None
    except OSError as exc:
        logger.debug(
            "numpy.load failed to memory-map %s: %s",
            file_path,
            exc,
        )
        return None

    if array.dtype != dtype:
        logger.debug(
            "numpy.load returned dtype %s for %s, expected %s; falling back to "
            "manual memmap handling.",
            array.dtype,
            file_path,
            dtype,
        )
        return None

    return array


def save_hcpe_array(
    array: np.ndarray,
    file_path: Path,
) -> None:
    """Save HCPE array to file.

    Args:
        array: HCPE numpy array to save
        file_path: Output file path

    Raises:
        DataIOError: If save operation fails
    """
    try:
        # Use high-performance binary format with tofile()
        if not file_path.suffix:
            # Add .raw extension if no extension is provided
            file_path = file_path.with_suffix(".npy")

        # High-performance binary save using tofile() for all formats
        array.tofile(file_path)
        logger.debug(
            f"Saved HCPE array to {file_path} using tofile()"
        )

    except Exception as e:
        raise DataIOError(
            f"Failed to save HCPE array to {file_path}: {e}"
        ) from e


def load_hcpe_array(
    file_path: Path,
    *,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
) -> Union[np.ndarray, np.memmap]:
    """Load HCPE array from file.

    Args:
        file_path: Input file path
        mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')

    Returns:
        numpy.ndarray or numpy.memmap: Loaded HCPE array

    Raises:
        DataIOError: If load operation fails
    """
    try:
        # Check for file existence, trying different extensions if needed
        if not file_path.exists():
            raise DataIOError(f"File not found: {file_path}")

        # Try to detect if it's a standard .npy file with header or raw binary
        dtype = get_hcpe_dtype()
        array: Union[np.ndarray, np.memmap]
        if mmap_mode:
            array = _load_numpy_memmap(
                file_path=file_path,
                dtype=dtype,
                mmap_mode=mmap_mode,
            )
            if array is None:
                array = np.memmap(
                    file_path, dtype=dtype, mode=mmap_mode
                )
                logger.debug(
                    f"Loaded HCPE array as memmap from {file_path}"
                )
            else:
                logger.debug(
                    "Loaded HCPE array using numpy.load with memory mapping "
                    f"from {file_path}"
                )
        else:
            array = np.fromfile(file_path, dtype=dtype)
            logger.debug(
                f"Loaded HCPE array using fromfile() from {file_path}"
            )

        logger.debug(
            f"Loaded HCPE array from {file_path}, shape: {array.shape}"
        )
        return array

    except Exception as e:
        raise DataIOError(
            f"Failed to load HCPE array from {file_path}: {e}"
        ) from e


def save_preprocessing_array(
    array: np.ndarray,
    file_path: Path,
    *,
    bit_pack: bool = True,
) -> None:
    """Save preprocessing array to file with optional bit packing compression.

    Args:
        array: Preprocessing numpy array to save
        file_path: Output file path
        bit_pack: Whether to use bit packing compression for binary fields

    Raises:
        DataIOError: If save operation fails
        CompressionError: If bit packing fails
    """
    try:
        if bit_pack:
            # Bit packing compression for binary fields

            # Convert to compressed format
            compressed_array = _convert_to_packed_format(array)

            compressed_array.tofile(file_path)
            logger.debug(
                f"Saved bit-packed preprocessing array to {file_path}"
            )
        else:
            # Standard format without bit packing
            # High-performance binary save using tofile() for all formats
            array.tofile(file_path)
            logger.debug(
                f"Saved preprocessing array to {file_path} using tofile()"
            )

    except Exception as e:
        raise DataIOError(
            f"Failed to save preprocessing array to {file_path}: {e}"
        ) from e


def save_preprocessing_array_with_unpacking(
    array: np.ndarray,
    file_path: Path,
) -> None:
    """Save unpacked preprocessing array to file.

    Args:
        array: Preprocessing numpy array to save
        file_path: Output file path

    Raises:
        DataIOError: If save operation fails
    """
    try:
        # Standard format without bit packing
        unpacked_array = convert_array_from_packed_format(array)
        # High-performance binary save using tofile() for all formats
        unpacked_array.tofile(file_path)
        logger.debug(
            f"Saved preprocessing array to {file_path} using tofile()"
        )
    except Exception as e:
        raise DataIOError(
            f"Failed to save preprocessing array to {file_path}: {e}"
        ) from e


def load_preprocessing_array(
    file_path: Path,
    *,
    bit_pack: bool,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
) -> Union[np.ndarray, np.memmap]:
    """Load preprocessing array from file.

    Args:
        file_path: Input file path
        bit_pack: Whether to use bit packing compression for binary fields
        mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')

    Returns:
        numpy.ndarray or numpy.memmap: Loaded preprocessing array in standard format

    Raises:
        DataIOError: If load operation fails
        CompressionError: If decompression fails
    """
    try:
        # Check for file existence, trying different extensions if needed
        if not file_path.exists():
            raise DataIOError(f"File not found: {file_path}")

        array: Union[np.ndarray, np.memmap]
        if bit_pack:
            # Load bit-packed format
            dtype = get_packed_preprocessing_dtype()

            if mmap_mode:
                logger.warning(
                    "Even when bit_pack is enabled, "
                    "the output will be a regular ndarray, not a memmap."
                )
                array = np.memmap(
                    file_path, dtype=dtype, mode=mmap_mode
                )
                logger.debug(
                    f"Loaded bit-packed preprocessing array as memmap from {file_path}"
                )
            else:
                array = np.fromfile(file_path, dtype=dtype)
                logger.debug(
                    f"Loaded bit-packed preprocessing array using "
                    f"fromfile() from {file_path}"
                )
            array = convert_array_from_packed_format(array)
        else:
            dtype = get_preprocessing_dtype()

            if mmap_mode:
                array = _load_numpy_memmap(
                    file_path=file_path,
                    dtype=dtype,
                    mmap_mode=mmap_mode,
                )
                if array is None:
                    array = np.memmap(
                        file_path, dtype=dtype, mode=mmap_mode
                    )
                    logger.debug(
                        "Loaded preprocessing array as memmap from "
                        f"{file_path}"
                    )
                else:
                    logger.debug(
                        "Loaded preprocessing array using numpy.load with "
                        f"memory mapping from {file_path}"
                    )
            else:
                array = np.fromfile(file_path, dtype=dtype)
                logger.debug(
                    f"Loaded preprocessing array using fromfile() from {file_path}"
                )

        logger.debug(
            f"Loaded preprocessing array from {file_path}, shape: {array.shape}"
        )
        return array

    except Exception as e:
        raise DataIOError(
            f"Failed to load preprocessing array from {file_path}: {e}"
        ) from e


def load_packed_preprocessing_array(
    file_path: Path,
    *,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
) -> Union[np.ndarray, np.memmap]:
    """Load packed preprocessing array from file.

    Args:
        file_path: Input file path
        mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')

    Returns:
        numpy.ndarray or numpy.memmap: Loaded preprocessing array in standard format

    Raises:
        DataIOError: If load operation fails
    """
    try:
        # Check for file existence, trying different extensions if needed
        if not file_path.exists():
            raise DataIOError(f"File not found: {file_path}")

        array: Union[np.ndarray, np.memmap]
        # Load bit-packed format
        dtype = get_packed_preprocessing_dtype()
        if mmap_mode:
            array = _load_numpy_memmap(
                file_path=file_path,
                dtype=dtype,
                mmap_mode=mmap_mode,
            )
            if array is None:
                array = np.memmap(
                    file_path, dtype=dtype, mode=mmap_mode
                )
                logger.debug(
                    "Loaded bit-packed preprocessing array as memmap "
                    f"from {file_path}"
                )
            else:
                logger.debug(
                    "Loaded bit-packed preprocessing array using numpy.load "
                    f"with memory mapping from {file_path}"
                )
        else:
            array = np.fromfile(file_path, dtype=dtype)
            logger.debug(
                f"Loaded bit-packed preprocessing array using "
                f"fromfile() from {file_path}"
            )

        logger.debug(
            f"Loaded packed preprocessing array from {file_path}, shape: {array.shape}"
        )
        return array

    except Exception as e:
        raise DataIOError(
            f"Failed to load preprocessing array from {file_path}: {e}"
        ) from e


def save_hcpe_array_to_buffer(
    array: np.ndarray,
) -> BytesIO:
    """Save HCPE array to BytesIO buffer in numpy binary format.

    This function provides a consistent way to serialize numpy arrays to BytesIO
    buffers for cloud storage, using numpy's native binary format for
    compatibility with np.fromfile().

    Args:
        array: HCPE array to save to buffer

    Returns:
        BytesIO: Buffer containing the serialized array data in numpy format

    Raises:
        DataIOError: If buffer creation fails
    """
    try:
        # Create buffer and save in numpy binary format
        buffer = BytesIO()
        buffer.write(array.tobytes())
        buffer.seek(0)

        logger.debug(
            f"Saved array to buffer using numpy.ndarray.tobytes, shape: {array.shape}"
        )
        return buffer

    except Exception as e:
        raise DataIOError(
            f"Failed to save HCPE array to buffer: {e}"
        ) from e


def load_hcpe_array_from_buffer(
    buffer: BytesIO,
    *,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
) -> Union[np.ndarray, np.memmap]:
    """Load HCPE array from BytesIO buffer in numpy binary format.

    This function provides a consistent way to deserialize numpy arrays from BytesIO
    buffers for cloud storage, using numpy's native binary format for
    compatibility with np.fromfile().

    Args:
        buffer: BytesIO buffer containing the HCPE array
        mmap_mode: Memory mapping mode for .npy files ('r', 'r+', 'w+', 'c')

    Returns:
        numpy.ndarray or numpy.memmap: Loaded HCPE array

    Raises:
        DataIOError: If loading fails
    """
    try:
        # Load from buffer using numpy's frombuffer()
        dtype = get_hcpe_dtype()
        array: Union[np.ndarray, np.memmap]
        if mmap_mode:
            logger.warning(
                "Even when bit_pack is enabled, "
                "the output will be a regular ndarray, not a memmap."
            )
            array = np.memmap(
                buffer, dtype=dtype, mode=mmap_mode
            )
            logger.debug(
                "Loaded HCPE array as memmap from buffer"
            )
        else:
            array = np.frombuffer(
                buffer.getvalue(), dtype=dtype
            )
            logger.debug(
                "Loaded HCPE array using frombuffer() from buffer"
            )

        logger.debug(
            f"Loaded HCPE array from buffer, shape: {array.shape}"
        )
        return array

    except Exception as e:
        raise DataIOError(
            f"Failed to load HCPE array from buffer: {e}"
        ) from e


def save_preprocessing_array_to_buffer(
    array: np.ndarray,
    *,
    bit_pack: bool,
) -> BytesIO:
    """Save preprocessing array with optional bit packing compression to BytesIO buffer in numpy binary format.

    This function provides a consistent way to serialize numpy arrays to BytesIO
    buffers for cloud storage, using numpy's native binary format for
    compatibility with np.fromfile().

    Args:
        array: preprocessing array to save to buffer
        bit_pack: Whether to use bit packing compression for binary fields

    Returns:
        BytesIO: Buffer containing the serialized array data in numpy format

    Raises:
        DataIOError: If buffer creation fails
    """
    try:
        if bit_pack:
            target_array = _convert_to_packed_format(array)
            # Create buffer and save in numpy binary format
            buffer = BytesIO()
            buffer.write(target_array.tobytes())
            buffer.seek(0)

            logger.debug(
                f"Saved array to buffer using numpy.ndarray.tobytes, shape: {target_array.shape}"
            )
            return buffer
        else:
            buffer = BytesIO()
            buffer.write(array.tobytes())
            buffer.seek(0)

            logger.debug(
                f"Saved array to buffer using numpy.ndarray.tobytes, shape: {array.shape}"
            )
            return buffer

    except Exception as e:
        raise DataIOError(
            f"Failed to save preprocessing array to buffer: {e}"
        ) from e


def load_preprocessing_array_from_buffer(
    buffer: BytesIO,
    *,
    bit_pack: bool,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
) -> Union[np.ndarray, np.memmap]:
    """Load preprocessing array from BytesIO buffer in numpy binary format.

    This function provides a consistent way to deserialize numpy arrays from BytesIO
    buffers for cloud storage, using numpy's native binary format for
    compatibility with np.fromfile().

    Args:
        buffer: BytesIO buffer containing the preprocessing array
        bit_pack: Whether to use bit packing compression for binary fields
        mmap_mode: Memory mapping mode for .npy files ('r', 'r+', 'w+', 'c')

    Returns:
        numpy.ndarray or numpy.memmap: Loaded preprocessing array

    Raises:
        DataIOError: If loading fails
    """
    try:
        # Load from buffer using numpy's frombuffer()
        array: Union[np.ndarray, np.memmap]
        if bit_pack:
            # Load bit-packed format
            dtype = get_packed_preprocessing_dtype()

            if mmap_mode:
                logger.warning(
                    "Even when bit_pack is enabled, "
                    "the output will be a regular ndarray, not a memmap."
                )
                array = np.memmap(
                    buffer, dtype=dtype, mode=mmap_mode
                )
                logger.debug(
                    "Loaded bit-packed preprocessing array as memmap from buffer"
                )
            else:
                array = np.frombuffer(
                    buffer.getvalue(), dtype=dtype
                )
                logger.debug(
                    "Loaded bit-packed preprocessing array using frombuffer() from buffer"
                )
            array = convert_array_from_packed_format(array)
        else:
            dtype = get_preprocessing_dtype()

            if mmap_mode:
                array = np.memmap(
                    buffer, dtype=dtype, mode=mmap_mode
                )
                logger.debug(
                    "Loaded preprocessing array as memmap from buffer"
                )
            else:
                array = np.frombuffer(
                    buffer.getvalue(), dtype=dtype
                )
                logger.debug(
                    "Loaded preprocessing array using frombuffer() from buffer"
                )

        logger.debug(
            f"Loaded preprocessing array from buffer, shape: {array.shape}"
        )
        return array

    except Exception as e:
        raise DataIOError(
            f"Failed to load preprocessing array from buffer: {e}"
        ) from e


def _convert_to_packed_format(array: np.ndarray) -> np.ndarray:
    """Convert standard preprocessing array to compressed format.

    Args:
        array: Standard preprocessing array

    Returns:
        Compressed preprocessing array with bit-packed fields
    """

    # Create empty compressed array
    compressed_dtype = get_packed_preprocessing_dtype()
    compressed_array = np.empty(
        len(array), dtype=compressed_dtype
    )

    # Copy non-packed fields directly
    compressed_array["id"] = array["id"]
    compressed_array["moveLabel"] = array["moveLabel"]
    compressed_array["resultValue"] = array["resultValue"]

    # Copy structured fields for each record with validation
    for i in range(len(array)):
        (
            board_positions,
            pieces_in_hand,
        ) = pack_preprocessing_record(array[i])
        compressed_array[i]["boardIdPositions"] = (
            board_positions
        )
        compressed_array[i]["piecesInHand"] = pieces_in_hand

    return compressed_array
