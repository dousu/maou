"""Centralized numpy I/O operations for Maou project.

This module provides standardized save and load functions for numpy arrays
with proper error handling, validation, and compression support.
"""

import logging
from io import BytesIO  # type: ignore[attr-defined]
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np

from .compression import (
    pack_preprocessing_record,
    unpack_preprocessing_fields,
)
from .schema import (
    SchemaValidationError,
    validate_compressed_preprocessing_array,
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
            if file_path.suffix in (".npy", ".raw"):
                file_path = file_path.with_suffix(".npz")

            np.savez_compressed(
                file_path, data=array, compression_level=compression_level
            )
            logger.debug(f"Saved compressed HCPE array to {file_path}")
        else:
            # Use high-performance binary format with tofile()
            if not file_path.suffix:
                # Add .raw extension if no extension is provided
                file_path = file_path.with_suffix(".raw")

            # High-performance binary save using tofile() for all formats
            array.tofile(file_path)
            logger.debug(f"Saved HCPE array to {file_path} using tofile()")

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
        file_path: Input file path (.raw, .npy or .npz)
        validate: Whether to validate schema after loading
        mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')

    Returns:
        numpy.ndarray: Loaded HCPE array

    Raises:
        DataIOError: If load operation fails
        SchemaValidationError: If validation fails
    """
    try:
        from .schema import get_hcpe_dtype

        file_path = Path(file_path)

        # Check for file existence, trying different extensions if needed
        if not file_path.exists():
            # Try .raw extension if original doesn't exist
            if file_path.suffix in (".npy", ".npz"):
                raw_path = file_path.with_suffix(".raw")
                if raw_path.exists():
                    file_path = raw_path
                else:
                    raise DataIOError(f"File not found: {file_path}")
            else:
                raise DataIOError(f"File not found: {file_path}")

        if file_path.suffix == ".npz":
            # Load compressed format
            with np.load(file_path) as data:
                if "data" not in data:
                    raise DataIOError(
                        f"Expected 'data' key in compressed file {file_path}"
                    )
                array = data["data"]
        elif file_path.suffix in (".raw", ".npy"):
            # Try to detect if it's a standard .npy file with header or raw binary
            dtype = get_hcpe_dtype()
            try:
                # First try to check if it's a standard .npy file
                with open(file_path, "rb") as f:
                    header = f.read(10)
                    if header.startswith(b"\x93NUMPY"):
                        # It's a standard .npy file with numpy header
                        if mmap_mode:
                            array = np.load(file_path, mmap_mode=mmap_mode)
                        else:
                            array = np.load(file_path)
                        logger.debug(
                            f"Loaded HCPE array from standard .npy file {file_path}"
                        )
                    else:
                        # It's a raw binary file (created with tofile())
                        if mmap_mode:
                            array = np.memmap(file_path, dtype=dtype, mode=mmap_mode)
                            logger.debug(
                                f"Loaded HCPE array as memmap from {file_path}"
                            )
                        else:
                            array = np.fromfile(file_path, dtype=dtype)
                            logger.debug(
                                f"Loaded HCPE array using fromfile() from {file_path}"
                            )
            except Exception:
                # Fallback to raw binary reading
                if mmap_mode:
                    array = np.memmap(file_path, dtype=dtype, mode=mmap_mode)
                    logger.debug(f"Loaded HCPE array as memmap from {file_path}")
                else:
                    array = np.fromfile(file_path, dtype=dtype)
                    logger.debug(f"Loaded HCPE array using fromfile() from {file_path}")
        else:
            # This should not happen with current implementation
            raise DataIOError(f"Unsupported file format: {file_path.suffix}")

        if validate:
            validate_hcpe_array(array)
            logger.debug(f"HCPE array validation passed for {file_path}")

        logger.debug(f"Loaded HCPE array from {file_path}, shape: {array.shape}")
        return array

    except Exception as e:
        raise DataIOError(f"Failed to load HCPE array from {file_path}: {e}") from e


def _convert_to_compressed_format(array: np.ndarray) -> np.ndarray:
    """Convert standard preprocessing array to compressed format.

    Args:
        array: Standard preprocessing array

    Returns:
        Compressed preprocessing array with bit-packed fields
    """
    from .schema import get_compressed_preprocessing_dtype

    # Create empty compressed array
    compressed_dtype = get_compressed_preprocessing_dtype()
    compressed_array = np.zeros(len(array), dtype=compressed_dtype)

    # Copy non-packed fields directly
    compressed_array["id"] = array["id"]
    compressed_array["eval"] = array["eval"]
    compressed_array["moveLabel"] = array["moveLabel"]
    compressed_array["resultValue"] = array["resultValue"]
    compressed_array["partitioningKey"] = array["partitioningKey"]

    # Pack binary fields for each record
    for i in range(len(array)):
        packed_features, packed_legal_moves = pack_preprocessing_record(array[i])
        compressed_array[i]["features_packed"] = packed_features
        compressed_array[i]["legalMoveMask_packed"] = packed_legal_moves

    return compressed_array


def _convert_from_compressed_format(compressed_array: np.ndarray) -> np.ndarray:
    """Convert compressed preprocessing array to standard format.

    Args:
        compressed_array: Compressed preprocessing array

    Returns:
        Standard preprocessing array with unpacked fields
    """
    from .schema import get_preprocessing_dtype

    # Create empty standard array
    standard_dtype = get_preprocessing_dtype()
    standard_array = np.zeros(len(compressed_array), dtype=standard_dtype)

    # Copy non-packed fields directly
    standard_array["id"] = compressed_array["id"]
    standard_array["eval"] = compressed_array["eval"]
    standard_array["moveLabel"] = compressed_array["moveLabel"]
    standard_array["resultValue"] = compressed_array["resultValue"]
    standard_array["partitioningKey"] = compressed_array["partitioningKey"]

    # Unpack binary fields for each record
    for i in range(len(compressed_array)):
        packed_features = compressed_array[i]["features_packed"]
        packed_legal_moves = compressed_array[i]["legalMoveMask_packed"]
        features, legal_moves = unpack_preprocessing_fields(
            packed_features, packed_legal_moves
        )
        standard_array[i]["features"] = features
        standard_array[i]["legalMoveMask"] = legal_moves

    return standard_array


def save_preprocessing_array(
    array: np.ndarray,
    file_path: Union[str, Path],
    *,
    validate: bool = True,
    compress: bool = False,
    compression_level: int = 6,
    bit_pack: bool = False,
) -> None:
    """Save preprocessing array to file with validation and optional compression.

    Args:
        array: Preprocessing numpy array to save
        file_path: Output file path
        validate: Whether to validate schema before saving
        compress: Whether to use compression (saves as .npz)
        compression_level: Compression level (0-9, higher is more compressed)
        bit_pack: Whether to use bit packing compression for binary fields

    Raises:
        DataIOError: If save operation fails
        SchemaValidationError: If validation fails
        CompressionError: If bit packing fails
    """
    try:
        file_path = Path(file_path)

        if bit_pack:
            # Bit pack compression for binary fields
            if validate:
                validate_preprocessing_array(array)
                logger.debug(f"Preprocessing array validation passed for {file_path}")

            # Convert to compressed format

            compressed_array = _convert_to_compressed_format(array)

            if compress:
                # Change extension to .npz for compressed format
                if file_path.suffix in (".npy", ".raw"):
                    file_path = file_path.with_suffix(".npz")

                np.savez_compressed(
                    file_path,
                    data=compressed_array,
                    compression_level=compression_level,
                )
                logger.debug(
                    f"Saved bit-packed + compressed preprocessing array to {file_path}"
                )
            else:
                # Use high-performance binary format with tofile()
                if not file_path.suffix:
                    # Add .packed extension for bit-packed format
                    file_path = file_path.with_suffix(".packed")

                compressed_array.tofile(file_path)
                logger.debug(f"Saved bit-packed preprocessing array to {file_path}")

        else:
            # Standard format without bit packing
            if validate:
                validate_preprocessing_array(array)
                logger.debug(f"Preprocessing array validation passed for {file_path}")

            if compress:
                # Change extension to .npz for compressed format
                if file_path.suffix in (".npy", ".raw"):
                    file_path = file_path.with_suffix(".npz")

                np.savez_compressed(
                    file_path, data=array, compression_level=compression_level
                )
                logger.debug(f"Saved compressed preprocessing array to {file_path}")
            else:
                # Use high-performance binary format with tofile()
                if not file_path.suffix:
                    # Add .raw extension if no extension is provided
                    file_path = file_path.with_suffix(".raw")

                # High-performance binary save using tofile() for all formats
                array.tofile(file_path)
                logger.debug(f"Saved preprocessing array to {file_path} using tofile()")

    except Exception as e:
        raise DataIOError(
            f"Failed to save preprocessing array to {file_path}: {e}"
        ) from e


def load_preprocessing_array(
    file_path: Union[str, Path],
    *,
    validate: bool = True,
    mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = None,
    auto_decompress: bool = True,
) -> np.ndarray:
    """Load preprocessing array from file with validation.

    Args:
        file_path: Input file path (.raw, .npy, .npz, .packed)
        validate: Whether to validate schema after loading
        mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')
        auto_decompress: Whether to automatically decompress bit-packed files

    Returns:
        numpy.ndarray: Loaded preprocessing array in standard format

    Raises:
        DataIOError: If load operation fails
        SchemaValidationError: If validation fails
        CompressionError: If decompression fails
    """
    try:
        from .schema import get_compressed_preprocessing_dtype, get_preprocessing_dtype

        file_path = Path(file_path)

        # Check for file existence, trying different extensions if needed
        if not file_path.exists():
            # Try different extensions if original doesn't exist
            if file_path.suffix in (".npy", ".npz") or not file_path.suffix:
                # Try .packed extension first (bit-packed), then .raw, then .npy, then
                # .npz
                extensions_to_try = [".packed", ".raw", ".npy", ".npz"]
                found_file = None

                for ext in extensions_to_try:
                    test_path = file_path.with_suffix(ext)
                    if test_path.exists():
                        found_file = test_path
                        break

                if found_file:
                    file_path = found_file
                else:
                    raise DataIOError(f"File not found: {file_path}")
            else:
                raise DataIOError(f"File not found: {file_path}")

        # Determine if this is a bit-packed file
        is_bit_packed = file_path.suffix == ".packed"

        if file_path.suffix == ".npz":
            # Load compressed format (could be standard or bit-packed)
            with np.load(file_path) as data:
                if "data" not in data:
                    raise DataIOError(
                        f"Expected 'data' key in compressed file {file_path}"
                    )
                array = data["data"]

                # Try to detect if this is bit-packed data by checking field names
                if hasattr(array, "dtype") and array.dtype.names:
                    is_bit_packed = "features_packed" in array.dtype.names

        elif file_path.suffix == ".packed":
            # Load bit-packed format
            dtype = get_compressed_preprocessing_dtype()
            try:
                # First try to check if it's a standard .npy file
                with open(file_path, "rb") as f:
                    header = f.read(10)
                    if header.startswith(b"\x93NUMPY"):
                        # It's a standard .npy file with numpy header
                        if mmap_mode:
                            array = np.load(file_path, mmap_mode=mmap_mode)
                        else:
                            array = np.load(file_path)
                        logger.debug(
                            f"Loaded bit-packed preprocessing array from standard "
                            f".npy file {file_path}"
                        )
                    else:
                        # It's a raw binary file (created with tofile())
                        if mmap_mode:
                            array = np.memmap(file_path, dtype=dtype, mode=mmap_mode)
                            logger.debug(
                                f"Loaded bit-packed preprocessing array as memmap "
                                f"from {file_path}"
                            )
                        else:
                            array = np.fromfile(file_path, dtype=dtype)
                            logger.debug(
                                f"Loaded bit-packed preprocessing array using "
                                f"fromfile() from {file_path}"
                            )
            except Exception:
                # Fallback to raw binary reading
                if mmap_mode:
                    array = np.memmap(file_path, dtype=dtype, mode=mmap_mode)
                    logger.debug(
                        f"Loaded bit-packed preprocessing array as memmap "
                        f"from {file_path}"
                    )
                else:
                    array = np.fromfile(file_path, dtype=dtype)
                    logger.debug(
                        f"Loaded bit-packed preprocessing array using "
                        f"fromfile() from {file_path}"
                    )
        elif file_path.suffix in (".raw", ".npy"):
            # Try to detect if it's a standard .npy file with header or raw binary
            dtype = get_preprocessing_dtype()
            try:
                # First try to check if it's a standard .npy file
                with open(file_path, "rb") as f:
                    header = f.read(10)
                    if header.startswith(b"\x93NUMPY"):
                        # It's a standard .npy file with numpy header
                        if mmap_mode:
                            array = np.load(file_path, mmap_mode=mmap_mode)
                        else:
                            array = np.load(file_path)
                        logger.debug(
                            f"Loaded preprocessing array from standard "
                            f".npy file {file_path}"
                        )
                    else:
                        # It's a raw binary file (created with tofile())
                        if mmap_mode:
                            array = np.memmap(file_path, dtype=dtype, mode=mmap_mode)
                            logger.debug(
                                f"Loaded preprocessing array as memmap from {file_path}"
                            )
                        else:
                            array = np.fromfile(file_path, dtype=dtype)
                            logger.debug(
                                f"Loaded preprocessing array using fromfile() from "
                                f"{file_path}"
                            )
            except Exception:
                # Fallback to raw binary reading
                if mmap_mode:
                    array = np.memmap(file_path, dtype=dtype, mode=mmap_mode)
                    logger.debug(
                        f"Loaded preprocessing array as memmap from {file_path}"
                    )
                else:
                    array = np.fromfile(file_path, dtype=dtype)
                    logger.debug(
                        f"Loaded preprocessing array using fromfile() from {file_path}"
                    )
        else:
            # This should not happen with current implementation
            raise DataIOError(f"Unsupported file format: {file_path.suffix}")

        # Handle bit-packed arrays - convert to standard format if auto_decompress
        # is True
        if is_bit_packed and auto_decompress:
            if validate:
                validate_compressed_preprocessing_array(array)
                logger.debug(
                    f"Compressed preprocessing array validation passed for {file_path}"
                )

            # Convert to standard format
            array = _convert_from_compressed_format(array)
            logger.debug(f"Auto-decompressed bit-packed array from {file_path}")

        elif is_bit_packed and not auto_decompress:
            # Return bit-packed format as-is
            if validate:
                validate_compressed_preprocessing_array(array)
                logger.debug(
                    f"Compressed preprocessing array validation passed for {file_path}"
                )
        else:
            # Standard format
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
        # Note: compression_level parameter is not used by np.savez_compressed
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


def save_array_to_buffer(
    array: np.ndarray,
    *,
    validate: bool = True,
    array_type: Optional[Literal["hcpe", "preprocessing"]] = None,
) -> BytesIO:
    """Save numpy array to BytesIO buffer in numpy binary format.

    This function provides a consistent way to serialize numpy arrays to BytesIO
    buffers for cloud storage, using numpy's native binary format for
    compatibility with np.load().

    Args:
        array: numpy array to save to buffer
        validate: Whether to validate schema before saving
        array_type: Type of array for validation ("hcpe" or "preprocessing")

    Returns:
        BytesIO: Buffer containing the serialized array data in numpy format

    Raises:
        DataIOError: If buffer creation fails
        SchemaValidationError: If validation fails
    """
    try:
        if validate and array_type:
            if array_type == "hcpe":
                validate_hcpe_array(array)
                logger.debug("HCPE array validation passed for buffer save")
            elif array_type == "preprocessing":
                validate_preprocessing_array(array)
                logger.debug("Preprocessing array validation passed for buffer save")
            else:
                raise ValueError(f"Unknown array type: {array_type}")

        # Create buffer and save in numpy binary format
        buffer = BytesIO()
        np.save(buffer, array)
        buffer.seek(0)

        logger.debug(f"Saved array to buffer using np.save(), shape: {array.shape}")
        return buffer

    except Exception as e:
        raise DataIOError(f"Failed to save array to buffer: {e}") from e


def load_array_from_buffer(
    buffer: BytesIO,
    *,
    validate: bool = True,
    array_type: Optional[Literal["hcpe", "preprocessing"]] = None,
) -> np.ndarray:
    """Load numpy array from BytesIO buffer.

    Args:
        buffer: BytesIO buffer containing numpy array data
        validate: Whether to validate schema after loading
        array_type: Expected array type for validation ("hcpe" or "preprocessing")

    Returns:
        numpy.ndarray: Loaded array

    Raises:
        DataIOError: If loading fails
        SchemaValidationError: If validation fails
    """
    try:
        buffer.seek(0)
        array = np.load(buffer, allow_pickle=False)

        if validate and array_type:
            if array_type == "hcpe":
                validate_hcpe_array(array)
                logger.debug("HCPE array validation passed after buffer load")
            elif array_type == "preprocessing":
                validate_preprocessing_array(array)
                logger.debug("Preprocessing array validation passed after buffer load")
            else:
                raise ValueError(f"Unknown array type: {array_type}")

        logger.debug(f"Loaded array from buffer, shape: {array.shape}")
        return array

    except Exception as e:
        raise DataIOError(f"Failed to load array from buffer: {e}") from e


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
        elif file_path.suffix == ".raw":
            # For .raw files, determine dtype and calculate shape from file size
            from .schema import get_hcpe_dtype, get_preprocessing_dtype

            # Try to determine array type and dtype
            file_name = file_path.name.lower()
            if "hcpe" in file_name or "game" in file_name:
                dtype = get_hcpe_dtype()
                info["array_type"] = "hcpe"
            elif (
                "pre" in file_name or "processing" in file_name or "train" in file_name
            ):
                dtype = get_preprocessing_dtype()
                info["array_type"] = "preprocessing"
            else:
                # Default to HCPE if can't determine
                dtype = get_hcpe_dtype()
                info["array_type"] = "unknown"

            # Calculate array shape from file size
            element_size = dtype.itemsize
            total_elements = file_path.stat().st_size // element_size

            info["shape"] = (total_elements,)
            info["dtype"] = str(dtype)
            info["size"] = total_elements
            info["format"] = "raw_binary"
        else:
            # For .npy files, try to read header first, but fallback to raw format
            # if no header
            try:
                with open(file_path, "rb") as f:
                    np.lib.format.read_magic(f)  # Read magic but don't store
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
                    info["shape"] = shape
                    info["dtype"] = str(dtype)
                    info["size"] = np.prod(shape) if shape else 0
                    info["fortran_order"] = fortran_order
            except (ValueError, OSError) as e:
                # If reading header fails, treat as raw binary file
                # (created with tofile())
                logger.debug(
                    f"Failed to read numpy header from {file_path}, "
                    f"treating as raw binary: {e}"
                )

                from .schema import get_hcpe_dtype, get_preprocessing_dtype

                # Try to determine array type and dtype from filename
                file_name = file_path.name.lower()
                if "hcpe" in file_name or "game" in file_name:
                    dtype = get_hcpe_dtype()
                    info["array_type"] = "hcpe"
                elif (
                    "pre" in file_name
                    or "processing" in file_name
                    or "train" in file_name
                ):
                    dtype = get_preprocessing_dtype()
                    info["array_type"] = "preprocessing"
                else:
                    # Default to HCPE if can't determine
                    dtype = get_hcpe_dtype()
                    info["array_type"] = "unknown"

                # Calculate array shape from file size
                element_size = dtype.itemsize
                total_elements = file_path.stat().st_size // element_size

                info["shape"] = (total_elements,)
                info["dtype"] = str(dtype)
                info["size"] = total_elements
                info["format"] = "raw_binary"

        return info

    except Exception as e:
        raise DataIOError(f"Failed to get file info for {file_path}: {e}") from e
