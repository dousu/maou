"""Bit packing compression utilities for Maou project.

This module provides high-performance bit packing compression for binary data
in the preprocessing dtype, primarily targeting the 'features' field that
contains only 0 or 1 values. Utilities for packing and unpacking legal move
masks are also provided for compatibility with other workflows.

The compression uses numpy's packbits/unpackbits functions to achieve 8x
storage reduction while maintaining high performance.

Also provides sparse array compression for move_label_count arrays which
are mostly zeros, achieving significant storage reduction.
"""

import logging
from typing import Tuple

import numpy as np

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board.shogi import FEATURES_NUM

logger: logging.Logger = logging.getLogger(__name__)


class CompressionError(Exception):
    """Raised when compression/decompression operations fail."""

    pass


# Constants for packed array sizes
FEATURES_PACKED_SIZE = (
    FEATURES_NUM * 9 * 9 + 7
) // 8  # 1053 bytes
LEGAL_MOVES_PACKED_SIZE = (
    MOVE_LABELS_NUM + 7
) // 8  # 187 bytes

# Original shapes for validation
FEATURES_SHAPE = (FEATURES_NUM, 9, 9)
LEGAL_MOVES_SHAPE = (MOVE_LABELS_NUM,)


def pack_features_array(features: np.ndarray) -> np.ndarray:
    """Pack features array using bit packing compression.

    Compresses features from (FEATURES_NUM, 9, 9) uint8 array containing
    only 0s and 1s into a packed bit array.

    Args:
        features: Features array with shape (FEATURES_NUM, 9, 9) and dtype uint8

    Returns:
        Packed features array with shape (FEATURES_PACKED_SIZE,) and dtype uint8

    Raises:
        CompressionError: If compression fails or input is invalid
    """
    try:
        if features.shape != FEATURES_SHAPE:
            raise CompressionError(
                f"Invalid features shape. Expected {FEATURES_SHAPE}, "
                f"got {features.shape}"
            )

        if features.dtype != np.uint8:
            raise CompressionError(
                f"Invalid features dtype. Expected uint8, got {features.dtype}"
            )

        # Validate that features only contain 0 or 1 values
        if not np.all(
            np.logical_or(features == 0, features == 1)
        ):
            raise CompressionError(
                "Features array contains values other than 0 or 1"
            )

        # Flatten the array and pack bits
        flattened = features.flatten()
        packed = np.packbits(flattened, axis=0)

        logger.debug(
            f"Packed features array: {features.size} bits → {packed.size} bytes "
            f"({features.size // packed.size:.1f}x compression)"
        )

        return packed

    except Exception as e:
        raise CompressionError(
            f"Failed to pack features array: {e}"
        ) from e


def unpack_features_array(
    packed_features: np.ndarray,
) -> np.ndarray:
    """Unpack features array from bit packed format.

    Decompresses packed bit array back to original features array format.

    Args:
        packed_features: Packed features array with shape (FEATURES_PACKED_SIZE,)

    Returns:
        Features array with shape (FEATURES_NUM, 9, 9) and dtype uint8

    Raises:
        CompressionError: If decompression fails or input is invalid
    """
    try:
        if packed_features.shape != (FEATURES_PACKED_SIZE,):
            raise CompressionError(
                f"Invalid packed features shape. Expected ({FEATURES_PACKED_SIZE},), "
                f"got {packed_features.shape}"
            )

        if packed_features.dtype != np.uint8:
            raise CompressionError(
                f"Invalid packed features dtype. Expected uint8, "
                f"got {packed_features.dtype}"
            )

        # Unpack bits and reshape
        unpacked = np.unpackbits(packed_features, axis=0)

        # Trim to exact size (packbits pads to byte boundary)
        expected_size = FEATURES_NUM * 9 * 9
        if unpacked.size > expected_size:
            unpacked = unpacked[:expected_size]

        # Reshape to original dimensions
        features = unpacked.reshape(FEATURES_SHAPE)

        logger.debug(
            f"Unpacked features array: {packed_features.size} bytes → "
            f"{features.size} bits"
        )

        return features

    except Exception as e:
        raise CompressionError(
            f"Failed to unpack features array: {e}"
        ) from e


def pack_legal_moves_mask(
    legal_moves: np.ndarray,
) -> np.ndarray:
    """Pack legal moves mask using bit packing compression.

    Compresses legal moves mask from (MOVE_LABELS_NUM,) uint8 array containing
    only 0s and 1s into a packed bit array.

    Args:
        legal_moves: Legal moves mask with shape (MOVE_LABELS_NUM,) and dtype uint8

    Returns:
        Packed legal moves array with shape (LEGAL_MOVES_PACKED_SIZE,) and dtype uint8

    Raises:
        CompressionError: If compression fails or input is invalid
    """
    try:
        if legal_moves.shape != LEGAL_MOVES_SHAPE:
            raise CompressionError(
                f"Invalid legal moves shape. Expected {LEGAL_MOVES_SHAPE}, "
                f"got {legal_moves.shape}"
            )

        if legal_moves.dtype != np.uint8:
            raise CompressionError(
                f"Invalid legal moves dtype. Expected uint8, got {legal_moves.dtype}"
            )

        # Validate that legal moves only contain 0 or 1 values
        if not np.all(
            np.logical_or(legal_moves == 0, legal_moves == 1)
        ):
            raise CompressionError(
                "Legal moves array contains values other than 0 or 1"
            )

        # Pack bits
        packed = np.packbits(legal_moves, axis=0)

        logger.debug(
            f"Packed legal moves mask: {legal_moves.size} bits → {packed.size} bytes "
            f"({legal_moves.size // packed.size:.1f}x compression)"
        )

        return packed

    except Exception as e:
        raise CompressionError(
            f"Failed to pack legal moves mask: {e}"
        ) from e


def unpack_legal_moves_mask(
    packed_legal_moves: np.ndarray,
) -> np.ndarray:
    """Unpack legal moves mask from bit packed format.

    Decompresses packed bit array back to original legal moves mask format.

    Args:
        packed_legal_moves: Packed legal moves with shape (LEGAL_MOVES_PACKED_SIZE,)

    Returns:
        Legal moves mask with shape (MOVE_LABELS_NUM,) and dtype uint8

    Raises:
        CompressionError: If decompression fails or input is invalid
    """
    try:
        if packed_legal_moves.shape != (
            LEGAL_MOVES_PACKED_SIZE,
        ):
            raise CompressionError(
                f"Invalid packed legal moves shape. Expected "
                f"({LEGAL_MOVES_PACKED_SIZE},), got {packed_legal_moves.shape}"
            )

        if packed_legal_moves.dtype != np.uint8:
            raise CompressionError(
                f"Invalid packed legal moves dtype. Expected uint8, "
                f"got {packed_legal_moves.dtype}"
            )

        # Unpack bits
        unpacked = np.unpackbits(packed_legal_moves, axis=0)

        # Trim to exact size (packbits pads to byte boundary)
        if unpacked.size > MOVE_LABELS_NUM:
            unpacked = unpacked[:MOVE_LABELS_NUM]

        logger.debug(
            f"Unpacked legal moves mask: {packed_legal_moves.size} bytes → "
            f"{unpacked.size} bits"
        )

        return unpacked

    except Exception as e:
        raise CompressionError(
            f"Failed to unpack legal moves mask: {e}"
        ) from e


def pack_preprocessing_record(record: np.ndarray) -> np.ndarray:
    """Pack features from a preprocessing record."""

    try:
        features = record["features"]
        return pack_features_array(features)

    except Exception as e:
        raise CompressionError(
            f"Failed to pack preprocessing record: {e}"
        ) from e


def unpack_preprocessing_fields(packed_features: np.ndarray) -> np.ndarray:
    """Unpack features array from a packed preprocessing record."""

    try:
        return unpack_features_array(packed_features)

    except Exception as e:
        raise CompressionError(
            f"Failed to unpack preprocessing fields: {e}"
        ) from e


def get_compression_stats() -> dict:
    """Get compression statistics and information.

    Returns:
        Dictionary containing compression statistics
    """
    return {
        "features": {
            "original_size": FEATURES_NUM * 9 * 9,
            "packed_size": FEATURES_PACKED_SIZE,
            "compression_ratio": (FEATURES_NUM * 9 * 9)
            / FEATURES_PACKED_SIZE,
        },
        "total": {
            "original_size": FEATURES_NUM * 9 * 9,
            "packed_size": FEATURES_PACKED_SIZE,
            "compression_ratio": (FEATURES_NUM * 9 * 9)
            / FEATURES_PACKED_SIZE,
        },
    }


def compress_sparse_int_array(
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compress sparse integer array by storing only non-zero elements.

    Args:
        arr: Integer array (typically int32)

    Returns:
        Tuple of (indices, values) where indices are positions of non-zero elements
    """
    # Ensure arr is a numpy array
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1D array, got {arr.ndim}D array"
        )

    nonzero_indices = np.nonzero(arr)[0].astype(np.uint16)
    nonzero_values = arr[nonzero_indices].astype(np.int32)
    return nonzero_indices, nonzero_values


def decompress_sparse_int_array(
    indices: np.ndarray, values: np.ndarray, size: int
) -> np.ndarray:
    """Decompress sparse integer array back to full array.

    Args:
        indices: Positions of non-zero elements (uint16)
        values: Values at those positions (int32)
        size: Size of the full array

    Returns:
        Full integer array with zeros and values restored
    """
    result = np.zeros(size, dtype=np.int32)
    result[indices] = values
    return result
