"""Compression utilities for preprocessing arrays.

This module provides validation helpers for preprocessing records and
high-performance bit packing for legal move masks. Sparse array helpers are
also available for move label count arrays which tend to be mostly zeros.
"""

import logging
from typing import Tuple

import numpy as np

from maou.app.pre_process.label import MOVE_LABELS_NUM

logger: logging.Logger = logging.getLogger(__name__)


class CompressionError(Exception):
    """Raised when compression/decompression operations fail."""

    pass


BOARD_ID_POSITIONS_SHAPE = (9, 9)
PIECES_IN_HAND_SHAPE = (7,)
LEGAL_MOVES_PACKED_SIZE = (MOVE_LABELS_NUM + 7) // 8
LEGAL_MOVES_SHAPE = (MOVE_LABELS_NUM,)


def pack_legal_moves_mask(
    legal_moves: np.ndarray,
) -> np.ndarray:
    """Pack legal moves mask using bit packing compression."""

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

        if not np.all(
            np.logical_or(legal_moves == 0, legal_moves == 1)
        ):
            raise CompressionError(
                "Legal moves array contains values other than 0 or 1"
            )

        packed = np.packbits(legal_moves, axis=0)

        logger.debug(
            "Packed legal moves mask: %s bits → %s bytes (%sx compression)",
            legal_moves.size,
            packed.size,
            legal_moves.size // packed.size,
        )

        return packed

    except (
        Exception
    ) as exc:  # pragma: no cover - defensive fallback
        raise CompressionError(
            f"Failed to pack legal moves mask: {exc}"
        ) from exc


def unpack_legal_moves_mask(
    packed_legal_moves: np.ndarray,
) -> np.ndarray:
    """Unpack legal moves mask from bit packed format."""

    try:
        if packed_legal_moves.shape != (
            LEGAL_MOVES_PACKED_SIZE,
        ):
            raise CompressionError(
                "Invalid packed legal moves shape. "
                f"Expected ({LEGAL_MOVES_PACKED_SIZE},), "
                f"got {packed_legal_moves.shape}"
            )

        if packed_legal_moves.dtype != np.uint8:
            raise CompressionError(
                "Invalid packed legal moves dtype. Expected uint8, "
                f"got {packed_legal_moves.dtype}"
            )

        unpacked = np.unpackbits(packed_legal_moves, axis=0)

        if unpacked.size > MOVE_LABELS_NUM:
            unpacked = unpacked[:MOVE_LABELS_NUM]

        logger.debug(
            "Unpacked legal moves mask: %s bytes → %s bits",
            packed_legal_moves.size,
            unpacked.size,
        )

        return unpacked

    except (
        Exception
    ) as exc:  # pragma: no cover - defensive fallback
        raise CompressionError(
            f"Failed to unpack legal moves mask: {exc}"
        ) from exc


def pack_preprocessing_record(
    record: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and copy preprocessing fields from a record."""

    try:
        board_positions = np.asarray(
            record["boardIdPositions"], dtype=np.uint8
        )
        if board_positions.shape != BOARD_ID_POSITIONS_SHAPE:
            raise CompressionError(
                "Invalid boardIdPositions shape. "
                f"Expected {BOARD_ID_POSITIONS_SHAPE}, "
                f"got {board_positions.shape}"
            )

        pieces_in_hand = np.asarray(
            record["piecesInHand"], dtype=np.uint8
        )
        if pieces_in_hand.shape != PIECES_IN_HAND_SHAPE:
            raise CompressionError(
                "Invalid piecesInHand shape. "
                f"Expected {PIECES_IN_HAND_SHAPE}, "
                f"got {pieces_in_hand.shape}"
            )

        return board_positions.copy(), pieces_in_hand.copy()

    except KeyError as exc:
        raise CompressionError(
            "Record missing required preprocessing fields"
        ) from exc
    except (
        Exception
    ) as exc:  # pragma: no cover - defensive fallback
        raise CompressionError(
            f"Failed to pack preprocessing record: {exc}"
        ) from exc


def unpack_preprocessing_fields(
    board_id_positions: np.ndarray,
    pieces_in_hand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and copy preprocessing fields from packed representation."""

    try:
        board_positions = np.asarray(
            board_id_positions, dtype=np.uint8
        )
        if board_positions.shape != BOARD_ID_POSITIONS_SHAPE:
            raise CompressionError(
                "Invalid boardIdPositions shape. "
                f"Expected {BOARD_ID_POSITIONS_SHAPE}, "
                f"got {board_positions.shape}"
            )

        hand = np.asarray(pieces_in_hand, dtype=np.uint8)
        if hand.shape != PIECES_IN_HAND_SHAPE:
            raise CompressionError(
                "Invalid piecesInHand shape. "
                f"Expected {PIECES_IN_HAND_SHAPE}, got {hand.shape}"
            )

        return board_positions.copy(), hand.copy()

    except (
        Exception
    ) as exc:  # pragma: no cover - defensive fallback
        raise CompressionError(
            f"Failed to unpack preprocessing fields: {exc}"
        ) from exc


def get_compression_stats() -> dict:
    """Get compression statistics and information."""

    board_size = (
        BOARD_ID_POSITIONS_SHAPE[0]
        * BOARD_ID_POSITIONS_SHAPE[1]
    )
    hand_size = PIECES_IN_HAND_SHAPE[0]

    return {
        "boardIdPositions": {
            "original_size": board_size,
            "packed_size": board_size,
            "compression_ratio": 1.0,
        },
        "piecesInHand": {
            "original_size": hand_size,
            "packed_size": hand_size,
            "compression_ratio": 1.0,
        },
        "total": {
            "original_size": board_size + hand_size,
            "packed_size": board_size + hand_size,
            "compression_ratio": 1.0,
        },
    }


def compress_sparse_int_array(
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compress sparse integer array by storing only non-zero elements."""

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
    """Decompress sparse integer array back to full array."""

    result = np.zeros(size, dtype=np.int32)
    result[indices] = values
    return result
