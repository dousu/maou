"""Centralized numpy dtype schemas for Maou project.

This module defines all numpy structured array schemas used throughout
the project for HCPE data and preprocessing features, ensuring consistency
and type safety across all layers.
"""

from typing import Any, Dict

import numpy as np

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.data.compression import (
    unpack_preprocessing_fields,
)


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    pass


def get_hcpe_dtype() -> np.dtype:
    """Get numpy dtype for HCPE (HuffmanCodedPosAndEval) data format.

    This schema is used for storing game positions, evaluations, and moves
    during the game record conversion process.

    Returns:
        numpy.dtype: Structured dtype for HCPE data
    """
    return np.dtype(
        [
            (
                "hcp",
                (np.uint8, 32),
            ),  # Huffman coded position (32 bytes)
            (
                "eval",
                np.int16,
            ),  # Position evaluation (-32767 to 32767)
            (
                "bestMove16",
                np.int16,
            ),  # Best move in 16-bit format
            (
                "gameResult",
                np.int8,
            ),  # Game result (BLACK_WIN, WHITE_WIN, DRAW)
            (
                "id",
                (np.str_, 128),
            ),  # Unique identifier for position
            (
                "partitioningKey",
                np.dtype("datetime64[D]"),
            ),  # Date for partitioning
            (
                "ratings",
                (np.uint16, 2),
            ),  # Player ratings [black, white]
            (
                "endgameStatus",
                (np.str_, 16),
            ),  # Endgame status description
            ("moves", np.int16),  # Number of moves in game
        ]
    )


def get_preprocessing_dtype() -> np.dtype:
    """Get numpy dtype for preprocessed training data.

    This schema is used for storing neural network training features
    after preprocessing HCPE data.

    Returns:
        numpy.dtype: Structured dtype for preprocessed training data
    """
    return np.dtype(
        [
            ("id", (np.str_, 128)),  # Unique identifier
            ("eval", np.int16),  # Position evaluation
            (
                "features",
                np.uint8,
                (FEATURES_NUM, 9, 9),
            ),  # Board feature representation
            ("moveLabel", np.uint16),  # Move label for training
            (
                "resultValue",
                np.float16,
            ),  # Game result value (-1 to 1)
            (
                "legalMoveMask",
                np.uint8,
                (MOVE_LABELS_NUM,),
            ),  # Legal move mask
            (
                "partitioningKey",
                np.dtype("datetime64[D]"),
            ),  # Date for partitioning
        ]
    )


def get_packed_preprocessing_dtype() -> np.dtype:
    """Get numpy dtype for bit-packed compressed preprocessed training data.

    This schema uses bit packing to compress the features and legalMoveMask
    fields, achieving approximately 8x storage reduction for these binary fields.

    The compressed fields are:
    - features: (104, 9, 9) uint8 → (1053,) uint8 packed bits
    - legalMoveMask: (1496,) uint8 → (187,) uint8 packed bits

    Returns:
        numpy.dtype: Structured dtype for compressed preprocessed training data
    """
    # Calculate packed sizes (bits rounded up to byte boundary)
    features_packed_size = (
        FEATURES_NUM * 9 * 9 + 7
    ) // 8  # 1053 bytes
    legal_moves_packed_size = (
        MOVE_LABELS_NUM + 7
    ) // 8  # 187 bytes

    return np.dtype(
        [
            ("id", (np.str_, 128)),  # Unique identifier
            ("eval", np.int16),  # Position evaluation
            (
                "features_packed",
                np.uint8,
                (features_packed_size,),
            ),  # Bit-packed board features
            ("moveLabel", np.uint16),  # Move label for training
            (
                "resultValue",
                np.float16,
            ),  # Game result value (-1 to 1)
            (
                "legalMoveMask_packed",
                np.uint8,
                (legal_moves_packed_size,),
            ),  # Bit-packed legal move mask
            (
                "partitioningKey",
                np.dtype("datetime64[D]"),
            ),  # Date for partitioning
        ]
    )


def validate_hcpe_array(array: np.ndarray) -> bool:
    """Validate that array conforms to HCPE schema.

    Args:
        array: numpy array to validate

    Returns:
        bool: True if array is valid

    Raises:
        SchemaValidationError: If validation fails
    """
    expected_dtype = get_hcpe_dtype()

    if not isinstance(array, np.ndarray):
        raise SchemaValidationError("Expected numpy ndarray")

    if array.dtype != expected_dtype:
        raise SchemaValidationError(
            f"Invalid dtype. Expected: {expected_dtype}, Got: {array.dtype}"
        )

    # Validate field constraints
    if len(array) > 0:
        # Check eval range
        if np.any(
            (array["eval"] < -32767) | (array["eval"] > 32767)
        ):
            raise SchemaValidationError(
                "eval values out of range [-32767, 32767]"
            )

        # Check gameResult values
        valid_results = {-1, 0, 1}  # BLACK_WIN, DRAW, WHITE_WIN
        if not all(
            result in valid_results
            for result in array["gameResult"]
        ):
            raise SchemaValidationError(
                "Invalid gameResult values"
            )

        # Check moves count
        if np.any(array["moves"] < 0):
            raise SchemaValidationError(
                "moves count cannot be negative"
            )

    return True


def numpy_dtype_to_bigquery_type(numpy_dtype: np.dtype) -> str:
    """Convert numpy dtype to BigQuery type string.

    Args:
        numpy_dtype: numpy dtype to convert

    Returns:
        str: BigQuery type string

    Raises:
        ValueError: If numpy dtype is not supported
    """
    # Handle special cases first
    if numpy_dtype.name == "datetime64[D]":
        return "DATE"
    elif numpy_dtype.name == "datetime64[ms]":
        return "TIMESTAMP"

    # Handle by kind
    kind = numpy_dtype.kind

    if kind in {"i", "u"}:  # signed and unsigned integers
        return "INTEGER"
    elif kind in {"f"}:  # floating point
        # BigQuery doesn't support float16 well, so we use FLOAT for all float types
        return "FLOAT"
    elif kind in {"U"}:  # Unicode string
        return "STRING"
    elif kind in {
        "S",
        "V",
    }:  # byte string, void (for numpy arrays)
        # numpy arrays that are nested are serialized as BYTES
        return "BYTES"
    elif kind in {"b"}:  # boolean
        return "BOOLEAN"
    else:
        raise ValueError(
            f"Unsupported numpy dtype: {numpy_dtype.name} (kind: {numpy_dtype.kind})"
        )


def get_bigquery_schema_for_hcpe() -> list[dict]:
    """Get BigQuery schema definition for HCPE data.

    Returns:
        list: BigQuery schema fields as dictionaries
    """
    hcpe_dtype = get_hcpe_dtype()
    schema = []

    if hcpe_dtype.fields is not None:
        for field_name, field_info in hcpe_dtype.fields.items():
            field_dtype = field_info[
                0
            ]  # First element is always dtype
            schema.append(
                {
                    "name": field_name,
                    "type": numpy_dtype_to_bigquery_type(
                        field_dtype
                    ),
                    "mode": "REQUIRED",
                }
            )

    return schema


def get_bigquery_schema_for_preprocessing() -> list[dict]:
    """Get BigQuery schema definition for preprocessing data.

    Returns:
        list: BigQuery schema fields as dictionaries
    """
    preprocessing_dtype = get_preprocessing_dtype()
    schema = []

    if preprocessing_dtype.fields is not None:
        for (
            field_name,
            field_info,
        ) in preprocessing_dtype.fields.items():
            field_dtype = field_info[
                0
            ]  # First element is always dtype
            schema.append(
                {
                    "name": field_name,
                    "type": numpy_dtype_to_bigquery_type(
                        field_dtype
                    ),
                    "mode": "REQUIRED",
                }
            )

    return schema


def validate_preprocessing_array(array: np.ndarray) -> bool:
    """Validate that array conforms to preprocessing schema.

    Args:
        array: numpy array to validate

    Returns:
        bool: True if array is valid

    Raises:
        SchemaValidationError: If validation fails
    """
    expected_dtype = get_preprocessing_dtype()

    if not isinstance(array, np.ndarray):
        raise SchemaValidationError("Expected numpy ndarray")

    if array.dtype != expected_dtype:
        raise SchemaValidationError(
            f"Invalid dtype. Expected: {expected_dtype}, Got: {array.dtype}"
        )

    # Validate field constraints
    if len(array) > 0:
        # Check eval range
        if np.any(
            (array["eval"] < -32767) | (array["eval"] > 32767)
        ):
            raise SchemaValidationError(
                "eval values out of range [-32767, 32767]"
            )

        # Check moveLabel range
        if np.any(
            (array["moveLabel"] < 0)
            | (array["moveLabel"] >= MOVE_LABELS_NUM)
        ):
            raise SchemaValidationError(
                f"moveLabel values out of range [0, {MOVE_LABELS_NUM})"
            )

        # Check resultValue range
        if np.any(
            (array["resultValue"] < 0.0)
            | (array["resultValue"] > 1.0)
        ):
            raise SchemaValidationError(
                "resultValue values out of range [0.0, 1.0]"
            )

        # Check features shape
        expected_features_shape = (FEATURES_NUM, 9, 9)
        if (
            array["features"].shape[1:]
            != expected_features_shape
        ):
            raise SchemaValidationError(
                f"Invalid features shape. Expected: {expected_features_shape}, "
                f"Got: {array['features'].shape[1:]}"
            )

    return True


def validate_compressed_preprocessing_array(
    array: np.ndarray,
) -> bool:
    """Validate that array conforms to compressed preprocessing schema.

    Args:
        array: numpy array to validate

    Returns:
        bool: True if array is valid

    Raises:
        SchemaValidationError: If validation fails
    """
    expected_dtype = get_packed_preprocessing_dtype()

    if not isinstance(array, np.ndarray):
        raise SchemaValidationError("Expected numpy ndarray")

    if array.dtype != expected_dtype:
        raise SchemaValidationError(
            f"Invalid dtype. Expected: {expected_dtype}, Got: {array.dtype}"
        )

    # Validate field constraints
    if len(array) > 0:
        # Check eval range
        if np.any(
            (array["eval"] < -32767) | (array["eval"] > 32767)
        ):
            raise SchemaValidationError(
                "eval values out of range [-32767, 32767]"
            )

        # Check moveLabel range
        if np.any(
            (array["moveLabel"] < 0)
            | (array["moveLabel"] >= MOVE_LABELS_NUM)
        ):
            raise SchemaValidationError(
                f"moveLabel values out of range [0, {MOVE_LABELS_NUM})"
            )

        # Check resultValue range
        if np.any(
            (array["resultValue"] < 0.0)
            | (array["resultValue"] > 1.0)
        ):
            raise SchemaValidationError(
                "resultValue values out of range [0.0, 1.0]"
            )

        # Check packed features shape
        features_packed_size = (FEATURES_NUM * 9 * 9 + 7) // 8
        expected_features_packed_shape = (features_packed_size,)
        if (
            array["features_packed"].shape[1:]
            != expected_features_packed_shape
        ):
            raise SchemaValidationError(
                f"Invalid features_packed shape. Expected: "
                f"{expected_features_packed_shape}, Got: "
                f"{array['features_packed'].shape[1:]}"
            )

        # Check packed legal moves shape
        legal_moves_packed_size = (MOVE_LABELS_NUM + 7) // 8
        expected_legal_moves_packed_shape = (
            legal_moves_packed_size,
        )
        if (
            array["legalMoveMask_packed"].shape[1:]
            != expected_legal_moves_packed_shape
        ):
            raise SchemaValidationError(
                f"Invalid legalMoveMask_packed shape. Expected: "
                f"{expected_legal_moves_packed_shape}, Got: "
                f"{array['legalMoveMask_packed'].shape[1:]}"
            )

    return True


def get_schema_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all available schemas.

    Returns:
        Dict containing schema information including field names,
        types, and descriptions for documentation purposes.
    """
    return {
        "hcpe": {
            "dtype": get_hcpe_dtype(),
            "description": "HCPE format for game positions and evaluations",
            "fields": {
                "hcp": "Huffman coded position (32 bytes)",
                "eval": "Position evaluation (-32767 to 32767)",
                "bestMove16": "Best move in 16-bit format",
                "gameResult": "Game result (BLACK_WIN, WHITE_WIN, DRAW)",
                "id": "Unique identifier for position",
                "partitioningKey": "Date for partitioning",
                "ratings": "Player ratings [black, white]",
                "endgameStatus": "Endgame status description",
                "moves": "Number of moves in game",
            },
        },
        "preprocessing": {
            "dtype": get_preprocessing_dtype(),
            "description": "Preprocessed training data for neural networks",
            "fields": {
                "id": "Unique identifier",
                "eval": "Position evaluation",
                "features": f"Board feature representation ({FEATURES_NUM}, 9, 9)",
                "moveLabel": f"Move label for training (0 to {MOVE_LABELS_NUM - 1})",
                "resultValue": "Game result value (0.0 to 1.0)",
                "legalMoveMask": f"Legal move mask ({MOVE_LABELS_NUM} elements)",
                "partitioningKey": "Date for partitioning",
            },
        },
        "packed_preprocessing": {
            "dtype": get_packed_preprocessing_dtype(),
            "description": (
                "Bit-packed compressed preprocessed training data (8x size reduction)"
            ),
            "fields": {
                "id": "Unique identifier",
                "eval": "Position evaluation",
                "features_packed": (
                    f"Bit-packed board features "
                    f"({(FEATURES_NUM * 9 * 9 + 7) // 8} bytes)"
                ),
                "moveLabel": f"Move label for training (0 to {MOVE_LABELS_NUM - 1})",
                "resultValue": "Game result value (0.0 to 1.0)",
                "legalMoveMask_packed": (
                    f"Bit-packed legal move mask ({(MOVE_LABELS_NUM + 7) // 8} bytes)"
                ),
                "partitioningKey": "Date for partitioning",
            },
        },
    }


def create_empty_hcpe_array(size: int) -> np.ndarray:
    """Create empty HCPE array with proper schema.

    Args:
        size: Number of elements in array

    Returns:
        numpy.ndarray: Empty array with HCPE schema
    """
    return np.zeros(size, dtype=get_hcpe_dtype())


def create_empty_preprocessing_array(size: int) -> np.ndarray:
    """Create empty preprocessing array with proper schema.

    Args:
        size: Number of elements in array

    Returns:
        numpy.ndarray: Empty array with preprocessing schema
    """
    return np.zeros(size, dtype=get_preprocessing_dtype())


def create_empty_packed_preprocessing_array(
    size: int,
) -> np.ndarray:
    """Create empty compressed preprocessing array with proper schema.

    Args:
        size: Number of elements in array

    Returns:
        numpy.ndarray: Empty array with bit-packing compressed preprocessing schema
    """
    return np.zeros(
        size, dtype=get_packed_preprocessing_dtype()
    )


def convert_array_from_packed_format(
    compressed_array: np.ndarray,
) -> np.ndarray:
    """Convert compressed preprocessing array to standard format.

    Args:
        compressed_array: Compressed preprocessing array

    Returns:
        Standard preprocessing array with unpacked fields
    """

    # Create empty standard array
    standard_dtype = get_preprocessing_dtype()
    standard_array = np.empty(
        len(compressed_array), dtype=standard_dtype
    )

    # Copy non-packed fields directly
    standard_array["id"] = compressed_array["id"]
    standard_array["eval"] = compressed_array["eval"]
    standard_array["moveLabel"] = compressed_array["moveLabel"]
    standard_array["resultValue"] = compressed_array[
        "resultValue"
    ]
    standard_array["partitioningKey"] = compressed_array[
        "partitioningKey"
    ]

    # Unpack binary fields for each record
    for i in range(len(compressed_array)):
        packed_features = compressed_array[i]["features_packed"]
        packed_legal_moves = compressed_array[i][
            "legalMoveMask_packed"
        ]
        features, legal_moves = unpack_preprocessing_fields(
            packed_features, packed_legal_moves
        )
        standard_array[i]["features"] = features
        standard_array[i]["legalMoveMask"] = legal_moves

    return standard_array


def convert_record_from_packed_format(
    compressed_record: np.ndarray,
) -> np.ndarray:
    """Convert compressed preprocessing record to standard format.

    Args:
        compressed_record: Compressed preprocessing record

    Returns:
        Standard preprocessing record with unpacked fields
    """

    # Create empty standard array
    standard_dtype = get_preprocessing_dtype()
    standard_array = np.empty(
        (),
        dtype=standard_dtype,
    )

    # Copy non-packed fields directly
    standard_array["id"] = compressed_record["id"]
    standard_array["eval"] = compressed_record["eval"]
    standard_array["moveLabel"] = compressed_record["moveLabel"]
    standard_array["resultValue"] = compressed_record[
        "resultValue"
    ]
    standard_array["partitioningKey"] = compressed_record[
        "partitioningKey"
    ]

    # Unpack binary fields for each record
    packed_features = compressed_record["features_packed"]
    packed_legal_moves = compressed_record[
        "legalMoveMask_packed"
    ]
    features, legal_moves = unpack_preprocessing_fields(
        packed_features, packed_legal_moves
    )
    standard_array["features"] = features
    standard_array["legalMoveMask"] = legal_moves

    return standard_array


# Constants for backward compatibility
HCPE_DTYPE = get_hcpe_dtype()
PREPROCESSING_DTYPE = get_preprocessing_dtype()
PACKED_PREPROCESSING_DTYPE = get_packed_preprocessing_dtype()
