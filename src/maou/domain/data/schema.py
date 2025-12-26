"""Centralized numpy dtype schemas for Maou project.

This module defines all numpy structured array schemas used throughout
the project for HCPE data and preprocessing features, ensuring consistency
and type safety across all layers.
"""

from typing import Any, Dict

import numpy as np

from maou.app.pre_process.label import MOVE_LABELS_NUM
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


def get_intermediate_dtype() -> np.dtype:
    """Get numpy dtype for intermediate data to process.

    Returns:
        numpy.dtype: Structured dtype for preprocessed training data
    """
    return np.dtype(
        [
            ("id", np.uint64),  # Unique identifier
            (
                "boardIdPositions",
                np.uint8,
                (9, 9),
            ),  # Board position identifiers
            (
                "piecesInHand",
                np.uint8,
                (14,),
            ),  # Pieces in hand counts for both players
            (
                "count",
                np.int32,
            ),  # count
            (
                "moveLabelCount",
                np.int32,
                (MOVE_LABELS_NUM,),
            ),  # Move label for training
            (
                "winCount",
                np.float32,
            ),  # Sum of result values
        ]
    )


def get_preprocessing_dtype() -> np.dtype:
    """Get numpy dtype for preprocessed training data.

    This schema is used for storing neural network training inputs
    after preprocessing HCPE data.

    Returns:
        numpy.dtype: Structured dtype for preprocessed training data
    """
    return np.dtype(
        [
            ("id", np.uint64),  # Unique identifier
            (
                "boardIdPositions",
                np.uint8,
                (9, 9),
            ),  # Board position identifiers
            (
                "piecesInHand",
                np.uint8,
                (14,),
            ),  # Pieces in hand for both players
            (
                "moveLabel",
                np.float16,
                (MOVE_LABELS_NUM,),
            ),  # Move label for training
            (
                "resultValue",
                np.float16,
            ),  # Game result value (0 to 1)
        ]
    )


def get_packed_preprocessing_dtype() -> np.dtype:
    """Get numpy dtype for bit-packed compressed preprocessed training data.

    This schema maintains compatibility with storage backends that
    expect the compressed preprocessing representation.

    Returns:
        numpy.dtype: Structured dtype for compressed preprocessed training data
    """
    return np.dtype(
        [
            ("id", np.uint64),  # Unique identifier
            (
                "boardIdPositions",
                np.uint8,
                (9, 9),
            ),  # Board position identifiers
            (
                "piecesInHand",
                np.uint8,
                (14,),
            ),  # Pieces in hand for both players
            (
                "moveLabel",
                np.float16,
                (MOVE_LABELS_NUM,),
            ),  # Move label for training
            (
                "resultValue",
                np.float16,
            ),  # Game result value (-1 to 1)
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
    elif numpy_dtype.name == "uint64":
        # BigQuery doesn't have unsigned 64-bit integers，
        # so we use STRING to avoid overflow for hash values
        return "STRING"

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
        # Check moveLabel range
        if np.any(
            (array["moveLabel"] < 0)
            | (array["moveLabel"] > 1.0)
        ):
            raise SchemaValidationError(
                "moveLabel values out of range [0, 1]"
            )

        # Check resultValue range
        if np.any(
            (array["resultValue"] < 0.0)
            | (array["resultValue"] > 1.0)
        ):
            raise SchemaValidationError(
                "resultValue values out of range [0.0, 1.0]"
            )

        # Check boardIdPositions shape
        expected_board_shape = (9, 9)
        if (
            array["boardIdPositions"].shape[1:]
            != expected_board_shape
        ):
            raise SchemaValidationError(
                "Invalid boardIdPositions shape. "
                f"Expected: {expected_board_shape}, "
                f"Got: {array['boardIdPositions'].shape[1:]}"
            )

        # Check piecesInHand shape
        expected_hand_shape = (14,)
        if (
            array["piecesInHand"].shape[1:]
            != expected_hand_shape
        ):
            raise SchemaValidationError(
                "Invalid piecesInHand shape. "
                f"Expected: {expected_hand_shape}, "
                f"Got: {array['piecesInHand'].shape[1:]}"
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
        # Check moveLabel range
        if np.any(
            (array["moveLabel"] < 0)
            | (array["moveLabel"] > 1.0)
        ):
            raise SchemaValidationError(
                "moveLabel values out of range [0, 1]"
            )

        # Check resultValue range
        if np.any(
            (array["resultValue"] < 0.0)
            | (array["resultValue"] > 1.0)
        ):
            raise SchemaValidationError(
                "resultValue values out of range [0.0, 1.0]"
            )

        # Check boardIdPositions shape
        expected_board_shape = (9, 9)
        if (
            array["boardIdPositions"].shape[1:]
            != expected_board_shape
        ):
            raise SchemaValidationError(
                "Invalid boardIdPositions shape. "
                f"Expected: {expected_board_shape}, "
                f"Got: {array['boardIdPositions'].shape[1:]}"
            )

        # Check piecesInHand shape
        expected_hand_shape = (14,)
        if (
            array["piecesInHand"].shape[1:]
            != expected_hand_shape
        ):
            raise SchemaValidationError(
                "Invalid piecesInHand shape. "
                f"Expected: {expected_hand_shape}, "
                f"Got: {array['piecesInHand'].shape[1:]}"
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
                "boardIdPositions": "Board position identifiers (9, 9)",
                "piecesInHand": "Pieces in hand counts (14,)",
                "moveLabel": (
                    "Move label for training "
                    f"({MOVE_LABELS_NUM} elements, 0.0 to 1.0)"
                ),
                "resultValue": "Game result value (0.0 to 1.0)",
            },
        },
        "packed_preprocessing": {
            "dtype": get_packed_preprocessing_dtype(),
            "description": (
                "Compressed-compatible preprocessed training data representation"
            ),
            "fields": {
                "id": "Unique identifier",
                "boardIdPositions": "Board position identifiers (9, 9)",
                "piecesInHand": "Pieces in hand counts (14,)",
                "moveLabel": (
                    "Move label for training "
                    f"({MOVE_LABELS_NUM} elements, 0.0 to 1.0)",
                ),
                "resultValue": "Game result value (0.0 to 1.0)",
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


def create_empty_intermediate_array(size: int) -> np.ndarray:
    """Create empty intermediate array with proper schema.

    Args:
        size: Number of elements in array

    Returns:
        numpy.ndarray: Empty array with intermediate schema
    """
    return np.zeros(size, dtype=get_intermediate_dtype())


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


def get_stage1_dtype() -> np.dtype:
    """Get numpy dtype for Stage 1 (reachable squares) training data.

    This schema is used for training the reachable squares prediction head，
    which learns which board squares pieces can move to.

    Returns:
        numpy.dtype: Structured dtype for Stage 1 data
    """
    return np.dtype(
        [
            ("id", np.uint64),  # Unique identifier
            (
                "boardIdPositions",
                np.uint8,
                (9, 9),
            ),  # Board position identifiers
            (
                "piecesInHand",
                np.uint8,
                (14,),
            ),  # Pieces in hand for both players
            (
                "reachableSquares",
                np.uint8,
                (9, 9),
            ),  # Binary: 1=reachable，0=not
        ]
    )


def get_stage2_dtype() -> np.dtype:
    """Get numpy dtype for Stage 2 (legal moves) training data.

    This schema is used for training the legal moves prediction head，
    which learns which moves are legal in a given position.

    Returns:
        numpy.dtype: Structured dtype for Stage 2 data
    """
    return np.dtype(
        [
            ("id", np.uint64),  # Unique identifier
            (
                "boardIdPositions",
                np.uint8,
                (9, 9),
            ),  # Board position identifiers
            (
                "piecesInHand",
                np.uint8,
                (14,),
            ),  # Pieces in hand for both players
            (
                "legalMovesLabel",
                np.uint8,
                (MOVE_LABELS_NUM,),
            ),  # Binary multi-label: 1=legal，0=illegal
        ]
    )


def validate_stage1_array(array: np.ndarray) -> bool:
    """Validate that array conforms to Stage 1 schema.

    Args:
        array: numpy array to validate

    Returns:
        bool: True if array is valid

    Raises:
        SchemaValidationError: If validation fails
    """
    expected_dtype = get_stage1_dtype()

    if not isinstance(array, np.ndarray):
        raise SchemaValidationError("Expected numpy ndarray")

    if array.dtype != expected_dtype:
        raise SchemaValidationError(
            f"Invalid dtype. Expected: {expected_dtype}, Got: {array.dtype}"
        )

    # Validate field constraints
    if len(array) > 0:
        # Check reachableSquares is binary
        if not np.all(
            np.isin(array["reachableSquares"], [0, 1])
        ):
            raise SchemaValidationError(
                "reachableSquares must contain only 0 or 1"
            )

        # Check boardIdPositions shape
        expected_board_shape = (9, 9)
        if (
            array["boardIdPositions"].shape[1:]
            != expected_board_shape
        ):
            raise SchemaValidationError(
                "Invalid boardIdPositions shape. "
                f"Expected: {expected_board_shape}，"
                f"Got: {array['boardIdPositions'].shape[1:]}"
            )

        # Check piecesInHand shape
        expected_hand_shape = (14,)
        if (
            array["piecesInHand"].shape[1:]
            != expected_hand_shape
        ):
            raise SchemaValidationError(
                "Invalid piecesInHand shape. "
                f"Expected: {expected_hand_shape}，"
                f"Got: {array['piecesInHand'].shape[1:]}"
            )

        # Check reachableSquares shape
        if (
            array["reachableSquares"].shape[1:]
            != expected_board_shape
        ):
            raise SchemaValidationError(
                "Invalid reachableSquares shape. "
                f"Expected: {expected_board_shape}，"
                f"Got: {array['reachableSquares'].shape[1:]}"
            )

    return True


def validate_stage2_array(array: np.ndarray) -> bool:
    """Validate that array conforms to Stage 2 schema.

    Args:
        array: numpy array to validate

    Returns:
        bool: True if array is valid

    Raises:
        SchemaValidationError: If validation fails
    """
    expected_dtype = get_stage2_dtype()

    if not isinstance(array, np.ndarray):
        raise SchemaValidationError("Expected numpy ndarray")

    if array.dtype != expected_dtype:
        raise SchemaValidationError(
            f"Invalid dtype. Expected: {expected_dtype}, Got: {array.dtype}"
        )

    # Validate field constraints
    if len(array) > 0:
        # Check legalMovesLabel is binary
        if not np.all(
            np.isin(array["legalMovesLabel"], [0, 1])
        ):
            raise SchemaValidationError(
                "legalMovesLabel must contain only 0 or 1"
            )

        # Check boardIdPositions shape
        expected_board_shape = (9, 9)
        if (
            array["boardIdPositions"].shape[1:]
            != expected_board_shape
        ):
            raise SchemaValidationError(
                "Invalid boardIdPositions shape. "
                f"Expected: {expected_board_shape}，"
                f"Got: {array['boardIdPositions'].shape[1:]}"
            )

        # Check piecesInHand shape
        expected_hand_shape = (14,)
        if (
            array["piecesInHand"].shape[1:]
            != expected_hand_shape
        ):
            raise SchemaValidationError(
                "Invalid piecesInHand shape. "
                f"Expected: {expected_hand_shape}，"
                f"Got: {array['piecesInHand'].shape[1:]}"
            )

        # Check legalMovesLabel shape
        expected_labels_shape = (MOVE_LABELS_NUM,)
        if (
            array["legalMovesLabel"].shape[1:]
            != expected_labels_shape
        ):
            raise SchemaValidationError(
                "Invalid legalMovesLabel shape. "
                f"Expected: {expected_labels_shape}，"
                f"Got: {array['legalMovesLabel'].shape[1:]}"
            )

    return True


def create_empty_stage1_array(size: int) -> np.ndarray:
    """Create empty Stage 1 array with proper schema.

    Args:
        size: Number of elements in array

    Returns:
        numpy.ndarray: Empty array with Stage 1 schema
    """
    return np.zeros(size, dtype=get_stage1_dtype())


def create_empty_stage2_array(size: int) -> np.ndarray:
    """Create empty Stage 2 array with proper schema.

    Args:
        size: Number of elements in array

    Returns:
        numpy.ndarray: Empty array with Stage 2 schema
    """
    return np.zeros(size, dtype=get_stage2_dtype())


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
    standard_array["moveLabel"] = compressed_array["moveLabel"]
    standard_array["resultValue"] = compressed_array[
        "resultValue"
    ]

    # Copy structured fields for each record with validation
    for i in range(len(compressed_array)):
        (
            board_positions,
            pieces_in_hand,
        ) = unpack_preprocessing_fields(
            compressed_array[i]["boardIdPositions"],
            compressed_array[i]["piecesInHand"],
        )
        standard_array[i]["boardIdPositions"] = board_positions
        standard_array[i]["piecesInHand"] = pieces_in_hand

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
    standard_array["moveLabel"] = compressed_record["moveLabel"]
    standard_array["resultValue"] = compressed_record[
        "resultValue"
    ]

    # Copy structured fields with validation
    (
        board_positions,
        pieces_in_hand,
    ) = unpack_preprocessing_fields(
        compressed_record["boardIdPositions"],
        compressed_record["piecesInHand"],
    )
    standard_array["boardIdPositions"] = board_positions
    standard_array["piecesInHand"] = pieces_in_hand

    return standard_array


# Constants for backward compatibility
HCPE_DTYPE = get_hcpe_dtype()
PREPROCESSING_DTYPE = get_preprocessing_dtype()
PACKED_PREPROCESSING_DTYPE = get_packed_preprocessing_dtype()


# ============================================================================
# Polars Schema Definitions
# ============================================================================


try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


def get_hcpe_polars_schema() -> dict[str, "pl.DataType"]:
    """Get Polars schema for HCPE format．

    HCPEフォーマット用のPolarsスキーマを返す．
    Feather形式での保存時にはArrowのFixed-sizeスキーマが使用される．

    Returns:
        dict[str, pl.DataType]: Polarsスキーマ定義

    Raises:
        ImportError: Polarsが利用不可の場合
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    return {
        "hcp": pl.Binary(),  # 32-byte fixed in Arrow
        "eval": pl.Int16(),
        "bestMove16": pl.Int16(),
        "gameResult": pl.Int8(),
        "id": pl.Utf8(),
        "partitioningKey": pl.Date(),
        "ratings": pl.List(pl.UInt16),  # Fixed size 2 in Arrow
        "endgameStatus": pl.Utf8(),
        "moves": pl.Int16(),
    }


def get_preprocessing_polars_schema() -> dict[
    str, "pl.DataType"
]:
    """Get Polars schema for preprocessing format．

    前処理済みデータ用のPolarsスキーマを返す．
    学習用の特徴量データ（盤面，持ち駒，指し手ラベル，結果値）を含む．

    Returns:
        dict[str, pl.DataType]: Polarsスキーマ定義

    Raises:
        ImportError: Polarsが利用不可の場合
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    return {
        "id": pl.UInt64(),
        "boardIdPositions": pl.List(
            pl.List(pl.UInt8)
        ),  # 9x9 board (fixed in Arrow)
        "piecesInHand": pl.List(
            pl.UInt8
        ),  # 14 elements (fixed in Arrow)
        "moveLabel": pl.List(
            pl.Float32
        ),  # MOVE_LABELS_NUM elements (fixed in Arrow)
        "resultValue": pl.Float32(),
    }


def create_empty_hcpe_df(size: int = 0) -> "pl.DataFrame":
    """Create empty HCPE DataFrame with proper schema．

    指定されたサイズの空のHCPE DataFrameを作成する．

    Args:
        size: 作成する行数（デフォルト: 0）

    Returns:
        pl.DataFrame: HCPEスキーマを持つ空のDataFrame

    Raises:
        ImportError: Polarsが利用不可の場合

    Example:
        >>> df = create_empty_hcpe_df(100)
        >>> len(df)
        100
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    schema = get_hcpe_polars_schema()

    if size == 0:
        return pl.DataFrame(schema=schema)

    # Create DataFrame with null values
    return pl.DataFrame(
        {
            col: pl.Series(
                values=[], dtype=dtype
            ).extend_constant(None, size)
            for col, dtype in schema.items()
        }
    )


def get_intermediate_polars_schema() -> dict[
    str, "pl.DataType"
]:
    """Get Polars schema for intermediate data format．

    中間データ用のPolarsスキーマを返す．
    前処理の中間段階で使用される集計データ（盤面ごとの統計情報）を含む．

    Returns:
        dict[str, pl.DataType]: Polarsスキーマ定義

    Raises:
        ImportError: Polarsが利用不可の場合
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    return {
        "id": pl.UInt64(),  # Board hash
        "boardIdPositions": pl.List(
            pl.List(pl.UInt8)
        ),  # 9x9 board (fixed in Arrow)
        "piecesInHand": pl.List(
            pl.UInt8
        ),  # 14 elements (fixed in Arrow)
        "count": pl.Int32(),  # Number of occurrences
        "moveLabelCount": pl.List(
            pl.Int32
        ),  # MOVE_LABELS_NUM elements (fixed in Arrow)
        "winCount": pl.Float32(),  # Sum of win values
    }


def get_stage1_polars_schema() -> dict[str, "pl.DataType"]:
    """Get Polars schema for Stage 1 (reachable squares) training data．

    Stage 1学習用のPolarsスキーマを返す．
    到達可能マス予測ヘッドの学習に使用される．

    Returns:
        dict[str, pl.DataType]: Stage 1データ用のPolarsスキーマ

    Raises:
        ImportError: Polarsが利用不可の場合

    Example:
        >>> schema = get_stage1_polars_schema()
        >>> df = pl.DataFrame(data, schema=schema)
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    return {
        "id": pl.UInt64(),
        "boardIdPositions": pl.List(
            pl.List(pl.UInt8)
        ),  # 9x9 board
        "piecesInHand": pl.List(pl.UInt8),  # 14 elements
        "reachableSquares": pl.List(
            pl.List(pl.UInt8)
        ),  # 9x9 binary
    }


def get_stage2_polars_schema() -> dict[str, "pl.DataType"]:
    """Get Polars schema for Stage 2 (legal moves) training data．

    Stage 2学習用のPolarsスキーマを返す．
    合法手予測ヘッドの学習に使用される．

    Returns:
        dict[str, pl.DataType]: Stage 2データ用のPolarsスキーマ

    Raises:
        ImportError: Polarsが利用不可の場合

    Example:
        >>> schema = get_stage2_polars_schema()
        >>> df = pl.DataFrame(data, schema=schema)
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    return {
        "id": pl.UInt64(),
        "boardIdPositions": pl.List(
            pl.List(pl.UInt8)
        ),  # 9x9 board
        "piecesInHand": pl.List(pl.UInt8),  # 14 elements
        "legalMovesLabel": pl.List(
            pl.UInt8
        ),  # MOVE_LABELS_NUM elements
    }


def create_empty_intermediate_df(
    size: int = 0,
) -> "pl.DataFrame":
    """Create empty intermediate DataFrame with proper schema．

    指定されたサイズの空の中間DataFrameを作成する．

    Args:
        size: 作成する行数（デフォルト: 0）

    Returns:
        pl.DataFrame: 中間データスキーマを持つ空のDataFrame

    Raises:
        ImportError: Polarsが利用不可の場合

    Example:
        >>> df = create_empty_intermediate_df(1000)
        >>> len(df)
        1000
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    schema = get_intermediate_polars_schema()

    if size == 0:
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(
        {
            col: pl.Series(
                values=[], dtype=dtype
            ).extend_constant(None, size)
            for col, dtype in schema.items()
        }
    )


def create_empty_preprocessing_df(
    size: int = 0,
) -> "pl.DataFrame":
    """Create empty preprocessing DataFrame with proper schema．

    指定されたサイズの空の前処理済みDataFrameを作成する．

    Args:
        size: 作成する行数（デフォルト: 0）

    Returns:
        pl.DataFrame: 前処理スキーマを持つ空のDataFrame

    Raises:
        ImportError: Polarsが利用不可の場合

    Example:
        >>> df = create_empty_preprocessing_df(1000)
        >>> len(df)
        1000
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    schema = get_preprocessing_polars_schema()

    if size == 0:
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(
        {
            col: pl.Series(
                values=[], dtype=dtype
            ).extend_constant(None, size)
            for col, dtype in schema.items()
        }
    )


def get_board_position_polars_schema() -> dict[
    str, "pl.DataType"
]:
    """Polars schema for board piece positions (9x9 grid)．

    盤面の駒配置を表すPolarsスキーマ．
    9x9のネストされたリストでPieceId値を格納する．

    Returns:
        dict[str, pl.DataType]: boardIdPositions列のスキーマ定義

    Raises:
        ImportError: Polarsが利用不可の場合

    Example:
        >>> schema = get_board_position_polars_schema()
        >>> schema["boardIdPositions"]
        List(List(UInt8))
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    import polars as pl

    return {
        "boardIdPositions": pl.List(
            pl.List(pl.UInt8)
        ),  # 9x9 nested lists
    }


def get_hcp_polars_schema() -> dict[str, "pl.DataType"]:
    """Polars schema for HuffmanCodedPos binary data．

    HuffmanCodedPos形式の局面データを表すPolarsスキーマ．
    32バイトのバイナリデータとして格納する．

    Returns:
        dict[str, pl.DataType]: hcp列のスキーマ定義

    Raises:
        ImportError: Polarsが利用不可の場合

    Example:
        >>> schema = get_hcp_polars_schema()
        >>> schema["hcp"]
        Binary
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    import polars as pl

    return {
        "hcp": pl.Binary(),  # 32-byte binary blob
    }


def get_piece_planes_polars_schema() -> dict[
    str, "pl.DataType"
]:
    """Polars schema for piece feature planes (104x9x9)．

    駒の特徴平面を表すPolarsスキーマ．
    104チャンネル×9×9のネストされたリストでfloat32値を格納する．

    Returns:
        dict[str, pl.DataType]: piecePlanes列のスキーマ定義

    Raises:
        ImportError: Polarsが利用不可の場合

    Example:
        >>> schema = get_piece_planes_polars_schema()
        >>> schema["piecePlanes"]
        List(List(List(Float32)))
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    import polars as pl

    return {
        "piecePlanes": pl.List(
            pl.List(pl.List(pl.Float32))
        ),  # 104x9x9 nested lists
    }
