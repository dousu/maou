"""Centralized data schemas for Maou project.

This module defines Polars-based schemas for all data structures:

**Polars Schemas (Primary)**:
- HCPE format: Game records with positions and evaluations
- Preprocessing format: Neural network training features
- Intermediate format: Aggregation data for preprocessing
- Stage1/Stage2 formats: Multi-stage training data

**Data Pipeline**:
- Arrow IPC (.feather) files for high-performance I/O
- Polars DataFrames for all processing
- Zero-copy integration with Rust backend
- Direct Polars → Parquet for BigQuery uploads

**Legacy Numpy Support (Minimal)**:
- Conversion functions for PyTorch Dataset compatibility
- Kept for stable Polars → numpy → PyTorch pipeline
- ONNX export utilities require numpy structured arrays

**Migration Status** (Phase 2-3 Complete):
- ✅ BigQueryFeatureStore: Now uses Polars → Parquet directly
- ✅ ObjectStorageFeatureStore: Accepts Polars DataFrames
- ✅ HCPE Converter: Outputs Polars DataFrames (.feather)
- ✅ Preprocessing: Outputs Polars DataFrames (.feather)
- ✅ Validation functions: Removed (use schema enforcement)
- ⚠️  PyTorch Dataset: Intentionally uses numpy (zero-copy to tensors)
"""

import numpy as np

from maou.app.pre_process.label import MOVE_LABELS_NUM

# ============================================================================
# Numpy dtype definitions (Required for BigQuery and cloud storage)
# ============================================================================
# These numpy schemas are maintained for:
# 1. BigQuery integration (numpy_dtype_to_bigquery_type)
# 2. Cloud storage FeatureStore (BigQuery, GCS, S3 expect numpy arrays)
# 3. Polars ↔ numpy conversions (convert_*_df_to_numpy)
# 4. Backward compatibility with existing cloud storage formats
#
# DO NOT DELETE unless all cloud storage backends are migrated to Polars.
# ============================================================================


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


# ============================================================================
# Polars DataFrame ↔ numpy structured array conversions
# ============================================================================


def convert_hcpe_df_to_numpy(df: "pl.DataFrame") -> np.ndarray:
    """Convert HCPE Polars DataFrame to numpy structured array．

    Args:
        df: Polars DataFrame with HCPE schema

    Returns:
        numpy.ndarray: Structured array with HCPE dtype

    Raises:
        ImportError: If polars is not installed
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    import polars as pl

    # Get target dtype
    dtype = get_hcpe_dtype()
    array = np.empty(len(df), dtype=dtype)

    # Convert fields
    array["hcp"] = np.array(
        [bytes(row) for row in df["hcp"].to_list()],
        dtype=(np.uint8, 32),
    )
    array["eval"] = df["eval"].to_numpy()
    array["bestMove16"] = df["bestMove16"].to_numpy()
    array["gameResult"] = df["gameResult"].to_numpy()
    array["id"] = df["id"].to_numpy()
    array["partitioningKey"] = (
        df["partitioningKey"].cast(pl.Date).to_numpy()
    )
    array["ratings"] = np.array(
        df["ratings"].to_list(), dtype=(np.uint16, 2)
    )
    array["endgameStatus"] = df["endgameStatus"].to_numpy()
    array["moves"] = df["moves"].to_numpy()

    return array


def convert_preprocessing_df_to_numpy(
    df: "pl.DataFrame",
) -> np.ndarray:
    """Convert preprocessing Polars DataFrame to numpy structured array．

    Args:
        df: Polars DataFrame with preprocessing schema

    Returns:
        numpy.ndarray: Structured array with preprocessing dtype

    Raises:
        ImportError: If polars is not installed
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    # Get target dtype
    dtype = get_preprocessing_dtype()
    array = np.empty(len(df), dtype=dtype)

    # Convert fields
    array["id"] = df["id"].to_numpy()
    array["boardIdPositions"] = np.array(
        df["boardIdPositions"].to_list(),
        dtype=(np.uint8, (9, 9)),
    )
    array["piecesInHand"] = np.array(
        df["piecesInHand"].to_list(), dtype=(np.uint8, 14)
    )
    array["moveLabel"] = np.array(
        df["moveLabel"].to_list(),
        dtype=(np.float32, MOVE_LABELS_NUM),
    )
    array["resultValue"] = df["resultValue"].to_numpy()

    return array


def convert_numpy_to_preprocessing_df(
    array: np.ndarray,
) -> "pl.DataFrame":
    """Convert numpy structured array to preprocessing Polars DataFrame．

    Args:
        array: numpy structured array with preprocessing dtype

    Returns:
        pl.DataFrame: Polars DataFrame with preprocessing schema

    Raises:
        ImportError: If polars is not installed
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is not installed. Install with: poetry add polars"
        )

    import polars as pl

    # Convert numpy structured array to dict
    data = {
        "id": array["id"].tolist(),
        "boardIdPositions": array["boardIdPositions"].tolist(),
        "piecesInHand": array["piecesInHand"].tolist(),
        "moveLabel": array["moveLabel"].tolist(),
        "resultValue": array["resultValue"].tolist(),
    }

    # Create DataFrame with proper schema
    df = pl.DataFrame(
        data, schema=get_preprocessing_polars_schema()
    )

    return df
