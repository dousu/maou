"""Polars DataFrame-based DataSource for PyTorch Dataset．

このモジュールは，Polars DataFrameをPyTorch Datasetで利用可能な形式に
変換するDataSourceラッパーを提供する．
"""

import logging
from typing import Literal, Union

import polars as pl
import numpy as np

from maou.domain.data.polars_tensor import polars_row_to_hcpe_arrays

logger = logging.getLogger(__name__)


class PolarsDataFrameSource:
    """DataSource wrapper for Polars DataFrames．

    Provides __getitem__ interface compatible with existing Dataset classes，
    but internally uses Polars DataFrame for efficient data access．
    """

    def __init__(
        self,
        *,
        dataframe: pl.DataFrame,
        array_type: Literal["hcpe", "preprocessing", "stage1", "stage2"],
    ):
        """Initialize Polars DataFrame source．

        Args:
            dataframe: Polars DataFrame with appropriate schema
            array_type: Type of data ("hcpe", "preprocessing", "stage1", "stage2")
        """
        self.dataframe = dataframe
        self.array_type = array_type
        self._length = len(dataframe)

        logger.info(
            f"PolarsDataFrameSource initialized: {self._length} samples, "
            f"type={array_type}"
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Union[np.ndarray, dict]:
        """Get a single row as numpy-compatible format．

        For preprocessing/stage1/stage2 data，returns a dict that mimics
        numpy structured array field access．

        For HCPE data，returns a dict with the required fields．

        Args:
            idx: Row index

        Returns:
            Dict mimicking numpy structured array access
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")

        # Get row as tuple (faster than named dict)
        row_tuple = self.dataframe.row(idx)

        if self.array_type == "hcpe":
            # HCPE data: Return dict with fields needed by Transform
            hcp, best_move16, game_result, eval_value = polars_row_to_hcpe_arrays(
                row_tuple, from_dict=False
            )

            # Create a dict that mimics numpy structured array
            return _PolarsRow(
                {
                    "hcp": hcp,
                    "bestMove16": best_move16,
                    "gameResult": game_result,
                    "eval": eval_value,
                }
            )

        elif self.array_type == "preprocessing":
            # Preprocessing schema: id, boardIdPositions, piecesInHand, moveLabel, resultValue
            return _PolarsRow(
                {
                    "id": row_tuple[0],
                    "boardIdPositions": row_tuple[1],
                    "piecesInHand": row_tuple[2],
                    "moveLabel": row_tuple[3],
                    "resultValue": row_tuple[4],
                }
            )

        elif self.array_type == "stage1":
            # Stage1 schema: id, boardIdPositions, piecesInHand, reachableSquares
            return _PolarsRow(
                {
                    "id": row_tuple[0],
                    "boardIdPositions": row_tuple[1],
                    "piecesInHand": row_tuple[2],
                    "reachableSquares": row_tuple[3],
                }
            )

        elif self.array_type == "stage2":
            # Stage2 schema: id, boardIdPositions, piecesInHand, legalMovesLabel
            return _PolarsRow(
                {
                    "id": row_tuple[0],
                    "boardIdPositions": row_tuple[1],
                    "piecesInHand": row_tuple[2],
                    "legalMovesLabel": row_tuple[3],
                }
            )

        else:
            raise ValueError(f"Unsupported array_type: {self.array_type}")


class _PolarsRow:
    """Wrapper class that mimics numpy structured array field access．

    This allows Polars DataFrame rows to be used with existing Dataset code
    that expects numpy structured arrays．
    """

    def __init__(self, data: dict):
        self._data = data
        self.dtype = _FakeDtype(list(data.keys()))

    def __getitem__(self, key: str):
        """Get field value．

        Returns a _PolarsField wrapper that provides .item() method
        for scalar values．
        """
        value = self._data[key]
        return _PolarsField(value)

    def __repr__(self):
        return f"_PolarsRow({self._data})"


class _PolarsField:
    """Wrapper for Polars field values that mimics numpy array/scalar behavior．"""

    def __init__(self, value):
        self._value = value
        # Convert Polars list to numpy array for tensor conversion
        if isinstance(value, list):
            # Infer dtype from field type
            # For board/pieces: uint8, for moveLabel: float32
            if value and isinstance(value[0], list):
                # Nested list (e.g., boardIdPositions)
                self._array = np.array(value, dtype=np.uint8)
            elif value and isinstance(value[0], float):
                # Float list (e.g., moveLabel)
                self._array = np.array(value, dtype=np.float32)
            else:
                # Integer list (e.g., piecesInHand)
                self._array = np.array(value, dtype=np.uint8)
        else:
            # Scalar value
            self._array = None

    def item(self):
        """Return scalar value (mimics numpy scalar .item() method)．"""
        if self._array is not None:
            raise ValueError("Cannot call .item() on array field")
        return self._value

    def tolist(self):
        """Convert to list (for array fields)．"""
        if self._array is not None:
            return self._array.tolist()
        return [self._value]

    @property
    def dtype(self):
        """Return dtype (mimics numpy array)．"""
        if self._array is not None:
            return self._array.dtype
        # Return dtype for scalar
        if isinstance(self._value, int):
            return np.dtype('int64')
        elif isinstance(self._value, float):
            return np.dtype('float64')
        return np.dtype('object')

    @property
    def flags(self):
        """Return flags (mimics numpy array)．"""
        if self._array is not None:
            return self._array.flags
        # For scalars, create fake flags
        return type('FakeFlags', (), {
            'c_contiguous': True,
            'writeable': True,
        })()

    @property
    def shape(self):
        """Return shape (mimics numpy array)．"""
        if self._array is not None:
            return self._array.shape
        return ()

    def __array__(self):
        """Return numpy array (allows np.asarray() to work)．"""
        if self._array is not None:
            return self._array
        return np.array(self._value)

    def __repr__(self):
        if self._array is not None:
            return repr(self._array)
        return repr(self._value)


class _FakeDtype:
    """Fake dtype object that mimics numpy dtype.names．"""

    def __init__(self, names: list[str]):
        self.names = names
