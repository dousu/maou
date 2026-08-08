"""Polars DataFrame-based DataSource for PyTorch Dataset．

このモジュールは，Polars DataFrameをPyTorch Datasetで利用可能な形式に
変換するDataSourceラッパーを提供する．
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import polars as pl

from maou.domain.data.polars_tensor import (
    polars_row_to_hcpe_arrays,
)

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
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ],
    ):
        """Initialize Polars DataFrame source．

        Args:
            dataframe: Polars DataFrame with appropriate schema
            array_type: Type of data ("hcpe", "preprocessing", "stage1", "stage2")
        """
        self.dataframe = dataframe
        self.array_type = array_type
        self._length = len(dataframe)
        # 列名 → タプル位置の対応を一度だけ作る．位置を直接
        # 埋め込むと，スキーマに列が増減した時点で黙って別の列を
        # 読むようになる (実際 preprocessing で発生していた)．
        self._col_idx: dict[str, int] = {
            name: i for i, name in enumerate(dataframe.columns)
        }

        logger.info(
            f"PolarsDataFrameSource initialized: {self._length} samples, "
            f"type={array_type}"
        )

    def __len__(self) -> int:
        return self._length

    def _row_by_names(
        self,
        row_tuple: tuple[Any, ...],
        *,
        required: tuple[str, ...],
        optional: tuple[str, ...] = (),
    ) -> _PolarsRow:
        """列名で行タプルから必要なフィールドだけを取り出す．

        Args:
            row_tuple: ``DataFrame.row(idx)`` が返す位置つきタプル
            required: 必須列名．欠けていれば ``KeyError``
            optional: 存在すれば含める列名

        Returns:
            _PolarsRow: numpy構造化配列風のアクセスを提供するラッパー

        Raises:
            KeyError: 必須列がDataFrameに存在しない場合
        """
        data: dict[str, Any] = {}
        for name in required:
            if name not in self._col_idx:
                raise KeyError(
                    f"DataFrame for array_type={self.array_type!r} "
                    f"lacks required column {name!r}; "
                    f"got {list(self._col_idx)}"
                )
            data[name] = row_tuple[self._col_idx[name]]
        for name in optional:
            if name in self._col_idx:
                data[name] = row_tuple[self._col_idx[name]]
        return _PolarsRow(data)

    def __getitem__(self, idx: int) -> _PolarsRow:
        """Get a single row as numpy-compatible format．

        For preprocessing/stage1/stage2 data，returns a dict that mimics
        numpy structured array field access．

        For HCPE data，returns a dict with the required fields．

        Args:
            idx: Row index

        Returns:
            _PolarsRow mimicking numpy structured array access
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(
                f"Index {idx} out of range [0, {self._length})"
            )

        # Get row as tuple (faster than named dict)
        row_tuple = self.dataframe.row(idx)

        if self.array_type == "hcpe":
            # HCPE data: Return dict with fields needed by Transform
            hcp, best_move16, game_result, eval_value = (
                polars_row_to_hcpe_arrays(
                    row_tuple, from_dict=False
                )
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
            # get_preprocessing_polars_schema() を参照．moveWinRate /
            # bestMoveWinRate は任意で，存在すれば KifDataset の
            # 4要素タプル経路が有効になる．
            return self._row_by_names(
                row_tuple,
                required=(
                    "id",
                    "boardIdPositions",
                    "piecesInHand",
                    "moveLabel",
                    "resultValue",
                ),
                optional=("moveWinRate", "bestMoveWinRate"),
            )

        elif self.array_type == "stage1":
            # get_stage1_polars_schema() を参照．
            return self._row_by_names(
                row_tuple,
                required=(
                    "id",
                    "boardIdPositions",
                    "piecesInHand",
                    "reachableSquares",
                ),
            )

        elif self.array_type == "stage2":
            # get_stage2_polars_schema() を参照．
            return self._row_by_names(
                row_tuple,
                required=(
                    "id",
                    "boardIdPositions",
                    "piecesInHand",
                    "legalMovesLabel",
                ),
            )

        else:
            raise ValueError(
                f"Unsupported array_type: {self.array_type}"
            )


class _PolarsRow:
    """Wrapper class that mimics numpy structured array field access．

    This allows Polars DataFrame rows to be used with existing Dataset code
    that expects numpy structured arrays．
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.dtype = _FakeDtype(list(data.keys()))

    def __getitem__(self, key: str) -> _PolarsField:
        """Get field value．

        Returns a _PolarsField wrapper that provides .item() method
        for scalar values．
        """
        value = self._data[key]
        return _PolarsField(value)

    def __repr__(self) -> str:
        return f"_PolarsRow({self._data})"


class _PolarsField:
    """Wrapper for Polars field values that mimics numpy array/scalar behavior．"""

    def __init__(self, value: Any) -> None:
        self._value = value
        # Convert Polars list to numpy array for tensor conversion
        if isinstance(value, list):
            # Infer dtype from field type
            # For board/pieces: uint8, for moveLabel: float32
            if value and isinstance(value[0], list):
                # Nested list (e.g., boardIdPositions)
                self._array: np.ndarray[Any, Any] | None = (
                    np.array(value, dtype=np.uint8)
                )
            elif value and isinstance(value[0], float):
                # Float list (e.g., moveLabel)
                self._array = np.array(value, dtype=np.float32)
            else:
                # Integer list (e.g., piecesInHand)
                self._array = np.array(value, dtype=np.uint8)
        else:
            # Scalar value
            self._array = None

    def item(self) -> Any:
        """Return scalar value (mimics numpy scalar .item() method)．"""
        if self._array is not None:
            raise ValueError(
                "Cannot call .item() on array field"
            )
        return self._value

    def tolist(self) -> list[Any]:
        """Convert to list (for array fields)．"""
        if self._array is not None:
            return self._array.tolist()
        return [self._value]

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return dtype (mimics numpy array)．"""
        if self._array is not None:
            return self._array.dtype
        # Return dtype for scalar
        if isinstance(self._value, int):
            return np.dtype("int64")
        elif isinstance(self._value, float):
            return np.dtype("float64")
        return np.dtype("object")

    @property
    def flags(self) -> Any:
        """Return flags (mimics numpy array)．"""
        if self._array is not None:
            return self._array.flags
        # For scalars, create fake flags
        return type(
            "FakeFlags",
            (),
            {
                "c_contiguous": True,
                "writeable": True,
            },
        )()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape (mimics numpy array)．"""
        if self._array is not None:
            return self._array.shape
        return ()

    def __array__(self) -> np.ndarray[Any, Any]:
        """Return numpy array (allows np.asarray() to work)．"""
        if self._array is not None:
            return self._array
        return np.array(self._value)

    def __repr__(self) -> str:
        if self._array is not None:
            return repr(self._array)
        return repr(self._value)


class _FakeDtype:
    """Fake dtype object that mimics numpy dtype.names．"""

    def __init__(self, names: list[str]):
        self.names = names
