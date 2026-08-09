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


def _leaf_numpy_dtype(
    dtype: Any,
) -> np.dtype[Any] | None:
    """Polars dtype から対応する numpy dtype を導出する．

    ``pl.List`` / ``pl.Array`` は入れ子を剥がして最内の型を見る
    (``List(List(UInt8))`` → ``uint8``)．

    変換表を書き下さず polars 自身に空 Series を numpy 化させて
    問い合わせるので，polars の型が増えても追従不要である．

    Args:
        dtype: 列の Polars dtype

    Returns:
        対応する numpy dtype．判定できない型では ``None``
        (呼び出し側は値からの推測にフォールバックする)．
    """
    leaf = dtype
    while isinstance(leaf, (pl.List, pl.Array)):
        leaf = leaf.inner
    try:
        return np.dtype(
            pl.Series("", [], dtype=leaf).to_numpy().dtype
        )
    except Exception:  # pragma: no cover - 未知のPolars型
        logger.debug(
            "Could not derive numpy dtype from %r", leaf
        )
        return None


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
        # 列名 → numpy dtype をスキーマから導出しておく．値の
        # Python 型から推測すると，例えば List(UInt16) の列が
        # uint8 として作られて値が 256 で折り返し，しかも
        # KifDataset 側の dtype ガードは (期待どおりの uint8 な
        # ので) 素通りする — 静かなデータ破壊になる．
        self._np_dtypes: dict[str, np.dtype[Any] | None] = {
            name: _leaf_numpy_dtype(dtype)
            for name, dtype in dataframe.schema.items()
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
        return _PolarsRow(
            data,
            np_dtypes={
                name: self._np_dtypes.get(name) for name in data
            },
        )

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

    def __init__(
        self,
        data: dict[str, Any],
        *,
        np_dtypes: dict[str, np.dtype[Any] | None]
        | None = None,
    ) -> None:
        """Initialize the row wrapper．

        Args:
            data: 列名 → 値
            np_dtypes: 列名 → スキーマ由来の numpy dtype．
                省略した列は値から推測される．
        """
        self._data = data
        self._np_dtypes = np_dtypes or {}
        self.dtype = _FakeDtype(list(data.keys()))

    def __getitem__(self, key: str) -> _PolarsField:
        """Get field value．

        Returns a _PolarsField wrapper that provides .item() method
        for scalar values．
        """
        value = self._data[key]
        return _PolarsField(
            value, np_dtype=self._np_dtypes.get(key)
        )

    def __repr__(self) -> str:
        return f"_PolarsRow({self._data})"


class _PolarsField:
    """Wrapper for Polars field values that mimics numpy array/scalar behavior．"""

    def __init__(
        self,
        value: Any,
        *,
        np_dtype: np.dtype[Any] | None = None,
    ) -> None:
        """Wrap one field value．

        Args:
            value: 列の値 (リストまたはスカラ)
            np_dtype: スキーマ由来の numpy dtype．``None`` の
                ときだけ値の形から推測する (HCPE 経路のように
                既に numpy 化された値を渡す場合)．
        """
        self._value = value
        self._np_dtype = np_dtype
        self._array: np.ndarray[Any, Any] | None = None
        # Convert Polars list to numpy array for tensor conversion
        if isinstance(value, list):
            if np_dtype is not None:
                # スキーマが正．入れ子リストでもそのまま渡せる．
                self._array = np.array(value, dtype=np_dtype)
            elif value and isinstance(value[0], list):
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
        if self._np_dtype is not None:
            return self._np_dtype
        # Return dtype for scalar
        if isinstance(self._value, int):
            return np.dtype("int64")
        elif isinstance(self._value, float):
            return np.dtype("float64")
        return np.dtype("object")

    # ``flags`` は意図的に実装しない．KifDataset の zero-copy
    # ガード (dataset.py の ``_numpy_to_tensor``) は
    # ``np.asarray(field)`` で本物の numpy 配列を作ってから
    # その ``.flags`` を見るので，このラッパの ``flags`` は
    # 一度も参照されない．以前ここにあった ``FakeFlags``
    # (c_contiguous/writeable を無条件に True と主張するもの) は
    # ガードを黙らせるために必要だと記録されていたが，実際には
    # 誰も読まない死んだコードだった．

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
