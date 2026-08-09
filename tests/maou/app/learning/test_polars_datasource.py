"""PolarsDataFrameSource のテスト．

``PolarsDataFrameSource`` は production からは呼ばれていないが，
``docs/rust-backend.md`` が Polars DataFrame を ``KifDataset`` に
直接食わせる入口として文書化している公開 API である．
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from maou.app.learning.dataset import KifDataset
from maou.app.learning.polars_datasource import (
    PolarsDataFrameSource,
    _leaf_numpy_dtype,
    _PolarsField,
)


def _preprocessing_schema(
    *,
    pieces_dtype: pl.DataType | None = None,
    with_win_rate: bool = False,
) -> dict[str, pl.DataType]:
    """テスト用の小さな preprocessing スキーマを作る．

    Args:
        pieces_dtype: ``piecesInHand`` の dtype を差し替える
            (dtype 推測の罠を再現するため)
        with_win_rate: ``moveWinRate`` 列を含めるか

    Returns:
        列名 → Polars dtype
    """
    schema: dict[str, pl.DataType] = {
        "id": pl.UInt64(),
        "boardIdPositions": pl.List(pl.List(pl.UInt8)),
        "piecesInHand": pieces_dtype or pl.List(pl.UInt8),
        "moveLabel": pl.List(pl.Float32()),
        "resultValue": pl.Float32(),
    }
    if with_win_rate:
        schema["moveWinRate"] = pl.List(pl.Float32())
    return schema


def _preprocessing_df(
    *,
    pieces: list[int] | None = None,
    pieces_dtype: pl.DataType | None = None,
    with_win_rate: bool = False,
) -> pl.DataFrame:
    """1 行だけの preprocessing DataFrame を作る．"""
    row: dict[str, object] = {
        "id": [1],
        "boardIdPositions": [[[1, 2], [3, 4]]],
        "piecesInHand": [pieces if pieces else [0, 1, 2, 3]],
        "moveLabel": [[0.5, 0.25, 0.25]],
        "resultValue": [1.0],
    }
    if with_win_rate:
        row["moveWinRate"] = [[0.8, 0.6, 0.1]]
    return pl.DataFrame(
        row,
        schema=_preprocessing_schema(
            pieces_dtype=pieces_dtype,
            with_win_rate=with_win_rate,
        ),
    )


class TestLeafNumpyDtype:
    """``_leaf_numpy_dtype`` のテスト．"""

    @pytest.mark.parametrize(
        ("polars_dtype", "expected"),
        [
            (pl.UInt8(), np.uint8),
            (pl.UInt16(), np.uint16),
            (pl.Int32(), np.int32),
            (pl.Float32(), np.float32),
            (pl.Float64(), np.float64),
            (pl.List(pl.UInt8), np.uint8),
            (pl.List(pl.List(pl.UInt8)), np.uint8),
            (pl.List(pl.Float32()), np.float32),
        ],
    )
    def test_derives_leaf_dtype(
        self,
        polars_dtype: pl.DataType,
        expected: type,
    ) -> None:
        """入れ子 List を剥がして最内の numpy dtype を返す．"""
        assert _leaf_numpy_dtype(polars_dtype) == np.dtype(
            expected
        )


class TestSchemaDrivenDtypes:
    """スキーマ由来の dtype が値の推測より優先されること．"""

    def test_array_fields_follow_schema(self) -> None:
        """配列列の dtype はスキーマどおりになる．"""
        source = PolarsDataFrameSource(
            dataframe=_preprocessing_df(),
            array_type="preprocessing",
        )
        row = source[0]

        assert (
            np.asarray(row["boardIdPositions"]).dtype
            == np.uint8
        )
        assert np.asarray(row["piecesInHand"]).dtype == np.uint8
        assert np.asarray(row["moveLabel"]).dtype == np.float32

    def test_wide_integer_column_is_not_truncated(
        self,
    ) -> None:
        """これがこの修正が塞いだ罠である．

        以前は「値が list で先頭が int なら uint8」と決め打ち
        していたため，``List(UInt16)`` の列は 256 で折り返して
        いた．しかも ``KifDataset`` の dtype ガードは期待どおり
        の uint8 を受け取るので何も言わない — 静かなデータ破壊
        になる．
        """
        source = PolarsDataFrameSource(
            dataframe=_preprocessing_df(
                pieces=[300, 1, 2, 3],
                pieces_dtype=pl.List(pl.UInt16()),
            ),
            array_type="preprocessing",
        )
        array = np.asarray(source[0]["piecesInHand"])

        assert array.dtype == np.uint16
        # 素朴な uint8 推測では 300 は 44 になっていた．
        assert array[0] == 300

    def test_scalar_dtype_follows_schema(self) -> None:
        """スカラ列の dtype もスキーマどおりになる．

        Python の float から推測すると float64 になるが，
        スキーマ上は Float32 である．
        """
        source = PolarsDataFrameSource(
            dataframe=_preprocessing_df(),
            array_type="preprocessing",
        )
        assert source[0]["resultValue"].dtype == np.float32

    def test_falls_back_to_value_shape_without_schema(
        self,
    ) -> None:
        """dtype 未指定なら従来どおり値の形から推測する．

        HCPE 経路は既に numpy 化された値を渡すので，この
        フォールバックが残っている必要がある．
        """
        field = _PolarsField([[1, 2], [3, 4]])
        assert np.asarray(field).dtype == np.uint8

        field = _PolarsField([0.5, 0.25])
        assert np.asarray(field).dtype == np.float32


class TestDeadNumpyMimicry:
    """使われていない numpy 模倣を持たないこと．"""

    def test_flags_is_not_implemented(self) -> None:
        """``flags`` は実装しない．

        ``KifDataset._numpy_to_tensor`` は ``np.asarray()`` で
        本物の numpy 配列を作ってからその ``.flags`` を見るので，
        このラッパの ``flags`` は一度も参照されない．以前ここに
        あった ``FakeFlags`` は「zero-copy ガードを通すために
        必要」と記録されていたが，実際は誰も読まない死んだ
        コードだった．
        """
        field = _PolarsField(1.0)
        assert not hasattr(field, "flags")

    def test_guards_still_see_real_numpy_flags(self) -> None:
        """ガードは本物の numpy 配列の flags を見る．"""
        source = PolarsDataFrameSource(
            dataframe=_preprocessing_df(),
            array_type="preprocessing",
        )
        array = np.asarray(source[0]["boardIdPositions"])

        assert array.flags.c_contiguous
        assert array.flags.writeable


class TestKifDatasetIntegration:
    """``docs/rust-backend.md`` が文書化している経路．"""

    def test_targets_omit_legal_move_mask(self) -> None:
        """targets は (move_label, result_value) の 2 要素．"""
        source = PolarsDataFrameSource(
            dataframe=_preprocessing_df(),
            array_type="preprocessing",
        )
        dataset = KifDataset(datasource=source)

        (board, pieces), targets = dataset[0]

        assert board.dtype == torch.uint8
        assert board.shape == (2, 2)
        assert pieces.dtype == torch.uint8
        assert len(targets) == 2
        move_label, result_value = targets
        assert move_label.dtype == torch.float32
        assert result_value.shape == (1,)

    def test_targets_include_move_win_rate(self) -> None:
        """moveWinRate があれば 3 要素目に入る．"""
        source = PolarsDataFrameSource(
            dataframe=_preprocessing_df(with_win_rate=True),
            array_type="preprocessing",
        )
        dataset = KifDataset(datasource=source)

        (_, _), targets = dataset[0]

        assert len(targets) == 3
        assert targets[2] is not None
        assert targets[2].dtype == torch.float32
