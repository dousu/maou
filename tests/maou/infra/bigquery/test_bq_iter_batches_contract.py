"""``BigQueryDataSource.iter_batches`` が preprocess 側の契約を守る回帰テスト．

`/audit-backlog` 2026-08-13 / backlog 行 N6-1．

``app.pre_process.hcpe_transform.DataSource`` の宣言は
``Generator[tuple[str, np.ndarray], None, None]`` だが，
``BigQueryDataSource.iter_batches`` は ``pl.DataFrame`` を yield して
いた．``PreProcess`` はこれを ``_process_single_array(data: np.ndarray)``
に渡して ``data["hcp"]`` / ``np.ascontiguousarray`` するので，
``maou pre-process --input-dataset-id … --input-table-name …`` は
最初のバッチで落ちる．

さらに ``iter_batches_df`` を override していなかったため，基底の既定実装
(``hcpe_transform.py:95``) に ``pl.DataFrame`` が渡り
``array.dtype.names`` で ``AttributeError`` になっていた．基底の既定実装は
HCPE スキーマ決め打ちでもあるので (backlog 行 N6-2)，BigQuery 側は
DataFrame をそのまま返す override を持つのが正しい．

``__getitem__`` については同じ形の不具合が O1 として既に直っている
(``test_bq_get_item_contract.py``)．ABC 化 (N3) では捕まらない —
ABC はメソッドの存在を見るが型は見ない．

BigQuery クライアントを構築せずに検証するため，``PageManager`` は
``object.__new__`` で生成し ``get_page`` だけ差し替える (O1 の前例と同じ)．
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from maou.domain.data.schema import (
    create_empty_hcpe_df,
    create_empty_preprocessing_df,
)
from maou.infra.bigquery.bq_data_source import (
    BigQueryDataSource,
)


def _make_page_manager(
    *,
    array_type: str,
    pages: list[pl.DataFrame],
) -> Any:
    """BigQuery に触れずに ``PageManager`` の取り出し部分だけを組み立てる．"""
    pm = object.__new__(BigQueryDataSource.PageManager)
    pm.array_type = array_type
    pm.batch_size = len(pages[0])
    pm.total_rows = sum(len(p) for p in pages)
    pm.total_pages = len(pages)
    pm._PageManager__pruning_info = []
    pm.get_page = lambda page_num: pages[page_num]
    return pm


def _preprocessing_pages(
    rows_per_page: int, page_count: int
) -> list[pl.DataFrame]:
    pages = []
    for page in range(page_count):
        df = create_empty_preprocessing_df(rows_per_page)
        df = df.with_columns(
            pl.Series(
                "id",
                [
                    page * rows_per_page + i
                    for i in range(rows_per_page)
                ],
            )
        )
        pages.append(df)
    return pages


def test_iter_batches_yields_structured_arrays() -> None:
    """``iter_batches`` が numpy structured array を yield すること．

    ``PreProcess._process_single_array`` は ``data["hcp"]`` のように
    フィールド名でアクセスするので，``dtype.names`` を契約として固定する．
    """
    pm = _make_page_manager(
        array_type="preprocessing",
        pages=_preprocessing_pages(3, 2),
    )
    datasource = BigQueryDataSource(page_manager=pm)

    batches = list(datasource.iter_batches())

    assert len(batches) == 2
    for name, array in batches:
        assert isinstance(name, str)
        assert not isinstance(array, pl.DataFrame)
        assert isinstance(array, np.ndarray)
        assert array.dtype.names is not None
        assert "boardIdPositions" in array.dtype.names
        assert len(array) == 3


def test_iter_batches_hcpe_uses_the_hcpe_converter() -> None:
    """array_type='hcpe' では HCPE の構造体になること．"""
    pm = _make_page_manager(
        array_type="hcpe",
        pages=[create_empty_hcpe_df(4)],
    )
    datasource = BigQueryDataSource(page_manager=pm)

    ((_, array),) = list(datasource.iter_batches())

    assert array.dtype.names is not None
    assert "hcp" in array.dtype.names
    assert len(array) == 4


def test_iter_batches_df_yields_dataframes() -> None:
    """``iter_batches_df`` が DataFrame をそのまま返すこと．

    override していないと基底の既定実装
    (``hcpe_transform.iter_batches_df``) が走り，``array.dtype.names``
    で ``AttributeError`` になる．さらに既定実装は HCPE スキーマ決め打ち
    なので preprocessing では黙って誤ったスキーマを当てる．
    """
    pages = _preprocessing_pages(3, 2)
    pm = _make_page_manager(
        array_type="preprocessing", pages=pages
    )
    datasource = BigQueryDataSource(page_manager=pm)

    batches = list(datasource.iter_batches_df())

    assert len(batches) == 2
    for (name, df), expected in zip(batches, pages):
        assert isinstance(name, str)
        assert isinstance(df, pl.DataFrame)
        assert df.schema == expected.schema
        assert df.equals(expected)


def test_iter_batches_preserves_row_count_and_order() -> None:
    """行の順序と件数が page 順のまま保たれること."""
    pages = _preprocessing_pages(2, 3)
    pm = _make_page_manager(
        array_type="preprocessing", pages=pages
    )
    datasource = BigQueryDataSource(page_manager=pm)

    ids = [
        int(row["id"])
        for _, array in datasource.iter_batches()
        for row in array
    ]

    assert ids == [0, 1, 2, 3, 4, 5]


def test_iter_batches_rejects_unknown_array_type() -> None:
    """未知の array_type は明示的に落ちること (黙って DataFrame を返さない)．"""
    pm = _make_page_manager(
        array_type="nonsense",
        pages=_preprocessing_pages(1, 1),
    )
    datasource = BigQueryDataSource(page_manager=pm)

    with pytest.raises(ValueError, match="array_type"):
        list(datasource.iter_batches())
