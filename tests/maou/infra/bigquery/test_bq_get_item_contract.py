"""``BigQueryDataSource`` が learn 側の DataSource 契約を守ることの回帰テスト．

`/audit-backlog` 2026-08-12 / backlog 行 O1．

``PageManager.get_item`` は ``pl.DataFrame`` の 1 行を返していたが，
``app.learning.dataset.DataSource`` の契約は 0 次元の numpy structured
array である．そのため ``KifDataset.__getitem__`` が ``data.dtype.names``
を読む時点で ``AttributeError`` になり，
``maou utility benchmark-dataloader --input-dataset-id … --input-table-name …``
がバッチ 0 から落ちていた．

構築時に捕まらなかったのは ``dataset.py:45`` が ``@abc.abstractmethod`` を
付けながら ``abc.ABCMeta`` を使っておらず，マーカーが不活性だったため．

BigQuery クライアントを構築せずに検証するため，``PageManager`` は
``object.__new__`` で生成し ``get_page`` だけ差し替える．
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from maou.app.learning.dataset import KifDataset
from maou.domain.data.schema import (
    create_empty_preprocessing_df,
)
from maou.infra.bigquery.bq_data_source import (
    BigQueryDataSource,
)


def _make_page_manager(rows: int) -> Any:
    """BigQuery に触れずに ``PageManager`` の取り出し部分だけを組み立てる．"""
    df = create_empty_preprocessing_df(rows)
    df = df.with_columns(
        [
            pl.Series("id", list(range(rows))),
            pl.Series(
                "resultValue",
                [float(i) for i in range(rows)],
            ),
        ]
    )

    pm = object.__new__(BigQueryDataSource.PageManager)
    pm.array_type = "preprocessing"
    pm.batch_size = rows
    pm.total_rows = rows
    pm.total_pages = 1
    pm._PageManager__pruning_info = []
    pm.get_page = lambda page_num: df
    return pm


def test_get_item_returns_structured_record() -> None:
    """``get_item`` が名前付きフィールドを持つレコードを返すこと．

    structured array の 1 要素なので実体は ``np.void`` になる
    (``FileDataSource`` が返すものと同じ形)．``KifDataset`` が読むのは
    ``dtype.names`` なので，そこを契約として固定する．
    """
    pm = _make_page_manager(4)
    datasource = BigQueryDataSource(page_manager=pm)

    record = datasource[2]

    assert not isinstance(record, pl.DataFrame)
    assert isinstance(record, np.void)
    assert record.dtype.names is not None
    assert "boardIdPositions" in record.dtype.names


def test_get_item_maps_index_to_the_right_row() -> None:
    """インデックスが対応する行に写ること．"""
    pm = _make_page_manager(4)
    datasource = BigQueryDataSource(page_manager=pm)

    for idx in range(4):
        assert float(datasource[idx]["resultValue"]) == float(
            idx
        )


def test_kif_dataset_can_consume_bigquery_records() -> None:
    """``KifDataset`` が BigQuery 由来のレコードを読めること．

    これが O1 の再現経路そのもの — 以前はここで
    ``AttributeError: 'DataFrame' object has no attribute 'dtype'``
    になっていた．
    """
    pm = _make_page_manager(3)
    datasource = BigQueryDataSource(page_manager=pm)

    dataset = KifDataset(datasource=datasource)

    assert len(dataset) == 3
    (inputs, targets) = dataset[0]
    assert len(inputs) == 2
    assert len(targets) >= 1


def test_get_item_rejects_out_of_range_index() -> None:
    """範囲外インデックスは IndexError で落ちること．"""
    pm = _make_page_manager(3)
    datasource = BigQueryDataSource(page_manager=pm)

    with pytest.raises(IndexError):
        datasource[3]
