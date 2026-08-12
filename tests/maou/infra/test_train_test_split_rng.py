"""``__train_test_split`` がグローバル RNG を汚さないことの回帰テスト．

`/audit-backlog` 2026-08-12 / backlog 行 O2．

``object_storage`` と ``bigquery`` の複製 2 件が ``random.seed(seed)`` で
モジュールグローバルの Mersenne Twister を差し替えており，同一プロセス内の
無関係な乱数消費者まで巻き添えにしていた．``infra/file_system`` 側だけが
`8c1417e` で ``random.Random(seed)`` に直っていて，修正が乖離していた．

3 つの実装は互いに文字単位で同一であるべきなので，同じテストを 3 つとも
に当てる．メソッドは ``self`` を使わないため，クラウドクライアントを
構築せずに未束縛関数として呼べる．
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import pytest

from maou.infra.bigquery.bq_data_source import (
    BigQueryDataSource,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.infra.object_storage.data_source import (
    ObjectStorageDataSource,
)

_SPLITTERS: dict[str, Callable[..., Any]] = {
    "bigquery": (
        BigQueryDataSource.BigQueryDataSourceSpliter._BigQueryDataSourceSpliter__train_test_split
    ),
    "object_storage": (
        ObjectStorageDataSource.DataSourceSpliter._DataSourceSpliter__train_test_split
    ),
    "file_system": (
        FileDataSource.FileDataSourceSpliter._FileDataSourceSpliter__train_test_split
    ),
}


@pytest.mark.parametrize("name", sorted(_SPLITTERS))
def test_split_does_not_disturb_global_rng(name: str) -> None:
    """seed 付きの分割がグローバル RNG の続きを変えないこと."""
    split = _SPLITTERS[name]

    random.seed(12345)
    expected = [random.random() for _ in range(3)]

    random.seed(12345)
    split(None, data=list(range(20)), test_ratio=0.25, seed=7)
    actual = [random.random() for _ in range(3)]

    assert actual == expected


@pytest.mark.parametrize("name", sorted(_SPLITTERS))
def test_split_is_reproducible_for_a_given_seed(
    name: str,
) -> None:
    """同じ seed なら同じ分割になること (置き換えの同値性)."""
    split = _SPLITTERS[name]

    first = split(
        None, data=list(range(20)), test_ratio=0.25, seed=7
    )
    second = split(
        None, data=list(range(20)), test_ratio=0.25, seed=7
    )

    assert first == second


@pytest.mark.parametrize("name", sorted(_SPLITTERS))
def test_split_sizes_and_partition(name: str) -> None:
    """分割が元データの過不足ない分割になっていること."""
    split = _SPLITTERS[name]

    train, test = split(
        None, data=list(range(20)), test_ratio=0.25, seed=7
    )

    assert len(train) == 15
    assert len(test) == 5
    assert sorted(train + test) == list(range(20))
