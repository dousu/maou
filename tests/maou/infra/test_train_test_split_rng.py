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

import logging
import random
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from maou.domain.data.rust_io import save_preprocessing_df
from maou.domain.data.schema import (
    create_empty_preprocessing_df,
)
from maou.infra.bigquery.bq_data_source import (
    BigQueryDataSource,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.infra.object_storage.data_source import (
    ObjectStorageDataSource,
)
from maou.interface import learn

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


_PUBLIC_SPLITTERS: dict[str, tuple[Callable[..., Any], str]] = {
    "bigquery": (
        BigQueryDataSource.BigQueryDataSourceSpliter.train_test_split,
        "_BigQueryDataSourceSpliter",
    ),
    "object_storage": (
        ObjectStorageDataSource.DataSourceSpliter.train_test_split,
        "_DataSourceSpliter",
    ),
    "file_system": (
        FileDataSource.FileDataSourceSpliter.train_test_split,
        "_FileDataSourceSpliter",
    ),
}


class _StubManager:
    total_rows = 20


@pytest.mark.parametrize("name", sorted(_PUBLIC_SPLITTERS))
def test_public_split_passes_the_default_seed(
    name: str,
) -> None:
    """公開 ``train_test_split`` が既定 seed を渡すこと.

    `/audit-backlog` 2026-08-13 / backlog 行 D2．
    ``__train_test_split`` は seed 引数を持っていたのに 3 実装とも
    渡しておらず，分割が実行ごとに変わっていた (学習を中断・再開すると
    前回の検証行で訓練し得る)．3 実装が同じ定数を引くことを固定する．
    """
    split, mangle_prefix = _PUBLIC_SPLITTERS[name]

    seen: dict[str, Any] = {}

    def spy(**kwargs: Any) -> tuple[list[int], list[int]]:
        seen.update(kwargs)
        data = kwargs["data"]
        return data[:15], data[15:]

    stub = types.SimpleNamespace(
        logger=logging.getLogger(__name__),
        **{
            f"{mangle_prefix}__train_test_split": spy,
            f"{mangle_prefix}__file_manager": _StubManager(),
            f"{mangle_prefix}__page_manager": _StubManager(),
        },
    )

    split(stub, test_ratio=0.25)

    assert seen["seed"] == learn.DEFAULT_SPLIT_SEED
    assert seen["seed"] is not None


def test_file_system_split_is_stable_across_instances(
    tmp_path: Path,
) -> None:
    """同じ入力なら別インスタンスでも同じ分割になること.

    `/audit-backlog` 2026-08-13 / backlog 行 D2 の実経路側の確認．
    spy を使わず本物のファイルで，再開に相当する 2 回目の構築が
    1 回目と同じ train/test を返すことを押さえる．
    """
    df = create_empty_preprocessing_df(20)
    path = tmp_path / "split.feather"
    save_preprocessing_df(df, str(path))

    def split_once() -> tuple[list[int], list[int]]:
        spliter = FileDataSource.FileDataSourceSpliter(
            file_paths=[path],
            array_type="preprocessing",
        )
        train, test = spliter.train_test_split(test_ratio=0.25)
        return (
            list(train.indicies),
            list(test.indicies),
        )

    assert split_once() == split_once()
