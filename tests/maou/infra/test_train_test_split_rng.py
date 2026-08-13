"""行単位 train/test 分割の回帰テスト．

`/audit-backlog` 2026-08-12 backlog 行 O2 / 2026-08-13 backlog 行 D2 /
2026-08-13 backlog 行 D8+D9 の 3 世代ぶんを 1 ファイルに束ねてある．

- **O2**: ``object_storage`` と ``bigquery`` の複製 2 件が
  ``random.seed(seed)`` でモジュールグローバルの Mersenne Twister を
  差し替えており，同一プロセス内の無関係な乱数消費者まで巻き添えに
  していた．
- **D2**: ``__train_test_split`` は seed 引数を持っていたのに 3 実装とも
  渡しておらず，分割が実行ごとに変わっていた．
- **D8+D9**: 3 実装が持っていた ``list(range(N))`` は索引だけで
  5000万行 ≒ 2.6GB を食う．``np.random.Generator.permutation`` へ
  切り替え，同時に 3 つの複製を
  :func:`maou.interface.learn.train_test_split_indices` へ 1 本化した．
"""

from __future__ import annotations

import logging
import random
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
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

# ============================================================================
# 共有ヘルパそのものの性質
# ============================================================================


class TestSplitIndices:
    """``learn.train_test_split_indices`` の性質."""

    def test_does_not_disturb_the_global_random_module(
        self,
    ) -> None:
        """``random`` のグローバル状態を汚さないこと (O2 の回帰)."""
        random.seed(12345)
        expected = [random.random() for _ in range(3)]

        random.seed(12345)
        learn.train_test_split_indices(
            total_rows=20, test_ratio=0.25, seed=7
        )
        actual = [random.random() for _ in range(3)]

        assert actual == expected

    def test_does_not_disturb_the_global_numpy_rng(
        self,
    ) -> None:
        """numpy のグローバル状態も汚さないこと.

        **trap**: 素朴な移植先は ``np.random.permutation`` だが，これは
        レガシーのグローバル ``RandomState`` を消費するので，O2 で直した
        「無関係な乱数消費者を巻き添えにする」問題が numpy 側で復活する．
        ``default_rng(seed)`` は独立した Generator を作るので汚さない．
        """
        np.random.seed(12345)
        expected = [float(np.random.random()) for _ in range(3)]

        np.random.seed(12345)
        learn.train_test_split_indices(
            total_rows=20, test_ratio=0.25, seed=7
        )
        actual = [float(np.random.random()) for _ in range(3)]

        assert actual == expected

    def test_is_reproducible_for_a_given_seed(self) -> None:
        """同じ seed なら同じ分割になること (D2 の中身)."""
        first = learn.train_test_split_indices(
            total_rows=20, test_ratio=0.25, seed=7
        )
        second = learn.train_test_split_indices(
            total_rows=20, test_ratio=0.25, seed=7
        )

        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])

    def test_different_seeds_give_different_splits(
        self,
    ) -> None:
        """seed が違えば分割も違うこと (seed が実際に効いていること)."""
        train_a, _ = learn.train_test_split_indices(
            total_rows=200, test_ratio=0.25, seed=7
        )
        train_b, _ = learn.train_test_split_indices(
            total_rows=200, test_ratio=0.25, seed=8
        )

        assert not np.array_equal(train_a, train_b)

    def test_sizes_and_partition(self) -> None:
        """分割が元データの過不足ない分割になっていること."""
        train, test = learn.train_test_split_indices(
            total_rows=20, test_ratio=0.25, seed=7
        )

        assert len(train) == 15
        assert len(test) == 5
        assert sorted(
            np.concatenate([train, test]).tolist()
        ) == list(range(20))

    def test_returns_int64_arrays_not_python_lists(
        self,
    ) -> None:
        """ndarray を返すこと (D8+D9 の中身).

        **trap**: ``list(range(N))`` は 5000万行で索引だけ約 2.6GB を
        食う (int オブジェクト + リストの参照が同時に生存する)．
        int64 の ndarray なら 400MB で済む．型が list に戻ると落ちる．
        """
        train, test = learn.train_test_split_indices(
            total_rows=20, test_ratio=0.25, seed=7
        )

        assert isinstance(train, np.ndarray)
        assert isinstance(test, np.ndarray)
        assert train.dtype == np.int64
        assert test.dtype == np.int64

    @pytest.mark.parametrize(
        ("total_rows", "test_ratio"),
        [(0, 0.25), (1, 0.25), (10, 0.0), (10, 1.0)],
    )
    def test_edge_cases_still_partition(
        self, total_rows: int, test_ratio: float
    ) -> None:
        """端の値でも全体を重複なく覆うこと."""
        train, test = learn.train_test_split_indices(
            total_rows=total_rows, test_ratio=test_ratio
        )

        assert sorted(
            np.concatenate([train, test]).tolist()
        ) == list(range(total_rows))


# ============================================================================
# 3 実装が共有ヘルパを引くこと
# ============================================================================

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

_SPLITTER_CLASSES: dict[str, type] = {
    "bigquery": BigQueryDataSource.BigQueryDataSourceSpliter,
    "object_storage": ObjectStorageDataSource.DataSourceSpliter,
    "file_system": FileDataSource.FileDataSourceSpliter,
}


class _StubManager:
    total_rows = 20


@pytest.mark.parametrize("name", sorted(_PUBLIC_SPLITTERS))
def test_public_split_uses_the_shared_helper_with_the_default_seed(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 実装とも共有ヘルパを既定 seed で呼ぶこと.

    D2 (seed を渡すこと) と D8+D9 (複製を 1 本に寄せたこと) の両方を
    ここで押さえる．
    """
    split, mangle_prefix = _PUBLIC_SPLITTERS[name]

    seen: dict[str, Any] = {}

    def spy(**kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        seen.update(kwargs)
        idx = np.arange(kwargs["total_rows"], dtype=np.int64)
        return idx[:15], idx[15:]

    monkeypatch.setattr(learn, "train_test_split_indices", spy)

    stub = types.SimpleNamespace(
        logger=logging.getLogger(__name__),
        **{
            f"{mangle_prefix}__file_manager": _StubManager(),
            f"{mangle_prefix}__page_manager": _StubManager(),
        },
    )

    split(stub, test_ratio=0.25)

    assert seen["seed"] == learn.DEFAULT_SPLIT_SEED
    assert seen["total_rows"] == 20
    assert seen["test_ratio"] == 0.25


@pytest.mark.parametrize("name", sorted(_SPLITTER_CLASSES))
def test_private_split_copies_are_gone(name: str) -> None:
    """各実装が private な複製を持ち直していないこと.

    3 つの ``__train_test_split`` は 1 文字違わず同一で，O2 のときは
    片方だけ直って乖離した．複製が復活したらここで落ちる．
    """
    cls = _SPLITTER_CLASSES[name]

    assert not [
        attr
        for attr in vars(cls)
        if attr.endswith("__train_test_split")
    ]


# ============================================================================
# 実経路
# ============================================================================


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


class TestBigQueryIndiciesAcceptance:
    """``BigQueryDataSource.indicies`` の受け口を他の 2 実装にそろえた.

    D8+D9 の後半．``file_system`` / ``object_storage`` は
    ``np.asarray(indicies, dtype=np.int64)`` を通すのに，BigQuery だけ
    ``list[int]`` 宣言のまま素通ししていたので，分割側が ndarray を
    返すと **BigQuery だけ型注釈と実体がずれる**状態だった．
    """

    def test_list_input_becomes_int64_array(self) -> None:
        source = BigQueryDataSource(
            page_manager=_StubManager(),  # type: ignore[arg-type]
            indicies=[3, 1, 2],
        )

        assert isinstance(source.indicies, np.ndarray)
        assert source.indicies.dtype == np.int64
        assert source.indicies.tolist() == [3, 1, 2]
        assert len(source) == 3

    def test_array_input_stays_an_int64_array(self) -> None:
        source = BigQueryDataSource(
            page_manager=_StubManager(),  # type: ignore[arg-type]
            indicies=np.array([5, 4], dtype=np.int64),
        )

        assert isinstance(source.indicies, np.ndarray)
        assert source.indicies.dtype == np.int64
        assert source.indicies.tolist() == [5, 4]

    def test_default_covers_every_row(self) -> None:
        source = BigQueryDataSource(
            page_manager=_StubManager(),  # type: ignore[arg-type]
        )

        assert isinstance(source.indicies, np.ndarray)
        assert source.indicies.dtype == np.int64
        assert source.indicies.tolist() == list(range(20))
