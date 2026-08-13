"""``hcpe_transform.DataSource`` が実際に ABC であることの回帰テスト．

`/audit-backlog` 2026-08-13 / backlog 行 N3．

以前は ``@abc.abstractmethod`` を付けながら ``abc.ABCMeta`` を使って
いなかったため，未実装の実装が**構築時に一切捕まらなかった**．
実際 ``tests/maou/app/pre_process/test_search_value.py`` の
``_NoDataSource`` は ``total_pages`` を持たないまま構築されており，
「触られないから」という理由で長らく気付かれずにいた．
"""

from __future__ import annotations

import abc
from collections.abc import Generator

import numpy as np
import pytest

from maou.app.pre_process.hcpe_transform import DataSource


def test_base_uses_abcmeta() -> None:
    assert isinstance(DataSource, abc.ABCMeta)


def test_all_three_abstract_methods_are_enforced() -> None:
    """抽象メソッドが 3 本とも登録されていること.

    **trap**: ``total_pages`` は具象 ``iter_batches_df`` の**下**に
    書かれているので，2 本だと思い込みやすい (実際そう読み違えた記録が
    残っている)．
    """
    assert DataSource.__abstractmethods__ == frozenset(
        {"__len__", "iter_batches", "total_pages"}
    )


def test_missing_total_pages_fails_at_construction() -> None:
    """``total_pages`` を埋めない実装は構築時に落ちること."""

    class _MissingTotalPages(DataSource):
        def __len__(self) -> int:
            return 0

        def iter_batches(
            self,
        ) -> Generator[tuple[str, np.ndarray], None, None]:
            yield from ()

    with pytest.raises(TypeError, match="total_pages"):
        _MissingTotalPages()  # type: ignore[abstract]


def test_complete_subclass_still_constructs() -> None:
    """全部埋めた実装はこれまでどおり構築できること (挙動不変)."""

    class _Complete(DataSource):
        def __len__(self) -> int:
            return 0

        def iter_batches(
            self,
        ) -> Generator[tuple[str, np.ndarray], None, None]:
            yield from ()

        def total_pages(self) -> int:
            return 0

    source = _Complete()

    assert len(source) == 0
    assert source.total_pages() == 0
    assert list(source.iter_batches()) == []


def test_iter_batches_df_default_is_still_inherited() -> None:
    """具象の既定実装が抽象扱いされていないこと.

    ABC 化で ``iter_batches_df`` まで abstract になると，これを
    override していない実装 (``StreamingHcpeDataSource`` など) が
    構築不能になる．
    """
    assert (
        "iter_batches_df" not in DataSource.__abstractmethods__
    )
    assert callable(DataSource.iter_batches_df)


def test_production_datasources_all_construct() -> None:
    """production の実装が ABC 化後も構築できること.

    ABC を入れる前に「何が壊れるか」を洗った結果，非準拠は
    テストの ``_NoDataSource`` 1 件だけだった．その結論を固定する．
    """
    from maou.infra.file_system.streaming_hcpe_source import (
        StreamingHcpeDataSource,
    )

    assert len(StreamingHcpeDataSource(file_paths=[])) == 0
