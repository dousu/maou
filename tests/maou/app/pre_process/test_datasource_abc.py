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
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import polars as pl

from maou.app.pre_process.hcpe_transform import DataSource


def test_base_uses_abcmeta() -> None:
    assert isinstance(DataSource, abc.ABCMeta)


def test_all_four_abstract_methods_are_enforced() -> None:
    """抽象メソッドが 4 本とも登録されていること.

    **trap**: ``total_pages`` は ``iter_batches_df`` の**下**に
    書かれているので，見落として本数を数え違えやすい (実際そう読み
    違えた記録が残っている)．
    """
    assert DataSource.__abstractmethods__ == frozenset(
        {
            "__len__",
            "iter_batches",
            "iter_batches_df",
            "total_pages",
        }
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

        def iter_batches_df(
            self,
        ) -> Generator[tuple[str, pl.DataFrame], None, None]:
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

        def iter_batches_df(
            self,
        ) -> Generator[tuple[str, pl.DataFrame], None, None]:
            yield from ()

        def total_pages(self) -> int:
            return 0

    source = _Complete()

    assert len(source) == 0
    assert source.total_pages() == 0
    assert list(source.iter_batches()) == []


def test_iter_batches_df_has_no_hcpe_hardcoded_default() -> (
    None
):
    """``iter_batches_df`` に既定実装が**無い**こと (backlog 行 N6-2).

    2026-08-14 まで基底は具象の既定実装を持っており，その本体は
    ``get_hcpe_polars_schema()`` 決め打ちだった．``preprocessing`` 型の
    ソースが override を忘れると，構築も呼び出しも成功したうえで
    **黙って誤ったスキーマ**を当てた．

    **trap**: これは以前の
    ``test_iter_batches_df_default_is_still_inherited`` が固定していた
    のと**正反対**の不変条件である．既定実装を戻すと，この取り違えが
    再び構築時をすり抜けるので落とす．
    """
    assert "iter_batches_df" in DataSource.__abstractmethods__

    class _MissingIterBatchesDf(DataSource):
        def __len__(self) -> int:
            return 0

        def iter_batches(
            self,
        ) -> Generator[tuple[str, np.ndarray], None, None]:
            yield from ()

        def total_pages(self) -> int:
            return 0

    with pytest.raises(TypeError, match="iter_batches_df"):
        _MissingIterBatchesDf()  # type: ignore[abstract]


def test_hcpe_default_moved_to_the_hcpe_only_class() -> None:
    """HCPE 決め打ちの本体が HCPE 専用クラスへ移っていること.

    基底が持っていた頃は「クラス名からは HCPE 専用と判らない」のが
    問題だった．``StreamingHcpeDataSource`` は名前が HCPE 専用を明示
    するので，同じ本体でも取り違えようがない．
    """
    from maou.infra.file_system.streaming_hcpe_source import (
        StreamingHcpeDataSource,
    )

    assert "iter_batches_df" in StreamingHcpeDataSource.__dict__


def test_production_datasources_all_construct() -> None:
    """production の実装が ABC 化後も構築できること.

    ABC を入れる前に「何が壊れるか」を洗った結果，非準拠は
    テストの ``_NoDataSource`` 1 件だけだった．その結論を固定する．
    """
    from maou.infra.file_system.streaming_hcpe_source import (
        StreamingHcpeDataSource,
    )

    assert len(StreamingHcpeDataSource(file_paths=[])) == 0


def test_file_data_source_is_not_a_preprocess_source() -> None:
    """``FileDataSource`` が pre-process の ABC を着ていないこと.

    backlog 行 D14(b)．pre-process のソースとしての役割は
    ``StreamingHcpeDataSource`` へ移っている
    (``console/pre_process.py`` が構築するのはそちら) のに，
    ``FileDataSource`` の継承だけが残っていたため「どちらを渡しても
    よい」ように見えていた．``FileDataSource`` は初期化時に全ファイルを
    メモリへロードするので，pre-process に渡すとピークメモリが入力
    全体に比例する．

    **trap**: 継承を戻しても ``iter_batches`` / ``iter_batches_df`` /
    ``total_pages`` は具象メソッドとして残るため，実行時には何も
    壊れない (破れるのは型検査の側だけ)．この不変条件だけが差を見る．
    """
    from maou.infra.file_system.file_data_source import (
        FileDataSource,
    )
    from maou.infra.file_system.streaming_hcpe_source import (
        StreamingHcpeDataSource,
    )

    assert not issubclass(FileDataSource, DataSource)
    assert issubclass(StreamingHcpeDataSource, DataSource)
