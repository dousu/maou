"""FileBackedListColumns の単体テスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maou.interface.lazy_list_columns import (
    FileBackedListColumns,
)


def _create_feather_file(
    path: Path,
    move_labels: list[list[float]],
    move_win_rates: list[list[float]],
) -> None:
    """テスト用の .feather ファイルを作成する．"""
    df = pl.DataFrame(
        {
            "moveLabel": move_labels,
            "moveWinRate": move_win_rates,
        }
    )
    df.write_ipc(path)


class TestFileBackedListColumns:
    """FileBackedListColumns のテスト．"""

    def test_single_file_access(self, tmp_path: Path) -> None:
        """単一ファイルから行データを正しく取得できる．"""
        labels = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        win_rates = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        path = tmp_path / "file_0.feather"
        _create_feather_file(path, labels, win_rates)

        accessor = FileBackedListColumns([path], [3])

        l0, w0 = accessor.get(0)
        assert l0 == pytest.approx([1.0, 2.0])
        assert w0 == pytest.approx([0.1, 0.2])

        l2, w2 = accessor.get(2)
        assert l2 == pytest.approx([5.0, 6.0])
        assert w2 == pytest.approx([0.5, 0.6])

    def test_multi_file_boundary(self, tmp_path: Path) -> None:
        """複数ファイルにまたがるアクセスで境界行が正しい．"""
        # ファイル0: 2行, ファイル1: 3行
        path0 = tmp_path / "file_0.feather"
        path1 = tmp_path / "file_1.feather"
        _create_feather_file(
            path0,
            [[1.0], [2.0]],
            [[0.1], [0.2]],
        )
        _create_feather_file(
            path1,
            [[3.0], [4.0], [5.0]],
            [[0.3], [0.4], [0.5]],
        )

        accessor = FileBackedListColumns([path0, path1], [2, 3])

        # ファイル0の最後の行
        l1, w1 = accessor.get(1)
        assert l1 == pytest.approx([2.0])
        assert w1 == pytest.approx([0.2])

        # ファイル1の最初の行(境界: global_row == 2)
        l2, w2 = accessor.get(2)
        assert l2 == pytest.approx([3.0])
        assert w2 == pytest.approx([0.3])

        # ファイル1の最後の行
        l4, w4 = accessor.get(4)
        assert l4 == pytest.approx([5.0])
        assert w4 == pytest.approx([0.5])

    def test_cache_hit_miss_counts(
        self, tmp_path: Path
    ) -> None:
        """キャッシュヒット/ミスカウントが正確に記録される．"""
        path0 = tmp_path / "file_0.feather"
        path1 = tmp_path / "file_1.feather"
        _create_feather_file(path0, [[1.0]], [[0.1]])
        _create_feather_file(path1, [[2.0]], [[0.2]])

        accessor = FileBackedListColumns([path0, path1], [1, 1])

        # 初回アクセス: ミス
        accessor.get(0)
        assert accessor.cache_hits == 0
        assert accessor.cache_misses == 1

        # 同一ファイル: ヒット(1行しかないが同ファイル内)
        accessor.get(0)
        assert accessor.cache_hits == 1
        assert accessor.cache_misses == 1

        # 別ファイル: ミス
        accessor.get(1)
        assert accessor.cache_hits == 1
        assert accessor.cache_misses == 2

        # 元のファイルに戻る: ミス
        accessor.get(0)
        assert accessor.cache_hits == 1
        assert accessor.cache_misses == 3

    def test_mismatched_lengths_raises(self) -> None:
        """file_paths と file_row_counts の長さが異なると ValueError."""
        with pytest.raises(ValueError, match="長さが一致"):
            FileBackedListColumns(
                [Path("a.feather"), Path("b.feather")],
                [10],
            )

    def test_empty_file_paths_raises(self) -> None:
        """空の file_paths で ValueError が発生する．"""
        with pytest.raises(ValueError, match="file_paths が空"):
            FileBackedListColumns([], [])

    def test_negative_global_row_raises(
        self, tmp_path: Path
    ) -> None:
        """負の global_row で IndexError が発生する．"""
        path = tmp_path / "file_0.feather"
        _create_feather_file(path, [[1.0]], [[0.1]])
        accessor = FileBackedListColumns([path], [1])

        with pytest.raises(IndexError, match="範囲外"):
            accessor.get(-1)

    def test_out_of_range_global_row_raises(
        self, tmp_path: Path
    ) -> None:
        """総行数以上の global_row で IndexError が発生する．"""
        path = tmp_path / "file_0.feather"
        _create_feather_file(
            path, [[1.0], [2.0]], [[0.1], [0.2]]
        )
        accessor = FileBackedListColumns([path], [2])

        with pytest.raises(IndexError, match="範囲外"):
            accessor.get(2)

    def test_numpy_dtype_is_float32(
        self, tmp_path: Path
    ) -> None:
        """返される配列のdtypeがfloat32である．"""
        path = tmp_path / "file_0.feather"
        _create_feather_file(path, [[1.0, 2.0]], [[0.1, 0.2]])

        accessor = FileBackedListColumns([path], [1])
        labels, win_rates = accessor.get(0)

        assert labels.dtype == np.float32
        assert win_rates.dtype == np.float32

    def test_log_stats_no_access(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """アクセスなしの状態で log_stats() が「アクセスなし」をログ出力する．"""
        path = tmp_path / "file_0.feather"
        _create_feather_file(path, [[1.0]], [[0.1]])
        accessor = FileBackedListColumns([path], [1])
        with caplog.at_level("INFO"):
            accessor.log_stats()
        assert "アクセスなし" in caplog.text

    def test_row_count_mismatch_raises(
        self, tmp_path: Path
    ) -> None:
        """List カラムの行数がスカラー行数と異なると ValueError."""
        path = tmp_path / "file_0.feather"
        _create_feather_file(
            path, [[1.0], [2.0]], [[0.1], [0.2]]
        )
        # 実際は2行だが file_row_counts=3 と不整合
        accessor = FileBackedListColumns([path], [3])
        with pytest.raises(ValueError, match="一致しません"):
            accessor.get(0)
