"""Tests for StreamingHcpeDataSource (preprocess向けストリーミングI/O)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from maou.domain.data.rust_io import save_hcpe_df
from maou.domain.data.schema import (
    create_empty_hcpe_df,
    get_hcpe_dtype,
)
from maou.infra.file_system.streaming_hcpe_source import (
    StreamingHcpeDataSource,
)

# ============================================================================
# Test data helpers
# ============================================================================


def _create_hcpe_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> list[Path]:
    """Create multiple HCPE .feather files for testing."""
    file_paths: list[Path] = []

    for i in range(file_count):
        df = create_empty_hcpe_df(rows_per_file)

        eval_values = list(
            range(i * rows_per_file, (i + 1) * rows_per_file)
        )
        ids = [f"file{i}_row{j}" for j in range(rows_per_file)]

        df = df.with_columns(
            [
                pl.Series("eval", eval_values),
                pl.Series("id", ids),
            ]
        )

        file_path = directory / f"hcpe_{i}.feather"
        save_hcpe_df(df, file_path)
        file_paths.append(file_path)

    return file_paths


# ============================================================================
# Initialization tests
# ============================================================================


class TestStreamingHcpeDataSourceInit:
    """Test StreamingHcpeDataSource initialization."""

    def test_init_basic(self, tmp_path: Path) -> None:
        """Initialize with HCPE files."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=3, rows_per_file=5
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        assert source.total_pages() == 3

    def test_init_empty(self) -> None:
        """Initialize with no files."""
        source = StreamingHcpeDataSource(file_paths=[])
        assert source.total_pages() == 0
        assert len(source) == 0

    def test_file_paths_returns_copy(
        self, tmp_path: Path
    ) -> None:
        """file_paths returns a copy, not the internal list."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=2, rows_per_file=3
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        paths = source.file_paths
        paths.clear()
        assert len(source.file_paths) == 2


# ============================================================================
# Lazy initialization tests
# ============================================================================


class TestLazyInitialization:
    """StreamingHcpeDataSource遅延初期化のテスト."""

    def test_init_does_not_scan(self, tmp_path: Path) -> None:
        """__init__直後は行数スキャンが実行されていない."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=2, rows_per_file=5
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        assert source._total_rows is None

    def test_len_triggers_scan(self, tmp_path: Path) -> None:
        """__len__()アクセス時にスキャンが実行される."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=2, rows_per_file=5
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        rows = len(source)
        assert rows == 10
        assert source._total_rows == 10

    def test_no_rescan_on_second_access(
        self, tmp_path: Path
    ) -> None:
        """2回目の__len__()アクセスで再スキャンされない."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=2, rows_per_file=5
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        _ = len(source)
        total_rows_ref = source._total_rows
        _ = len(source)
        assert source._total_rows is total_rows_ref


# ============================================================================
# iter_batches tests
# ============================================================================


class TestIterBatches:
    """Test iter_batches generator."""

    def test_yields_correct_count(self, tmp_path: Path) -> None:
        """Yields one batch per file."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=3, rows_per_file=5
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        batches = list(source.iter_batches())
        assert len(batches) == 3

    def test_batch_names(self, tmp_path: Path) -> None:
        """Batch names correspond to file names."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=2, rows_per_file=3
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        names = [name for name, _ in source.iter_batches()]
        assert names == ["hcpe_0.feather", "hcpe_1.feather"]

    def test_batch_dtype(self, tmp_path: Path) -> None:
        """Batches have correct HCPE numpy dtype."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=1, rows_per_file=5
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        expected_dtype = get_hcpe_dtype()
        for _, arr in source.iter_batches():
            assert arr.dtype == expected_dtype

    def test_batch_row_counts(self, tmp_path: Path) -> None:
        """Each batch has correct number of rows."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=2, rows_per_file=7
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        for _, arr in source.iter_batches():
            assert len(arr) == 7

    def test_total_rows_match(self, tmp_path: Path) -> None:
        """Sum of batch lengths matches __len__()."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=3, rows_per_file=4
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        total = sum(
            len(arr) for _, arr in source.iter_batches()
        )
        assert total == len(source)

    def test_empty_source_yields_nothing(self) -> None:
        """Empty source yields no batches."""
        source = StreamingHcpeDataSource(file_paths=[])
        batches = list(source.iter_batches())
        assert len(batches) == 0

    def test_multiple_iterations(self, tmp_path: Path) -> None:
        """Generator can be called multiple times."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=2, rows_per_file=3
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        first_pass = list(source.iter_batches())
        second_pass = list(source.iter_batches())

        assert len(first_pass) == len(second_pass) == 2
        for (n1, a1), (n2, a2) in zip(first_pass, second_pass):
            assert n1 == n2
            np.testing.assert_array_equal(
                a1["eval"], a2["eval"]
            )

    def test_eval_values_preserved(
        self, tmp_path: Path
    ) -> None:
        """Data values are correctly preserved through streaming."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=2, rows_per_file=3
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        all_evals = []
        for _, arr in source.iter_batches():
            all_evals.extend(arr["eval"].tolist())

        # File 0: [0, 1, 2], File 1: [3, 4, 5]
        assert all_evals == [0, 1, 2, 3, 4, 5]


# ============================================================================
# total_pages tests
# ============================================================================


class TestTotalPages:
    """Test total_pages method."""

    def test_matches_file_count(self, tmp_path: Path) -> None:
        """total_pages returns the number of files."""
        file_paths = _create_hcpe_files(
            tmp_path, file_count=5, rows_per_file=2
        )
        source = StreamingHcpeDataSource(
            file_paths=file_paths,
        )
        assert source.total_pages() == 5

    def test_empty_returns_zero(self) -> None:
        """total_pages returns 0 for empty source."""
        source = StreamingHcpeDataSource(file_paths=[])
        assert source.total_pages() == 0
