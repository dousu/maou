"""Tests for FileDataSourceSpliter.file_level_split()."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from maou.domain.data.rust_io import save_preprocessing_df
from maou.domain.data.schema import (
    create_empty_preprocessing_df,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.infra.file_system.streaming_file_source import (
    StreamingFileSource,
)


def _create_preprocessing_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> list[Path]:
    """Create multiple preprocessing .feather files for testing."""
    file_paths: list[Path] = []

    for i in range(file_count):
        df = create_empty_preprocessing_df(rows_per_file)

        result_values = [
            float(i * rows_per_file + j)
            for j in range(rows_per_file)
        ]
        ids = list(
            range(
                i * rows_per_file,
                (i + 1) * rows_per_file,
            )
        )

        df = df.with_columns(
            [
                pl.Series("resultValue", result_values),
                pl.Series("id", ids),
            ]
        )

        file_path = directory / f"preprocessing_{i}.feather"
        save_preprocessing_df(df, file_path)
        file_paths.append(file_path)

    return file_paths


class TestFileLevelSplit:
    """Tests for file_level_split() method."""

    def test_split_returns_streaming_sources(
        self, tmp_path: Path
    ) -> None:
        """file_level_split returns two StreamingFileSource instances."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=4, rows_per_file=5
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train_source, val_source = splitter.file_level_split(
            test_ratio=0.25
        )

        assert isinstance(train_source, StreamingFileSource)
        assert isinstance(val_source, StreamingFileSource)

    def test_split_file_counts(self, tmp_path: Path) -> None:
        """File counts match expected train/val ratio."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=10, rows_per_file=5
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train_source, val_source = splitter.file_level_split(
            test_ratio=0.2
        )

        assert len(train_source.file_paths) == 8
        assert len(val_source.file_paths) == 2

    def test_split_total_rows_preserved(
        self, tmp_path: Path
    ) -> None:
        """Total rows across train+val equal original total."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=5, rows_per_file=10
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train_source, val_source = splitter.file_level_split(
            test_ratio=0.2
        )

        assert (
            train_source.total_rows + val_source.total_rows
            == 50
        )

    def test_split_no_file_overlap(
        self, tmp_path: Path
    ) -> None:
        """Train and val files do not overlap."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=6, rows_per_file=5
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train_source, val_source = splitter.file_level_split(
            test_ratio=0.3
        )

        train_set = set(train_source.file_paths)
        val_set = set(val_source.file_paths)
        assert len(train_set & val_set) == 0

    def test_split_all_files_covered(
        self, tmp_path: Path
    ) -> None:
        """All original files appear in either train or val."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=6, rows_per_file=5
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train_source, val_source = splitter.file_level_split(
            test_ratio=0.3
        )

        all_split_files = set(train_source.file_paths) | set(
            val_source.file_paths
        )
        assert all_split_files == set(file_paths)

    def test_split_minimum_two_files(
        self, tmp_path: Path
    ) -> None:
        """Two files split into 1 train + 1 val."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=5
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train_source, val_source = splitter.file_level_split(
            test_ratio=0.5
        )

        assert len(train_source.file_paths) == 1
        assert len(val_source.file_paths) == 1

    def test_split_single_file_raises(
        self, tmp_path: Path
    ) -> None:
        """Single file raises ValueError."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=1, rows_per_file=10
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        with pytest.raises(ValueError, match="at least 2"):
            splitter.file_level_split(test_ratio=0.2)

    def test_split_deterministic_with_seed(
        self, tmp_path: Path
    ) -> None:
        """Same seed produces same split."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=8, rows_per_file=5
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train1, val1 = splitter.file_level_split(
            test_ratio=0.25, seed=42
        )
        train2, val2 = splitter.file_level_split(
            test_ratio=0.25, seed=42
        )

        assert train1.file_paths == train2.file_paths
        assert val1.file_paths == val2.file_paths

    def test_split_different_seeds_differ(
        self, tmp_path: Path
    ) -> None:
        """Different seeds produce different splits."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=8, rows_per_file=5
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train1, _ = splitter.file_level_split(
            test_ratio=0.25, seed=42
        )
        train2, _ = splitter.file_level_split(
            test_ratio=0.25, seed=99
        )

        # With 8 files, different seeds should produce different orderings
        assert train1.file_paths != train2.file_paths

    def test_split_val_at_least_one_file(
        self, tmp_path: Path
    ) -> None:
        """Very small test_ratio still produces at least 1 val file."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=10, rows_per_file=5
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train_source, val_source = splitter.file_level_split(
            test_ratio=0.01
        )

        assert len(val_source.file_paths) >= 1
        assert len(train_source.file_paths) >= 1

    def test_split_iter_files_works(
        self, tmp_path: Path
    ) -> None:
        """Resulting StreamingFileSources can iterate and yield data."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=4, rows_per_file=5
        )
        splitter = FileDataSource.FileDataSourceSpliter(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        train_source, val_source = splitter.file_level_split(
            test_ratio=0.25
        )

        # Train source should yield batches
        train_batches = list(train_source.iter_files_columnar())
        assert len(train_batches) == len(
            train_source.file_paths
        )

        # Val source should yield batches
        val_batches = list(val_source.iter_files_columnar())
        assert len(val_batches) == len(val_source.file_paths)
