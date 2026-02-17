"""Tests for the file-system based data source.

Updated to use DataFrame-based I/O with Polars.
Simplified from original numpy-based tests to focus on DataFrame functionality.
"""

import logging
from pathlib import Path

import polars as pl
import pytest

from maou.domain.data.rust_io import (
    save_hcpe_df,
    save_preprocessing_df,
)
from maou.domain.data.schema import (
    create_empty_hcpe_df,
    create_empty_preprocessing_df,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)


def _create_hcpe_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> tuple[list[Path], list[pl.DataFrame]]:
    """Create multiple HCPE .feather files for testing."""
    file_paths: list[Path] = []
    dataframes: list[pl.DataFrame] = []

    for i in range(file_count):
        df = create_empty_hcpe_df(rows_per_file)

        # Add some test data
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
        dataframes.append(df)

    return file_paths, dataframes


def _create_preprocessing_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> tuple[list[Path], list[pl.DataFrame]]:
    """Create multiple preprocessing .feather files for testing."""
    file_paths: list[Path] = []
    dataframes: list[pl.DataFrame] = []

    for i in range(file_count):
        df = create_empty_preprocessing_df(rows_per_file)

        # Add some test data
        result_values = [
            float(i * rows_per_file + j)
            for j in range(rows_per_file)
        ]
        ids = list(
            range(i * rows_per_file, (i + 1) * rows_per_file)
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
        dataframes.append(df)

    return file_paths, dataframes


def test_file_data_source_basic_loading(tmp_path: Path) -> None:
    """Test basic FileDataSource loading functionality."""
    file_paths, _ = _create_hcpe_files(
        tmp_path, file_count=2, rows_per_file=3
    )

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="hcpe",
    )

    # Should have 6 total records (2 files * 3 rows)
    assert len(datasource) == 6


def test_file_data_source_indexing(tmp_path: Path) -> None:
    """Test FileDataSource indexing."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=2, rows_per_file=5
    )

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="preprocessing",
    )

    # Test indexing
    first_record = datasource[0]
    assert first_record is not None

    last_record = datasource[9]
    assert last_record is not None


def test_file_data_source_train_test_split(
    tmp_path: Path,
) -> None:
    """Test FileDataSource train/test split functionality."""
    file_paths, _ = _create_hcpe_files(
        tmp_path, file_count=1, rows_per_file=10
    )

    splitter = FileDataSource.FileDataSourceSpliter(
        file_paths=file_paths,
        array_type="hcpe",
    )

    train_ds, test_ds = splitter.train_test_split(
        test_ratio=0.3
    )

    # Verify split sizes (approximately)
    total_size = len(train_ds) + len(test_ds)
    assert total_size == 10
    assert len(test_ds) >= 2  # At least 30% of 10


def test_file_data_source_iter_batches(tmp_path: Path) -> None:
    """Test FileDataSource batch iteration."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=2, rows_per_file=4
    )

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="preprocessing",
    )

    # Iterate through batches
    batch_count = 0
    total_records = 0

    for file_name, batch in datasource.iter_batches():
        # file_name is just the filename (str), not the full path
        assert isinstance(file_name, str)
        assert len(batch) > 0
        total_records += len(batch)
        batch_count += 1

    assert batch_count == 2  # 2 files
    assert total_records == 8  # 2 files * 4 rows


def test_file_data_source_memory_cache(tmp_path: Path) -> None:
    """Test FileDataSource with memory cache mode."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=1, rows_per_file=5
    )

    datasource = FileDataSource.FileDataSourceSpliter(
        file_paths=file_paths,
        array_type="preprocessing",
        cache_mode="memory",
    )

    train_ds, _ = datasource.train_test_split(test_ratio=0.0)

    # Access records multiple times (should use cache)
    record1_first = train_ds[0]
    record1_second = train_ds[0]

    # Both accesses should return valid data
    assert record1_first is not None
    assert record1_second is not None


def test_file_data_source_mmap_cache(tmp_path: Path) -> None:
    """Test FileDataSource with file cache mode."""
    file_paths, _ = _create_preprocessing_files(
        tmp_path, file_count=1, rows_per_file=4
    )

    datasource = FileDataSource.FileDataSourceSpliter(
        file_paths=file_paths,
        array_type="preprocessing",
        cache_mode="file",
    )

    train_ds, _ = datasource.train_test_split(test_ratio=0.0)

    # Access records
    record = train_ds[0]
    assert record is not None


def test_file_data_source_empty_file_list() -> None:
    """Test FileDataSource with empty file list."""
    # Empty file list is now allowed (creates empty datasource)
    datasource = FileDataSource(
        file_paths=[],
        array_type="hcpe",
    )
    assert len(datasource) == 0


def test_file_data_source_nonexistent_file(
    tmp_path: Path,
) -> None:
    """Test FileDataSource with non-existent file."""
    fake_path = tmp_path / "nonexistent.feather"

    with pytest.raises((FileNotFoundError, Exception)):
        datasource = FileDataSource(
            file_paths=[fake_path],
            array_type="hcpe",
        )
        # Try to access data
        len(datasource)


def test_file_data_source_progress_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that FileManager emits progress logs during initialization."""
    file_paths, _ = _create_hcpe_files(
        tmp_path, file_count=2, rows_per_file=3
    )

    # maouロガーはpropagate=Falseかつlevel=INFOのため，
    # DEBUGログ捕捉にはpropagateとlevelの両方を一時変更する
    maou_logger = logging.getLogger("maou")
    original_propagate = maou_logger.propagate
    original_level = maou_logger.level
    maou_logger.propagate = True
    maou_logger.setLevel(logging.DEBUG)
    try:
        with caplog.at_level(logging.DEBUG):
            FileDataSource(
                file_paths=file_paths,
                array_type="hcpe",
            )
    finally:
        maou_logger.propagate = original_propagate
        maou_logger.setLevel(original_level)

    # INFOレベルのログ検証
    info_messages = [
        r.message
        for r in caplog.records
        if r.levelno >= logging.INFO
    ]
    info_text = "\n".join(info_messages)

    # 初期化開始ログ
    assert "Initializing FileManager with 2 files" in info_text
    # マイルストーン進捗ログ (n=2, interval=1なので全ファイル表示)
    assert "Progress: 1/2 files" in info_text
    assert "Progress: 2/2 files" in info_text
    # サマリーログ
    assert (
        "FileManager initialized: 6 rows from 2 files"
        in info_text
    )

    # DEBUGレベルのログ検証
    debug_messages = [
        r.message
        for r in caplog.records
        if r.levelno == logging.DEBUG
    ]
    debug_text = "\n".join(debug_messages)

    assert "Loading file 1/2" in debug_text
    assert "Loading file 2/2" in debug_text
    assert "Loaded" in debug_text
    assert "Converted to numpy array" in debug_text
