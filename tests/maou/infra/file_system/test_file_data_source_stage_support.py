"""Tests for FileDataSource stage1 and stage2 array type support.

Updated to use DataFrame-based I/O with Polars.
Tests support for multi-stage training with stage1 and stage2 data formats.
"""

from pathlib import Path

import polars as pl

from maou.domain.data.rust_io import save_stage1_df, save_stage2_df
from maou.domain.data.schema import (
    create_empty_stage1_df,
    create_empty_stage2_df,
)
from maou.infra.file_system.file_data_source import (
    FileDataSource,
)


def test_file_data_source_stage1_support(tmp_path: Path) -> None:
    """Test FileDataSource with stage1 array type."""
    # Create stage1 DataFrame
    df = create_empty_stage1_df(5)
    df = df.with_columns([
        pl.Series("id", [1, 2, 3, 4, 5]),
    ])

    file_path = tmp_path / "stage1.feather"
    save_stage1_df(df, file_path)

    # Create FileDataSource with stage1 type
    datasource = FileDataSource(
        file_paths=[file_path],
        array_type="stage1",
    )

    assert len(datasource) == 5

    # Access a record
    record = datasource[0]
    assert record is not None


def test_file_data_source_stage2_support(tmp_path: Path) -> None:
    """Test FileDataSource with stage2 array type."""
    # Create stage2 DataFrame
    df = create_empty_stage2_df(3)
    df = df.with_columns([
        pl.Series("id", [10, 20, 30]),
    ])

    file_path = tmp_path / "stage2.feather"
    save_stage2_df(df, file_path)

    # Create FileDataSource with stage2 type
    datasource = FileDataSource(
        file_paths=[file_path],
        array_type="stage2",
    )

    assert len(datasource) == 3

    # Access a record
    record = datasource[2]
    assert record is not None


def test_file_data_source_spliter_stage1_support(
    tmp_path: Path,
) -> None:
    """Test FileDataSourceSpliter with stage1 array type."""
    # Create stage1 DataFrame
    df = create_empty_stage1_df(10)
    df = df.with_columns([
        pl.Series("id", list(range(10))),
    ])

    file_path = tmp_path / "stage1_split.feather"
    save_stage1_df(df, file_path)

    # Create splitter
    splitter = FileDataSource.FileDataSourceSpliter(
        file_paths=[file_path],
        array_type="stage1",
    )

    # Perform train/test split
    train_ds, test_ds = splitter.train_test_split(test_ratio=0.2)

    assert len(train_ds) + len(test_ds) == 10
    assert len(test_ds) >= 1  # At least 20% of 10


def test_file_data_source_spliter_stage2_support(
    tmp_path: Path,
) -> None:
    """Test FileDataSourceSpliter with stage2 array type."""
    # Create stage2 DataFrame
    df = create_empty_stage2_df(8)
    df = df.with_columns([
        pl.Series("id", list(range(100, 108))),
    ])

    file_path = tmp_path / "stage2_split.feather"
    save_stage2_df(df, file_path)

    # Create splitter
    splitter = FileDataSource.FileDataSourceSpliter(
        file_paths=[file_path],
        array_type="stage2",
    )

    # Perform train/test split
    train_ds, test_ds = splitter.train_test_split(test_ratio=0.25)

    assert len(train_ds) + len(test_ds) == 8
    assert len(test_ds) >= 2  # At least 25% of 8


def test_file_data_source_stage1_iter_batches(
    tmp_path: Path,
) -> None:
    """Test stage1 FileDataSource batch iteration."""
    # Create multiple stage1 files
    file_paths = []
    for i in range(2):
        df = create_empty_stage1_df(4)
        df = df.with_columns([
            pl.Series("id", [i * 10 + j for j in range(4)]),
        ])

        file_path = tmp_path / f"stage1_{i}.feather"
        save_stage1_df(df, file_path)
        file_paths.append(file_path)

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="stage1",
    )

    # Iterate through batches
    batch_count = 0
    for _, batch in datasource.iter_batches():
        assert len(batch) > 0
        batch_count += 1

    assert batch_count == 2


def test_file_data_source_stage2_iter_batches(
    tmp_path: Path,
) -> None:
    """Test stage2 FileDataSource batch iteration."""
    # Create multiple stage2 files
    file_paths = []
    for i in range(3):
        df = create_empty_stage2_df(2)
        df = df.with_columns([
            pl.Series("id", [i * 100 + j for j in range(2)]),
        ])

        file_path = tmp_path / f"stage2_{i}.feather"
        save_stage2_df(df, file_path)
        file_paths.append(file_path)

    datasource = FileDataSource(
        file_paths=file_paths,
        array_type="stage2",
    )

    # Iterate through batches
    batch_count = 0
    total_records = 0
    for _, batch in datasource.iter_batches():
        assert len(batch) > 0
        total_records += len(batch)
        batch_count += 1

    assert batch_count == 3
    assert total_records == 6  # 3 files * 2 records
