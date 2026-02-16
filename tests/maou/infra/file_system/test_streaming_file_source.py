"""Tests for StreamingFileSource (file-level streaming I/O)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from maou.domain.data.columnar_batch import ColumnarBatch
from maou.domain.data.rust_io import (
    save_preprocessing_df,
    save_stage1_df,
    save_stage2_df,
)
from maou.domain.data.schema import (
    create_empty_preprocessing_df,
    create_empty_stage1_df,
    create_empty_stage2_df,
)
from maou.domain.move.label import MOVE_LABELS_NUM
from maou.infra.file_system.streaming_file_source import (
    StreamingFileSource,
)

# ============================================================================
# Test data helpers
# ============================================================================


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

    return file_paths


def _create_stage1_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> list[Path]:
    """Create multiple Stage 1 .feather files for testing."""
    file_paths: list[Path] = []

    for i in range(file_count):
        df = create_empty_stage1_df(rows_per_file)

        ids = list(
            range(i * rows_per_file, (i + 1) * rows_per_file)
        )
        df = df.with_columns(
            [
                pl.Series("id", ids),
            ]
        )

        file_path = directory / f"stage1_{i}.feather"
        save_stage1_df(df, file_path)
        file_paths.append(file_path)

    return file_paths


def _create_stage2_files(
    directory: Path,
    *,
    file_count: int,
    rows_per_file: int,
) -> list[Path]:
    """Create multiple Stage 2 .feather files for testing."""
    file_paths: list[Path] = []

    for i in range(file_count):
        df = create_empty_stage2_df(rows_per_file)

        ids = list(
            range(i * rows_per_file, (i + 1) * rows_per_file)
        )
        df = df.with_columns(
            [
                pl.Series("id", ids),
            ]
        )

        file_path = directory / f"stage2_{i}.feather"
        save_stage2_df(df, file_path)
        file_paths.append(file_path)

    return file_paths


# ============================================================================
# Initialization tests
# ============================================================================


class TestStreamingFileSourceInit:
    """Test StreamingFileSource initialization."""

    def test_init_preprocessing(self, tmp_path: Path) -> None:
        """Initialize with preprocessing files."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )
        assert source.total_rows == 10
        assert len(source.file_paths) == 2

    def test_init_stage1(self, tmp_path: Path) -> None:
        """Initialize with stage1 files."""
        file_paths = _create_stage1_files(
            tmp_path, file_count=3, rows_per_file=4
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="stage1",
        )
        assert source.total_rows == 12
        assert len(source.file_paths) == 3

    def test_init_stage2(self, tmp_path: Path) -> None:
        """Initialize with stage2 files."""
        file_paths = _create_stage2_files(
            tmp_path, file_count=2, rows_per_file=6
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="stage2",
        )
        assert source.total_rows == 12
        assert len(source.file_paths) == 2

    def test_init_empty(self) -> None:
        """Initialize with no files."""
        source = StreamingFileSource(
            file_paths=[],
            array_type="preprocessing",
        )
        assert source.total_rows == 0
        assert len(source.file_paths) == 0

    def test_init_invalid_array_type(self) -> None:
        """Reject unsupported array_type."""
        with pytest.raises(ValueError, match="Unsupported"):
            StreamingFileSource(
                file_paths=[],
                array_type="invalid",  # type: ignore[arg-type]
            )

    def test_init_hcpe_rejected(self) -> None:
        """Reject hcpe array_type (no columnar converter)."""
        with pytest.raises(
            ValueError, match="hcpe.*not supported"
        ):
            StreamingFileSource(
                file_paths=[],
                array_type="hcpe",
            )


# ============================================================================
# file_paths property tests
# ============================================================================


class TestFilePathsProperty:
    """Test file_paths property."""

    def test_returns_copy(self, tmp_path: Path) -> None:
        """file_paths returns a copy, not the internal list."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=3
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        paths = source.file_paths
        paths.clear()
        # Internal list should not be affected
        assert len(source.file_paths) == 2


# ============================================================================
# iter_files_columnar tests
# ============================================================================


class TestIterFilesColumnar:
    """Test iter_files_columnar generator."""

    def test_yields_correct_count(self, tmp_path: Path) -> None:
        """Yields one ColumnarBatch per file."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=3, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        batches = list(source.iter_files_columnar())
        assert len(batches) == 3

    def test_total_rows_match(self, tmp_path: Path) -> None:
        """Sum of batch lengths matches total_rows."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=7
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        total = sum(
            len(b) for b in source.iter_files_columnar()
        )
        assert total == source.total_rows

    def test_preprocessing_batch_shapes(
        self, tmp_path: Path
    ) -> None:
        """Preprocessing batches have correct shapes and dtypes."""
        n = 5
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=1, rows_per_file=n
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        batches = list(source.iter_files_columnar())
        assert len(batches) == 1
        batch = batches[0]

        assert isinstance(batch, ColumnarBatch)
        assert batch.board_positions.shape == (n, 9, 9)
        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.shape == (n, 14)
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.move_label is not None
        assert batch.move_label.shape == (
            n,
            MOVE_LABELS_NUM,
        )
        assert batch.move_label.dtype == np.float16
        assert batch.result_value is not None
        assert batch.result_value.shape == (n,)
        assert batch.result_value.dtype == np.float16
        # Stage-specific fields should be None
        assert batch.reachable_squares is None
        assert batch.legal_moves_label is None

    def test_stage1_batch_shapes(self, tmp_path: Path) -> None:
        """Stage 1 batches have correct shapes and dtypes."""
        n = 4
        file_paths = _create_stage1_files(
            tmp_path, file_count=1, rows_per_file=n
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="stage1",
        )

        batches = list(source.iter_files_columnar())
        batch = batches[0]

        assert batch.board_positions.shape == (n, 9, 9)
        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.shape == (n, 14)
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.reachable_squares is not None
        assert batch.reachable_squares.shape == (n, 9, 9)
        assert batch.reachable_squares.dtype == np.uint8
        # Other fields should be None
        assert batch.move_label is None
        assert batch.result_value is None
        assert batch.legal_moves_label is None

    def test_stage2_batch_shapes(self, tmp_path: Path) -> None:
        """Stage 2 batches have correct shapes and dtypes."""
        n = 6
        file_paths = _create_stage2_files(
            tmp_path, file_count=1, rows_per_file=n
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="stage2",
        )

        batches = list(source.iter_files_columnar())
        batch = batches[0]

        assert batch.board_positions.shape == (n, 9, 9)
        assert batch.board_positions.dtype == np.uint8
        assert batch.pieces_in_hand.shape == (n, 14)
        assert batch.pieces_in_hand.dtype == np.uint8
        assert batch.legal_moves_label is not None
        assert batch.legal_moves_label.shape == (
            n,
            MOVE_LABELS_NUM,
        )
        assert batch.legal_moves_label.dtype == np.uint8
        # Other fields should be None
        assert batch.move_label is None
        assert batch.result_value is None
        assert batch.reachable_squares is None

    def test_contiguity(self, tmp_path: Path) -> None:
        """Arrays in yielded batches are C-contiguous."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=1, rows_per_file=5
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        for batch in source.iter_files_columnar():
            assert batch.board_positions.flags.c_contiguous
            assert batch.pieces_in_hand.flags.c_contiguous
            if batch.move_label is not None:
                assert batch.move_label.flags.c_contiguous
            if batch.result_value is not None:
                assert batch.result_value.flags.c_contiguous

    def test_empty_source_yields_nothing(self) -> None:
        """Empty source yields no batches."""
        source = StreamingFileSource(
            file_paths=[],
            array_type="preprocessing",
        )
        batches = list(source.iter_files_columnar())
        assert len(batches) == 0

    def test_multiple_iterations(self, tmp_path: Path) -> None:
        """Generator can be called multiple times."""
        file_paths = _create_preprocessing_files(
            tmp_path, file_count=2, rows_per_file=3
        )
        source = StreamingFileSource(
            file_paths=file_paths,
            array_type="preprocessing",
        )

        first_pass = list(source.iter_files_columnar())
        second_pass = list(source.iter_files_columnar())

        assert len(first_pass) == len(second_pass) == 2
        for b1, b2 in zip(first_pass, second_pass):
            np.testing.assert_array_equal(
                b1.board_positions, b2.board_positions
            )
