"""Tests for StreamingDataset (IterableDataset implementations)."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import logging
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from maou.app.learning.streaming_dataset import (
    StreamingDataSource,
    StreamingKifDataset,
    StreamingStage1Dataset,
    StreamingStage2Dataset,
    _compute_total_batches,
    _resolve_worker_files,
)
from maou.domain.data.columnar_batch import ColumnarBatch
from maou.domain.move.label import MOVE_LABELS_NUM

# ============================================================================
# Fake StreamingDataSource for testing
# ============================================================================


class FakePreprocessingSource:
    """Fake source for preprocessing data tests."""

    def __init__(
        self, *, n_files: int = 2, rows_per_file: int = 10
    ) -> None:
        self._n_files = n_files
        self._rows_per_file = rows_per_file
        self._file_paths = [
            Path(f"/fake/file_{i}.feather")
            for i in range(n_files)
        ]

    @property
    def file_paths(self) -> list[Path]:
        return list(self._file_paths)

    @property
    def total_rows(self) -> int:
        return self._n_files * self._rows_per_file

    @property
    def row_counts(self) -> list[int]:
        return [self._rows_per_file] * self._n_files

    def _make_batch(
        self, rng: np.random.Generator
    ) -> ColumnarBatch:
        n = self._rows_per_file
        return ColumnarBatch(
            board_positions=rng.integers(
                0, 30, size=(n, 9, 9), dtype=np.uint8
            ),
            pieces_in_hand=rng.integers(
                0, 5, size=(n, 14), dtype=np.uint8
            ),
            move_label=rng.random((n, MOVE_LABELS_NUM)).astype(
                np.float16
            ),
            result_value=rng.random(n).astype(np.float16),
        )

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(123)
        for i in range(self._n_files):
            yield self._make_batch(rng)

    def iter_files_columnar_subset(
        self,
        file_paths: list[Path],
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(123)
        all_paths = self._file_paths
        target_set = set(str(fp) for fp in file_paths)
        for fp in all_paths:
            batch = self._make_batch(rng)
            if str(fp) in target_set:
                yield batch


class FakeStage1Source:
    """Fake source for stage1 data tests."""

    def __init__(
        self, *, n_files: int = 2, rows_per_file: int = 10
    ) -> None:
        self._n_files = n_files
        self._rows_per_file = rows_per_file
        self._file_paths = [
            Path(f"/fake/stage1_{i}.feather")
            for i in range(n_files)
        ]

    @property
    def file_paths(self) -> list[Path]:
        return list(self._file_paths)

    @property
    def total_rows(self) -> int:
        return self._n_files * self._rows_per_file

    @property
    def row_counts(self) -> list[int]:
        return [self._rows_per_file] * self._n_files

    def _make_batch(
        self, rng: np.random.Generator
    ) -> ColumnarBatch:
        n = self._rows_per_file
        return ColumnarBatch(
            board_positions=rng.integers(
                0, 30, size=(n, 9, 9), dtype=np.uint8
            ),
            pieces_in_hand=rng.integers(
                0, 5, size=(n, 14), dtype=np.uint8
            ),
            reachable_squares=rng.integers(
                0, 2, size=(n, 9, 9), dtype=np.uint8
            ),
        )

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(456)
        for i in range(self._n_files):
            yield self._make_batch(rng)

    def iter_files_columnar_subset(
        self,
        file_paths: list[Path],
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(456)
        all_paths = self._file_paths
        target_set = set(str(fp) for fp in file_paths)
        for fp in all_paths:
            batch = self._make_batch(rng)
            if str(fp) in target_set:
                yield batch


class FakeStage2Source:
    """Fake source for stage2 data tests."""

    def __init__(
        self, *, n_files: int = 2, rows_per_file: int = 10
    ) -> None:
        self._n_files = n_files
        self._rows_per_file = rows_per_file
        self._file_paths = [
            Path(f"/fake/stage2_{i}.feather")
            for i in range(n_files)
        ]

    @property
    def file_paths(self) -> list[Path]:
        return list(self._file_paths)

    @property
    def total_rows(self) -> int:
        return self._n_files * self._rows_per_file

    @property
    def row_counts(self) -> list[int]:
        return [self._rows_per_file] * self._n_files

    def _make_batch(
        self, rng: np.random.Generator
    ) -> ColumnarBatch:
        n = self._rows_per_file
        return ColumnarBatch(
            board_positions=rng.integers(
                0, 30, size=(n, 9, 9), dtype=np.uint8
            ),
            pieces_in_hand=rng.integers(
                0, 5, size=(n, 14), dtype=np.uint8
            ),
            legal_moves_label=rng.integers(
                0,
                2,
                size=(n, MOVE_LABELS_NUM),
                dtype=np.uint8,
            ),
        )

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(789)
        for i in range(self._n_files):
            yield self._make_batch(rng)

    def iter_files_columnar_subset(
        self,
        file_paths: list[Path],
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(789)
        all_paths = self._file_paths
        target_set = set(str(fp) for fp in file_paths)
        for fp in all_paths:
            batch = self._make_batch(rng)
            if str(fp) in target_set:
                yield batch


# ============================================================================
# Protocol conformance tests
# ============================================================================


class TestStreamingDataSourceProtocol:
    """Test StreamingDataSource protocol conformance."""

    def test_fake_source_conforms(self) -> None:
        """Fake sources implement StreamingDataSource protocol."""
        assert isinstance(
            FakePreprocessingSource(), StreamingDataSource
        )
        assert isinstance(
            FakeStage1Source(), StreamingDataSource
        )
        assert isinstance(
            FakeStage2Source(), StreamingDataSource
        )


# ============================================================================
# StreamingKifDataset tests
# ============================================================================


class TestStreamingKifDataset:
    """Test StreamingKifDataset."""

    def test_batch_shapes(self) -> None:
        """Yielded batches have correct shapes."""
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=10
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )

        batches = list(dataset)
        assert len(batches) == 3  # ceil(10/4) = 3

        features, targets = batches[0]
        board, pieces = features
        move_label, result_value, legal_mask = targets

        assert board.shape == (4, 9, 9)
        assert board.dtype == torch.uint8
        assert pieces.shape == (4, 14)
        assert pieces.dtype == torch.uint8
        assert move_label.shape == (4, MOVE_LABELS_NUM)
        assert result_value.shape == (4, 1)
        assert legal_mask.shape == (4, MOVE_LABELS_NUM)

    def test_last_batch_smaller(self) -> None:
        """Last batch is smaller when data doesn't divide evenly."""
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=10
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )

        batches = list(dataset)
        # Last batch has 10 - 2*4 = 2 records
        features, _ = batches[-1]
        assert features[0].shape[0] == 2

    def test_total_records(self) -> None:
        """Total records across all batches equals total_rows."""
        source = FakePreprocessingSource(
            n_files=2, rows_per_file=7
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=3,
            shuffle=False,
            seed=42,
        )

        total = sum(batch[0][0].shape[0] for batch in dataset)
        assert total == 14

    def test_len(self) -> None:
        """__len__ returns correct batch count (per-file ceil sum)."""
        source = FakePreprocessingSource(
            n_files=2, rows_per_file=10
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )
        # Per-file: ceil(10/4) + ceil(10/4) = 3 + 3 = 6
        assert len(dataset) == 6

    def test_shuffle_changes_order(self) -> None:
        """Shuffle produces different record orders."""
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=20
        )
        ds_shuffled = StreamingKifDataset(
            streaming_source=source,
            batch_size=20,
            shuffle=True,
            seed=42,
        )
        ds_unshuffled = StreamingKifDataset(
            streaming_source=source,
            batch_size=20,
            shuffle=False,
            seed=42,
        )

        shuffled_batch = list(ds_shuffled)[0]
        unshuffled_batch = list(ds_unshuffled)[0]

        # Board tensors should differ in order
        assert not torch.equal(
            shuffled_batch[0][0], unshuffled_batch[0][0]
        )

    def test_set_epoch_changes_shuffle(self) -> None:
        """Different epochs produce different shuffle orders."""
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=20
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=20,
            shuffle=True,
            seed=42,
        )

        dataset.set_epoch(0)
        epoch0_batch = list(dataset)[0]

        dataset.set_epoch(1)
        epoch1_batch = list(dataset)[0]

        assert not torch.equal(
            epoch0_batch[0][0], epoch1_batch[0][0]
        )

    def test_result_value_shape(self) -> None:
        """result_value has shape (N, 1)."""
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=5
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=5,
            shuffle=False,
            seed=42,
        )

        _, targets = list(dataset)[0]
        assert targets[1].shape == (5, 1)
        assert targets[1].dtype == torch.float32

    def test_legal_move_mask_is_ones(self) -> None:
        """Legal move mask is all ones for preprocessing data."""
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=5
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=5,
            shuffle=False,
            seed=42,
        )

        _, targets = list(dataset)[0]
        assert torch.all(targets[2] == 1.0)


# ============================================================================
# StreamingStage1Dataset tests
# ============================================================================


class TestStreamingStage1Dataset:
    """Test StreamingStage1Dataset."""

    def test_batch_shapes(self) -> None:
        """Yielded batches have correct shapes."""
        source = FakeStage1Source(n_files=1, rows_per_file=8)
        dataset = StreamingStage1Dataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )

        batches = list(dataset)
        assert len(batches) == 2

        features, target = batches[0]
        board, pieces = features

        assert board.shape == (4, 9, 9)
        assert board.dtype == torch.uint8
        assert pieces.shape == (4, 14)
        assert pieces.dtype == torch.uint8
        assert target.shape == (4, 81)  # (9*9) flattened
        assert target.dtype == torch.float32

    def test_total_records(self) -> None:
        """Total records match source."""
        source = FakeStage1Source(n_files=2, rows_per_file=6)
        dataset = StreamingStage1Dataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )

        total = sum(batch[0][0].shape[0] for batch in dataset)
        assert total == 12

    def test_len(self) -> None:
        """__len__ returns correct batch count (per-file ceil sum)."""
        source = FakeStage1Source(n_files=2, rows_per_file=10)
        dataset = StreamingStage1Dataset(
            streaming_source=source,
            batch_size=3,
            shuffle=False,
            seed=42,
        )
        # Per-file: ceil(10/3) + ceil(10/3) = 4 + 4 = 8
        assert len(dataset) == 8

    def test_set_epoch(self) -> None:
        """Different epochs produce different orders."""
        source = FakeStage1Source(n_files=1, rows_per_file=20)
        dataset = StreamingStage1Dataset(
            streaming_source=source,
            batch_size=20,
            shuffle=True,
            seed=42,
        )

        dataset.set_epoch(0)
        epoch0 = list(dataset)[0]

        dataset.set_epoch(1)
        epoch1 = list(dataset)[0]

        assert not torch.equal(epoch0[0][0], epoch1[0][0])


# ============================================================================
# StreamingStage2Dataset tests
# ============================================================================


class TestStreamingStage2Dataset:
    """Test StreamingStage2Dataset."""

    def test_batch_shapes(self) -> None:
        """Yielded batches have correct shapes."""
        source = FakeStage2Source(n_files=1, rows_per_file=10)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )

        batches = list(dataset)
        assert len(batches) == 3

        features, target = batches[0]
        board, pieces = features

        assert board.shape == (4, 9, 9)
        assert board.dtype == torch.uint8
        assert pieces.shape == (4, 14)
        assert pieces.dtype == torch.uint8
        assert target.shape == (4, MOVE_LABELS_NUM)
        assert target.dtype == torch.float32

    def test_total_records(self) -> None:
        """Total records match source."""
        source = FakeStage2Source(n_files=2, rows_per_file=5)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=3,
            shuffle=False,
            seed=42,
        )

        total = sum(batch[0][0].shape[0] for batch in dataset)
        assert total == 10

    def test_len(self) -> None:
        """__len__ returns correct batch count (per-file ceil sum)."""
        source = FakeStage2Source(n_files=3, rows_per_file=10)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=7,
            shuffle=False,
            seed=42,
        )
        # Per-file: ceil(10/7) * 3 = 2 * 3 = 6
        assert len(dataset) == 6

    def test_set_epoch(self) -> None:
        """Different epochs produce different orders."""
        source = FakeStage2Source(n_files=1, rows_per_file=20)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=20,
            shuffle=True,
            seed=42,
        )

        dataset.set_epoch(0)
        epoch0 = list(dataset)[0]

        dataset.set_epoch(1)
        epoch1 = list(dataset)[0]

        assert not torch.equal(epoch0[0][0], epoch1[0][0])


# ============================================================================
# DataLoader integration tests
# ============================================================================


class TestDataLoaderIntegration:
    """Test DataLoader integration with streaming datasets."""

    def test_batch_size_none_passthrough(self) -> None:
        """DataLoader with batch_size=None passes batches through."""
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=10
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 3

        features, targets = batches[0]
        assert features[0].shape == (4, 9, 9)

    def test_total_records_through_dataloader(
        self,
    ) -> None:
        """All records pass through DataLoader."""
        source = FakePreprocessingSource(
            n_files=2, rows_per_file=7
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=3,
            shuffle=False,
            seed=42,
        )

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        total = sum(batch[0][0].shape[0] for batch in loader)
        assert total == 14

    def test_stage1_dataloader(self) -> None:
        """Stage1 dataset works with DataLoader."""
        source = FakeStage1Source(n_files=1, rows_per_file=8)
        dataset = StreamingStage1Dataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 2
        features, target = batches[0]
        assert target.shape == (4, 81)

    def test_stage2_dataloader(self) -> None:
        """Stage2 dataset works with DataLoader."""
        source = FakeStage2Source(n_files=1, rows_per_file=6)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=3,
            shuffle=False,
            seed=42,
        )

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 2
        features, target = batches[0]
        assert target.shape == (3, MOVE_LABELS_NUM)


# ============================================================================
# Empty source tests
# ============================================================================


class TestEmptySource:
    """Test behavior with empty data source."""

    def test_kif_empty_source(self) -> None:
        """Empty source yields no batches."""
        source = FakePreprocessingSource(
            n_files=0, rows_per_file=0
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )
        assert list(dataset) == []

    def test_stage1_empty_source(self) -> None:
        """Empty source yields no batches."""
        source = FakeStage1Source(n_files=0, rows_per_file=0)
        dataset = StreamingStage1Dataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )
        assert list(dataset) == []

    def test_stage2_empty_source(self) -> None:
        """Empty source yields no batches."""
        source = FakeStage2Source(n_files=0, rows_per_file=0)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )
        assert list(dataset) == []


# ============================================================================
# Worker file splitting tests (Phase 1)
# ============================================================================


class TestResolveWorkerFiles:
    """Test _resolve_worker_files helper function."""

    def test_no_worker_returns_all_files(self) -> None:
        """Without worker context, all files are returned."""
        source = FakePreprocessingSource(
            n_files=4, rows_per_file=10
        )
        # No worker context (main process)
        files = _resolve_worker_files(
            source, shuffle=False, epoch_seed=0
        )
        assert len(files) == 4

    def test_shuffle_changes_file_order(self) -> None:
        """Shuffle reorders files."""
        source = FakePreprocessingSource(
            n_files=10, rows_per_file=5
        )
        files_epoch0 = _resolve_worker_files(
            source, shuffle=True, epoch_seed=0
        )
        files_epoch1 = _resolve_worker_files(
            source, shuffle=True, epoch_seed=1
        )
        # Different epoch seeds should produce different orders
        assert files_epoch0 != files_epoch1

    def test_shuffle_preserves_all_files(self) -> None:
        """Shuffle preserves the complete set of files."""
        source = FakePreprocessingSource(
            n_files=6, rows_per_file=5
        )
        original = source.file_paths
        shuffled = _resolve_worker_files(
            source, shuffle=True, epoch_seed=42
        )
        assert set(str(f) for f in shuffled) == set(
            str(f) for f in original
        )

    def test_no_shuffle_preserves_order(self) -> None:
        """Without shuffle, file order is preserved."""
        source = FakePreprocessingSource(
            n_files=4, rows_per_file=5
        )
        files = _resolve_worker_files(
            source, shuffle=False, epoch_seed=0
        )
        assert files == source.file_paths

    def test_total_records_across_workers_sum_correctly(
        self,
    ) -> None:
        """Total records from all datasets equal source total (no worker context)."""
        source = FakePreprocessingSource(
            n_files=4, rows_per_file=10
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=5,
            shuffle=False,
            seed=42,
        )
        total = sum(batch[0][0].shape[0] for batch in dataset)
        assert total == 40


class TestWorkerFileSplitIntegration:
    """Integration tests for worker file splitting with datasets."""

    def test_kif_single_worker_all_data(self) -> None:
        """Single-worker mode processes all files."""
        source = FakePreprocessingSource(
            n_files=3, rows_per_file=10
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=5,
            shuffle=False,
            seed=42,
        )
        total = sum(batch[0][0].shape[0] for batch in dataset)
        assert total == 30

    def test_stage1_single_worker_all_data(self) -> None:
        """Stage1 single-worker mode processes all files."""
        source = FakeStage1Source(n_files=3, rows_per_file=8)
        dataset = StreamingStage1Dataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )
        total = sum(batch[0][0].shape[0] for batch in dataset)
        assert total == 24

    def test_stage2_single_worker_all_data(self) -> None:
        """Stage2 single-worker mode processes all files."""
        source = FakeStage2Source(n_files=3, rows_per_file=6)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=3,
            shuffle=False,
            seed=42,
        )
        total = sum(batch[0][0].shape[0] for batch in dataset)
        assert total == 18


# ============================================================================
# Zero-copy optimization tests (Phase 2)
# ============================================================================


class TestZeroCopyOptimization:
    """Test that .copy() removal doesn't affect correctness."""

    def test_kif_batch_values_correct(self) -> None:
        """Kif batch tensor values are correct without .copy()."""
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=5
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=5,
            shuffle=False,
            seed=42,
        )
        batches = list(dataset)
        features, targets = batches[0]
        board, pieces = features
        move_label, result_value, _ = targets

        # Values should be non-trivial (not all zeros)
        assert board.sum() > 0
        assert pieces.sum() > 0
        assert move_label.sum() > 0
        assert result_value.sum() > 0

    def test_stage1_batch_values_correct(self) -> None:
        """Stage1 batch tensor values are correct without .copy()."""
        source = FakeStage1Source(n_files=1, rows_per_file=5)
        dataset = StreamingStage1Dataset(
            streaming_source=source,
            batch_size=5,
            shuffle=False,
            seed=42,
        )
        batches = list(dataset)
        features, target = batches[0]
        board, pieces = features

        assert board.sum() > 0
        assert pieces.sum() > 0
        assert target.sum() > 0

    def test_stage2_batch_values_correct(self) -> None:
        """Stage2 batch tensor values are correct without .copy()."""
        source = FakeStage2Source(n_files=1, rows_per_file=5)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=5,
            shuffle=False,
            seed=42,
        )
        batches = list(dataset)
        features, target = batches[0]
        board, pieces = features

        assert board.sum() > 0
        assert pieces.sum() > 0
        assert target.sum() > 0

    def test_columnar_batch_slice_is_contiguous(
        self,
    ) -> None:
        """ColumnarBatch.slice() returns C-contiguous arrays."""
        rng = np.random.default_rng(42)
        batch = ColumnarBatch(
            board_positions=rng.integers(
                0, 30, size=(10, 9, 9), dtype=np.uint8
            ),
            pieces_in_hand=rng.integers(
                0, 5, size=(10, 14), dtype=np.uint8
            ),
            move_label=rng.random((10, MOVE_LABELS_NUM)).astype(
                np.float16
            ),
            result_value=rng.random(10).astype(np.float16),
        )

        indices = np.array([3, 1, 7, 5])
        sliced = batch.slice(indices)

        assert sliced.board_positions.flags["C_CONTIGUOUS"]
        assert sliced.pieces_in_hand.flags["C_CONTIGUOUS"]
        assert sliced.move_label is not None
        assert sliced.move_label.flags["C_CONTIGUOUS"]
        assert sliced.result_value is not None
        assert sliced.result_value.flags["C_CONTIGUOUS"]


class TestComputeTotalBatches:
    """_compute_total_batches ヘルパー関数のテスト."""

    def test_exact_division(self) -> None:
        """全ファイルがbatch_sizeの倍数の場合."""
        assert (
            _compute_total_batches([100, 200], batch_size=100)
            == 3
        )

    def test_remainder_per_file(self) -> None:
        """各ファイルに端数がある場合，ファイルごとにceilされる."""
        # ceil(150/100) + ceil(150/100) = 2 + 2 = 4
        assert (
            _compute_total_batches([150, 150], batch_size=100)
            == 4
        )

    def test_single_file(self) -> None:
        """単一ファイルの場合."""
        assert (
            _compute_total_batches([250], batch_size=100) == 3
        )

    def test_empty(self) -> None:
        """ファイルなしの場合."""
        assert _compute_total_batches([], batch_size=100) == 0

    def test_zero_row_file(self) -> None:
        """0行ファイルを含む場合."""
        assert (
            _compute_total_batches([0, 100], batch_size=100)
            == 1
        )

    def test_batch_size_larger_than_file(self) -> None:
        """batch_sizeが各ファイルの行数より大きい場合，各ファイルが1バッチになる."""
        # ceil(50/1000) + ceil(80/1000) + ceil(30/1000) = 1 + 1 + 1 = 3
        assert (
            _compute_total_batches(
                [50, 80, 30], batch_size=1000
            )
            == 3
        )


# ============================================================================
# Fix A: __iter__ 例外ハンドリングテスト
# ============================================================================


class ErrorSource:
    """イテレーション中にエラーを発生させるテスト用ソース．"""

    def __init__(self, *, n_files: int = 2) -> None:
        self._file_paths = [
            Path(f"/fake/err_{i}.feather")
            for i in range(n_files)
        ]

    @property
    def file_paths(self) -> list[Path]:
        return list(self._file_paths)

    @property
    def total_rows(self) -> int:
        return 100

    @property
    def row_counts(self) -> list[int]:
        return [50, 50]

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        raise RuntimeError("Simulated file read error")

    def iter_files_columnar_subset(
        self,
        file_paths: list[Path],
    ) -> Generator[ColumnarBatch, None, None]:
        raise RuntimeError("Simulated file read error")


_STREAMING_LOGGER = "maou.app.learning.streaming_dataset"


def test_streaming_kif_dataset_exception_logged_and_reraised() -> (
    None
):
    """StreamingKifDataset.__iter__ で例外がログ出力されて再raiseされること．"""
    source = ErrorSource()
    dataset = StreamingKifDataset(
        streaming_source=source,  # type: ignore[arg-type]
        batch_size=32,
        shuffle=False,
    )

    target_logger = logging.getLogger(_STREAMING_LOGGER)
    with patch.object(target_logger, "error") as mock_error:
        with pytest.raises(
            RuntimeError, match="Simulated file read error"
        ):
            list(dataset)

    mock_error.assert_called_once()
    assert (
        "crashed during iteration" in mock_error.call_args[0][0]
    )


def test_streaming_stage1_dataset_exception_logged_and_reraised() -> (
    None
):
    """StreamingStage1Dataset.__iter__ で例外がログ出力されて再raiseされること．"""
    source = ErrorSource()
    dataset = StreamingStage1Dataset(
        streaming_source=source,  # type: ignore[arg-type]
        batch_size=32,
        shuffle=False,
    )

    target_logger = logging.getLogger(_STREAMING_LOGGER)
    with patch.object(target_logger, "error") as mock_error:
        with pytest.raises(
            RuntimeError, match="Simulated file read error"
        ):
            list(dataset)

    mock_error.assert_called_once()
    assert (
        "crashed during iteration" in mock_error.call_args[0][0]
    )


def test_streaming_stage2_dataset_exception_logged_and_reraised() -> (
    None
):
    """StreamingStage2Dataset.__iter__ で例外がログ出力されて再raiseされること．"""
    source = ErrorSource()
    dataset = StreamingStage2Dataset(
        streaming_source=source,  # type: ignore[arg-type]
        batch_size=32,
        shuffle=False,
    )

    target_logger = logging.getLogger(_STREAMING_LOGGER)
    with patch.object(target_logger, "error") as mock_error:
        with pytest.raises(
            RuntimeError, match="Simulated file read error"
        ):
            list(dataset)

    mock_error.assert_called_once()
    assert (
        "crashed during iteration" in mock_error.call_args[0][0]
    )


def test_streaming_kif_dataset_normal_operation_with_exception_handling() -> (
    None
):
    """例外ハンドリング追加後も正常動作が維持されること．"""
    source = FakePreprocessingSource(
        n_files=2, rows_per_file=10
    )
    dataset = StreamingKifDataset(
        streaming_source=source,  # type: ignore[arg-type]
        batch_size=5,
        shuffle=False,
    )

    batches = list(dataset)
    # 2 files × 10 rows / 5 batch_size = 4 batches
    assert len(batches) == 4
    for (board, pieces), (move, value, mask) in batches:
        assert board.shape[0] == 5
        assert pieces.shape[0] == 5
