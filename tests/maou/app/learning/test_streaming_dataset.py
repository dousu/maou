"""Tests for StreamingDataset (IterableDataset implementations)."""

from __future__ import annotations

import math
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from maou.app.learning.streaming_dataset import (
    StreamingDataSource,
    StreamingKifDataset,
    StreamingStage1Dataset,
    StreamingStage2Dataset,
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

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(123)
        for i in range(self._n_files):
            n = self._rows_per_file
            yield ColumnarBatch(
                board_positions=rng.integers(
                    0, 30, size=(n, 9, 9), dtype=np.uint8
                ),
                pieces_in_hand=rng.integers(
                    0, 5, size=(n, 14), dtype=np.uint8
                ),
                move_label=rng.random(
                    (n, MOVE_LABELS_NUM)
                ).astype(np.float16),
                result_value=rng.random(n).astype(np.float16),
            )


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

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(456)
        for i in range(self._n_files):
            n = self._rows_per_file
            yield ColumnarBatch(
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

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(789)
        for i in range(self._n_files):
            n = self._rows_per_file
            yield ColumnarBatch(
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
        """__len__ returns correct batch count."""
        source = FakePreprocessingSource(
            n_files=2, rows_per_file=10
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=4,
            shuffle=False,
            seed=42,
        )
        assert len(dataset) == math.ceil(20 / 4)

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
        """__len__ returns correct batch count."""
        source = FakeStage1Source(n_files=2, rows_per_file=10)
        dataset = StreamingStage1Dataset(
            streaming_source=source,
            batch_size=3,
            shuffle=False,
            seed=42,
        )
        assert len(dataset) == math.ceil(20 / 3)

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
        """__len__ returns correct batch count."""
        source = FakeStage2Source(n_files=3, rows_per_file=10)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=7,
            shuffle=False,
            seed=42,
        )
        assert len(dataset) == math.ceil(30 / 7)

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
