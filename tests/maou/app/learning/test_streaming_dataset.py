"""Tests for StreamingDataset (IterableDataset implementations)."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
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
    _compute_concat_total_batches,
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
        target_set = {str(fp) for fp in file_paths}
        for fp in all_paths:
            batch = self._make_batch(rng)
            if str(fp) in target_set:
                yield batch


class FakePreprocessingSourceWithWinRate:
    """Fake source for preprocessing data with move_win_rate."""

    def __init__(
        self, *, n_files: int = 2, rows_per_file: int = 10
    ) -> None:
        self._n_files = n_files
        self._rows_per_file = rows_per_file
        self._file_paths = [
            Path(f"/fake/wr_file_{i}.feather")
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
            move_win_rate=rng.random(
                (n, MOVE_LABELS_NUM)
            ).astype(np.float32),
        )

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(321)
        for i in range(self._n_files):
            yield self._make_batch(rng)

    def iter_files_columnar_subset(
        self,
        file_paths: list[Path],
    ) -> Generator[ColumnarBatch, None, None]:
        rng = np.random.default_rng(321)
        all_paths = self._file_paths
        target_set = {str(fp) for fp in file_paths}
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
        target_set = {str(fp) for fp in file_paths}
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
        target_set = {str(fp) for fp in file_paths}
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
        move_label, result_value, _move_win_rate = targets

        assert board.shape == (4, 9, 9)
        assert board.dtype == torch.uint8
        assert pieces.shape == (4, 14)
        assert pieces.dtype == torch.uint8
        assert move_label.shape == (4, MOVE_LABELS_NUM)
        assert result_value.shape == (4, 1)
        # legal_move_mask は yield されない (常に全 1 の no-op
        # だったため targets から外した)．
        assert len(targets) == 3

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

        shuffled_batch = next(iter(ds_shuffled))
        unshuffled_batch = next(iter(ds_unshuffled))

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
        epoch0_batch = next(iter(dataset))

        dataset.set_epoch(1)
        epoch1_batch = next(iter(dataset))

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

        _, targets = next(iter(dataset))
        assert targets[1].shape == (5, 1)
        assert targets[1].dtype == torch.float32

    def test_legal_move_mask_not_yielded(self) -> None:
        """legal_move_mask は targets に含まれない．

        この経路が作れるマスクは常に全 1 で消費側では
        no-op なのに，バッチ毎に moveLabel と同じサイズを
        PCIe 上に流していた．3 要素目は move_win_rate．
        """
        source = FakePreprocessingSource(
            n_files=1, rows_per_file=5
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=5,
            shuffle=False,
            seed=42,
        )

        _, targets = next(iter(dataset))
        assert len(targets) == 3
        assert targets[2] is None


class TestStreamingKifDatasetMoveWinRate:
    """Test StreamingKifDataset with move_win_rate."""

    def test_move_win_rate_present(self) -> None:
        """3rd target element is a tensor when move_win_rate exists."""
        source = FakePreprocessingSourceWithWinRate(
            n_files=1, rows_per_file=6
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=6,
            shuffle=False,
            seed=42,
        )

        batches = list(dataset)
        _, targets = batches[0]
        (
            _move_label,
            _result_value,
            move_win_rate,
        ) = targets

        assert move_win_rate is not None
        assert move_win_rate.shape == (6, MOVE_LABELS_NUM)
        assert move_win_rate.dtype == torch.float32

    def test_move_win_rate_absent(self) -> None:
        """3rd target element is None when move_win_rate is absent."""
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
        _, targets = batches[0]
        assert targets[2] is None

    def test_move_win_rate_values_nonzero(self) -> None:
        """move_win_rate tensor has non-trivial values."""
        source = FakePreprocessingSourceWithWinRate(
            n_files=1, rows_per_file=5
        )
        dataset = StreamingKifDataset(
            streaming_source=source,
            batch_size=5,
            shuffle=False,
            seed=42,
        )

        batches = list(dataset)
        _, targets = batches[0]
        assert targets[2] is not None
        assert targets[2].sum() > 0


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
        epoch0 = next(iter(dataset))

        dataset.set_epoch(1)
        epoch1 = next(iter(dataset))

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
        """__len__ counts concat groups, not files.

        ``__iter__`` は ``_FILES_PER_CONCAT`` 個までのファイルを結合
        してからバッチ化するので，端数バッチはファイルごとではなく
        結合グループごとにしか出ない．
        """
        source = FakeStage2Source(n_files=3, rows_per_file=10)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=7,
            shuffle=False,
            seed=42,
        )
        # 3 files < _FILES_PER_CONCAT なので 1 グループ:
        # ceil(30 / 7) = 5．ファイル単位で数えると
        # ceil(10/7) * 3 = 6 で過大評価になる．
        assert len(dataset) == 5

    def test_len_matches_actual_iteration(self) -> None:
        """__len__ は実際に yield されるバッチ数と一致する．

        これが崩れると ``steps_per_epoch`` 経由で LR スケジューラの
        総ステップ数がずれる (cosine decay が下がりきらない)．
        """
        for n_files, rows, batch in [
            (3, 10, 7),
            (1, 10, 4),
            (5, 33, 16),
            (12, 10, 7),
        ]:
            source = FakeStage2Source(
                n_files=n_files, rows_per_file=rows
            )
            dataset = StreamingStage2Dataset(
                streaming_source=source,
                batch_size=batch,
                shuffle=False,
                seed=42,
            )
            assert len(dataset) == len(list(dataset)), (
                f"n_files={n_files} rows={rows} batch={batch}"
            )

    def test_set_num_workers_changes_len(self) -> None:
        """ワーカー数を上げると端数グループが増えてバッチ数が増える．"""
        source = FakeStage2Source(n_files=20, rows_per_file=100)
        dataset = StreamingStage2Dataset(
            streaming_source=source,
            batch_size=64,
            shuffle=False,
            seed=42,
        )
        # 1 shard: 2 グループ (10, 10) → ceil(1000/64) * 2 = 32
        assert len(dataset) == 32

        # 20 workers: 各 shard 1 ファイル → ceil(100/64) * 20 = 40
        dataset.set_num_workers(20)
        assert len(dataset) == 40

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
        epoch0 = next(iter(dataset))

        dataset.set_epoch(1)
        epoch1 = next(iter(dataset))

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

        features, _targets = batches[0]
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
        _features, target = batches[0]
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
        _features, target = batches[0]
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
        assert {str(f) for f in shuffled} == {
            str(f) for f in original
        }

    def test_no_shuffle_preserves_order(self) -> None:
        """Without shuffle, file order is preserved."""
        source = FakePreprocessingSource(
            n_files=4, rows_per_file=5
        )
        files = _resolve_worker_files(
            source, shuffle=False, epoch_seed=0
        )
        assert files == source.file_paths

    def test_shuffle_multiworker_disjoint_partition(
        self,
    ) -> None:
        """All workers produce disjoint file sets covering all files.

        PyTorchは worker_info.seed = base_seed + worker_id を設定する．
        ファイルシャッフルは base_seed のみに依存し，
        全ワーカーが同一の並び順を共有する必要がある．
        """
        source = FakePreprocessingSource(
            n_files=20, rows_per_file=10
        )
        n_workers = 4
        base_seed = 12345
        all_files: list[list[Path]] = []
        for wid in range(n_workers):
            worker_seed = base_seed + wid
            info = type(
                "WorkerInfo",
                (),
                {
                    "id": wid,
                    "num_workers": n_workers,
                    "seed": worker_seed,
                    "dataset": None,
                },
            )()
            with patch(
                "maou.app.learning.streaming_dataset"
                ".torch.utils.data.get_worker_info",
                return_value=info,
            ):
                files = _resolve_worker_files(
                    source,
                    shuffle=True,
                    epoch_seed=worker_seed,
                )
            all_files.append(files)

        # 各ワーカーのファイルが重複なく全ファイルをカバーする
        flat = [str(f) for group in all_files for f in group]
        original = [str(f) for f in source.file_paths]
        assert sorted(flat) == sorted(original), (
            f"Files not partitioned correctly: "
            f"got {len(flat)} files, expected {len(original)}. "
            f"Duplicates: {len(flat) - len(set(flat))}"
        )

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
        move_label, result_value, *_ = targets

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


class TestComputeConcatTotalBatches:
    """_compute_concat_total_batches ヘルパー関数のテスト."""

    def test_group_is_ceiled_once_not_per_file(self) -> None:
        """結合グループごとに 1 回だけ ceil される."""
        # 1 グループ (3 ファイル) → ceil(30/7) = 5
        assert (
            _compute_concat_total_batches(
                [10, 10, 10], batch_size=7, num_workers=0
            )
            == 5
        )

    def test_group_boundary_at_group_size(self) -> None:
        """group_size を超えると新しいグループになる."""
        # 12 ファイル → グループ (10, 2)
        # ceil(100/7) + ceil(20/7) = 15 + 3 = 18
        assert (
            _compute_concat_total_batches(
                [10] * 12, batch_size=7, num_workers=0
            )
            == 18
        )

    def test_workers_shard_round_robin(self) -> None:
        """worker 数だけ shard に分かれ，端数グループが shard ごとに出る."""
        # 20 ファイル x 100 行, batch=64
        # 1 shard: グループ (10, 10) → ceil(1000/64) * 2 = 32
        assert (
            _compute_concat_total_batches(
                [100] * 20, batch_size=64, num_workers=0
            )
            == 32
        )
        # 2 shards: 各 10 ファイル → ceil(1000/64) * 2 = 32
        assert (
            _compute_concat_total_batches(
                [100] * 20, batch_size=64, num_workers=2
            )
            == 32
        )
        # 20 shards: 各 1 ファイル → ceil(100/64) * 20 = 40
        assert (
            _compute_concat_total_batches(
                [100] * 20, batch_size=64, num_workers=20
            )
            == 40
        )

    def test_more_workers_than_files(self) -> None:
        """ファイル数より worker 数が多い場合，空 shard は 0 を足す."""
        assert (
            _compute_concat_total_batches(
                [100, 100], batch_size=64, num_workers=8
            )
            == 4
        )

    def test_empty(self) -> None:
        """ファイルなしの場合."""
        assert (
            _compute_concat_total_batches(
                [], batch_size=100, num_workers=4
            )
            == 0
        )

    def test_zero_row_file(self) -> None:
        """0 行ファイルはバッチを生まない."""
        assert (
            _compute_concat_total_batches(
                [0, 0], batch_size=100, num_workers=0
            )
            == 0
        )

    def test_shard_round_robin_matches_resolve_worker_files(
        self,
    ) -> None:
        """shard の切り方が _resolve_worker_files と同じ規則である."""
        row_counts = [10, 20, 30, 40, 50]
        n_workers = 2
        for worker_id in range(n_workers):
            # _resolve_worker_files は i % n_workers == worker_id
            expected = [
                n
                for i, n in enumerate(row_counts)
                if i % n_workers == worker_id
            ]
            assert row_counts[worker_id::n_workers] == expected


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
    with (
        patch.object(target_logger, "error") as mock_error,
        pytest.raises(
            RuntimeError, match="Simulated file read error"
        ),
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
    with (
        patch.object(target_logger, "error") as mock_error,
        pytest.raises(
            RuntimeError, match="Simulated file read error"
        ),
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
    with (
        patch.object(target_logger, "error") as mock_error,
        pytest.raises(
            RuntimeError, match="Simulated file read error"
        ),
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
    for (board, pieces), (move, value, _) in batches:
        assert board.shape[0] == 5
        assert pieces.shape[0] == 5
