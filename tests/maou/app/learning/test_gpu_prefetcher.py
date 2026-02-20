# -*- coding: utf-8 -*-
"""Tests for maou.app.learning.gpu_prefetcher."""

import queue
from collections.abc import Iterator

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from maou.app.learning.gpu_prefetcher import (
    DataPrefetcher,
    calculate_recommended_buffer_size,
)


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (64, 3),
        (128, 3),
        (256, 5),
        (512, 8),
        (1024, 12),
        (2048, 8),
        (4096, 4),
    ],
)
def test_calculate_recommended_buffer_size_ranges(
    batch_size: int, expected: int
) -> None:
    """Verify recommended buffer size for various batch sizes."""
    assert (
        calculate_recommended_buffer_size(batch_size)
        == expected
    )


class _FixedBatchDataset(IterableDataset):
    """Yields a fixed number of simple batches."""

    def __init__(self, num_batches: int) -> None:
        self.num_batches = num_batches

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ]:  # type: ignore[override]
        for _ in range(self.num_batches):
            features = torch.zeros(2, 3)
            targets = (
                torch.zeros(2),
                torch.zeros(2),
                torch.zeros(2),
            )
            yield features, targets


def test_prefetcher_iter_drains_multiple_sentinels() -> None:
    """Iteration must drain stale None sentinels without deadlocking."""
    num_batches = 4
    dataset = _FixedBatchDataset(num_batches)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    prefetcher = DataPrefetcher(
        loader, device="cpu", buffer_size=3
    )

    # First pass: consume all batches normally.
    first_pass_count = sum(1 for _ in prefetcher)
    assert first_pass_count == num_batches

    # After iteration the loader thread has finished.  Inject extra
    # None sentinels into the queue to simulate the scenario where
    # multiple sentinels are left over.
    prefetcher.queue.put(None)
    prefetcher.queue.put(None)
    prefetcher.queue.put(None)

    # Second pass must still work correctly: __iter__ resets the
    # queue, so the stale sentinels are discarded.
    second_pass_count = sum(1 for _ in prefetcher)
    assert second_pass_count == num_batches


def test_prefetcher_queue_empty_after_iteration() -> None:
    """Queue must be fully drained after a normal iteration completes."""
    num_batches = 5
    dataset = _FixedBatchDataset(num_batches)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    prefetcher = DataPrefetcher(
        loader, device="cpu", buffer_size=3
    )

    collected: list[
        tuple[torch.Tensor, tuple[torch.Tensor, ...]]
    ] = []
    for batch in prefetcher:
        collected.append(batch)

    assert len(collected) == num_batches

    # The queue must be empty — no residual items or sentinels.
    assert prefetcher.queue.empty()
    with pytest.raises(queue.Empty):
        prefetcher.queue.get_nowait()


def test_prefetcher_timeout_constants() -> None:
    """タイムアウト定数が期待値と整合することを検証する．"""
    assert DataPrefetcher.DEFAULT_TIMEOUT == 120.0
    assert DataPrefetcher.FIRST_BATCH_TIMEOUT == 300.0
    assert (
        DataPrefetcher.FIRST_BATCH_TIMEOUT
        > DataPrefetcher.DEFAULT_TIMEOUT
    )


def test_log_timeout_diagnostics_no_error() -> None:
    """_log_timeout_diagnosticsがシステム状態によらず例外を投げないことを検証する．"""
    dataset = _FixedBatchDataset(1)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    prefetcher = DataPrefetcher(
        loader, device="cpu", buffer_size=2
    )

    # diagnosticsが検査するqueue/threadを初期化
    prefetcher.queue = queue.Queue(maxsize=2)
    prefetcher.thread = None  # type: ignore[assignment]

    # is_first_batch=True/False の両方で例外なく完了すること
    prefetcher._log_timeout_diagnostics(is_first_batch=True)
    prefetcher._log_timeout_diagnostics(is_first_batch=False)
