# -*- coding: utf-8 -*-
"""Tests for maou.app.learning.gpu_prefetcher."""

import queue
import threading
from collections.abc import Iterator
from unittest.mock import patch

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


# --- Track 1-3: バッファサイズのワーカー数連動テスト ---


@pytest.mark.parametrize(
    ("batch_size", "num_workers", "expected"),
    [
        pytest.param(1024, 12, 6, id="1024_12w_halved"),
        pytest.param(1024, 4, 8, id="1024_4w_two_thirds"),
        pytest.param(1024, 0, 12, id="1024_0w_no_suppression"),
        pytest.param(1024, 2, 12, id="1024_2w_no_suppression"),
        pytest.param(
            128, 12, 3, id="128_12w_small_batch_no_suppression"
        ),
        pytest.param(
            256, 12, 5, id="256_12w_small_batch_no_suppression"
        ),
        pytest.param(512, 8, 4, id="512_8w_halved"),
        pytest.param(512, 4, 5, id="512_4w_two_thirds"),
        pytest.param(2048, 8, 4, id="2048_8w_halved"),
        pytest.param(4096, 8, 2, id="4096_8w_halved_min2"),
    ],
)
def test_buffer_size_with_num_workers(
    batch_size: int, num_workers: int, expected: int
) -> None:
    """ワーカー数によるバッファサイズ抑制の検証．"""
    result = calculate_recommended_buffer_size(
        batch_size, num_workers=num_workers
    )
    assert result == expected


def test_buffer_size_backward_compatible() -> None:
    """num_workers=0(デフォルト)で既存動作と同一であることを検証する．"""
    for bs in [64, 128, 256, 512, 1024, 2048, 4096]:
        assert calculate_recommended_buffer_size(
            bs, num_workers=0
        ) == calculate_recommended_buffer_size(bs)


def test_buffer_size_minimum_is_two() -> None:
    """バッファサイズが最小2を下回らないことを検証する．"""
    # 最小バッファテスト: 大バッチ+多ワーカーでも2以上
    result = calculate_recommended_buffer_size(
        4096, num_workers=16
    )
    assert result >= 2


# --- Fix 1: ローダースレッド例外伝播テスト ---


def _make_error_loader_thread(
    error: Exception,
    batches: list | None = None,
    delay_error_until: "threading.Event | None" = None,
):
    """_loader_thread の差し替え用関数を生成する．

    キューにバッチを投入した後，例外を設定して None センチネルを投入する．

    Args:
        error: 発生させる例外
        batches: 投入するバッチのリスト(None の場合は空)
        delay_error_until: 設定時，このイベントが set されるまで
            例外発生を遅延させる(部分成功テスト用)
    """

    def _fake_loader_thread(self) -> None:
        try:
            for batch in batches or []:
                self.queue.put(batch)
            if delay_error_until is not None:
                delay_error_until.wait(timeout=5.0)
            raise error
        except Exception as e:
            self.exception = e
        finally:
            self.queue.put(None)

    return _fake_loader_thread


def test_prefetcher_propagates_loader_thread_exception() -> (
    None
):
    """ローダースレッドでの例外がイテレータに伝播することを検証する．"""
    dataset = _FixedBatchDataset(0)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    prefetcher = DataPrefetcher(
        loader, device="cpu", buffer_size=2
    )

    error = RuntimeError("simulated loader error")
    fake_thread = _make_error_loader_thread(error)

    with (
        patch.object(
            type(prefetcher),
            "_loader_thread",
            fake_thread,
        ),
        pytest.raises(
            RuntimeError, match="simulated loader error"
        ),
    ):
        for _ in prefetcher:
            pass


def test_prefetcher_propagates_exception_after_partial_iteration() -> (
    None
):
    """途中まで正常にバッチを返した後に例外が発生するケースを検証する．"""
    dataset = _FixedBatchDataset(0)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    prefetcher = DataPrefetcher(
        loader, device="cpu", buffer_size=4
    )

    # 2バッチ成功後にエラーが発生するシナリオ
    batch1 = (
        torch.zeros(2, 3),
        (torch.zeros(2), torch.zeros(2), torch.zeros(2)),
    )
    batch2 = (
        torch.ones(2, 3),
        (torch.ones(2), torch.ones(2), torch.ones(2)),
    )
    error = RuntimeError("I/O error after 2 batches")

    # バッチ消費後まで例外発生を遅延させる
    consume_done = threading.Event()
    fake_thread = _make_error_loader_thread(
        error,
        batches=[batch1, batch2],
        delay_error_until=consume_done,
    )

    from unittest.mock import patch

    with patch.object(
        type(prefetcher),
        "_loader_thread",
        fake_thread,
    ):
        it = iter(prefetcher)

        # 最初の2バッチは正常に取得できる
        first = next(it)
        assert first[0].shape == torch.Size([2, 3])

        second = next(it)
        assert second[0].shape == torch.Size([2, 3])

        # バッチ消費完了をシグナルし，例外発生を許可
        consume_done.set()

        # 3回目の next() で None を受信し，例外が伝播する
        with pytest.raises(
            RuntimeError, match="I/O error after 2 batches"
        ):
            next(it)
