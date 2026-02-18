"""Tests for TimingCallback warmup exclusion."""

from __future__ import annotations

from unittest.mock import patch

import torch

from maou.app.learning.callbacks import (
    TimingCallback,
    TrainingContext,
)


def _make_dummy_context(batch_idx: int) -> TrainingContext:
    """テスト用の最小限の TrainingContext を生成する."""
    dummy_tensor = torch.zeros(1)
    return TrainingContext(
        batch_idx=batch_idx,
        epoch_idx=0,
        inputs=dummy_tensor,
        labels_policy=dummy_tensor,
        labels_value=dummy_tensor,
        legal_move_mask=None,
        loss=torch.tensor(0.5),
        batch_size=32,
    )


class TestTimingCallbackWarmupExclusion:
    """TimingCallback のウォームアップ除外テスト."""

    def _simulate_batches(
        self,
        callback: TimingCallback,
        n_batches: int,
        time_sequence: list[float],
    ) -> None:
        """指定した時間シーケンスでバッチを模擬実行する.

        time_sequence は time.perf_counter の返り値のリスト．
        各バッチで on_batch_start -> (forward等) -> on_batch_end が呼ばれ，
        その都度 perf_counter が呼ばれる．
        """
        time_iter = iter(time_sequence)

        def mock_perf_counter() -> float:
            return next(time_iter)

        with patch(
            "time.perf_counter", side_effect=mock_perf_counter
        ):
            callback.on_epoch_start(epoch_idx=0)

            for i in range(n_batches):
                ctx = _make_dummy_context(batch_idx=i)
                callback.on_batch_start(ctx)

                # on_forward_start 〜 on_optimizer_end の
                # タイミングを _temp_timings に直接設定
                callback._temp_timings = {
                    "data_loading": 0.001,
                    "gpu_transfer": 0.001,
                    "forward_pass": 0.005,
                    "loss_computation": 0.001,
                    "backward_pass": 0.003,
                    "optimizer_step": 0.002,
                }

                callback.on_batch_end(ctx)

    def test_measurement_start_time_set_after_warmup(
        self,
    ) -> None:
        """ウォームアップ完了後に _measurement_start_time が設定される."""
        callback = TimingCallback(warmup_batches=2)

        # 時間シーケンス:
        # on_epoch_start: t=0.0
        # batch 0 (warmup): on_batch_start=1.0, on_batch_end=2.0, 2.0
        # batch 1 (warmup): on_batch_start=3.0, on_batch_end=4.0, 4.0
        # batch 2 (measured): on_batch_start=5.0, on_batch_end=6.0, 6.0
        # batch 3 (measured): on_batch_start=7.0, on_batch_end=8.0, 8.0
        time_seq = [
            0.0,  # on_epoch_start
            1.0,  # batch 0: on_batch_start (perf_counter)
            2.0,
            2.0,  # batch 0: on_batch_end (perf_counter x2)
            3.0,  # batch 1: on_batch_start
            4.0,
            4.0,  # batch 1: on_batch_end
            5.0,  # batch 2: on_batch_start
            6.0,
            6.0,  # batch 2: on_batch_end
            7.0,  # batch 3: on_batch_start
            8.0,
            8.0,  # batch 3: on_batch_end
        ]

        self._simulate_batches(
            callback, n_batches=4, time_sequence=time_seq
        )

        # ウォームアップ完了後のbatch 2の開始時刻が記録されている
        assert callback._measurement_start_time == 5.0
        assert callback.measured_batches == 2

    def test_measurement_start_time_none_during_warmup_only(
        self,
    ) -> None:
        """ウォームアップバッチのみ実行した場合，_measurement_start_time は None."""
        callback = TimingCallback(warmup_batches=3)

        # 3バッチ = 全てウォームアップ
        time_seq = [
            0.0,  # on_epoch_start
            1.0,
            2.0,
            2.0,  # batch 0
            3.0,
            4.0,
            4.0,  # batch 1
            5.0,
            6.0,
            6.0,  # batch 2
        ]

        self._simulate_batches(
            callback, n_batches=3, time_sequence=time_seq
        )

        assert callback._measurement_start_time is None
        assert callback.measured_batches == 0

    def test_get_performance_metrics_excludes_warmup(
        self,
    ) -> None:
        """get_performance_metrics がウォームアップを除外した平均を返す."""
        callback = TimingCallback(warmup_batches=2)

        # ウォームアップ2バッチ(遅い) + 計測2バッチ(速い)
        time_seq = [
            0.0,  # on_epoch_start
            # batch 0 (warmup, slow): 10秒
            1.0,
            11.0,
            11.0,
            # batch 1 (warmup, slow): 10秒
            12.0,
            22.0,
            22.0,
            # batch 2 (measured, fast): 1秒
            23.0,
            24.0,
            24.0,
            # batch 3 (measured, fast): 1秒
            25.0,
            26.0,
            26.0,
        ]

        self._simulate_batches(
            callback, n_batches=4, time_sequence=time_seq
        )

        # get_performance_metrics 呼び出し時の perf_counter
        with patch("time.perf_counter", return_value=26.0):
            metrics = callback.get_performance_metrics(
                total_batches=4
            )

        # measured_time = 26.0 - 23.0 = 3.0秒
        assert metrics["measured_time"] == 3.0
        # actual_average_batch_time = 3.0 / 2 = 1.5秒/batch
        assert metrics["actual_average_batch_time"] == 1.5
        # warmup_time = 23.0 - 0.0 = 23.0秒
        assert metrics["warmup_time"] == 23.0
        # total_epoch_time = 26.0 - 0.0 = 26.0秒
        assert metrics["total_epoch_time"] == 26.0
        # measured_batches
        assert metrics["measured_batches"] == 2.0

    def test_get_performance_metrics_zero_warmup(self) -> None:
        """warmup_batches=0 の場合，全バッチが計測対象."""
        callback = TimingCallback(warmup_batches=0)

        time_seq = [
            0.0,  # on_epoch_start
            # batch 0 (measured): 1秒
            1.0,
            2.0,
            2.0,
            # batch 1 (measured): 1秒
            3.0,
            4.0,
            4.0,
        ]

        self._simulate_batches(
            callback, n_batches=2, time_sequence=time_seq
        )

        with patch("time.perf_counter", return_value=4.0):
            metrics = callback.get_performance_metrics(
                total_batches=2
            )

        # warmup_time ≈ 1.0 - 0.0 = 1.0秒（batch 0開始時刻 - エポック開始時刻）
        assert metrics["warmup_time"] == 1.0
        # measured_time = 4.0 - 1.0 = 3.0秒
        assert metrics["measured_time"] == 3.0
        # actual_average_batch_time = 3.0 / 2 = 1.5秒/batch
        assert metrics["actual_average_batch_time"] == 1.5
        assert metrics["measured_batches"] == 2.0
