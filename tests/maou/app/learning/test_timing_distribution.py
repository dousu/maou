"""Tests for TimingCallback.get_timing_distribution edge cases."""

from __future__ import annotations

from maou.app.learning.callbacks import TimingCallback


class TestGetTimingDistribution:
    """get_timing_distribution のエッジケーステスト."""

    def _populate_callback(
        self, cb: TimingCallback, values: list[float]
    ) -> None:
        """timing_stats にテストデータを直接注入する."""
        for key in cb.timing_stats:
            cb.timing_stats[key] = list(values)

    def test_empty_returns_none(self) -> None:
        """計測バッチがない場合は None を返す."""
        cb = TimingCallback(warmup_batches=0)
        result = cb.get_timing_distribution()
        assert result is None

    def test_single_value(self) -> None:
        """n=1: std=0, p50=value, min=max=value."""
        cb = TimingCallback(warmup_batches=0)
        self._populate_callback(cb, [0.05])
        result = cb.get_timing_distribution()
        assert result is not None
        stats = result["total_batch"]
        assert stats["std"] == 0.0
        assert stats["p50"] == 0.05
        assert stats["min"] == stats["max"] == 0.05

    def test_two_values_p50_is_median(self) -> None:
        """n=2: p50 は2値の平均 (正しい中央値)."""
        cb = TimingCallback(warmup_batches=0)
        self._populate_callback(cb, [0.04, 0.06])
        result = cb.get_timing_distribution()
        assert result is not None
        stats = result["total_batch"]
        assert stats["p50"] == 0.05  # (0.04 + 0.06) / 2

    def test_three_values_p50_is_middle(self) -> None:
        """n=3 (奇数): p50 は中央の値."""
        cb = TimingCallback(warmup_batches=0)
        self._populate_callback(cb, [0.03, 0.05, 0.07])
        result = cb.get_timing_distribution()
        assert result is not None
        stats = result["total_batch"]
        assert stats["p50"] == 0.05

    def test_large_n_percentiles_ordered(self) -> None:
        """十分なデータ: min <= p50 <= p95 <= p99 <= max."""
        cb = TimingCallback(warmup_batches=0)
        values = [float(i) for i in range(100)]
        self._populate_callback(cb, values)
        result = cb.get_timing_distribution()
        assert result is not None
        stats = result["total_batch"]
        assert stats["min"] <= stats["p50"]
        assert stats["p50"] <= stats["p95"]
        assert stats["p95"] <= stats["p99"]
        assert stats["p99"] <= stats["max"]

    def test_all_keys_present(self) -> None:
        """全タイミングカテゴリが結果に含まれる."""
        cb = TimingCallback(warmup_batches=0)
        self._populate_callback(cb, [0.05, 0.06])
        result = cb.get_timing_distribution()
        assert result is not None
        expected_keys = {
            "data_loading",
            "gpu_transfer",
            "forward_pass",
            "loss_computation",
            "backward_pass",
            "optimizer_step",
            "total_batch",
        }
        assert set(result.keys()) == expected_keys

    def test_stats_keys_present(self) -> None:
        """各カテゴリに必要な統計キーが含まれる."""
        cb = TimingCallback(warmup_batches=0)
        self._populate_callback(cb, [0.05, 0.06])
        result = cb.get_timing_distribution()
        assert result is not None
        expected_stat_keys = {
            "mean",
            "std",
            "min",
            "max",
            "p50",
            "p95",
            "p99",
        }
        for key, stats in result.items():
            assert set(stats.keys()) == expected_stat_keys, (
                f"Missing keys in {key}"
            )
