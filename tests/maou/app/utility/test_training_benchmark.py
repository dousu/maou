"""Tests for training benchmark components."""

from __future__ import annotations

import pytest

from maou.app.utility.training_benchmark import (
    BenchmarkResult,
    TrainingBenchmarkConfig,
)
from maou.interface import utility_interface


class TestBenchmarkResult:
    """BenchmarkResult dataclass のテスト."""

    def _make_result(
        self,
        *,
        warmup_time: float = 34.0,
        warmup_batches: int = 5,
        measured_time: float = 4.75,
        measured_batches: int = 95,
        actual_average_batch_time: float = 0.05,
        data_load_method: str = "map-style",
    ) -> BenchmarkResult:
        """テスト用の BenchmarkResult を生成する."""
        return BenchmarkResult(
            total_epoch_time=warmup_time + measured_time,
            average_batch_time=0.04,
            actual_average_batch_time=actual_average_batch_time,
            total_batches=warmup_batches + measured_batches,
            warmup_time=warmup_time,
            warmup_batches=warmup_batches,
            measured_time=measured_time,
            measured_batches=measured_batches,
            data_loading_time=0.01,
            gpu_transfer_time=0.005,
            forward_pass_time=0.015,
            loss_computation_time=0.002,
            backward_pass_time=0.005,
            optimizer_step_time=0.003,
            final_loss=0.5,
            average_loss=0.6,
            samples_per_second=640.0,
            batches_per_second=20.0,
            data_load_method=data_load_method,
        )

    def test_warmup_fields_present(self) -> None:
        """ウォームアップ関連フィールドが正しく設定される."""
        result = self._make_result()
        assert result.warmup_time == 34.0
        assert result.warmup_batches == 5
        assert result.measured_time == 4.75
        assert result.measured_batches == 95

    def test_actual_average_batch_time_excludes_warmup(
        self,
    ) -> None:
        """actual_average_batch_time がウォームアップを除外した値."""
        result = self._make_result(
            warmup_time=34.0,
            measured_time=4.75,
            measured_batches=95,
            actual_average_batch_time=4.75 / 95,
        )
        assert result.actual_average_batch_time == 4.75 / 95

    def test_estimated_epoch_time_excludes_warmup(self) -> None:
        """推定エポック時間がウォームアップを含まない定常速度で算出される.

        format_timing_summary 内の推定式:
            estimated = actual_average_batch_time * total_batches_in_dataset
        ウォームアップ時間は含めない（初回エポック限りのコスト）．
        """
        total_batches_in_dataset = 10000
        avg_batch_time = 0.05  # 50ms/batch

        result = self._make_result(
            warmup_time=34.0,
            actual_average_batch_time=avg_batch_time,
        )

        # 推定式（format_timing_summary と同じ）
        estimated = (
            result.actual_average_batch_time
            * total_batches_in_dataset
        )

        # 500秒（warmup 34秒は含まない）
        assert (
            estimated
            == avg_batch_time * total_batches_in_dataset
        )
        assert estimated == 500.0

    def test_to_dict_includes_warmup_fields(self) -> None:
        """to_dict にウォームアップ関連フィールドが含まれる."""
        result = self._make_result()
        d = result.to_dict()
        assert "warmup_time" in d
        assert "warmup_batches" in d
        assert "measured_time" in d
        assert "measured_batches" in d
        assert d["warmup_time"] == 34.0
        assert d["warmup_batches"] == 5.0
        assert d["measured_time"] == 4.75
        assert d["measured_batches"] == 95.0

    def test_zero_warmup(self) -> None:
        """warmup_batches=0 の場合のフィールド値."""
        result = self._make_result(
            warmup_time=0.0,
            warmup_batches=0,
            measured_time=10.0,
            measured_batches=100,
            actual_average_batch_time=0.1,
        )
        assert result.warmup_time == 0.0
        assert result.warmup_batches == 0
        assert result.total_epoch_time == 10.0

    def test_data_load_method_default(self) -> None:
        """data_load_method のデフォルト値が map-style."""
        result = self._make_result()
        assert result.data_load_method == "map-style"

    def test_data_load_method_streaming(self) -> None:
        """data_load_method に streaming を設定できる."""
        result = self._make_result(data_load_method="streaming")
        assert result.data_load_method == "streaming"

    def test_to_dict_includes_data_load_method(self) -> None:
        """to_dict に data_load_method が含まれる."""
        result = self._make_result(data_load_method="streaming")
        d = result.to_dict()
        assert "data_load_method" in d
        assert d["data_load_method"] == "streaming"


class TestTrainingBenchmarkConfig:
    """TrainingBenchmarkConfig dataclass のテスト."""

    def test_warmup_batches_default_is_10(self) -> None:
        """warmup_batches デフォルトが10."""
        config = TrainingBenchmarkConfig()
        assert config.warmup_batches == 10

    def test_streaming_defaults(self) -> None:
        """streaming関連フィールドのデフォルト値."""
        config = TrainingBenchmarkConfig()
        assert config.streaming is False
        assert config.streaming_train_source is None
        assert config.streaming_val_source is None

    def test_datasource_optional(self) -> None:
        """datasource が None を許容する."""
        config = TrainingBenchmarkConfig(datasource=None)
        assert config.datasource is None


class TestBenchmarkTrainingValidation:
    """Interface層バリデーションのテスト."""

    def test_non_streaming_without_datasource_raises_value_error(
        self,
    ) -> None:
        """streaming=False かつ datasource=None で ValueError."""
        with pytest.raises(
            ValueError, match="datasource is required"
        ):
            utility_interface.benchmark_training(
                datasource=None,
                streaming=False,
            )

    def test_streaming_without_train_source_raises_value_error(
        self,
    ) -> None:
        """streaming=True かつ streaming_train_source=None で ValueError."""
        with pytest.raises(
            ValueError,
            match="streaming_train_source is required",
        ):
            utility_interface.benchmark_training(
                datasource=None,
                streaming=True,
                streaming_train_source=None,
            )
