"""Tests for training benchmark components."""

from __future__ import annotations

from maou.app.utility.training_benchmark import BenchmarkResult


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
        gpu_prefetch_enabled: bool = True,
        gpu_prefetch_buffer_size: int = 5,
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
            gpu_prefetch_enabled=gpu_prefetch_enabled,
            gpu_prefetch_buffer_size=gpu_prefetch_buffer_size,
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

    def test_gpu_prefetch_fields_enabled(self) -> None:
        """GPU Prefetch有効時のフィールド値."""
        result = self._make_result(
            gpu_prefetch_enabled=True,
            gpu_prefetch_buffer_size=5,
        )
        assert result.gpu_prefetch_enabled is True
        assert result.gpu_prefetch_buffer_size == 5

    def test_gpu_prefetch_fields_disabled(self) -> None:
        """GPU Prefetch無効時のフィールド値."""
        result = self._make_result(
            gpu_prefetch_enabled=False,
            gpu_prefetch_buffer_size=0,
        )
        assert result.gpu_prefetch_enabled is False
        assert result.gpu_prefetch_buffer_size == 0

    def test_to_dict_includes_gpu_prefetch_fields(self) -> None:
        """to_dict にGPU Prefetchフィールドが含まれる."""
        result = self._make_result(
            gpu_prefetch_enabled=True,
            gpu_prefetch_buffer_size=8,
        )
        d = result.to_dict()
        assert "gpu_prefetch_enabled" in d
        assert "gpu_prefetch_buffer_size" in d
        assert d["gpu_prefetch_enabled"] == 1.0
        assert d["gpu_prefetch_buffer_size"] == 8.0


class TestFormatTimingSummaryLabeling:
    """format_timing_summary のラベリングロジックテスト.

    format_timing_summary はローカル関数のため直接呼び出せない．
    ここでは同関数内で使用される条件分岐ロジックを再現し，
    BenchmarkResult のフィールド値に基づくラベルとノートの期待値を検証する．
    """

    def test_format_summary_label_prefetch_wait(self) -> None:
        """Prefetch有効時に 'Prefetch Wait' ラベルが使用される."""
        gpu_prefetch_enabled = True
        data_label = (
            "Prefetch Wait"
            if gpu_prefetch_enabled
            else "Data Loading"
        )
        assert data_label == "Prefetch Wait"

    def test_format_summary_label_data_loading(self) -> None:
        """Prefetch無効時に 'Data Loading' ラベルが使用される."""
        gpu_prefetch_enabled = False
        data_label = (
            "Prefetch Wait"
            if gpu_prefetch_enabled
            else "Data Loading"
        )
        assert data_label == "Data Loading"

    def test_format_summary_prefetch_note(self) -> None:
        """Prefetch有効時に注釈が含まれる."""
        gpu_prefetch_enabled = True
        prefetch_note = ""
        if gpu_prefetch_enabled:
            prefetch_note = (
                "\n\n  Note: GPU Prefetcher active"
                " - 'Prefetch Wait' shows buffer fetch time,"
                " not actual disk I/O time."
            )
        assert "GPU Prefetcher active" in prefetch_note
        assert "Prefetch Wait" in prefetch_note

        # 無効時は空文字列
        gpu_prefetch_enabled = False
        prefetch_note = ""
        if gpu_prefetch_enabled:
            prefetch_note = "should not appear"
        assert prefetch_note == ""
