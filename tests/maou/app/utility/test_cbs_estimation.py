"""Tests for CBS estimation in TrainingBenchmarkUseCase."""

from __future__ import annotations

from maou.app.utility.training_benchmark import (
    TrainingBenchmarkUseCase,
)


class TestEstimateCbsFromSweep:
    """_estimate_cbs_from_sweep の数値テスト."""

    def _make_result(
        self,
        bs: int,
        sps: float,
    ) -> dict:
        """Create a minimal sweep result dict."""
        return {
            "sweep_batch_size": bs,
            "training_metrics": {
                "samples_per_second": sps,
            },
        }

    def test_linear_scaling_returns_exceeds_tested(
        self,
    ) -> None:
        """全バッチサイズが線形スケーリング → cbs_exceeds_tested=True."""
        # Perfect linear: sps doubles when bs doubles
        results = [
            self._make_result(128, 1000.0),
            self._make_result(256, 2000.0),
            self._make_result(512, 4000.0),
        ]
        est = TrainingBenchmarkUseCase._estimate_cbs_from_sweep(
            results
        )
        assert est is not None
        assert est["cbs_exceeds_tested"] is True
        assert est["gradient_noise_scale"] is None

    def test_diminishing_returns_estimates_cbs(self) -> None:
        """効率が低下するパターン → CBS が推定される."""
        # Sublinear scaling: throughput increases less than proportionally
        results = [
            self._make_result(64, 640.0),  # eff = 10.0
            self._make_result(128, 1024.0),  # eff = 8.0
            self._make_result(256, 1536.0),  # eff = 6.0
        ]
        est = TrainingBenchmarkUseCase._estimate_cbs_from_sweep(
            results
        )
        assert est is not None
        assert est["cbs_exceeds_tested"] is False
        assert isinstance(est["estimated_cbs"], int)
        assert est["estimated_cbs"] > 0

    def test_single_result_returns_none(self) -> None:
        """単一結果 → 推定不可で None."""
        results = [self._make_result(256, 2000.0)]
        est = TrainingBenchmarkUseCase._estimate_cbs_from_sweep(
            results
        )
        assert est is None

    def test_empty_results_returns_none(self) -> None:
        """空リスト → None."""
        est = TrainingBenchmarkUseCase._estimate_cbs_from_sweep(
            []
        )
        assert est is None

    def test_json_serializable(self) -> None:
        """結果が JSON シリアライズ可能であることを確認."""
        import json

        results = [
            self._make_result(64, 640.0),
            self._make_result(128, 1024.0),
            self._make_result(256, 1536.0),
        ]
        est = TrainingBenchmarkUseCase._estimate_cbs_from_sweep(
            results
        )
        assert est is not None
        # Should not raise (float('inf') was a previous bug)
        json.dumps(est)

    def test_two_results_sufficient(self) -> None:
        """2データポイントで推定可能."""
        results = [
            self._make_result(128, 1280.0),  # eff = 10.0
            self._make_result(256, 1792.0),  # eff = 7.0
        ]
        est = TrainingBenchmarkUseCase._estimate_cbs_from_sweep(
            results
        )
        assert est is not None
        assert isinstance(est["estimated_cbs"], int)

    def test_scaling_efficiency_keys(self) -> None:
        """scaling_efficiency に全テスト済みバッチサイズが含まれる."""
        results = [
            self._make_result(128, 1280.0),
            self._make_result(256, 1792.0),
        ]
        est = TrainingBenchmarkUseCase._estimate_cbs_from_sweep(
            results
        )
        assert est is not None
        eff = est["scaling_efficiency"]
        assert "128" in eff
        assert "256" in eff
