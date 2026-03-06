"""GradientNoiseScaleEstimator のテスト．"""

import torch
import torch.nn as nn

from maou.app.learning.gradient_noise_scale import (
    GNSEstimate,
    GradientNoiseScaleEstimator,
)


def _make_simple_model() -> nn.Module:
    """テスト用の簡易モデルを作成する．"""
    return nn.Linear(10, 2, bias=False)


class TestGradientNoiseScaleEstimator:
    """GradientNoiseScaleEstimator のテスト．"""

    def test_basic_gns_estimation(self) -> None:
        """2 micro-batch で GNS が推定できることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # Simulate 2 micro-batches with gradient accumulation
        # Micro-batch 0
        model.zero_grad()
        x0 = torch.randn(32, 10)
        y0 = model(x0)
        loss0 = y0.sum() / 2  # Normalize by accum steps
        loss0.backward()
        estimator.on_backward_end(model, accumulation_step=0)

        # Micro-batch 1
        x1 = torch.randn(32, 10)
        y1 = model(x1)
        loss1 = y1.sum() / 2
        loss1.backward()
        estimator.on_backward_end(model, accumulation_step=1)

        # Compute GNS
        result = estimator.compute(model)

        assert result is not None
        assert isinstance(result, GNSEstimate)
        assert result.b_noise > 0
        assert result.physical_batch_size == 32
        assert result.micro_batch_count == 2

    def test_single_step_returns_none(self) -> None:
        """micro-batch が 1 つしかない場合は None が返ることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        model.zero_grad()
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        estimator.on_backward_end(model, accumulation_step=0)

        result = estimator.compute(model)
        assert result is None

    def test_measurement_interval(self) -> None:
        """measurement_interval=2 で交互にスキップされることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
            measurement_interval=2,
        )

        def _run_cycle() -> GNSEstimate | None:
            model.zero_grad()
            for step in range(2):
                x = torch.randn(32, 10)
                y = model(x)
                loss = y.sum() / 2
                loss.backward()
                estimator.on_backward_end(
                    model, accumulation_step=step
                )
            return estimator.compute(model)

        # Cycle 1: step_count=0, 0%2==0 → measured
        result1 = _run_cycle()
        assert result1 is not None

        # Cycle 2: step_count=1, 1%2!=0 → skipped
        result2 = _run_cycle()
        assert result2 is None

        # Cycle 3: step_count=2, 2%2==0 → measured
        result3 = _run_cycle()
        assert result3 is not None

    def test_reset_between_cycles(self) -> None:
        """compute 後に内部状態がリセットされることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # Run first cycle
        model.zero_grad()
        for step in range(3):
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum() / 3
            loss.backward()
            estimator.on_backward_end(
                model, accumulation_step=step
            )
        result1 = estimator.compute(model)
        assert result1 is not None

        # Run second cycle - should work independently
        model.zero_grad()
        for step in range(3):
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.sum() / 3
            loss.backward()
            estimator.on_backward_end(
                model, accumulation_step=step
            )
        result2 = estimator.compute(model)
        assert result2 is not None

    def test_no_grad_returns_none(self) -> None:
        """勾配がない場合に None が返ることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # No backward call - no gradients
        result = estimator.compute(model)
        assert result is None

    def test_identical_gradients_returns_none(self) -> None:
        """全 micro-batch の勾配が同一(ratio <= 1.0)のとき None が返ることを確認する．"""
        model = _make_simple_model()
        estimator = GradientNoiseScaleEstimator(
            physical_batch_size=32,
        )

        # 同一入力で同一勾配を生成
        x = torch.randn(32, 10)

        model.zero_grad()
        y0 = model(x)
        loss0 = y0.sum() / 2
        loss0.backward()
        estimator.on_backward_end(model, accumulation_step=0)

        y1 = model(x)
        loss1 = y1.sum() / 2
        loss1.backward()
        estimator.on_backward_end(model, accumulation_step=1)

        result = estimator.compute(model)
        # ratio = K * S / G ≈ 1.0 (同一勾配 → ノイズなし)
        # ratio <= 1.0 のため None が返る
        assert result is None
