"""AdaptiveBatchConfig / AdaptiveBatchController / round_to_power_of_two のテスト．"""

import pytest

from maou.app.learning.adaptive_batch import (
    AdaptiveBatchConfig,
    AdaptiveBatchController,
    round_to_power_of_two,
)


class TestAdaptiveBatchConfig:
    """AdaptiveBatchConfig のバリデーションテスト．"""

    def test_default_config(self) -> None:
        """デフォルト設定で正常に作成できることを確認する．"""
        config = AdaptiveBatchConfig()
        assert config.min_accumulation_steps == 2
        assert config.max_accumulation_steps == 8
        assert config.adjustment_interval == 50
        assert config.smoothing_factor == 0.1

    def test_min_steps_validation(self) -> None:
        """min_accumulation_steps < 2 でエラーになることを確認する．"""
        with pytest.raises(
            ValueError, match="min_accumulation_steps"
        ):
            AdaptiveBatchConfig(min_accumulation_steps=1)

    def test_max_less_than_min_validation(self) -> None:
        """max < min でエラーになることを確認する．"""
        with pytest.raises(
            ValueError, match="max_accumulation_steps"
        ):
            AdaptiveBatchConfig(
                min_accumulation_steps=4,
                max_accumulation_steps=2,
            )

    def test_smoothing_factor_validation(self) -> None:
        """smoothing_factor が (0, 1] 外でエラーになることを確認する．"""
        with pytest.raises(
            ValueError, match="smoothing_factor"
        ):
            AdaptiveBatchConfig(smoothing_factor=0.0)
        with pytest.raises(
            ValueError, match="smoothing_factor"
        ):
            AdaptiveBatchConfig(smoothing_factor=1.5)


class TestAdaptiveBatchController:
    """AdaptiveBatchController のテスト．"""

    def _make_controller(
        self,
        physical_batch_size: int = 256,
        min_steps: int = 2,
        max_steps: int = 8,
        interval: int = 1,
        smoothing: float = 1.0,
    ) -> AdaptiveBatchController:
        config = AdaptiveBatchConfig(
            min_accumulation_steps=min_steps,
            max_accumulation_steps=max_steps,
            adjustment_interval=interval,
            smoothing_factor=smoothing,
        )
        return AdaptiveBatchController(
            config=config,
            physical_batch_size=physical_batch_size,
        )

    def test_initial_state(self) -> None:
        """初期状態が min_accumulation_steps であることを確認する．"""
        ctrl = self._make_controller(min_steps=2, max_steps=8)
        assert ctrl.current_accumulation_steps == 2
        assert ctrl.current_effective_batch_size == 512

    def test_gns_triggers_increase(self) -> None:
        """高 GNS で accumulation steps が増加することを確認する．"""
        ctrl = self._make_controller(
            physical_batch_size=256,
            min_steps=2,
            max_steps=8,
            interval=1,
            smoothing=1.0,
        )

        # GNS = 1024 → target = 1024/256 = 4
        new_steps = ctrl.update(1024.0)
        assert new_steps == 4

    def test_gns_triggers_decrease(self) -> None:
        """低 GNS で accumulation steps が減少することを確認する．"""
        ctrl = self._make_controller(
            physical_batch_size=256,
            min_steps=2,
            max_steps=8,
            interval=1,
            smoothing=1.0,
        )

        # First increase to 4
        ctrl.update(1024.0)
        assert ctrl.current_accumulation_steps == 4

        # GNS drops to 400 → target = 400/256 ≈ 1.5 → rounded to 2
        new_steps = ctrl.update(400.0)
        assert new_steps == 2

    def test_clamp_to_min(self) -> None:
        """min_steps 以下にならないことを確認する．"""
        ctrl = self._make_controller(
            physical_batch_size=256,
            min_steps=2,
            max_steps=8,
            interval=1,
            smoothing=1.0,
        )

        # Very small GNS
        new_steps = ctrl.update(10.0)
        assert new_steps == 2

    def test_clamp_to_max(self) -> None:
        """max_steps 以上にならないことを確認する．"""
        ctrl = self._make_controller(
            physical_batch_size=256,
            min_steps=2,
            max_steps=8,
            interval=1,
            smoothing=1.0,
        )

        # Very large GNS
        new_steps = ctrl.update(100000.0)
        assert new_steps == 8

    def test_adjustment_interval(self) -> None:
        """interval に達するまで値が変わらないことを確認する．"""
        ctrl = self._make_controller(
            physical_batch_size=256,
            min_steps=2,
            max_steps=8,
            interval=3,
            smoothing=1.0,
        )

        # Steps 1 and 2: no adjustment
        result1 = ctrl.update(2048.0)
        assert result1 == 2  # Not yet at interval
        result2 = ctrl.update(2048.0)
        assert result2 == 2  # Not yet at interval

        # Step 3: adjustment happens
        result3 = ctrl.update(2048.0)
        assert result3 == 8  # 2048/256 = 8

    def test_ema_smoothing(self) -> None:
        """EMA 平滑化が機能することを確認する．"""
        ctrl = self._make_controller(
            physical_batch_size=256,
            min_steps=2,
            max_steps=8,
            interval=1,
            smoothing=0.5,
        )

        # First: smoothed = 1024
        ctrl.update(1024.0)
        assert ctrl.smoothed_gns == 1024.0

        # Second: smoothed = 0.5 * 256 + 0.5 * 1024 = 640
        ctrl.update(256.0)
        assert ctrl.smoothed_gns is not None
        assert abs(ctrl.smoothed_gns - 640.0) < 1e-6

    def test_power_of_two_rounding(self) -> None:
        """2の冪乗に丸められることを確認する．"""
        ctrl = self._make_controller(
            physical_batch_size=256,
            min_steps=2,
            max_steps=16,
            interval=1,
            smoothing=1.0,
        )

        # GNS = 768 → 768/256 = 3 → rounded to 4 (nearest power of 2)
        new_steps = ctrl.update(768.0)
        assert (
            new_steps == 4
        )  # 2^round(log2(3)) = 2^1.58 ≈ 2^2 = 4

    def test_none_gns_skips_ema_update(self) -> None:
        """gns=None のとき EMA が更新されないことを確認する．"""
        ctrl = self._make_controller(
            physical_batch_size=256,
            min_steps=2,
            max_steps=8,
            interval=1,
            smoothing=1.0,
        )

        # None → EMA 未初期化のまま
        result = ctrl.update(None)
        assert result == 2  # min_steps のまま
        assert ctrl.smoothed_gns is None

        # 有効な GNS → EMA 初期化
        ctrl.update(1024.0)
        assert ctrl.smoothed_gns == 1024.0

        # None → EMA は前の値を維持
        ctrl.update(None)
        assert ctrl.smoothed_gns == 1024.0

    def test_adjustment_interval_counts_all_steps(self) -> None:
        """adjustment_interval が None を含む全 step をカウントすることを確認する．"""
        ctrl = self._make_controller(
            physical_batch_size=256,
            min_steps=2,
            max_steps=8,
            interval=3,
            smoothing=1.0,
        )

        # Step 1: GNS=2048, not at interval
        ctrl.update(2048.0)
        assert ctrl.current_accumulation_steps == 2

        # Step 2: None (measurement skipped), not at interval
        ctrl.update(None)
        assert ctrl.current_accumulation_steps == 2

        # Step 3: None, but at interval → adjusts using smoothed_gns=2048
        ctrl.update(None)
        assert (
            ctrl.current_accumulation_steps == 8
        )  # 2048/256 = 8


class TestRoundToPowerOfTwo:
    """round_to_power_of_two のテスト．"""

    def test_exact_power_of_two(self) -> None:
        """2の冪乗はそのまま返ることを確認する．"""
        assert round_to_power_of_two(4.0) == 4
        assert round_to_power_of_two(8.0) == 8

    def test_rounds_to_nearest(self) -> None:
        """最も近い2の冪乗に丸められることを確認する．"""
        assert (
            round_to_power_of_two(3.0) == 4
        )  # log2(3) ≈ 1.58 → 2
        assert (
            round_to_power_of_two(5.0) == 4
        )  # log2(5) ≈ 2.32 → 2
        assert (
            round_to_power_of_two(6.0) == 8
        )  # log2(6) ≈ 2.58 → 3

    def test_minimum_clamp(self) -> None:
        """minimum 以下にならないことを確認する．"""
        assert round_to_power_of_two(0.5, minimum=2) == 2
        assert round_to_power_of_two(1.0, minimum=4) == 4

    def test_very_small_value(self) -> None:
        """非常に小さい値でも minimum が適用されることを確認する．"""
        assert round_to_power_of_two(0.001) == 1
        assert round_to_power_of_two(0.001, minimum=2) == 2

    def test_large_value(self) -> None:
        """大きい値が正しく丸められることを確認する．"""
        assert round_to_power_of_two(1000.0) == 1024
        assert round_to_power_of_two(100.0) == 128
