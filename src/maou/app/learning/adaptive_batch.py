"""Adaptive Batch Size コントローラ．

Gradient Noise Scale (GNS) に基づいて gradient_accumulation_steps を
動的に調整し，実効バッチサイズを最適化する．

実効バッチサイズ = physical_batch_size × gradient_accumulation_steps

GNS (B_noise) が大きい場合は実効バッチサイズを増やし，
小さい場合は減らすことで，訓練効率を最大化する．
"""

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def round_to_power_of_two(
    value: float, *, minimum: int = 1
) -> int:
    """値を最も近い2の冪乗に丸める．

    Args:
        value: 丸める対象の値．
        minimum: 下限値(デフォルト 1)．結果はこの値以上になる．

    Returns:
        2の冪乗に丸められた整数．
    """
    clamped = max(minimum, value)
    return max(minimum, int(2 ** round(math.log2(clamped))))


@dataclass(frozen=True)
class AdaptiveBatchConfig:
    """Adaptive batch size の設定．

    Attributes:
        min_accumulation_steps: 最小 gradient accumulation steps．
            GNS 推定に最低2サンプル必要なため 2 以上．
        max_accumulation_steps: 最大 gradient accumulation steps．
        adjustment_interval: accumulation steps を調整する
            optimizer step 間隔．
        smoothing_factor: GNS の EMA (Exponential Moving Average) 係数．
            0 に近いほど平滑化が強い．
        measurement_interval: GNS を計測する optimizer step 間隔．
            adjustment_interval の約数であることが望ましい．
    """

    min_accumulation_steps: int = 2
    max_accumulation_steps: int = 8
    adjustment_interval: int = 50
    smoothing_factor: float = 0.1
    measurement_interval: int = 1

    def __post_init__(self) -> None:
        """バリデーション．"""
        if self.min_accumulation_steps < 2:
            msg = (
                f"min_accumulation_steps must be >= 2 for GNS estimation, "
                f"got {self.min_accumulation_steps}"
            )
            raise ValueError(msg)
        if (
            self.max_accumulation_steps
            < self.min_accumulation_steps
        ):
            msg = (
                f"max_accumulation_steps ({self.max_accumulation_steps}) "
                f"must be >= min_accumulation_steps ({self.min_accumulation_steps})"
            )
            raise ValueError(msg)
        if self.adjustment_interval < 1:
            msg = (
                f"adjustment_interval must be >= 1, "
                f"got {self.adjustment_interval}"
            )
            raise ValueError(msg)
        if not (0.0 < self.smoothing_factor <= 1.0):
            msg = (
                f"smoothing_factor must be in (0, 1], "
                f"got {self.smoothing_factor}"
            )
            raise ValueError(msg)
        if self.measurement_interval < 1:
            msg = (
                f"measurement_interval must be >= 1, "
                f"got {self.measurement_interval}"
            )
            raise ValueError(msg)


class AdaptiveBatchController:
    """GNS に基づく adaptive batch size コントローラ．

    optimizer step ごとに GNS 推定値を受け取り，
    一定間隔で gradient_accumulation_steps を調整する．

    Args:
        config: Adaptive batch の設定．
        physical_batch_size: DataLoader の物理バッチサイズ．
    """

    def __init__(
        self,
        *,
        config: AdaptiveBatchConfig,
        physical_batch_size: int,
    ) -> None:
        self._config = config
        self._physical_batch_size = physical_batch_size
        self._current_steps = config.min_accumulation_steps

        # EMA 平滑化された GNS
        self._smoothed_gns: float | None = None

        # optimizer step カウンタ
        self._step_count: int = 0

    @property
    def current_accumulation_steps(self) -> int:
        """現在の gradient accumulation steps．"""
        return self._current_steps

    @property
    def current_effective_batch_size(self) -> int:
        """現在の実効バッチサイズ．"""
        return self._physical_batch_size * self._current_steps

    @property
    def smoothed_gns(self) -> float | None:
        """EMA 平滑化された GNS．"""
        return self._smoothed_gns

    def update(self, gns: float | None) -> int:
        """Optimizer step ごとに呼び出し，GNS に基づいて調整する．

        毎 optimizer step で呼び出すこと．GNS 推定値がない場合
        (measurement_interval によるスキップ等)は gns=None を渡す．
        adjustment_interval は全 optimizer step に対する間隔として
        機能する．

        Args:
            gns: Gradient Noise Scale の推定値 (B_noise)．
                計測されなかった step では None．

        Returns:
            更新後の gradient_accumulation_steps．
        """
        self._step_count += 1

        # EMA 更新(GNS 推定値がある場合のみ)
        if gns is not None:
            alpha = self._config.smoothing_factor
            if self._smoothed_gns is None:
                self._smoothed_gns = gns
            else:
                self._smoothed_gns = (
                    alpha * gns
                    + (1 - alpha) * self._smoothed_gns
                )

        # 調整間隔でない場合はスキップ
        if (
            self._step_count % self._config.adjustment_interval
            != 0
        ):
            return self._current_steps

        # EMA が未初期化(まだ一度も GNS を受け取っていない)場合はスキップ
        if self._smoothed_gns is None:
            return self._current_steps

        # 目標 accumulation steps を計算
        target_steps = self._compute_target_steps(
            self._smoothed_gns
        )

        if target_steps != self._current_steps:
            old_steps = self._current_steps
            self._current_steps = target_steps
            logger.info(
                "Adaptive batch: accumulation_steps %d → %d "
                "(effective_bs: %d → %d, smoothed_GNS: %.1f)",
                old_steps,
                target_steps,
                self._physical_batch_size * old_steps,
                self._physical_batch_size * target_steps,
                self._smoothed_gns,
            )

        return self._current_steps

    def _compute_target_steps(self, smoothed_gns: float) -> int:
        """平滑化された GNS から目標 accumulation steps を計算する．

        目標実効バッチサイズ ≈ B_noise とし，
        accumulation_steps = round(B_noise / physical_batch_size) を
        [min, max] 範囲にクリップする．

        2の冪乗に丸めることで GPU tensor core の効率を維持する．
        """
        if smoothed_gns <= 0:
            return self._config.min_accumulation_steps

        raw_steps = smoothed_gns / self._physical_batch_size

        # 2の冪乗に丸める
        target = round_to_power_of_two(raw_steps)

        # [min, max] にクリップ
        target = max(
            self._config.min_accumulation_steps, target
        )
        target = min(
            self._config.max_accumulation_steps, target
        )

        return target
