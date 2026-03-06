"""Gradient Noise Scale (GNS) のオンライン推定．

勾配蓄積(gradient accumulation)の各 micro-batch から勾配を差分抽出し，
勾配の分散と平均から B_noise (Gradient Noise Scale) を推定する．

B_noise は Critical Batch Size (CBS) の近似であり，
実効バッチサイズが B_noise を大幅に超えると学習効率が低下する
(McCandlish et al. 2018)．

推定手法:
    各 micro-batch の backward 後に勾配スナップショットとの差分から
    micro-batch 勾配の二乗ノルムを蓄積する．accumulation cycle 完了時に
    分散と平均二乗ノルムから B_noise を算出する．

    B_noise = b * K/(K-1) * (K * S / G - 1)

    ここで:
        b = 物理バッチサイズ
        K = gradient_accumulation_steps
        S = Σ_k |micro_grad_k|² (micro-batch 勾配の二乗ノルム和)
        G = |mean_grad|² (蓄積済み勾配の二乗ノルム)

メモリオーバーヘッド:
    モデルパラメータ1コピー分(prev_grad snapshot) + スカラー数個
"""

import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class GNSEstimate:
    """GNS推定結果．

    Attributes:
        b_noise: Gradient Noise Scale の推定値．
        sum_micro_norm_sq: micro-batch 勾配の二乗ノルム和．
        mean_grad_norm_sq: 蓄積済み勾配の二乗ノルム．
        accumulation_steps: 使用した gradient accumulation steps 数．
        physical_batch_size: 物理バッチサイズ．
    """

    b_noise: float
    sum_micro_norm_sq: float
    mean_grad_norm_sq: float
    accumulation_steps: int
    physical_batch_size: int


class GradientNoiseScaleEstimator:
    """Gradient Noise Scale のオンライン推定器．

    gradient accumulation の各 micro-batch から勾配統計を収集し，
    accumulation cycle 完了時に B_noise を算出する．

    Args:
        physical_batch_size: DataLoader の物理バッチサイズ．
        measurement_interval: GNS を計測する optimizer step 間隔．
            1 なら毎ステップ計測する．
        device: 計算デバイス．
    """

    def __init__(
        self,
        *,
        physical_batch_size: int,
        measurement_interval: int = 1,
        device: torch.device | None = None,
    ) -> None:
        self._physical_batch_size = physical_batch_size
        self._measurement_interval = max(1, measurement_interval)
        self._device = device

        # accumulation cycle 内の状態
        self._sum_micro_norm_sq: float = 0.0
        self._prev_grad: torch.Tensor | None = None
        self._micro_batch_count: int = 0

        # optimizer step カウンタ
        self._optimizer_step_count: int = 0

    @property
    def should_measure(self) -> bool:
        """現在の optimizer step で GNS を計測すべきかどうか．"""
        return (
            self._optimizer_step_count % self._measurement_interval
            == 0
        )

    def on_backward_end(
        self,
        model: torch.nn.Module,
        accumulation_step: int,
    ) -> None:
        """backward 完了後に呼び出し，micro-batch 勾配統計を収集する．

        計測対象の cycle でない場合はスキップする．

        Args:
            model: 勾配が蓄積されたモデル．
            accumulation_step: 現在の accumulation cycle 内のステップ番号
                (0-indexed)．
        """
        if not self.should_measure:
            return

        curr_grad = self._flatten_grads(model)
        if curr_grad is None:
            return

        if accumulation_step == 0:
            # cycle の最初: param.grad がこの micro-batch の勾配そのもの
            micro_grad = curr_grad
        else:
            if self._prev_grad is None:
                logger.warning(
                    "prev_grad is None at accumulation_step=%d, "
                    "skipping GNS measurement for this cycle",
                    accumulation_step,
                )
                return
            # 差分で micro-batch 勾配を抽出
            micro_grad = curr_grad - self._prev_grad

        self._sum_micro_norm_sq += (
            micro_grad.dot(micro_grad).item()
        )
        self._prev_grad = curr_grad.clone()
        self._micro_batch_count = accumulation_step + 1

    def compute(
        self,
        model: torch.nn.Module,
        accumulation_steps: int,
    ) -> GNSEstimate | None:
        """accumulation cycle 完了時に GNS を算出する．

        optimizer step の直前(勾配クリッピング後，step() 前)に呼び出す．

        Args:
            model: 勾配が蓄積されたモデル．
            accumulation_steps: 現在の gradient_accumulation_steps 値．

        Returns:
            GNS推定結果．計測対象外または計算不能な場合は None．
        """
        # should_measure は backward_end 時点の step_count で判定済み．
        # compute 内では同じ step_count で判定してからインクリメントする．
        should_compute = self.should_measure
        self._optimizer_step_count += 1

        if not should_compute:
            self._reset()
            return None

        if accumulation_steps < 2:
            self._reset()
            return None

        if self._micro_batch_count < 2:
            logger.debug(
                "Insufficient micro-batch samples (%d) for GNS estimation",
                self._micro_batch_count,
            )
            self._reset()
            return None

        # 蓄積済み勾配の二乗ノルム
        mean_grad = self._flatten_grads(model)
        if mean_grad is None:
            self._reset()
            return None

        mean_grad_norm_sq = mean_grad.dot(mean_grad).item()

        if mean_grad_norm_sq < 1e-30:
            logger.debug(
                "Mean gradient norm too small (%.2e) for GNS estimation",
                mean_grad_norm_sq,
            )
            self._reset()
            return None

        k = accumulation_steps
        s = self._sum_micro_norm_sq
        g = mean_grad_norm_sq
        b = self._physical_batch_size

        # B_noise = b * K/(K-1) * (K * S / G - 1)
        ratio = k * s / g
        if ratio <= 1.0:
            # 全 micro-batch の勾配がほぼ同一方向 → ノイズ極小
            # B_noise は非常に大きい(制限なくバッチを増やせる)
            logger.debug(
                "GNS ratio <= 1.0 (%.4f), gradient noise is negligible",
                ratio,
            )
            self._reset()
            return None

        b_noise = b * k / (k - 1) * (ratio - 1)

        estimate = GNSEstimate(
            b_noise=b_noise,
            sum_micro_norm_sq=s,
            mean_grad_norm_sq=g,
            accumulation_steps=k,
            physical_batch_size=b,
        )

        self._reset()
        return estimate

    def _reset(self) -> None:
        """accumulation cycle の状態をリセットする．"""
        self._sum_micro_norm_sq = 0.0
        self._prev_grad = None
        self._micro_batch_count = 0

    @staticmethod
    def _flatten_grads(
        model: torch.nn.Module,
    ) -> torch.Tensor | None:
        """モデルの全パラメータの勾配を1次元テンソルに結合する．"""
        grads: list[torch.Tensor] = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.reshape(-1))
        if not grads:
            return None
        return torch.cat(grads)
