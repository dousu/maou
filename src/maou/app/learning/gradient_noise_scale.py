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
    モデルパラメータ1コピー分(prev_grads snapshot) + スカラー数個
"""

import logging
import math
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class GNSEstimate:
    """GNS推定結果．

    Attributes:
        b_noise: Gradient Noise Scale の推定値．
        sum_micro_norm_sq: micro-batch 勾配の二乗ノルム和．
        mean_grad_norm_sq: 蓄積済み勾配の二乗ノルム．
        micro_batch_count: 実際に計測された micro-batch 数．
        physical_batch_size: 物理バッチサイズ．
    """

    b_noise: float
    sum_micro_norm_sq: float
    mean_grad_norm_sq: float
    micro_batch_count: int
    physical_batch_size: int


class GradientNoiseScaleEstimator:
    """Gradient Noise Scale のオンライン推定器．

    gradient accumulation の各 micro-batch から勾配統計を収集し，
    accumulation cycle 完了時に B_noise を算出する．

    勾配ノルムはパラメータごとに計算し，torch.cat による
    全パラメータ結合を避けることでメモリ割り当てを最小化する．

    Args:
        physical_batch_size: DataLoader の物理バッチサイズ．
        measurement_interval: GNS を計測する optimizer step 間隔．
            1 なら毎ステップ計測する．計測中はモデルパラメータ1コピー分の
            追加メモリを使用するため，大規模モデル(数百M〜数Bパラメータ)
            では 5〜10 を推奨する．
    """

    def __init__(
        self,
        *,
        physical_batch_size: int,
        measurement_interval: int = 1,
    ) -> None:
        self._physical_batch_size = physical_batch_size
        self._measurement_interval = max(
            1, measurement_interval
        )

        # accumulation cycle 内の状態
        self._sum_micro_norm_sq: float = 0.0
        self._prev_grads: list[torch.Tensor | None] | None = (
            None
        )
        self._micro_batch_count: int = 0

        # optimizer step カウンタ
        self._optimizer_step_count: int = 0

    @property
    def optimizer_step_count(self) -> int:
        """累積 optimizer step 数．"""
        return self._optimizer_step_count

    @property
    def should_measure(self) -> bool:
        """現在の optimizer step で GNS を計測すべきかどうか．"""
        return (
            self._optimizer_step_count
            % self._measurement_interval
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

        if accumulation_step == 0:
            # cycle の最初: param.grad がこの micro-batch の勾配そのもの
            micro_norm_sq = 0.0
            prev_grads: list[torch.Tensor | None] = []
            has_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    has_grad = True
                    g = param.grad.detach()
                    micro_norm_sq += g.pow(2).sum().item()
                    prev_grads.append(g.clone())
                else:
                    prev_grads.append(None)
            if not has_grad:
                return
            self._sum_micro_norm_sq += micro_norm_sq
            self._prev_grads = prev_grads
        else:
            if self._prev_grads is None:
                logger.warning(
                    "prev_grads is None at accumulation_step=%d, "
                    "skipping GNS measurement for this cycle",
                    accumulation_step,
                )
                return
            # 差分で micro-batch 勾配のノルムを計算
            # prev_grads はすべてのパラメータに対応するエントリを持つ
            # (grad=None のパラメータも含む)ため，idx は全パラメータで
            # インクリメントする
            micro_norm_sq = 0.0
            new_prev: list[torch.Tensor | None] = []
            params = list(model.parameters())
            if len(self._prev_grads) != len(params):
                logger.warning(
                    "prev_grads length (%d) != model parameters (%d), "
                    "resetting GNS state",
                    len(self._prev_grads),
                    len(params),
                )
                self._reset()
                return
            for idx, param in enumerate(params):
                if param.grad is not None:
                    g = param.grad.detach()
                    prev = self._prev_grads[idx]
                    if prev is not None:
                        diff = g - prev
                        micro_norm_sq += (
                            diff.pow(2).sum().item()
                        )
                    else:
                        micro_norm_sq += g.pow(2).sum().item()
                    new_prev.append(g.clone())
                else:
                    new_prev.append(None)
            self._sum_micro_norm_sq += micro_norm_sq
            self._prev_grads = new_prev

        # accumulation_step は 0-indexed なので +1 で処理済み micro-batch 数になる．
        # インクリメントではなく代入: skip された step があっても正しい値を保持する．
        self._micro_batch_count = accumulation_step + 1

    def compute(
        self,
        model: torch.nn.Module,
    ) -> GNSEstimate | None:
        """accumulation cycle 完了時に GNS を算出する．

        勾配クリッピング前に呼び出すこと．S と G を同じ基準で
        計測するため，クリッピング後では G が変化してしまう．

        数式の K には引数ではなく実際に on_backward_end が呼ばれた
        micro-batch 数(_micro_batch_count)を使用する．非有限損失で
        一部の backward がスキップされた場合でも S と K が整合する．

        Args:
            model: 勾配が蓄積されたモデル．

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

        if self._micro_batch_count < 2:
            logger.debug(
                "Insufficient micro-batch samples (%d) for GNS estimation",
                self._micro_batch_count,
            )
            self._reset()
            return None

        # 蓄積済み勾配の二乗ノルムをパラメータごとに計算
        mean_grad_norm_sq = 0.0
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                mean_grad_norm_sq += (
                    param.grad.detach().pow(2).sum().item()
                )

        if not has_grad:
            self._reset()
            return None

        if (
            math.isnan(mean_grad_norm_sq)
            or mean_grad_norm_sq < 1e-30
        ):
            logger.debug(
                "Mean gradient norm too small (%.2e) for GNS estimation",
                mean_grad_norm_sq,
            )
            self._reset()
            return None

        k = self._micro_batch_count
        s = self._sum_micro_norm_sq
        g = mean_grad_norm_sq
        b = self._physical_batch_size

        # B_noise = b * K/(K-1) * (K * S / G - 1)
        # オーバーフロー防止: S が極大の場合は計算をスキップ
        if s > 1e30 or math.isnan(s):
            logger.debug(
                "sum_micro_norm_sq overflow or NaN (%.2e), "
                "skipping GNS estimation",
                s,
            )
            self._reset()
            return None

        ratio = k * s / g
        if ratio <= 1.0:
            # 全 micro-batch の勾配がほぼ同一方向 → ノイズ極小．
            # 理論的には B_noise → ∞ (バッチサイズを増やし続けて良い)だが，
            # 保守的に None を返すことで controller の EMA を更新しない．
            # 学習安定期に max_accumulation_steps まで増やす挙動が望ましい場合は
            # ここで大きな B_noise を返す設計に変更すること．
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
            micro_batch_count=k,
            physical_batch_size=b,
        )

        self._reset()
        return estimate

    def reset_cycle(self) -> None:
        """accumulation cycle の状態をリセットする．

        非有限損失等で backward がスキップされた場合に外部から呼び出し，
        _sum_micro_norm_sq の stale データが次サイクルに混入するのを防ぐ．
        """
        self._reset()

    def _reset(self) -> None:
        """accumulation cycle の状態をリセットする(内部用)．"""
        self._sum_micro_norm_sq = 0.0
        self._prev_grads = None
        self._micro_batch_count = 0
