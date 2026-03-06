import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Union

import torch
from torch.utils.tensorboard import (
    SummaryWriter,  # type: ignore
)

from maou.app.learning.policy_targets import (
    normalize_policy_targets,
)
from maou.app.learning.resource_monitor import (
    GPUResourceMonitor,
    ResourceUsage,
    SystemResourceMonitor,
)

ModelInputs = Union[torch.Tensor, Sequence[torch.Tensor]]


def _safe_average(total: float, count: int) -> Optional[float]:
    """countが正の場合のみ平均を返す．countが0ならNone．"""
    return total / count if count > 0 else None


@dataclass
class TrainingContext:
    """Training context passed to callbacks."""

    batch_idx: int
    epoch_idx: int
    inputs: ModelInputs
    labels_policy: torch.Tensor
    labels_value: torch.Tensor
    legal_move_mask: Optional[torch.Tensor]
    outputs_policy: Optional[torch.Tensor] = None
    outputs_value: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None
    policy_target_distribution: Optional[torch.Tensor] = None
    move_win_rate: Optional[torch.Tensor] = None


class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    def on_epoch_start(self, epoch_idx: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_batch_start(self, context: TrainingContext) -> None:
        """Called at the start of each batch."""
        ...

    def on_data_transfer_start(
        self, context: TrainingContext
    ) -> None:
        """Called before GPU data transfer."""
        ...

    def on_data_transfer_end(
        self, context: TrainingContext
    ) -> None:
        """Called after GPU data transfer."""
        ...

    def on_forward_pass_start(
        self, context: TrainingContext
    ) -> None:
        """Called before forward pass."""
        ...

    def on_forward_pass_end(
        self, context: TrainingContext
    ) -> None:
        """Called after forward pass."""
        ...

    def on_loss_computation_start(
        self, context: TrainingContext
    ) -> None:
        """Called before loss computation."""
        ...

    def on_loss_computation_end(
        self, context: TrainingContext
    ) -> None:
        """Called after loss computation."""
        ...

    def on_backward_pass_start(
        self, context: TrainingContext
    ) -> None:
        """Called before backward pass."""
        ...

    def on_backward_pass_end(
        self, context: TrainingContext
    ) -> None:
        """Called after backward pass."""
        ...

    def on_optimizer_step_start(
        self, context: TrainingContext
    ) -> None:
        """Called before optimizer step."""
        ...

    def on_optimizer_step_end(
        self, context: TrainingContext
    ) -> None:
        """Called after optimizer step."""
        ...

    def on_batch_end(self, context: TrainingContext) -> None:
        """Called at the end of each batch."""
        ...

    def on_epoch_end(self, epoch_idx: int) -> None:
        """Called at the end of each epoch."""
        ...

    def get_postfix(self) -> dict[str, str] | None:
        """tqdm プログレスバーに表示するメトリクスを返す．

        Returns:
            キーと値が文字列の dict，または None（表示不要の場合）．
            例: ``{"acc": "99.5%", "loss": "0.0012"}``
        """
        ...


class BaseCallback:
    """Base callback implementation with no-op methods."""

    _postfix_sync_interval: int = 50
    """``get_postfix`` で ``.item()`` を呼ぶ間隔(バッチ数)．"""

    def on_epoch_start(self, epoch_idx: int) -> None:
        pass

    def on_batch_start(self, context: TrainingContext) -> None:
        pass

    def on_data_transfer_start(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_data_transfer_end(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_forward_pass_start(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_forward_pass_end(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_loss_computation_start(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_loss_computation_end(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_backward_pass_start(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_backward_pass_end(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_optimizer_step_start(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_optimizer_step_end(
        self, context: TrainingContext
    ) -> None:
        pass

    def on_batch_end(self, context: TrainingContext) -> None:
        pass

    def on_epoch_end(self, epoch_idx: int) -> None:
        pass

    def get_postfix(self) -> dict[str, str] | None:
        """tqdm プログレスバーに表示するメトリクスを返す．"""
        return None


class LoggingCallback(BaseCallback):
    """Callback for logging training progress.

    GPU同期を抑制するため，lossをGPUテンソル上で蓄積し，
    TensorBoard書き込み時(Nバッチごと)にのみ ``.item()`` を呼ぶ．
    """

    def __init__(
        self,
        writer: SummaryWriter,
        dataloader_length: int,
        logger: Optional[logging.Logger] = None,
    ):
        self.writer = writer
        self.dataloader_length = dataloader_length
        self.logger = logger or logging.getLogger(__name__)
        self._running_loss: torch.Tensor = torch.tensor(0.0)
        self.record_num = max(1, dataloader_length // 10)
        self.last_loss = 0.0
        self._device_initialized: bool = False

    def _ensure_device(self, device: torch.device) -> None:
        """初回バッチで蓄積テンソルをGPUデバイスに移動する."""
        if not self._device_initialized:
            self._running_loss = self._running_loss.to(device)
            self._device_initialized = True

    def on_batch_end(self, context: TrainingContext) -> None:
        if context.loss is not None:
            self._ensure_device(context.loss.device)
            self._running_loss += context.loss.detach()
            if (
                context.batch_idx % self.record_num
                == self.record_num - 1
            ):
                self.last_loss = float(
                    self._running_loss.item()
                ) / float(self.record_num)
                tb_x = (
                    context.epoch_idx * self.dataloader_length
                    + context.batch_idx
                    + 1
                )
                self.writer.add_scalar(
                    "Loss/train", self.last_loss, tb_x
                )
                self._running_loss.zero_()


@dataclass(frozen=True)
class ValidationMetrics:
    """Validation metrics aggregated over a single epoch."""

    policy_cross_entropy: float
    value_brier_score: float
    policy_top5_accuracy: float
    policy_f1_score: float
    value_high_confidence_rate: float
    policy_top1_win_rate: Optional[float] = None
    policy_move_label_ce: Optional[float] = None
    policy_expected_win_rate: Optional[float] = None

    def format_log_lines(self) -> str:
        """Format metrics as multi-line log output for console."""
        lines = [
            "METRICS",
            f"  policy_cross_entropy         = {self.policy_cross_entropy}",
            f"  policy_f1_score              = {self.policy_f1_score}",
            f"  policy_top5_accuracy         = {self.policy_top5_accuracy}",
            f"  value_brier_score            = {self.value_brier_score}",
            f"  value_high_confidence_rate   = {self.value_high_confidence_rate}",
        ]
        if self.policy_move_label_ce is not None:
            lines.append(
                f"  policy_move_label_ce         = {self.policy_move_label_ce}"
            )
        if self.policy_top1_win_rate is not None:
            lines.append(
                f"  policy_top1_win_rate         = {self.policy_top1_win_rate}"
            )
        if self.policy_expected_win_rate is not None:
            lines.append(
                f"  policy_expected_win_rate     = {self.policy_expected_win_rate}"
            )
        return "\n".join(lines)


class ValidationCallback(BaseCallback):
    """Callback for validation with policy and value quality metrics.

    GPU同期を抑制するため，全メトリクスをGPUテンソル上で蓄積し，
    エポック終了時の ``get_average_metrics()`` でのみ ``.item()`` を呼ぶ．
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._running_vloss: torch.Tensor = torch.tensor(0.0)
        self._policy_cross_entropy_sum: torch.Tensor = (
            torch.tensor(0.0)
        )
        self._value_brier_score_sum: torch.Tensor = (
            torch.tensor(0.0)
        )
        self.batch_count = 0
        self.policy_sample_count = 0
        self.value_sample_count = 0
        self._policy_top5_ratio_sum: torch.Tensor = (
            torch.tensor(0.0)
        )
        self.policy_top5_sample_count = 0
        self._value_high_conf_pred_count: torch.Tensor = (
            torch.tensor(0, dtype=torch.long)
        )
        self._value_high_conf_correct: torch.Tensor = (
            torch.tensor(0, dtype=torch.long)
        )
        self._policy_f1_tp: torch.Tensor = torch.tensor(
            0, dtype=torch.long
        )
        self._policy_f1_fp: torch.Tensor = torch.tensor(
            0, dtype=torch.long
        )
        self._policy_f1_fn: torch.Tensor = torch.tensor(
            0, dtype=torch.long
        )
        # move_win_rate metrics
        self._policy_top1_win_rate_sum: torch.Tensor = (
            torch.tensor(0.0)
        )
        self._policy_move_label_ce_sum: torch.Tensor = (
            torch.tensor(0.0)
        )
        self.policy_move_label_ce_count = 0
        self._policy_expected_win_rate_sum: torch.Tensor = (
            torch.tensor(0.0)
        )
        self.move_win_rate_sample_count = 0
        self._device_initialized: bool = False

    def _ensure_device(self, device: torch.device) -> None:
        """初回バッチで蓄積テンソルをGPUデバイスに移動する."""
        if self._device_initialized:
            return
        self._running_vloss = self._running_vloss.to(device)
        self._policy_cross_entropy_sum = (
            self._policy_cross_entropy_sum.to(device)
        )
        self._value_brier_score_sum = (
            self._value_brier_score_sum.to(device)
        )
        self._policy_top5_ratio_sum = (
            self._policy_top5_ratio_sum.to(device)
        )
        self._value_high_conf_pred_count = (
            self._value_high_conf_pred_count.to(device)
        )
        self._value_high_conf_correct = (
            self._value_high_conf_correct.to(device)
        )
        self._policy_f1_tp = self._policy_f1_tp.to(device)
        self._policy_f1_fp = self._policy_f1_fp.to(device)
        self._policy_f1_fn = self._policy_f1_fn.to(device)
        self._policy_top1_win_rate_sum = (
            self._policy_top1_win_rate_sum.to(device)
        )
        self._policy_move_label_ce_sum = (
            self._policy_move_label_ce_sum.to(device)
        )
        self._policy_expected_win_rate_sum = (
            self._policy_expected_win_rate_sum.to(device)
        )
        self._device_initialized = True

    def on_batch_end(self, context: TrainingContext) -> None:
        if (
            context.loss is not None
            and context.outputs_policy is not None
            and context.outputs_value is not None
        ):
            self._ensure_device(context.loss.device)
            self._running_vloss += context.loss.detach()
            policy_targets = (
                context.policy_target_distribution
                if context.policy_target_distribution
                is not None
                else normalize_policy_targets(
                    context.labels_policy,
                    context.legal_move_mask,
                )
            )
            self._policy_cross_entropy_sum += (
                self._policy_cross_entropy_gpu(
                    context.outputs_policy, policy_targets
                )
            )
            policy_batch_size = int(policy_targets.size(0))
            value_batch_size = int(context.labels_value.numel())
            self._value_brier_score_sum += (
                self._value_brier_score_gpu(
                    context.outputs_value, context.labels_value
                )
            )
            self.policy_sample_count += policy_batch_size
            self.value_sample_count += value_batch_size
            ratio_sum_t, sample_count = (
                self._compute_policy_top5_accuracy_stats_gpu(
                    logits=context.outputs_policy,
                    targets=policy_targets,
                )
            )
            self._policy_top5_ratio_sum += ratio_sum_t
            self.policy_top5_sample_count += sample_count
            # Accumulate F1 score components on GPU
            tp_t, fp_t, fn_t = (
                self._compute_policy_f1_components_gpu(
                    logits=context.outputs_policy,
                    targets=policy_targets,
                )
            )
            self._policy_f1_tp += tp_t
            self._policy_f1_fp += fp_t
            self._policy_f1_fn += fn_t
            labels_value = context.labels_value.view(-1)
            predicted_value = torch.sigmoid(
                context.outputs_value
            ).view(-1)
            prediction_high_confidence_mask = (
                predicted_value >= 0.8
            )
            self._value_high_conf_pred_count += (
                prediction_high_confidence_mask.sum().long()
            )
            self._value_high_conf_correct += (
                (
                    labels_value[
                        prediction_high_confidence_mask
                    ]
                    >= 0.8
                )
                .sum()
                .long()
            )
            self.batch_count += 1

            # move_win_rate metrics
            if context.move_win_rate is not None:
                with torch.no_grad():
                    win_rate = context.move_win_rate
                    logits = context.outputs_policy
                    batch_n = int(logits.size(0))
                    self.move_win_rate_sample_count += batch_n

                    # policy_top1_win_rate: top-1予測手の実勝率
                    top1_indices = logits.argmax(dim=1)
                    top1_win_rates = win_rate.gather(
                        1, top1_indices.unsqueeze(1)
                    ).squeeze(1)
                    self._policy_top1_win_rate_sum += (
                        top1_win_rates.sum()
                    )

                    # policy_move_label_ce: moveLabelとのCE(参考値)
                    # legal_move_maskがある場合のみ計算する．
                    # maskなしで正規化したmoveLabelはマスク付き確率空間と不整合になるため．
                    if context.legal_move_mask is not None:
                        move_label_targets = (
                            normalize_policy_targets(
                                context.labels_policy,
                                context.legal_move_mask,
                            )
                        )
                        self._policy_move_label_ce_sum += (
                            self._policy_cross_entropy_gpu(
                                logits, move_label_targets
                            )
                        )
                        self.policy_move_label_ce_count += (
                            batch_n
                        )

                    # policy_expected_win_rate: Σ softmax(logits)[i] × moveWinRate[i]
                    # log_softmax + exp で softmax を得る(数値安定性のため)
                    log_probs = torch.log_softmax(logits, dim=1)
                    probs = log_probs.exp()
                    expected_wr = (probs * win_rate).sum(dim=1)
                    self._policy_expected_win_rate_sum += (
                        expected_wr.sum()
                    )

    def get_average_loss(self) -> float:
        """Get average validation loss."""
        if self.batch_count == 0:
            return 0.0
        return float(self._running_vloss.item()) / float(
            self.batch_count
        )

    def get_average_metrics(self) -> ValidationMetrics:
        """Get aggregated policy and value metrics for the epoch.

        エポック終了時にのみ呼ばれる前提のため，
        ここでの ``.item()`` によるGPU同期は許容範囲．
        """
        avg_policy_cross_entropy = float(
            self._policy_cross_entropy_sum.item()
        ) / float(max(1, self.policy_sample_count))
        avg_value_brier = float(
            self._value_brier_score_sum.item()
        ) / float(max(1, self.value_sample_count))
        policy_top5_accuracy = float(
            self._policy_top5_ratio_sum.item()
        ) / float(max(1, self.policy_top5_sample_count))
        policy_f1 = self._calculate_f1_from_components(
            tp=int(self._policy_f1_tp.item()),
            fp=int(self._policy_f1_fp.item()),
            fn=int(self._policy_f1_fn.item()),
        )
        high_conf_pred = int(
            self._value_high_conf_pred_count.item()
        )
        high_conf_correct = int(
            self._value_high_conf_correct.item()
        )
        value_high_confidence_rate = float(
            high_conf_correct
        ) / float(max(1, high_conf_pred))
        # move_win_rate metrics (None when data lacks moveWinRate)
        policy_top1_win_rate = _safe_average(
            float(self._policy_top1_win_rate_sum.item()),
            self.move_win_rate_sample_count,
        )
        policy_move_label_ce = _safe_average(
            float(self._policy_move_label_ce_sum.item()),
            self.policy_move_label_ce_count,
        )
        policy_expected_win_rate = _safe_average(
            float(self._policy_expected_win_rate_sum.item()),
            self.move_win_rate_sample_count,
        )

        return ValidationMetrics(
            policy_cross_entropy=avg_policy_cross_entropy,
            value_brier_score=avg_value_brier,
            policy_top5_accuracy=policy_top5_accuracy,
            policy_f1_score=policy_f1,
            value_high_confidence_rate=value_high_confidence_rate,
            policy_top1_win_rate=policy_top1_win_rate,
            policy_move_label_ce=policy_move_label_ce,
            policy_expected_win_rate=policy_expected_win_rate,
        )

    def reset(self) -> None:
        """Reset all counters for next epoch."""
        if self._device_initialized:
            self._running_vloss.zero_()
            self._policy_cross_entropy_sum.zero_()
            self._value_brier_score_sum.zero_()
            self._policy_top5_ratio_sum.zero_()
            self._value_high_conf_pred_count.zero_()
            self._value_high_conf_correct.zero_()
            self._policy_f1_tp.zero_()
            self._policy_f1_fp.zero_()
            self._policy_f1_fn.zero_()
            self._policy_top1_win_rate_sum.zero_()
            self._policy_move_label_ce_sum.zero_()
            self._policy_expected_win_rate_sum.zero_()
        else:
            self._running_vloss = torch.tensor(0.0)
            self._policy_cross_entropy_sum = torch.tensor(0.0)
            self._value_brier_score_sum = torch.tensor(0.0)
            self._policy_top5_ratio_sum = torch.tensor(0.0)
            self._value_high_conf_pred_count = torch.tensor(
                0, dtype=torch.long
            )
            self._value_high_conf_correct = torch.tensor(
                0, dtype=torch.long
            )
            self._policy_f1_tp = torch.tensor(
                0, dtype=torch.long
            )
            self._policy_f1_fp = torch.tensor(
                0, dtype=torch.long
            )
            self._policy_f1_fn = torch.tensor(
                0, dtype=torch.long
            )
            self._policy_top1_win_rate_sum = torch.tensor(0.0)
            self._policy_move_label_ce_sum = torch.tensor(0.0)
            self._policy_expected_win_rate_sum = torch.tensor(
                0.0
            )
        self.batch_count = 0
        self.policy_sample_count = 0
        self.value_sample_count = 0
        self.policy_top5_sample_count = 0
        self.policy_move_label_ce_count = 0
        self.move_win_rate_sample_count = 0

    def _policy_cross_entropy_gpu(
        self,
        logits: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate total cross entropy as a GPU tensor (no sync)."""
        if logits.ndim != 2 or target_distribution.ndim != 2:
            raise ValueError(
                "Tensors y and t must be 2-dimensional."
            )
        logits_detached = logits.detach()
        targets_detached = target_distribution.detach()
        log_probs = torch.log_softmax(logits_detached, dim=1)
        cross_entropy = -torch.sum(
            targets_detached * log_probs, dim=1
        )
        return torch.sum(cross_entropy)

    def _value_brier_score_gpu(
        self, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Calculate sum of Brier Score as a GPU tensor (no sync)."""
        y_detached = y.detach()
        t_detached = t.detach()
        probabilities = torch.sigmoid(y_detached)
        squared_error = torch.square(probabilities - t_detached)
        return torch.sum(squared_error)

    def _compute_policy_top5_accuracy_stats_gpu(
        self, *, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Return cumulative ratio sum as GPU tensor and sample count."""
        if logits.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                "Tensors logits and targets must be 2-dimensional."
            )

        batch_size = int(targets.size(0))
        if batch_size == 0:
            return torch.tensor(0.0, device=logits.device), 0

        topk_pred = min(5, int(logits.size(1)))
        if topk_pred == 0:
            return torch.tensor(0.0, device=logits.device), 0

        max_label_topk = min(5, int(targets.size(1)))
        if max_label_topk == 0:
            return torch.tensor(0.0, device=logits.device), 0

        logits_detached = logits.detach()
        targets_detached = targets.detach()

        positive_mask = targets_detached > 0
        positive_counts = torch.sum(positive_mask, dim=1)
        effective_label_topk = torch.minimum(
            positive_counts,
            torch.tensor(
                max_label_topk,
                device=targets_detached.device,
                dtype=positive_counts.dtype,
            ),
        )

        label_top_indices = torch.topk(
            targets_detached.masked_fill(
                ~positive_mask, float("-inf")
            ),
            k=max_label_topk,
            dim=1,
        ).indices

        prediction_top_indices = torch.topk(
            logits_detached,
            k=topk_pred,
            dim=1,
        ).indices

        current_topk = torch.minimum(
            effective_label_topk,
            torch.tensor(
                topk_pred,
                device=targets_detached.device,
                dtype=positive_counts.dtype,
            ),
        )

        label_positions = torch.arange(
            max_label_topk,
            device=targets_detached.device,
            dtype=positive_counts.dtype,
        )
        pred_positions = torch.arange(
            topk_pred,
            device=targets_detached.device,
            dtype=positive_counts.dtype,
        )

        label_mask = label_positions.unsqueeze(
            0
        ) < effective_label_topk.unsqueeze(1)
        pred_mask = pred_positions.unsqueeze(
            0
        ) < current_topk.unsqueeze(1)

        matches = label_top_indices.unsqueeze(
            -1
        ) == prediction_top_indices.unsqueeze(-2)
        valid_matches = (
            matches
            & label_mask.unsqueeze(-1)
            & pred_mask.unsqueeze(1)
        )
        match_counts = torch.sum(valid_matches, dim=(1, 2))

        ratios = torch.zeros(
            batch_size,
            device=targets_detached.device,
            dtype=targets_detached.dtype,
        )
        positive_samples = effective_label_topk > 0
        ratios[positive_samples] = match_counts[
            positive_samples
        ].to(targets_detached.dtype) / effective_label_topk[
            positive_samples
        ].to(targets_detached.dtype)

        return torch.sum(ratios), batch_size

    def _compute_policy_f1_components_gpu(
        self, *, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute F1 score components as GPU tensors (no sync).

        Returns:
            Tuple of (total_tp, total_fp, total_fn) as long tensors on GPU.
        """
        if logits.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                "Tensors logits and targets must be 2-dimensional."
            )

        batch_size = int(targets.size(0))
        zero = torch.tensor(
            0, dtype=torch.long, device=logits.device
        )
        if batch_size == 0:
            return zero, zero.clone(), zero.clone()

        topk_pred = min(5, int(logits.size(1)))
        if topk_pred == 0:
            return zero, zero.clone(), zero.clone()

        logits_detached = logits.detach()
        targets_detached = targets.detach()

        positive_mask = targets_detached > 0
        positive_counts = torch.sum(positive_mask, dim=1)

        prediction_top_indices = torch.topk(
            logits_detached,
            k=topk_pred,
            dim=1,
        ).indices

        pred_one_hot = torch.zeros_like(targets_detached)
        pred_one_hot.scatter_(1, prediction_top_indices, 1.0)

        tp_mask = pred_one_hot * positive_mask.float()
        tp_per_sample = torch.sum(tp_mask, dim=1)

        fp_per_sample = topk_pred - tp_per_sample
        fn_per_sample = positive_counts - tp_per_sample

        total_tp = torch.sum(tp_per_sample).long()
        total_fp = torch.sum(fp_per_sample).long()
        total_fn = torch.sum(fn_per_sample).long()

        return total_tp, total_fp, total_fn

    def _calculate_f1_from_components(
        self, *, tp: int, fp: int, fn: int
    ) -> float:
        """Calculate F1 score from accumulated TP, FP, FN counts.

        Args:
            tp: Total true positives across all batches.
            fp: Total false positives across all batches.
            fn: Total false negatives across all batches.

        Returns:
            F1 score in range [0.0, 1.0].
        """
        # Edge case: no predictions and no labels
        if tp == 0 and fp == 0 and fn == 0:
            return 0.0

        # Calculate precision with zero-division protection
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = float(tp) / float(tp + fp)

        # Calculate recall with zero-division protection
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = float(tp) / float(tp + fn)

        # Calculate F1 with zero-division protection
        if precision + recall == 0.0:
            return 0.0

        f1 = 2.0 * (precision * recall) / (precision + recall)
        return f1


@dataclass(frozen=True)
class TimingData:
    """Timing data for a single measurement."""

    value: float


class TimingCallback(BaseCallback):
    """Callback for timing training operations."""

    def __init__(self, warmup_batches: int = 5):
        self.warmup_batches = warmup_batches
        self.timing_stats: Dict[str, List[float]] = {
            "data_loading": [],
            "gpu_transfer": [],
            "forward_pass": [],
            "loss_computation": [],
            "backward_pass": [],
            "optimizer_step": [],
            "total_batch": [],
        }
        self.measured_batches = 0
        self.total_samples = 0
        self._total_loss: torch.Tensor = torch.tensor(0.0)
        self._last_batch_loss: torch.Tensor = torch.tensor(0.0)
        self.epoch_start_time = 0.0
        self.batch_start_time = 0.0
        self.previous_batch_end_time: Optional[float] = None
        self._measurement_start_time: Optional[float] = None
        self._temp_timings: Dict[str, float] = {}
        self._device_initialized: bool = False

    def on_epoch_start(self, epoch_idx: int) -> None:
        self.epoch_start_time = time.perf_counter()

    def on_batch_start(self, context: TrainingContext) -> None:
        self.batch_start_time = time.perf_counter()

        # データローディング時間の測定
        if self.previous_batch_end_time is not None:
            data_load_time = (
                self.batch_start_time
                - self.previous_batch_end_time
            )
        else:
            data_load_time = 0.0

        self._temp_timings["data_loading"] = data_load_time

        if context.batch_size is not None:
            self.total_samples += context.batch_size

    def on_data_transfer_start(
        self, context: TrainingContext
    ) -> None:
        self._temp_timings["gpu_transfer_start"] = (
            time.perf_counter()
        )

    def on_data_transfer_end(
        self, context: TrainingContext
    ) -> None:
        self._temp_timings["gpu_transfer"] = (
            time.perf_counter()
            - self._temp_timings["gpu_transfer_start"]
        )

    def on_forward_pass_start(
        self, context: TrainingContext
    ) -> None:
        self._temp_timings["forward_start"] = (
            time.perf_counter()
        )

    def on_forward_pass_end(
        self, context: TrainingContext
    ) -> None:
        total_forward_time = (
            time.perf_counter()
            - self._temp_timings["forward_start"]
        )
        # 純粋なモデル順伝播時間 = 全体時間 - 損失計算時間
        self._temp_timings["forward_pass"] = (
            total_forward_time
            - self._temp_timings.get("loss_computation", 0.0)
        )

    def on_loss_computation_start(
        self, context: TrainingContext
    ) -> None:
        self._temp_timings["loss_start"] = time.perf_counter()

    def on_loss_computation_end(
        self, context: TrainingContext
    ) -> None:
        self._temp_timings["loss_computation"] = (
            time.perf_counter()
            - self._temp_timings["loss_start"]
        )

    def on_backward_pass_start(
        self, context: TrainingContext
    ) -> None:
        self._temp_timings["backward_start"] = (
            time.perf_counter()
        )

    def on_backward_pass_end(
        self, context: TrainingContext
    ) -> None:
        self._temp_timings["backward_pass"] = (
            time.perf_counter()
            - self._temp_timings["backward_start"]
        )

    def on_optimizer_step_start(
        self, context: TrainingContext
    ) -> None:
        self._temp_timings["optimizer_start"] = (
            time.perf_counter()
        )

    def on_optimizer_step_end(
        self, context: TrainingContext
    ) -> None:
        self._temp_timings["optimizer_step"] = (
            time.perf_counter()
            - self._temp_timings["optimizer_start"]
        )

    def _ensure_device(self, device: torch.device) -> None:
        """初回バッチで蓄積テンソルをGPUデバイスに移動する."""
        if not self._device_initialized:
            self._total_loss = self._total_loss.to(device)
            self._last_batch_loss = self._last_batch_loss.to(
                device
            )
            self._device_initialized = True

    def on_batch_end(self, context: TrainingContext) -> None:
        batch_total_time = (
            time.perf_counter() - self.batch_start_time
        )
        batch_end_time = time.perf_counter()

        if context.loss is not None:
            self._ensure_device(context.loss.device)
            loss_detached = context.loss.detach()
            self._total_loss += loss_detached
            self._last_batch_loss = loss_detached

        # ウォームアップ期間後のタイミング統計を記録
        if context.batch_idx >= self.warmup_batches:
            if self._measurement_start_time is None:
                self._measurement_start_time = (
                    self.batch_start_time
                )
            self.timing_stats["data_loading"].append(
                self._temp_timings["data_loading"]
            )
            # GPU prefetch使用時はGPU転送がスキップされるため，存在する場合のみ記録
            if "gpu_transfer" in self._temp_timings:
                self.timing_stats["gpu_transfer"].append(
                    self._temp_timings["gpu_transfer"]
                )
            self.timing_stats["forward_pass"].append(
                self._temp_timings["forward_pass"]
            )
            self.timing_stats["loss_computation"].append(
                self._temp_timings["loss_computation"]
            )
            # backward_pass / optimizer_step are absent in validation mode
            if "backward_pass" in self._temp_timings:
                self.timing_stats["backward_pass"].append(
                    self._temp_timings["backward_pass"]
                )
            if "optimizer_step" in self._temp_timings:
                self.timing_stats["optimizer_step"].append(
                    self._temp_timings["optimizer_step"]
                )
            self.timing_stats["total_batch"].append(
                batch_total_time
            )
            self.measured_batches += 1

        self.previous_batch_end_time = batch_end_time

    def get_timing_statistics(
        self,
    ) -> Dict[str, float]:
        """Get averaged timing statistics."""
        if not self.timing_stats["total_batch"]:
            raise RuntimeError(
                "No batches were processed for timing measurement"
            )

        def average(values: List[float]) -> float:
            return (
                float(sum(values)) / float(len(values))
                if len(values) > 0
                else 0.0
            )

        return {
            "data_loading_time": average(
                self.timing_stats["data_loading"]
            ),
            "gpu_transfer_time": average(
                self.timing_stats["gpu_transfer"]
            ),
            "forward_pass_time": average(
                self.timing_stats["forward_pass"]
            ),
            "loss_computation_time": average(
                self.timing_stats["loss_computation"]
            ),
            "backward_pass_time": average(
                self.timing_stats["backward_pass"]
            ),
            "optimizer_step_time": average(
                self.timing_stats["optimizer_step"]
            ),
            "average_batch_time": average(
                self.timing_stats["total_batch"]
            ),
        }

    def get_timing_distribution(
        self,
    ) -> Optional[dict[str, dict[str, float]]]:
        """Get timing distribution statistics (std dev, min, max, percentiles).

        Returns:
            各タイミングカテゴリごとの分布統計量を含む辞書．
            キーはタイミングカテゴリ名，値は統計量の辞書．
            計測バッチがない場合は None を返す．
        """
        if not self.timing_stats["total_batch"]:
            return None

        def compute_stats(
            values: list[float],
        ) -> dict[str, float]:
            if not values:
                return {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }
            n = len(values)
            mean = sum(values) / n
            # Population std dev (divided by n, not n-1).
            # Returns 0 for single measurement (n=1).
            # For small n, percentiles use nearest-rank: p50 rounds up
            # (e.g. n=2 → index 1 = max), p95/p99 coincide with max.
            variance = (
                sum((v - mean) ** 2 for v in values) / n
                if n > 1
                else 0.0
            )
            std = math.sqrt(variance)
            sorted_vals = sorted(values)
            # Median: average of two middle values for even n
            mid = n // 2
            if n % 2 == 0:
                p50 = (
                    sorted_vals[mid - 1] + sorted_vals[mid]
                ) / 2
            else:
                p50 = sorted_vals[mid]
            return {
                "mean": mean,
                "std": std,
                "min": sorted_vals[0],
                "max": sorted_vals[-1],
                "p50": p50,
                "p95": sorted_vals[
                    min(
                        max(int(math.ceil(n * 0.95)) - 1, 0),
                        n - 1,
                    )
                ],
                "p99": sorted_vals[
                    min(
                        max(int(math.ceil(n * 0.99)) - 1, 0),
                        n - 1,
                    )
                ],
            }

        return {
            key: compute_stats(values)
            for key, values in self.timing_stats.items()
        }

    def get_performance_metrics(
        self, total_batches: int
    ) -> Dict[str, float]:
        """Get performance metrics.

        ウォームアップ除外の計測区間に基づくメトリクスを返す．
        ``actual_average_batch_time`` はウォームアップバッチを除外した
        壁時計時間ベースの平均バッチ時間で，推定エポック時間の算出に使用する．
        """
        epoch_end_time = time.perf_counter()
        epoch_total_time = (
            epoch_end_time - self.epoch_start_time
        )

        # ウォームアップ除外の計測区間
        if (
            self._measurement_start_time is not None
            and self.measured_batches > 0
        ):
            measured_time = (
                epoch_end_time - self._measurement_start_time
            )
            actual_avg_batch_time = measured_time / float(
                self.measured_batches
            )
            warmup_time = (
                self._measurement_start_time
                - self.epoch_start_time
            )
        else:
            measured_time = epoch_total_time
            actual_avg_batch_time = (
                epoch_total_time / float(total_batches)
                if total_batches > 0
                else 0.0
            )
            warmup_time = 0.0

        return {
            "total_epoch_time": epoch_total_time,
            "total_batches": float(total_batches),
            "measured_batches": float(self.measured_batches),
            "actual_average_batch_time": actual_avg_batch_time,
            "warmup_time": warmup_time,
            "measured_time": measured_time,
            "samples_per_second": (
                float(self.total_samples) / measured_time
                if measured_time > 0
                else 0.0
            ),
            "batches_per_second": (
                float(self.measured_batches) / measured_time
                if measured_time > 0
                else 0.0
            ),
        }

    def get_loss_metrics(
        self, total_batches: int
    ) -> Dict[str, float]:
        """Get loss metrics.

        エポック終了時にのみ呼ばれる前提のため，
        ここでの ``.item()`` によるGPU同期は許容範囲．
        """
        total_loss_val = float(self._total_loss.item())
        return {
            "total_loss": total_loss_val,
            "last_batch_loss": float(
                self._last_batch_loss.item()
            ),
            "average_loss": (
                total_loss_val / float(total_batches)
                if total_batches > 0
                else 0.0
            ),
        }


class ResourceMonitoringCallback(BaseCallback):
    """リソース使用率を監視するCallback．"""

    def __init__(
        self,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            device: 学習で使用しているデバイス
            logger: ロガー
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # システムリソース監視
        self.system_monitor = SystemResourceMonitor(
            monitoring_interval=0.5
        )

        # GPU監視（CUDA使用時のみ）
        self.gpu_monitor: Optional[GPUResourceMonitor] = None
        if self.device.type == "cuda":
            gpu_index = (
                self.device.index
                if self.device.index is not None
                else 0
            )
            self.gpu_monitor = GPUResourceMonitor(
                gpu_index=gpu_index,
                monitoring_interval=0.5,
            )

    def on_epoch_start(self, epoch_idx: int) -> None:
        """エポック開始時にリソース監視を開始する．"""
        self.logger.debug("Starting resource monitoring")
        self.system_monitor.start_monitoring()

        if self.gpu_monitor is not None:
            self.gpu_monitor.start_monitoring()

    def on_epoch_end(self, epoch_idx: int) -> None:
        """エポック終了時にリソース監視を停止する．"""
        self.logger.debug("Stopping resource monitoring")
        self.system_monitor.stop_monitoring()

        if self.gpu_monitor is not None:
            self.gpu_monitor.stop_monitoring()

    def get_resource_usage(self) -> ResourceUsage:
        """統合されたリソース使用率統計を取得する．"""
        # システムリソース使用率を取得
        system_usage = self.system_monitor.get_resource_usage()

        # GPU使用率を取得（利用可能な場合）
        gpu_usage = None
        if self.gpu_monitor is not None:
            gpu_usage = self.gpu_monitor.get_resource_usage()

        # 統合されたResourceUsageを作成
        return ResourceUsage(
            cpu_max_percent=system_usage.cpu_max_percent,
            memory_max_bytes=system_usage.memory_max_bytes,
            memory_max_percent=system_usage.memory_max_percent,
            gpu_max_percent=(
                gpu_usage.gpu_max_percent if gpu_usage else None
            ),
            gpu_avg_percent=(
                gpu_usage.gpu_avg_percent if gpu_usage else None
            ),
            gpu_memory_max_bytes=(
                gpu_usage.gpu_memory_max_bytes
                if gpu_usage
                else None
            ),
            gpu_memory_total_bytes=(
                gpu_usage.gpu_memory_total_bytes
                if gpu_usage
                else None
            ),
            gpu_memory_max_percent=(
                gpu_usage.gpu_memory_max_percent
                if gpu_usage
                else None
            ),
        )


class LRSchedulerStepCallback(BaseCallback):
    """Per-step LR scheduler callback.

    optimizer.step()の直後にscheduler.step()を呼び出し，
    バッチ単位で学習率を更新する．
    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        self._scheduler = scheduler

    def on_optimizer_step_end(
        self, context: TrainingContext
    ) -> None:
        """Optimizer step完了後に学習率スケジューラをステップする．"""
        self._scheduler.step()


class Stage2F1Callback(BaseCallback):
    """Stage 2 (Legal Moves) 用のF1スコア計算コールバック．

    ``on_batch_end`` でサンプル平均F1スコアをGPUテンソル上に蓄積し，
    エポック終了後に ``get_epoch_f1()`` で取得する．

    GPU同期(`.item()`)はエポック終了時のみ行い，
    バッチごとのGPUパイプラインストールを回避する．

    F1計算ロジックは Stage 2 の multi-label binary classification に対応する．
    """

    def __init__(self) -> None:
        self._total_f1: torch.Tensor = torch.tensor(0.0)
        self._total_samples: int = 0
        self._total_loss: torch.Tensor = torch.tensor(0.0)
        self._num_batches: int = 0
        self._device_initialized: bool = False
        self._cached_f1: float = 0.0
        self._cached_loss: float = 0.0

    def _ensure_device(self, device: torch.device) -> None:
        """初回バッチで蓄積テンソルをGPUデバイスに移動する."""
        if not self._device_initialized:
            self._total_f1 = self._total_f1.to(device)
            self._total_loss = self._total_loss.to(device)
            self._device_initialized = True

    def on_batch_end(self, context: TrainingContext) -> None:
        """バッチごとにF1スコアとlossをGPUテンソル上に蓄積する."""
        if context.loss is not None:
            self._ensure_device(context.loss.device)
            self._total_loss += context.loss.detach()
            self._num_batches += 1

        if context.outputs_policy is None:
            return

        # loss が None でも outputs_policy がある場合にデバイスを初期化
        if not self._device_initialized:
            self._ensure_device(context.outputs_policy.device)

        with torch.no_grad():
            predictions = (
                torch.sigmoid(context.outputs_policy) > 0.5
            )
            targets = context.labels_policy.bool()

            tp = (predictions & targets).float().sum(dim=1)
            fp = (predictions & ~targets).float().sum(dim=1)
            fn = (~predictions & targets).float().sum(dim=1)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = (
                2
                * precision
                * recall
                / (precision + recall + 1e-8)
            )

            # 予測・正解ともに空の場合はF1=1.0とする
            both_empty = (predictions.sum(dim=1) == 0) & (
                targets.sum(dim=1) == 0
            )
            f1 = torch.where(
                both_empty, torch.ones_like(f1), f1
            )

            self._total_f1 += f1.sum()
            self._total_samples += f1.numel()

    def get_epoch_f1(self) -> float:
        """エポックのサンプル平均F1スコアを返す."""
        if self._total_samples == 0:
            return 0.0
        return (self._total_f1 / self._total_samples).item()

    def get_average_loss(self) -> float:
        """エポックの平均lossを返す."""
        if self._num_batches == 0:
            return 0.0
        return (self._total_loss / self._num_batches).item()

    def reset(self) -> None:
        """エポック間でカウンタをリセットする."""
        if self._device_initialized:
            self._total_f1.zero_()
            self._total_loss.zero_()
        else:
            self._total_f1 = torch.tensor(0.0)
            self._total_loss = torch.tensor(0.0)
        self._total_samples = 0
        self._num_batches = 0

    def get_postfix(self) -> dict[str, str] | None:
        """Running F1 スコアと loss を tqdm 用に返す．

        GPU同期を抑制するため，Nバッチごとにのみ ``.item()`` で
        スカラー値を取得し，それ以外はキャッシュ値を返す．
        """
        if self._total_samples == 0:
            return None
        if self._num_batches == 1 or (
            self._num_batches % self._postfix_sync_interval == 0
        ):
            self._cached_f1 = (
                self._total_f1 / self._total_samples
            ).item()
            self._cached_loss = (
                self._total_loss / self._num_batches
            ).item()
        return {
            "f1": f"{self._cached_f1:.4f}",
            "loss": f"{self._cached_loss:.4f}",
        }


class Stage1AccuracyCallback(BaseCallback):
    """Stage 1 (Reachable Squares) 用のバイナリ精度計算コールバック．

    ``on_batch_end`` で element-wise バイナリ精度を GPU テンソル上に蓄積し，
    エポック終了後に ``get_epoch_accuracy()`` で取得する．

    GPU同期(`.item()`)はエポック終了時のみ行い，
    バッチごとのGPUパイプラインストールを回避する．

    精度計算ロジックは Stage 1 の element-wise binary classification に対応する:
    ``(sigmoid(logits) > 0.5) == targets.bool()``
    """

    def __init__(self) -> None:
        self._total_correct: torch.Tensor = torch.tensor(0.0)
        self._total_elements: int = 0
        self._total_loss: torch.Tensor = torch.tensor(0.0)
        self._num_batches: int = 0
        self._device_initialized: bool = False
        self._cached_acc: float = 0.0
        self._cached_loss: float = 0.0

    def _ensure_device(self, device: torch.device) -> None:
        """初回バッチで蓄積テンソルをGPUデバイスに移動する."""
        if not self._device_initialized:
            self._total_correct = self._total_correct.to(device)
            self._total_loss = self._total_loss.to(device)
            self._device_initialized = True

    def on_batch_end(self, context: TrainingContext) -> None:
        """バッチごとに精度とlossをGPUテンソル上に蓄積する."""
        if context.loss is not None:
            self._ensure_device(context.loss.device)
            self._total_loss += context.loss.detach()
            self._num_batches += 1

        if context.outputs_policy is None:
            return

        # loss が None でも outputs_policy がある場合にデバイスを初期化
        if not self._device_initialized:
            self._ensure_device(context.outputs_policy.device)

        with torch.no_grad():
            predictions = (
                torch.sigmoid(context.outputs_policy) > 0.5
            )
            targets = context.labels_policy.bool()
            correct = (predictions == targets).float().sum()
            self._total_correct += correct
            self._total_elements += (
                context.labels_policy.numel()
            )

    def get_epoch_accuracy(self) -> float:
        """エポックの element-wise バイナリ精度を返す."""
        if self._total_elements == 0:
            return 0.0
        return (
            self._total_correct / self._total_elements
        ).item()

    def get_average_loss(self) -> float:
        """エポックの平均lossを返す."""
        if self._num_batches == 0:
            return 0.0
        return (self._total_loss / self._num_batches).item()

    def reset(self) -> None:
        """エポック間でカウンタをリセットする."""
        if self._device_initialized:
            self._total_correct.zero_()
            self._total_loss.zero_()
        else:
            self._total_correct = torch.tensor(0.0)
            self._total_loss = torch.tensor(0.0)
        self._total_elements = 0
        self._num_batches = 0

    def get_postfix(self) -> dict[str, str] | None:
        """Running accuracy と loss を tqdm 用に返す．

        GPU同期を抑制するため，Nバッチごとにのみ ``.item()`` で
        スカラー値を取得し，それ以外はキャッシュ値を返す．
        """
        if self._total_elements == 0:
            return None
        if self._num_batches == 1 or (
            self._num_batches % self._postfix_sync_interval == 0
        ):
            self._cached_acc = (
                self._total_correct / self._total_elements
            ).item()
            self._cached_loss = (
                self._total_loss / self._num_batches
            ).item()
        return {
            "acc": f"{self._cached_acc:.1%}",
            "loss": f"{self._cached_loss:.4f}",
        }


class Stage3LossCallback(BaseCallback):
    """Stage 3 (Policy+Value) 用の損失表示コールバック．

    ``on_batch_end`` で損失をGPUテンソル上に蓄積し，
    ``get_postfix()`` でtqdmプログレスバーにrunning lossを表示する．

    GPU同期(``.item()``)はエポック終了時のみ行い，
    バッチごとのGPUパイプラインストールを回避する．
    """

    def __init__(self) -> None:
        self._total_loss: torch.Tensor = torch.tensor(0.0)
        self._num_batches: int = 0
        self._device_initialized: bool = False
        self._cached_loss: float = 0.0

    def _ensure_device(self, device: torch.device) -> None:
        """初回バッチで蓄積テンソルをGPUデバイスに移動する."""
        if not self._device_initialized:
            self._total_loss = self._total_loss.to(device)
            self._device_initialized = True

    def on_batch_end(self, context: TrainingContext) -> None:
        """バッチごとにlossをGPUテンソル上に蓄積する."""
        if context.loss is None:
            return
        self._ensure_device(context.loss.device)
        self._total_loss += context.loss.detach()
        self._num_batches += 1

    def get_average_loss(self) -> float:
        """エポックの平均lossを返す."""
        if self._num_batches == 0:
            return 0.0
        return (self._total_loss / self._num_batches).item()

    def reset(self) -> None:
        """エポック間でカウンタをリセットする."""
        if self._device_initialized:
            self._total_loss.zero_()
        else:
            self._total_loss = torch.tensor(0.0)
        self._num_batches = 0

    def get_postfix(self) -> dict[str, str] | None:
        """Running loss を tqdm 用に返す．

        GPU同期を抑制するため，Nバッチごとにのみ ``.item()`` で
        スカラー値を取得し，それ以外はキャッシュ値を返す．
        """
        if self._num_batches == 0:
            return None
        if self._num_batches == 1 or (
            self._num_batches % self._postfix_sync_interval == 0
        ):
            self._cached_loss = (
                self._total_loss / self._num_batches
            ).item()
        return {"loss": f"{self._cached_loss:.4f}"}
