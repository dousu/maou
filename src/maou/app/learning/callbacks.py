import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, Union

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
    """Callback for logging training progress."""

    def __init__(
        self,
        writer: SummaryWriter,
        dataloader_length: int,
        logger: Optional[logging.Logger] = None,
    ):
        self.writer = writer
        self.dataloader_length = dataloader_length
        self.logger = logger or logging.getLogger(__name__)
        self.running_loss = 0.0
        self.record_num = max(1, dataloader_length // 10)
        self.last_loss = 0.0

    def on_batch_end(self, context: TrainingContext) -> None:
        if context.loss is not None:
            self.running_loss += context.loss.item()
            if (
                context.batch_idx % self.record_num
                == self.record_num - 1
            ):
                self.last_loss = float(
                    self.running_loss
                ) / float(self.record_num)
                self.logger.info(
                    f"  batch {context.batch_idx + 1} loss: {self.last_loss}"
                )
                tb_x = (
                    context.epoch_idx * self.dataloader_length
                    + context.batch_idx
                    + 1
                )
                self.writer.add_scalar(
                    "Loss/train", self.last_loss, tb_x
                )
                self.running_loss = 0.0


@dataclass(frozen=True)
class ValidationMetrics:
    """Validation metrics aggregated over a single epoch."""

    policy_cross_entropy: float
    value_brier_score: float
    policy_top5_accuracy: float
    policy_f1_score: float
    value_high_confidence_rate: float


class ValidationCallback(BaseCallback):
    """Callback for validation with policy and value quality metrics."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.running_vloss = 0.0
        self.policy_cross_entropy_sum = 0.0
        self.value_brier_score_sum = 0.0
        self.batch_count = 0
        self.policy_sample_count = 0
        self.value_sample_count = 0
        self.policy_top5_ratio_sum = 0.0
        self.policy_top5_sample_count = 0
        self.value_high_confidence_prediction_count = 0
        self.value_high_confidence_correct = 0
        self.policy_f1_true_positives = 0
        self.policy_f1_false_positives = 0
        self.policy_f1_false_negatives = 0

    def on_batch_end(self, context: TrainingContext) -> None:
        if (
            context.loss is not None
            and context.outputs_policy is not None
            and context.outputs_value is not None
        ):
            self.running_vloss += context.loss.item()
            policy_targets = (
                context.policy_target_distribution
                if context.policy_target_distribution
                is not None
                else normalize_policy_targets(
                    context.labels_policy,
                    context.legal_move_mask,
                )
            )
            self.policy_cross_entropy_sum += (
                self._policy_cross_entropy(
                    context.outputs_policy, policy_targets
                )
            )
            policy_batch_size = int(policy_targets.size(0))
            value_batch_size = int(context.labels_value.numel())
            self.value_brier_score_sum += (
                self._value_brier_score(
                    context.outputs_value, context.labels_value
                )
            )
            self.policy_sample_count += policy_batch_size
            self.value_sample_count += value_batch_size
            ratio_sum, sample_count = (
                self._compute_policy_top5_accuracy_stats(
                    logits=context.outputs_policy,
                    targets=policy_targets,
                )
            )
            self.policy_top5_ratio_sum += ratio_sum
            self.policy_top5_sample_count += sample_count
            # Accumulate F1 score components
            tp, fp, fn = self._compute_policy_f1_components(
                logits=context.outputs_policy,
                targets=policy_targets,
            )
            self.policy_f1_true_positives += tp
            self.policy_f1_false_positives += fp
            self.policy_f1_false_negatives += fn
            labels_value = context.labels_value.view(-1)
            predicted_value = torch.sigmoid(
                context.outputs_value
            ).view(-1)
            prediction_high_confidence_mask = (
                predicted_value >= 0.8
            )
            self.value_high_confidence_prediction_count += int(
                torch.sum(
                    prediction_high_confidence_mask
                ).item()
            )
            if torch.any(prediction_high_confidence_mask):
                self.value_high_confidence_correct += int(
                    torch.sum(
                        labels_value[
                            prediction_high_confidence_mask
                        ]
                        >= 0.8
                    ).item()
                )
            self.batch_count += 1

    def get_average_loss(self) -> float:
        """Get average validation loss."""
        return float(self.running_vloss) / float(
            max(1, self.batch_count)
        )

    def get_average_metrics(self) -> ValidationMetrics:
        """Get aggregated policy and value metrics for the epoch."""
        avg_policy_cross_entropy = float(
            self.policy_cross_entropy_sum
        ) / float(max(1, self.policy_sample_count))
        avg_value_brier = float(
            self.value_brier_score_sum
        ) / float(max(1, self.value_sample_count))
        policy_top5_accuracy = float(
            self.policy_top5_ratio_sum
        ) / float(max(1, self.policy_top5_sample_count))
        policy_f1 = self._calculate_f1_from_components(
            tp=self.policy_f1_true_positives,
            fp=self.policy_f1_false_positives,
            fn=self.policy_f1_false_negatives,
        )
        value_high_confidence_rate = float(
            self.value_high_confidence_correct
        ) / float(
            max(1, self.value_high_confidence_prediction_count)
        )
        return ValidationMetrics(
            policy_cross_entropy=avg_policy_cross_entropy,
            value_brier_score=avg_value_brier,
            policy_top5_accuracy=policy_top5_accuracy,
            policy_f1_score=policy_f1,
            value_high_confidence_rate=value_high_confidence_rate,
        )

    def reset(self) -> None:
        """Reset all counters for next epoch."""
        self.running_vloss = 0.0
        self.policy_cross_entropy_sum = 0.0
        self.value_brier_score_sum = 0.0
        self.batch_count = 0
        self.policy_sample_count = 0
        self.value_sample_count = 0
        self.policy_top5_ratio_sum = 0.0
        self.policy_top5_sample_count = 0
        self.value_high_confidence_prediction_count = 0
        self.value_high_confidence_correct = 0
        self.policy_f1_true_positives = 0
        self.policy_f1_false_positives = 0
        self.policy_f1_false_negatives = 0

    def _policy_cross_entropy(
        self,
        logits: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> float:
        """Calculate total cross entropy between logits and target distribution."""
        if logits.ndim != 2 or target_distribution.ndim != 2:
            raise ValueError(
                "Tensors y and t must be 2-dimensional."
            )

        log_probs = torch.log_softmax(logits, dim=1)
        cross_entropy = -torch.sum(
            target_distribution * log_probs, dim=1
        )
        return float(torch.sum(cross_entropy).item())

    def _policy_accuracy(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Calculate average policy Top-5 accuracy for the provided batch."""

        ratio_sum, sample_count = (
            self._compute_policy_top5_accuracy_stats(
                logits=logits, targets=targets
            )
        )
        if sample_count == 0:
            return 0.0
        return ratio_sum / float(sample_count)

    def _compute_policy_top5_accuracy_stats(
        self, *, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, int]:
        """Return cumulative ratio sum and sample count for Top-5 accuracy."""

        if logits.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                "Tensors logits and targets must be 2-dimensional."
            )

        batch_size = int(targets.size(0))
        if batch_size == 0:
            return 0.0, 0

        topk_pred = min(5, int(logits.size(1)))
        if topk_pred == 0:
            return 0.0, 0

        max_label_topk = min(5, int(targets.size(1)))
        if max_label_topk == 0:
            return 0.0, 0

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

        ratio_sum = float(torch.sum(ratios).item())
        sample_count = batch_size

        return ratio_sum, sample_count

    def _value_brier_score(
        self, y: torch.Tensor, t: torch.Tensor
    ) -> float:
        """Calculate sum of Brier Score components for a batch."""
        probabilities = torch.sigmoid(y)
        squared_error = torch.square(probabilities - t)
        return torch.sum(squared_error).item()

    def _compute_policy_f1_components(
        self, *, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[int, int, int]:
        """Compute F1 score components (TP, FP, FN) for Top-5 predictions.

        Args:
            logits: Policy network output logits (batch_size, num_classes).
            targets: Normalized policy target distribution (batch_size, num_classes).

        Returns:
            Tuple of (true_positives, false_positives, false_negatives).
        """
        if logits.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                "Tensors logits and targets must be 2-dimensional."
            )

        batch_size = int(targets.size(0))
        if batch_size == 0:
            return 0, 0, 0

        # Get top-5 predictions
        topk_pred = min(5, int(logits.size(1)))
        if topk_pred == 0:
            return 0, 0, 0

        logits_detached = logits.detach()
        targets_detached = targets.detach()

        # Extract positive labels (targets > 0)
        positive_mask = targets_detached > 0
        positive_counts = torch.sum(
            positive_mask, dim=1
        )  # (batch_size,)

        # Get top-5 prediction indices
        prediction_top_indices = torch.topk(
            logits_detached,
            k=topk_pred,
            dim=1,
        ).indices  # (batch_size, topk_pred)

        # Create one-hot encoding of predictions
        pred_one_hot = torch.zeros_like(
            targets_detached
        )  # (batch_size, num_classes)
        pred_one_hot.scatter_(1, prediction_top_indices, 1.0)

        # True Positives: predictions that are also positive labels
        tp_mask = (
            pred_one_hot * positive_mask.float()
        )  # Element-wise AND
        tp_per_sample = torch.sum(
            tp_mask, dim=1
        )  # (batch_size,)

        # False Positives: predictions that are NOT positive labels
        fp_per_sample = (
            topk_pred - tp_per_sample
        )  # Total predictions - TP

        # False Negatives: positive labels that were NOT predicted
        fn_per_sample = (
            positive_counts - tp_per_sample
        )  # Total labels - TP

        # Sum across batch
        total_tp = int(torch.sum(tp_per_sample).item())
        total_fp = int(torch.sum(fp_per_sample).item())
        total_fn = int(torch.sum(fn_per_sample).item())

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
        self.running_loss = 0.0
        self.total_loss = 0.0
        self.epoch_start_time = 0.0
        self.batch_start_time = 0.0
        self.previous_batch_end_time: Optional[float] = None
        self._measurement_start_time: Optional[float] = None
        self._temp_timings: Dict[str, float] = {}

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

    def on_batch_end(self, context: TrainingContext) -> None:
        batch_total_time = (
            time.perf_counter() - self.batch_start_time
        )
        batch_end_time = time.perf_counter()

        if context.loss is not None:
            loss_value = context.loss.item()
            self.running_loss += loss_value
            self.total_loss += loss_value

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
            self.timing_stats["backward_pass"].append(
                self._temp_timings["backward_pass"]
            )
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
        """Get loss metrics."""
        return {
            "total_loss": self.total_loss,
            "average_loss": (
                float(self.total_loss) / float(total_batches)
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
        """Running F1 スコアと loss を tqdm 用に返す．"""
        if self._total_samples == 0:
            return None
        f1 = (self._total_f1 / self._total_samples).item()
        loss = (
            (self._total_loss / self._num_batches).item()
            if self._num_batches > 0
            else 0.0
        )
        return {"f1": f"{f1:.4f}", "loss": f"{loss:.4f}"}


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
        """Running accuracy と loss を tqdm 用に返す．"""
        if self._total_elements == 0:
            return None
        acc = (
            self._total_correct / self._total_elements
        ).item()
        loss = (
            (self._total_loss / self._num_batches).item()
            if self._num_batches > 0
            else 0.0
        )
        return {"acc": f"{acc:.1%}", "loss": f"{loss:.4f}"}


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
        """Running loss を tqdm 用に返す．"""
        if self._num_batches == 0:
            return None
        loss = (self._total_loss / self._num_batches).item()
        return {"loss": f"{loss:.4f}"}
