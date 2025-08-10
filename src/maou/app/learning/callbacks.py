import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore


@dataclass
class TrainingContext:
    """Training context passed to callbacks."""

    batch_idx: int
    epoch_idx: int
    inputs: torch.Tensor
    labels_policy: torch.Tensor
    labels_value: torch.Tensor
    legal_move_mask: torch.Tensor
    outputs_policy: Optional[torch.Tensor] = None
    outputs_value: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None


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

    def on_batch_end(self, context: TrainingContext) -> None:
        if context.loss is not None:
            self.running_loss += context.loss.item()
            if (
                context.batch_idx % self.record_num
                == self.record_num - 1
            ):
                last_loss = float(self.running_loss) / float(
                    self.record_num
                )
                self.logger.info(
                    f"  batch {context.batch_idx + 1} loss: {last_loss}"
                )
                tb_x = (
                    context.epoch_idx * self.dataloader_length
                    + context.batch_idx
                    + 1
                )
                self.writer.add_scalar(
                    "Loss/train", last_loss, tb_x
                )
                self.running_loss = 0.0


class ValidationCallback(BaseCallback):
    """Callback for validation with accuracy tracking."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.running_vloss = 0.0
        self.test_accuracy_policy = 0.0
        self.test_accuracy_value = 0.0
        self.batch_count = 0

    def on_batch_end(self, context: TrainingContext) -> None:
        if (
            context.loss is not None
            and context.outputs_policy is not None
            and context.outputs_value is not None
        ):
            self.running_vloss += context.loss.item()
            self.test_accuracy_policy += self._policy_accuracy(
                context.outputs_policy, context.labels_policy
            )
            self.test_accuracy_value += self._value_accuracy(
                context.outputs_value, context.labels_value
            )
            self.batch_count += 1

    def get_average_loss(self) -> float:
        """Get average validation loss."""
        return float(self.running_vloss) / float(
            max(1, self.batch_count)
        )

    def get_average_accuracies(self) -> tuple[float, float]:
        """Get average policy and value accuracies."""
        avg_policy = float(self.test_accuracy_policy) / float(
            max(1, self.batch_count)
        )
        avg_value = float(self.test_accuracy_value) / float(
            max(1, self.batch_count)
        )
        return avg_policy, avg_value

    def reset(self) -> None:
        """Reset all counters for next epoch."""
        self.running_vloss = 0.0
        self.test_accuracy_policy = 0.0
        self.test_accuracy_value = 0.0
        self.batch_count = 0

    def _policy_accuracy(
        self, y: torch.Tensor, t: torch.Tensor
    ) -> float:
        """Calculate policy accuracy."""
        return (torch.max(y, 1)[1] == t).sum().item() / len(t)

    def _value_accuracy(
        self, y: torch.Tensor, t: torch.Tensor
    ) -> float:
        """Calculate value accuracy."""
        pred = y >= 0
        truth = t >= 0.5
        return pred.eq(truth).sum().item() / len(t)


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
            self.timing_stats["data_loading"].append(
                self._temp_timings["data_loading"]
            )
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

    def get_timing_statistics(self) -> Dict[str, float]:
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
        """Get performance metrics."""
        epoch_total_time = (
            time.perf_counter() - self.epoch_start_time
        )
        return {
            "total_epoch_time": epoch_total_time,
            "total_batches": float(total_batches),
            "samples_per_second": float(self.total_samples)
            / epoch_total_time,
            "batches_per_second": float(total_batches)
            / epoch_total_time,
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
