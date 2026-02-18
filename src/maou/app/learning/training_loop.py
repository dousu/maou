import logging
from collections.abc import Sequence
from typing import List, Optional, cast

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from maou.app.learning.callbacks import (
    ModelInputs,
    TrainingCallback,
    TrainingContext,
)
from maou.app.learning.gpu_prefetcher import (
    DataPrefetcher,
    calculate_recommended_buffer_size,
)
from maou.app.learning.policy_targets import (
    normalize_policy_targets,
)


class TrainingLoop:
    """Generic training loop with callback support."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn_policy: torch.nn.Module,
        loss_fn_value: torch.nn.Module,
        policy_loss_ratio: float,
        value_loss_ratio: float,
        callbacks: Optional[List[TrainingCallback]] = None,
        logger: Optional[logging.Logger] = None,
        enable_gpu_prefetch: bool = True,
        gpu_prefetch_buffer_size: int = 0,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn_policy = loss_fn_policy
        self.loss_fn_value = loss_fn_value
        self.policy_loss_ratio = policy_loss_ratio
        self.value_loss_ratio = value_loss_ratio
        self.callbacks = callbacks or []
        self.logger = logger or logging.getLogger(__name__)
        self._cuda_sync_enabled = False

        # GPU prefetch設定
        self.enable_gpu_prefetch = (
            enable_gpu_prefetch and device.type == "cuda"
        )
        self.gpu_prefetch_buffer_size = gpu_prefetch_buffer_size

        if self.enable_gpu_prefetch:
            if gpu_prefetch_buffer_size <= 0:
                self.logger.info(
                    "GPU prefetching enabled with auto-calculated buffer size "
                    "(determined per batch size)"
                )
            else:
                self.logger.info(
                    f"GPU prefetching enabled with buffer_size={gpu_prefetch_buffer_size}"
                )

        # Gradient accumulation設定
        self.gradient_accumulation_steps = max(
            1, gradient_accumulation_steps
        )
        if self.gradient_accumulation_steps > 1:
            self.logger.info(
                f"Gradient accumulation enabled: {gradient_accumulation_steps} steps "
                f"(effective batch size multiplied by {gradient_accumulation_steps})"
            )

        # Loss finitude check interval (reduces GPU sync points)
        self._finitude_check_interval: int = 100

        # Mixed precision training用のGradScalerを初期化（GPU使用時のみ）
        if self.device.type == "cuda":
            self.scaler: Optional[GradScaler] = GradScaler(
                "cuda"
            )
        else:
            self.scaler = None

    def run_epoch(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
        *,
        max_batches: Optional[int] = None,
        enable_profiling: bool = False,
        progress_bar: bool = True,
        train_mode: bool = True,
        force_cuda_sync: Optional[bool] = None,
    ) -> None:
        """Run a single epoch of training or validation."""
        previous_sync_state = self._cuda_sync_enabled
        resolved_sync_state = (
            force_cuda_sync
            if force_cuda_sync is not None
            else enable_profiling
        )
        self._cuda_sync_enabled = resolved_sync_state

        if (
            self.device.type == "cuda"
            and previous_sync_state != resolved_sync_state
        ):
            self.logger.debug(
                "CUDA synchronization %s for epoch %d (profiling=%s, forced=%s)",
                "enabled"
                if resolved_sync_state
                else "disabled",
                epoch_idx,
                enable_profiling,
                force_cuda_sync,
            )

        # モデルのモード設定
        self.model.train(train_mode)

        # エポック開始のコールバック
        for callback in self.callbacks:
            callback.on_epoch_start(epoch_idx)

        # PyTorchプロファイラーの設定
        profiler = None
        if enable_profiling:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=3, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./profiler_logs"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler.start()

        try:
            # GPU prefetchを有効化している場合はDataPrefetcherでラップ
            if self.enable_gpu_prefetch:
                # バッファサイズの決定（0以下の場合は自動計算）
                if self.gpu_prefetch_buffer_size <= 0:
                    effective_batch_size = (
                        dataloader.batch_size
                        if dataloader.batch_size is not None
                        else 256
                    )
                    buffer_size = (
                        calculate_recommended_buffer_size(
                            effective_batch_size
                        )
                    )
                    self.logger.info(
                        f"Auto-calculated GPU prefetch buffer size: {buffer_size} "
                        f"(based on batch_size={effective_batch_size})"
                    )
                else:
                    buffer_size = self.gpu_prefetch_buffer_size

                prefetcher = DataPrefetcher(
                    dataloader,
                    device=self.device,
                    buffer_size=buffer_size,
                    pin_memory_override=True,
                )
                data_iterator = prefetcher
            else:
                data_iterator = dataloader

            dataloader_iter = (
                tqdm(
                    enumerate(data_iterator),
                    desc="Training"
                    if train_mode
                    else "Validation",
                    total=len(dataloader),
                )
                if progress_bar
                else enumerate(data_iterator)
            )

            for batch_idx, data in dataloader_iter:
                if (
                    max_batches is not None
                    and batch_idx >= max_batches
                ):
                    break

                # データの展開
                (
                    inputs,
                    (
                        labels_policy,
                        labels_value,
                        legal_move_mask,
                    ),
                ) = data
                batch_size = self._resolve_batch_size(inputs)

                # コンテキストの作成
                context = TrainingContext(
                    batch_idx=batch_idx,
                    epoch_idx=epoch_idx,
                    inputs=inputs,
                    labels_policy=labels_policy,
                    labels_value=labels_value,
                    legal_move_mask=legal_move_mask,
                    batch_size=batch_size,
                )

                # バッチ開始のコールバック
                for callback in self.callbacks:
                    callback.on_batch_start(context)

                # GPU転送（prefetch使用時は既にGPU上にあるためスキップ）
                if not self.enable_gpu_prefetch:
                    self._transfer_to_device(context)

                if train_mode:
                    # 学習モード：勾配計算あり
                    self._train_batch(context)
                else:
                    # 評価モード：勾配計算なし
                    self._eval_batch(context)

                # バッチ終了のコールバック
                for callback in self.callbacks:
                    callback.on_batch_end(context)

                # プロファイラーのステップ
                if profiler is not None:
                    profiler.step()

        finally:
            # GPU prefetcherのクリーンアップ
            if (
                self.enable_gpu_prefetch
                and "prefetcher" in locals()
            ):
                prefetcher.shutdown()

            if (
                self.device.type == "cuda"
                and previous_sync_state
                != self._cuda_sync_enabled
            ):
                self.logger.debug(
                    "CUDA synchronization restored to %s after epoch %d",
                    "enabled"
                    if previous_sync_state
                    else "disabled",
                    epoch_idx,
                )
            self._cuda_sync_enabled = previous_sync_state
            if profiler is not None:
                profiler.stop()

        # エポック終了のコールバック
        for callback in self.callbacks:
            callback.on_epoch_end(epoch_idx)

    def _transfer_to_device(
        self, context: TrainingContext
    ) -> None:
        """Transfer data to device with callback hooks."""
        for callback in self.callbacks:
            callback.on_data_transfer_start(context)

        if self.device.type == "cuda":
            # GPU転送（DataLoaderのpin_memoryと非同期転送を活用）
            context.inputs = self._move_inputs_to_device(
                context.inputs,
                self.device,
            )
            context.labels_policy = context.labels_policy.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            context.labels_value = context.labels_value.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if context.legal_move_mask is not None:
                context.legal_move_mask = (
                    context.legal_move_mask.to(
                        self.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                )

            # GPU同期（正確な転送時間測定のため）
            self._maybe_synchronize("post_data_transfer")

        for callback in self.callbacks:
            callback.on_data_transfer_end(context)

    def _train_batch(self, context: TrainingContext) -> None:
        """Train a single batch with gradient computation."""
        # Gradient accumulationのステップを計算
        accumulation_step = (
            context.batch_idx % self.gradient_accumulation_steps
        )
        is_accumulation_step = accumulation_step < (
            self.gradient_accumulation_steps - 1
        )

        # 勾配蓄積サイクルの最初のステップでのみ勾配をクリア
        if accumulation_step == 0:
            self.optimizer.zero_grad()

        # Mixed precision training with autocast
        if self.scaler is not None:
            self._train_batch_mixed_precision(
                context, is_accumulation_step
            )
        else:
            self._train_batch_full_precision(
                context, is_accumulation_step
            )

    def _train_batch_mixed_precision(
        self,
        context: TrainingContext,
        is_accumulation_step: bool,
    ) -> None:
        """Train batch with mixed precision."""
        if self.scaler is None:
            raise RuntimeError(
                "GradScaler is None but mixed precision training was requested"
            )

        # 順伝播
        for callback in self.callbacks:
            callback.on_forward_pass_start(context)

        with autocast(self.device.type):
            context.outputs_policy, context.outputs_value = (
                self.model(context.inputs)
            )

            # 損失計算
            for callback in self.callbacks:
                callback.on_loss_computation_start(context)

            policy_loss = self._compute_policy_loss(context)
            value_loss = self.loss_fn_value(
                context.outputs_value, context.labels_value
            )
            # Gradient accumulation: 損失を蓄積ステップ数で正規化
            context.loss = (
                self.policy_loss_ratio * policy_loss
                + self.value_loss_ratio * value_loss
            ) / self.gradient_accumulation_steps
            for callback in self.callbacks:
                callback.on_loss_computation_end(context)

        self._maybe_synchronize("post_forward_mixed_precision")

        # Check loss finitude periodically to reduce GPU sync points
        if (
            context.batch_idx % self._finitude_check_interval
            == 0
        ):
            loss_is_finite = torch.isfinite(context.loss)
            if not bool(loss_is_finite.item()):
                policy_loss_is_finite = torch.isfinite(
                    policy_loss
                )
                value_loss_is_finite = torch.isfinite(
                    value_loss
                )
                for callback in self.callbacks:
                    callback.on_forward_pass_end(context)

                self.logger.warning(
                    "Non-finite loss detected (epoch=%d, batch=%d): loss=%s "
                    "policy_loss=%s (finite=%s) value_loss=%s (finite=%s)",
                    context.epoch_idx,
                    context.batch_idx,
                    context.loss.detach(),
                    policy_loss.detach(),
                    bool(policy_loss_is_finite.item()),
                    value_loss.detach(),
                    bool(value_loss_is_finite.item()),
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                return

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 逆伝播
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)

        if context.loss is None:
            raise RuntimeError(
                "Loss computation failed - context.loss is None"
            )

        self.scaler.scale(context.loss).backward()

        self._maybe_synchronize("post_backward_mixed_precision")

        for callback in self.callbacks:
            callback.on_backward_pass_end(context)

        # Gradient accumulation: 蓄積ステップの最後でのみオプティマイザを実行
        if not is_accumulation_step:
            # 勾配クリッピング (scaled gradients)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            # オプティマイザステップ
            for callback in self.callbacks:
                callback.on_optimizer_step_start(context)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._maybe_synchronize(
                "post_optimizer_step_mixed_precision"
            )

            for callback in self.callbacks:
                callback.on_optimizer_step_end(context)

    def _move_inputs_to_device(
        self,
        inputs: ModelInputs,
        device: torch.device,
        *,
        non_blocking: bool = True,
    ) -> ModelInputs:
        def move(
            value: object, path: tuple[int, ...]
        ) -> object:
            if isinstance(value, torch.Tensor):
                target_dtype = self._infer_input_dtype(
                    path, value
                )
                if (
                    target_dtype is not None
                    and value.dtype != target_dtype
                ):
                    return value.to(
                        device=device,
                        dtype=target_dtype,
                        non_blocking=non_blocking,
                    )
                return value.to(
                    device=device,
                    non_blocking=non_blocking,
                )
            if isinstance(value, tuple):
                return tuple(
                    move(item, path + (index,))
                    for index, item in enumerate(value)
                )
            if isinstance(value, list):
                return [
                    move(item, path + (index,))
                    for index, item in enumerate(value)
                ]
            return value

        moved = move(inputs, tuple())
        return cast(ModelInputs, moved)

    @staticmethod
    def _infer_input_dtype(
        index_path: tuple[int, ...], tensor: torch.Tensor
    ) -> torch.dtype | None:
        if not index_path:
            return None

        root_index = index_path[0]
        if root_index == 0 and not torch.is_floating_point(
            tensor
        ):
            return torch.long
        if root_index == 1 and tensor.dtype != torch.float32:
            return torch.float32
        return None

    @staticmethod
    def _resolve_batch_size(inputs: ModelInputs) -> int:
        if isinstance(inputs, torch.Tensor):
            return int(inputs.size(0))
        if isinstance(inputs, Sequence):
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    return int(element.size(0))
                if isinstance(element, Sequence):
                    try:
                        return TrainingLoop._resolve_batch_size(
                            element
                        )
                    except (
                        AttributeError,
                        TypeError,
                        ValueError,
                    ):
                        continue
            msg = "Unable to resolve batch size from nested inputs."
            raise TypeError(msg)
        msg = f"Unsupported input type for batch size resolution: {type(inputs)!r}"
        raise TypeError(msg)

    def _train_batch_full_precision(
        self,
        context: TrainingContext,
        is_accumulation_step: bool,
    ) -> None:
        """Train batch with full precision."""
        # 順伝播
        for callback in self.callbacks:
            callback.on_forward_pass_start(context)

        context.outputs_policy, context.outputs_value = (
            self.model(context.inputs)
        )

        self._maybe_synchronize("post_forward_full_precision")

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 損失計算
        for callback in self.callbacks:
            callback.on_loss_computation_start(context)

        policy_loss = self._compute_policy_loss(context)
        value_loss = self.loss_fn_value(
            context.outputs_value, context.labels_value
        )
        # Gradient accumulation: 損失を蓄積ステップ数で正規化
        context.loss = (
            self.policy_loss_ratio * policy_loss
            + self.value_loss_ratio * value_loss
        ) / self.gradient_accumulation_steps
        for callback in self.callbacks:
            callback.on_loss_computation_end(context)

        # Check loss finitude periodically to reduce GPU sync points
        if (
            context.batch_idx % self._finitude_check_interval
            == 0
        ):
            loss_is_finite = torch.isfinite(context.loss)
            if not bool(loss_is_finite.item()):
                policy_loss_is_finite = torch.isfinite(
                    policy_loss
                )
                value_loss_is_finite = torch.isfinite(
                    value_loss
                )
                self.logger.warning(
                    "Non-finite loss detected (epoch=%d, batch=%d): loss=%s "
                    "policy_loss=%s (finite=%s) value_loss=%s (finite=%s)",
                    context.epoch_idx,
                    context.batch_idx,
                    context.loss.detach(),
                    policy_loss.detach(),
                    bool(policy_loss_is_finite.item()),
                    value_loss.detach(),
                    bool(value_loss_is_finite.item()),
                )
                self.optimizer.zero_grad(set_to_none=True)
                return

        # 逆伝播
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)

        if context.loss is None:
            raise RuntimeError(
                "Loss computation failed - context.loss is None"
            )

        context.loss.backward()

        self._maybe_synchronize("post_backward_full_precision")

        for callback in self.callbacks:
            callback.on_backward_pass_end(context)

        # Gradient accumulation: 蓄積ステップの最後でのみオプティマイザを実行
        if not is_accumulation_step:
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            # オプティマイザステップ
            for callback in self.callbacks:
                callback.on_optimizer_step_start(context)

            self.optimizer.step()

            self._maybe_synchronize(
                "post_optimizer_step_full_precision"
            )

            for callback in self.callbacks:
                callback.on_optimizer_step_end(context)

    def _eval_batch(self, context: TrainingContext) -> None:
        """Evaluate a single batch without gradient computation."""
        with torch.no_grad():
            # Mixed precision for validation (memory efficiency)
            if self.scaler is not None:
                self._eval_batch_mixed_precision(context)
            else:
                self._eval_batch_full_precision(context)

    def _eval_batch_mixed_precision(
        self, context: TrainingContext
    ) -> None:
        """Evaluate batch with mixed precision."""
        # 順伝播
        for callback in self.callbacks:
            callback.on_forward_pass_start(context)

        with autocast(self.device.type):
            context.outputs_policy, context.outputs_value = (
                self.model(context.inputs)
            )

            # 損失計算
            for callback in self.callbacks:
                callback.on_loss_computation_start(context)

            policy_loss = self._compute_policy_loss(context)
            context.loss = (
                self.policy_loss_ratio * policy_loss
                + self.value_loss_ratio
                * self.loss_fn_value(
                    context.outputs_value, context.labels_value
                )
            )

            for callback in self.callbacks:
                callback.on_loss_computation_end(context)

        self._maybe_synchronize(
            "post_eval_forward_mixed_precision"
        )

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 評価モードでは逆伝播とオプティマイザステップはスキップ
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)
            callback.on_backward_pass_end(context)
            callback.on_optimizer_step_start(context)
            callback.on_optimizer_step_end(context)

    def _maybe_synchronize(self, reason: str) -> None:
        if self.device.type != "cuda":
            return

        if not self._cuda_sync_enabled:
            return

        torch.cuda.synchronize()

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "torch.cuda.synchronize() executed (%s)",
                reason,
            )

    def _eval_batch_full_precision(
        self, context: TrainingContext
    ) -> None:
        """Evaluate batch with full precision."""
        # 順伝播
        for callback in self.callbacks:
            callback.on_forward_pass_start(context)

        context.outputs_policy, context.outputs_value = (
            self.model(context.inputs)
        )

        self._maybe_synchronize(
            "post_eval_forward_full_precision"
        )

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 損失計算
        for callback in self.callbacks:
            callback.on_loss_computation_start(context)

        policy_loss = self._compute_policy_loss(context)
        context.loss = (
            self.policy_loss_ratio * policy_loss
            + self.value_loss_ratio
            * self.loss_fn_value(
                context.outputs_value, context.labels_value
            )
        )

        for callback in self.callbacks:
            callback.on_loss_computation_end(context)

        # 評価モードでは逆伝播とオプティマイザステップはスキップ
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)
            callback.on_backward_pass_end(context)
            callback.on_optimizer_step_start(context)
            callback.on_optimizer_step_end(context)

    def _compute_policy_loss(
        self, context: TrainingContext
    ) -> torch.Tensor:
        if context.outputs_policy is None:
            raise RuntimeError(
                "Policy outputs are required before computing the loss"
            )

        if context.legal_move_mask is not None:
            masked_logits = context.outputs_policy.masked_fill(
                ~context.legal_move_mask.bool(), float("-inf")
            )
        else:
            masked_logits = context.outputs_policy

        policy_log_probs = torch.nn.functional.log_softmax(
            masked_logits,
            dim=1,
        )
        policy_targets = normalize_policy_targets(
            context.labels_policy,
            context.legal_move_mask,
            dtype=policy_log_probs.dtype,
            device=policy_log_probs.device,
        )
        context.policy_target_distribution = (
            policy_targets.detach()
        )
        return self.loss_fn_policy(
            policy_log_probs, policy_targets
        )
