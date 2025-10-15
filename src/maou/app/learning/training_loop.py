import logging
from typing import List, Optional

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from maou.app.learning.callbacks import (
    TrainingCallback,
    TrainingContext,
)
from maou.domain.loss.loss_fn import GCEwithNegativePenaltyLoss


class TrainingLoop:
    """Generic training loop with callback support."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn_policy: GCEwithNegativePenaltyLoss,
        loss_fn_value: torch.nn.Module,
        policy_loss_ratio: float,
        value_loss_ratio: float,
        callbacks: Optional[List[TrainingCallback]] = None,
        logger: Optional[logging.Logger] = None,
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
    ) -> None:
        """Run a single epoch of training or validation."""
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
            dataloader_iter = (
                tqdm(
                    enumerate(dataloader),
                    desc="Training"
                    if train_mode
                    else "Validation",
                    total=len(dataloader),
                )
                if progress_bar
                else enumerate(dataloader)
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
                batch_size = inputs.size(0)

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

                # GPU転送
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
            context.inputs = context.inputs.to(
                self.device, non_blocking=True
            )
            context.labels_policy = context.labels_policy.to(
                self.device, non_blocking=True
            )
            context.labels_value = context.labels_value.to(
                self.device, non_blocking=True
            )
            context.legal_move_mask = (
                context.legal_move_mask.to(
                    self.device, non_blocking=True
                )
            )

            # GPU同期（正確な転送時間測定のため）
            torch.cuda.synchronize()

        for callback in self.callbacks:
            callback.on_data_transfer_end(context)

    def _train_batch(self, context: TrainingContext) -> None:
        """Train a single batch with gradient computation."""
        # 前のバッチの勾配をクリア
        self.optimizer.zero_grad()

        # Mixed precision training with autocast
        if self.scaler is not None:
            self._train_batch_mixed_precision(context)
        else:
            self._train_batch_full_precision(context)

    def _train_batch_mixed_precision(
        self, context: TrainingContext
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

            context.loss = (
                self.policy_loss_ratio
                * self.loss_fn_policy(
                    context.outputs_policy,
                    context.labels_policy,
                    context.legal_move_mask,
                )
                + self.value_loss_ratio
                * self.loss_fn_value(
                    context.outputs_value, context.labels_value
                )
            )

            for callback in self.callbacks:
                callback.on_loss_computation_end(context)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

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

        # 勾配クリッピング (scaled gradients)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for callback in self.callbacks:
            callback.on_backward_pass_end(context)

        # オプティマイザステップ
        for callback in self.callbacks:
            callback.on_optimizer_step_start(context)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for callback in self.callbacks:
            callback.on_optimizer_step_end(context)

    def _train_batch_full_precision(
        self, context: TrainingContext
    ) -> None:
        """Train batch with full precision."""
        # 順伝播
        for callback in self.callbacks:
            callback.on_forward_pass_start(context)

        context.outputs_policy, context.outputs_value = (
            self.model(context.inputs)
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 損失計算
        for callback in self.callbacks:
            callback.on_loss_computation_start(context)

        context.loss = (
            self.policy_loss_ratio
            * self.loss_fn_policy(
                context.outputs_policy,
                context.labels_policy,
                context.legal_move_mask,
            )
            + self.value_loss_ratio
            * self.loss_fn_value(
                context.outputs_value, context.labels_value
            )
        )

        for callback in self.callbacks:
            callback.on_loss_computation_end(context)

        # 逆伝播
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)

        if context.loss is None:
            raise RuntimeError(
                "Loss computation failed - context.loss is None"
            )

        context.loss.backward()

        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for callback in self.callbacks:
            callback.on_backward_pass_end(context)

        # オプティマイザステップ
        for callback in self.callbacks:
            callback.on_optimizer_step_start(context)

        self.optimizer.step()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

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

            context.loss = (
                self.policy_loss_ratio
                * self.loss_fn_policy(
                    context.outputs_policy,
                    context.labels_policy,
                    context.legal_move_mask,
                )
                + self.value_loss_ratio
                * self.loss_fn_value(
                    context.outputs_value, context.labels_value
                )
            )

            for callback in self.callbacks:
                callback.on_loss_computation_end(context)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 評価モードでは逆伝播とオプティマイザステップはスキップ
        for callback in self.callbacks:
            callback.on_backward_pass_start(context)
            callback.on_backward_pass_end(context)
            callback.on_optimizer_step_start(context)
            callback.on_optimizer_step_end(context)

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

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for callback in self.callbacks:
            callback.on_forward_pass_end(context)

        # 損失計算
        for callback in self.callbacks:
            callback.on_loss_computation_start(context)

        context.loss = (
            self.policy_loss_ratio
            * self.loss_fn_policy(
                context.outputs_policy,
                context.labels_policy,
                context.legal_move_mask,
            )
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
