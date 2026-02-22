"""Multi-stage training orchestration for progressive learning.

This module implements a 3-stage training system:
1. Stage 1: Reachable Squares Learning - Learn basic piece movement
2. Stage 2: Legal Moves Learning - Learn move legality constraints
3. Stage 3: Policy + Value Learning - Learn optimal play (existing training)

The orchestrator manages automatic progression between stages based on
accuracy thresholds,with fail-fast error handling if thresholds aren't met.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, cast

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from maou.app.learning.network import (
    HeadlessNetwork,
    LegalMovesHead,
    ReachableSquaresHead,
)


class TrainingStage(IntEnum):
    """Training stage enumeration for multi-stage training."""

    REACHABLE_SQUARES = 1  # Stage 1: Learn reachable squares
    LEGAL_MOVES = 2  # Stage 2: Learn legal moves
    POLICY_VALUE = 3  # Stage 3: Learn policy + value (existing)


@dataclass(frozen=True)
class StageConfig:
    """Configuration for a single training stage.

    This dataclass encapsulates all parameters needed to train one stage,
    including data,loss function,and stopping criteria.
    The optimizer is created internally by the training loop using this config.
    """

    stage: TrainingStage
    max_epochs: int
    accuracy_threshold: float  # e.g.,0.99 for 99% accuracy
    dataloader: DataLoader
    loss_fn: torch.nn.Module
    learning_rate: float
    lr_scheduler_name: Optional[str] = None
    base_batch_size: int = 256
    actual_batch_size: int = 256
    compilation: bool = False
    head_hidden_dim: int | None = None
    head_dropout: float = 0.0
    val_dataloader: Optional[DataLoader] = None


@dataclass(frozen=True)
class StageResult:
    """Result of training a single stage.

    Contains performance metrics and completion status for one training stage.
    """

    stage: TrainingStage
    achieved_accuracy: float
    final_loss: float
    epochs_trained: int
    threshold_met: bool


class Stage2ModelAdapter(torch.nn.Module):
    """Stage 2 用のモデルアダプタ．

    HeadlessNetwork と LegalMovesHead をラップし，
    TrainingLoop が期待する ``(policy, value)`` の2タプルを返す．
    ``value`` 出力はダミーゼロテンソルで，value loss は ``value_loss_ratio=0.0`` で無視される．

    Args:
        backbone: 共有バックボーンネットワーク
        head: Stage 2 用の LegalMovesHead
    """

    def __init__(
        self,
        backbone: HeadlessNetwork,
        head: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """フォワードパスを実行し，(policy, dummy_value) を返す．

        Args:
            inputs: (board, hand) のタプル

        Returns:
            (logits, dummy_value) のタプル
        """
        features = self.backbone.forward_features(inputs)
        logits = self.head(features)
        dummy_value = torch.zeros(
            logits.shape[0], 1, device=logits.device
        )
        return logits, dummy_value


class SingleStageTrainingLoop:
    """Training loop for a single stage with threshold checking.

    This class handles the training loop for one stage of multi-stage training,
    including:
    - Forward/backward passes
    - Loss computation
    - Accuracy calculation
    - Threshold-based early stopping
    - Mixed precision training support
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        model: HeadlessNetwork,
        head: torch.nn.Module,
        device: torch.device,
        config: StageConfig,
    ):
        """Initialize single stage training loop.

        Args:
            model: Backbone model (shared across stages)
            head: Stage-specific output head
            device: Training device (CPU or CUDA)
            config: Stage configuration
        """
        self.model = model
        self.head = head
        self.device = device
        self.config = config

        # Move model and head to device
        self.model.to(device)
        self.head.to(device)

        # Sqrt scaling for larger batch sizes
        effective_lr = self.config.learning_rate
        if (
            self.config.actual_batch_size
            > self.config.base_batch_size
        ):
            scale = math.sqrt(
                self.config.actual_batch_size
                / self.config.base_batch_size
            )
            effective_lr = self.config.learning_rate * scale
            self.logger.info(
                "LR sqrt scaling: base_lr=%.6f, scale=%.2f, "
                "effective_lr=%.6f",
                self.config.learning_rate,
                scale,
                effective_lr,
            )

        # Create optimizer with all trainable parameters (backbone + head)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters())
            + list(self.head.parameters()),
            lr=effective_lr,
        )

        # LR scheduler (per-step stepping)
        self.lr_scheduler: Optional[LRScheduler] = None
        if self.config.lr_scheduler_name is not None:
            from maou.app.learning.setup import SchedulerFactory

            steps_per_epoch = len(self.config.dataloader)
            self.lr_scheduler = SchedulerFactory.create_scheduler(
                self.optimizer,
                lr_scheduler_name=self.config.lr_scheduler_name,
                max_epochs=self.config.max_epochs,
                steps_per_epoch=steps_per_epoch,
            )

        # torch.compile (Stage 3 parity)
        if self.config.compilation:
            from maou.app.learning.compilation import (
                compile_module,
                warmup_compiled_model,
            )

            self.logger.info(
                "Compiling backbone model with torch.compile"
            )
            self.model = cast(
                HeadlessNetwork, compile_module(self.model)
            )
            self.logger.info(
                "Compiling head model with torch.compile"
            )
            self.head = compile_module(self.head)
            # Warmup: backbone + head を一括コンパイル
            dummy_board = torch.zeros(
                self.config.actual_batch_size,
                9,
                9,
                dtype=torch.int64,
                device=self.device,
            )
            with torch.no_grad():
                dummy_features = self.model.forward_features(
                    dummy_board
                )
            warmup_compiled_model(self.head, dummy_features)

        # Mixed precision scaler for GPU training
        self.scaler: torch.amp.GradScaler | None
        if self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    def run(self) -> StageResult:
        """Run training for this stage until threshold or max epochs.

        Returns:
            StageResult with performance metrics and completion status

        Note: This method will train until either:
            1. Accuracy threshold is met (success)
            2. Max epochs reached (may or may not meet threshold)
        """
        self.logger.info(
            f"Starting Stage {self.config.stage}: "
            f"max_epochs={self.config.max_epochs},"
            f"threshold={self.config.accuracy_threshold:.1%}"
        )

        has_val = self.config.val_dataloader is not None
        if has_val:
            self.logger.info(
                "Validation dataloader provided; "
                "metrics will be reported on validation set"
            )

        best_accuracy = 0.0
        final_loss = 0.0

        for epoch in range(self.config.max_epochs):
            # Streaming IterableDatasetのエポックシード更新
            ds = self.config.dataloader.dataset
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

            self.model.train()
            self.head.train()

            epoch_loss, epoch_accuracy = self._train_epoch(
                epoch
            )

            # Use validation metrics if val_dataloader is provided
            if has_val:
                val_loss, val_accuracy = self._validate_epoch()
                report_loss = val_loss
                report_accuracy = val_accuracy
            else:
                report_loss = epoch_loss
                report_accuracy = epoch_accuracy

            metric_label = (
                "F1"
                if self.config.stage
                == TrainingStage.LEGAL_MOVES
                else "Accuracy"
            )
            if has_val:
                self.logger.info(
                    f"Stage {self.config.stage} Epoch {epoch + 1}/{self.config.max_epochs}: "
                    f"Train Loss={epoch_loss:.4f}, "
                    f"Val Loss={report_loss:.4f}, "
                    f"Val {metric_label}={report_accuracy:.2%}"
                )
            else:
                self.logger.info(
                    f"Stage {self.config.stage} Epoch {epoch + 1}/{self.config.max_epochs}: "
                    f"Loss={report_loss:.4f}, {metric_label}={report_accuracy:.2%}"
                )

            # Log current LR (scheduler is stepped per-batch in _train_epoch)
            if self.lr_scheduler is not None:
                current_lr = self.optimizer.param_groups[0][
                    "lr"
                ]
                self.logger.info(
                    "Stage %s Epoch %d: LR = %.6f",
                    self.config.stage.name,
                    epoch + 1,
                    current_lr,
                )

            best_accuracy = max(best_accuracy, report_accuracy)
            final_loss = report_loss

            # Check if threshold met (early stopping)
            if (
                report_accuracy
                >= self.config.accuracy_threshold
            ):
                self.logger.info(
                    f"Stage {self.config.stage} {metric_label} threshold achieved! "
                    f"({report_accuracy:.2%} >= {self.config.accuracy_threshold:.2%})"
                )
                return StageResult(
                    stage=self.config.stage,
                    achieved_accuracy=report_accuracy,
                    final_loss=final_loss,
                    epochs_trained=epoch + 1,
                    threshold_met=True,
                )

        # Max epochs reached - check if threshold met
        threshold_met = (
            best_accuracy >= self.config.accuracy_threshold
        )

        return StageResult(
            stage=self.config.stage,
            achieved_accuracy=best_accuracy,
            final_loss=final_loss,
            epochs_trained=self.config.max_epochs,
            threshold_met=threshold_met,
        )

    def _train_epoch(
        self, epoch_idx: int
    ) -> tuple[float, float]:
        """Train one epoch and return (loss, metric_value).

        Args:
            epoch_idx: Current epoch index (0-based)

        Returns:
            Tuple of (average_loss, metric_value) where metric_value is
            F1 score for Stage2 (LEGAL_MOVES) or accuracy for other stages.
        """
        total_loss_tensor = torch.tensor(
            0.0, device=self.device, dtype=torch.float64
        )
        total_correct_tensor = torch.tensor(
            0.0, device=self.device, dtype=torch.float64
        )
        total_samples = 0

        # マイルストーンログの間隔を計算
        # IterableDataset対応: len()がTypeErrorになる場合がある
        try:
            total_batches: int | None = len(
                self.config.dataloader
            )
        except TypeError:
            total_batches = None

        # 時間ベースフォールバック: 最低30秒間隔でログ出力
        last_log_time = time.monotonic()
        _LOG_INTERVAL_SEC = 30.0

        # バッチ数ベースのマイルストーン（既知の場合）
        if total_batches is not None and total_batches > 0:
            milestone_interval: int | None = max(
                1, total_batches // 10
            )
        else:
            milestone_interval = None

        batch_idx = -1
        for batch_idx, (inputs, targets) in enumerate(
            self.config.dataloader
        ):
            # Move to device
            board_tensor, hand_tensor = inputs
            board_tensor = board_tensor.to(
                self.device, non_blocking=True
            )
            hand_tensor = (
                hand_tensor.to(self.device, non_blocking=True)
                if hand_tensor is not None
                else None
            )
            targets = targets.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            use_amp = self.device.type == "cuda"
            with torch.amp.autocast(
                device_type=self.device.type, enabled=use_amp
            ):
                # Get features from backbone
                features = self.model.forward_features(
                    (board_tensor, hand_tensor)
                )

                # Get predictions from head
                logits = self.head(features)

                # Compute loss
                loss = self.config.loss_fn(logits, targets)

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Per-step LR scheduling
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Compute metric (threshold at 0.5 for binary classification)
            with torch.no_grad():
                predictions = torch.sigmoid(logits) > 0.5
                if (
                    self.config.stage
                    == TrainingStage.LEGAL_MOVES
                ):
                    # Stage2: sample-averaged F1 score
                    pred_bool = predictions
                    tgt_bool = targets.bool()
                    tp = (
                        (pred_bool & tgt_bool)
                        .float()
                        .sum(dim=1)
                    )
                    fp = (
                        (pred_bool & ~tgt_bool)
                        .float()
                        .sum(dim=1)
                    )
                    fn = (
                        (~pred_bool & tgt_bool)
                        .float()
                        .sum(dim=1)
                    )
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = (
                        2
                        * precision
                        * recall
                        / (precision + recall + 1e-8)
                    )
                    both_empty = (pred_bool.sum(dim=1) == 0) & (
                        tgt_bool.sum(dim=1) == 0
                    )
                    f1 = torch.where(
                        both_empty,
                        torch.ones_like(f1),
                        f1,
                    )
                    total_correct_tensor += f1.sum()
                    total_samples += f1.numel()
                else:
                    # Stage1/3: element-wise accuracy
                    correct = (
                        (predictions == targets.bool())
                        .float()
                        .sum()
                    )
                    total_correct_tensor += correct
                    total_samples += targets.numel()

            total_loss_tensor += loss.detach()

            # マイルストーンログ出力
            now = time.monotonic()
            should_log = False

            if milestone_interval is not None:
                if (batch_idx + 1) % milestone_interval == 0:
                    should_log = True

            if (
                not should_log
                and (now - last_log_time) >= _LOG_INTERVAL_SEC
            ):
                should_log = True

            if should_log:
                running_loss = total_loss_tensor.item() / (
                    batch_idx + 1
                )
                running_acc = (
                    total_correct_tensor.item() / total_samples
                    if total_samples > 0
                    else 0.0
                )
                progress = (
                    f"{batch_idx + 1}/{total_batches}"
                    if total_batches is not None
                    else f"{batch_idx + 1}/?"
                )
                metric_label = (
                    "F1"
                    if self.config.stage
                    == TrainingStage.LEGAL_MOVES
                    else "Accuracy"
                )
                self.logger.info(
                    "Stage %s Epoch %d Batch %s: "
                    "Loss=%.4f, %s=%.2f%%",
                    self.config.stage,
                    epoch_idx + 1,
                    progress,
                    running_loss,
                    metric_label,
                    running_acc * 100,
                )
                last_log_time = now

        # avg_loss計算: IterableDataset対応
        # batch_idxが-1の場合(空のDataLoader)はゼロ除算を防止
        num_batches = batch_idx + 1 if batch_idx >= 0 else 1
        avg_loss = total_loss_tensor.item() / num_batches
        accuracy = (
            total_correct_tensor.item() / total_samples
            if total_samples > 0
            else 0.0
        )

        return avg_loss, accuracy

    def _validate_epoch(self) -> tuple[float, float]:
        """Run validation and return (loss, metric_value).

        Returns:
            Tuple of (average_loss, metric_value) where metric_value is
            F1 score for Stage2 (LEGAL_MOVES) or accuracy for other stages.
        """
        if self.config.val_dataloader is None:
            raise RuntimeError(
                "val_dataloader is required for validation"
            )

        self.model.eval()
        self.head.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0
        num_batches = 0

        use_amp = self.device.type == "cuda"

        with torch.no_grad():
            for inputs, targets in self.config.val_dataloader:
                board_tensor, hand_tensor = inputs
                board_tensor = board_tensor.to(
                    self.device, non_blocking=True
                )
                hand_tensor = (
                    hand_tensor.to(
                        self.device, non_blocking=True
                    )
                    if hand_tensor is not None
                    else None
                )
                targets = targets.to(
                    self.device, non_blocking=True
                )

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=use_amp,
                ):
                    features = self.model.forward_features(
                        (board_tensor, hand_tensor)
                    )
                    logits = self.head(features)
                    loss = self.config.loss_fn(logits, targets)

                total_loss += loss.item()
                num_batches += 1

                predictions = torch.sigmoid(logits) > 0.5
                if (
                    self.config.stage
                    == TrainingStage.LEGAL_MOVES
                ):
                    pred_bool = predictions
                    tgt_bool = targets.bool()
                    tp = (
                        (pred_bool & tgt_bool)
                        .float()
                        .sum(dim=1)
                    )
                    fp = (
                        (pred_bool & ~tgt_bool)
                        .float()
                        .sum(dim=1)
                    )
                    fn = (
                        (~pred_bool & tgt_bool)
                        .float()
                        .sum(dim=1)
                    )
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = (
                        2
                        * precision
                        * recall
                        / (precision + recall + 1e-8)
                    )
                    both_empty = (pred_bool.sum(dim=1) == 0) & (
                        tgt_bool.sum(dim=1) == 0
                    )
                    f1 = torch.where(
                        both_empty,
                        torch.ones_like(f1),
                        f1,
                    )
                    total_correct += f1.sum().item()
                    total_samples += f1.numel()
                else:
                    correct = (
                        (predictions == targets.bool())
                        .float()
                        .sum()
                        .item()
                    )
                    total_correct += correct
                    total_samples += targets.numel()

        avg_loss = (
            total_loss / num_batches if num_batches > 0 else 0.0
        )
        metric = (
            total_correct / total_samples
            if total_samples > 0
            else 0.0
        )

        return avg_loss, metric


def run_stage2_with_training_loop(
    *,
    backbone: HeadlessNetwork,
    config: StageConfig,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> tuple[StageResult, LegalMovesHead]:
    """TrainingLoop を使用して Stage 2 (Legal Moves) を学習する．

    CUDA stream overlap，tqdm 進捗表示，コールバックアーキテクチャを活用し，
    SingleStageTrainingLoop より高スループットな学習を実現する．

    Args:
        backbone: 共有バックボーンネットワーク
        config: Stage 2 の学習設定
        device: 学習デバイス (CPU or CUDA)
        logger: ロガー

    Returns:
        (StageResult, LegalMovesHead) のタプル．
        ヘッドはチェックポイント保存に使用される．
    """
    from maou.app.learning.callbacks import (
        LRSchedulerStepCallback,
        Stage2F1Callback,
    )
    from maou.app.learning.setup import SchedulerFactory
    from maou.app.learning.training_loop import (
        Stage2TrainingLoop,
    )

    _logger = logger or logging.getLogger(__name__)

    # Head 作成
    legal_moves_head = LegalMovesHead(
        input_dim=backbone.embedding_dim,
        hidden_dim=config.head_hidden_dim,
        dropout=config.head_dropout,
    )

    # Model adapter (TrainingLoop 互換)
    model = Stage2ModelAdapter(backbone, legal_moves_head)
    model.to(device)

    # torch.compile (オプション)
    if config.compilation:
        from maou.app.learning.compilation import (
            compile_module,
            warmup_compiled_model,
        )

        _logger.info(
            "Compiling Stage 2 model with torch.compile"
        )
        model = cast(Stage2ModelAdapter, compile_module(model))
        dummy_board = torch.zeros(
            config.actual_batch_size,
            9,
            9,
            dtype=torch.int64,
            device=device,
        )
        warmup_compiled_model(model, (dummy_board, None))

    # Sqrt scaling for larger batch sizes
    effective_lr = config.learning_rate
    if config.actual_batch_size > config.base_batch_size:
        scale = math.sqrt(
            config.actual_batch_size / config.base_batch_size
        )
        effective_lr = config.learning_rate * scale
        _logger.info(
            "LR sqrt scaling: base_lr=%.6f, scale=%.2f, "
            "effective_lr=%.6f",
            config.learning_rate,
            scale,
            effective_lr,
        )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=effective_lr
    )

    # Callbacks
    f1_callback = Stage2F1Callback()
    callbacks: list = [f1_callback]

    # LR Scheduler (optional)
    if config.lr_scheduler_name is not None:
        steps_per_epoch = len(config.dataloader)
        scheduler = SchedulerFactory.create_scheduler(
            optimizer,
            lr_scheduler_name=config.lr_scheduler_name,
            max_epochs=config.max_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        if scheduler is not None:
            callbacks.append(LRSchedulerStepCallback(scheduler))

    # TrainingLoop 作成
    training_loop = Stage2TrainingLoop(
        model=model,
        device=device,
        optimizer=optimizer,
        loss_fn_policy=config.loss_fn,
        loss_fn_value=torch.nn.MSELoss(),
        policy_loss_ratio=1.0,
        value_loss_ratio=0.0,
        callbacks=callbacks,
        logger=_logger,
    )

    # Epoch loop
    _logger.info(
        "Starting Stage 2 (TrainingLoop): max_epochs=%d, "
        "threshold=%.1f%%",
        config.max_epochs,
        config.accuracy_threshold * 100,
    )

    best_f1 = 0.0
    final_loss = 0.0

    for epoch in range(config.max_epochs):
        # IterableDataset のエポックシード更新
        ds = config.dataloader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

        f1_callback.reset()

        training_loop.run_epoch(
            dataloader=config.dataloader,
            epoch_idx=epoch,
            progress_bar=True,
            train_mode=True,
        )

        epoch_f1 = f1_callback.get_epoch_f1()
        epoch_loss = f1_callback.get_average_loss()
        final_loss = epoch_loss

        _logger.info(
            "Stage 2 Epoch %d/%d: Loss=%.4f, F1=%.2f%%",
            epoch + 1,
            config.max_epochs,
            epoch_loss,
            epoch_f1 * 100,
        )

        if optimizer.param_groups:
            current_lr = optimizer.param_groups[0]["lr"]
            _logger.info(
                "Stage 2 Epoch %d: LR = %.6f",
                epoch + 1,
                current_lr,
            )

        best_f1 = max(best_f1, epoch_f1)

        # Threshold check (early stopping)
        if epoch_f1 >= config.accuracy_threshold:
            _logger.info(
                "Stage 2 F1 threshold achieved! "
                "(%.2f%% >= %.2f%%)",
                epoch_f1 * 100,
                config.accuracy_threshold * 100,
            )
            return (
                StageResult(
                    stage=config.stage,
                    achieved_accuracy=epoch_f1,
                    final_loss=final_loss,
                    epochs_trained=epoch + 1,
                    threshold_met=True,
                ),
                legal_moves_head,
            )

    # Max epochs reached
    threshold_met = best_f1 >= config.accuracy_threshold

    return (
        StageResult(
            stage=config.stage,
            achieved_accuracy=best_f1,
            final_loss=final_loss,
            epochs_trained=config.max_epochs,
            threshold_met=threshold_met,
        ),
        legal_moves_head,
    )


class MultiStageTrainingOrchestrator:
    """Orchestrator for multi-stage training with automatic progression.

    This class manages the sequential execution of training stages,
    automatically progressing from Stage 1 → 2 → 3 when thresholds are met.

    Key features:
    - Automatic stage progression based on accuracy thresholds
    - Fail-fast error handling if thresholds aren't met
    - Checkpoint saving per stage
    - Backbone parameter transfer between stages
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        backbone: HeadlessNetwork,
        device: torch.device,
        model_dir: Path,
        trainable_layers: Optional[int] = None,
    ):
        """Initialize multi-stage training orchestrator.

        Args:
            backbone: Shared backbone model
            device: Training device
            model_dir: Directory for saving checkpoints
            trainable_layers: Number of trailing backbone layer groups
                to keep trainable in Stage 3. None = no freezing.
        """
        self.backbone = backbone
        self.device = device
        self.model_dir = model_dir
        self.trainable_layers = trainable_layers
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def run_all_stages(
        self,
        *,
        stage1_config: Optional[StageConfig] = None,
        stage2_config: Optional[StageConfig] = None,
        stage3_config: Optional[StageConfig] = None,
        save_checkpoints: bool = True,
    ) -> dict[TrainingStage, StageResult]:
        """Run all configured stages sequentially.

        Args:
            stage1_config: Configuration for Stage 1 (reachable squares)
            stage2_config: Configuration for Stage 2 (legal moves)
            stage3_config: Configuration for Stage 3 (policy + value)
            save_checkpoints: Whether to save checkpoints after each stage

        Returns:
            Dictionary mapping TrainingStage to StageResult

        Raises:
            RuntimeError: If Stage 1 or 2 fails to meet threshold
        """
        results: dict[TrainingStage, StageResult] = {}

        # Stage 1: Reachable Squares
        if stage1_config is not None:
            self.logger.info("=" * 60)
            self.logger.info(
                "STAGE 1: REACHABLE SQUARES LEARNING"
            )
            self.logger.info("=" * 60)

            reachable_head = ReachableSquaresHead(
                input_dim=self.backbone.embedding_dim
            )
            stage1_loop = SingleStageTrainingLoop(
                model=self.backbone,
                head=reachable_head,
                device=self.device,
                config=stage1_config,
            )
            result = stage1_loop.run()
            results[TrainingStage.REACHABLE_SQUARES] = result

            if not result.threshold_met:
                raise RuntimeError(
                    f"Stage 1 failed to meet accuracy threshold "
                    f"after {result.epochs_trained} epochs.\n"
                    f"  achieved: {result.achieved_accuracy:.2%}"
                    f" / required: "
                    f"{stage1_config.accuracy_threshold:.2%}"
                    f" (reached "
                    f"{result.achieved_accuracy / stage1_config.accuracy_threshold:.1%}"
                    f" of target)\n"
                    f"  Hint: try increasing --stage1-max-epochs,"
                    f" adjusting --stage1-learning-rate,"
                    f" or lowering --stage1-threshold"
                    f" or setting --stage1-batch-size 32"
                )

            if save_checkpoints:
                self._save_stage_checkpoint(
                    stage=TrainingStage.REACHABLE_SQUARES,
                    head=reachable_head,
                )

            self.logger.info(
                f"Stage 1 completed successfully: "
                f"accuracy={result.achieved_accuracy:.2%},"
                f"epochs={result.epochs_trained}"
            )

        # Stage 2: Legal Moves
        if stage2_config is not None:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 2: LEGAL MOVES LEARNING")
            self.logger.info("=" * 60)

            result, legal_moves_head = (
                run_stage2_with_training_loop(
                    backbone=self.backbone,
                    config=stage2_config,
                    device=self.device,
                    logger=self.logger,
                )
            )
            results[TrainingStage.LEGAL_MOVES] = result

            if not result.threshold_met:
                raise RuntimeError(
                    f"Stage 2 failed to meet accuracy threshold "
                    f"after {result.epochs_trained} epochs.\n"
                    f"  achieved: {result.achieved_accuracy:.2%}"
                    f" / required: "
                    f"{stage2_config.accuracy_threshold:.2%}"
                    f" (reached "
                    f"{result.achieved_accuracy / stage2_config.accuracy_threshold:.1%}"
                    f" of target)\n"
                    f"  Hint: try increasing --stage2-max-epochs,"
                    f" adjusting --stage2-learning-rate,"
                    f" or lowering --stage2-threshold"
                    f" or setting --stage2-batch-size 32"
                )

            if save_checkpoints:
                self._save_stage_checkpoint(
                    stage=TrainingStage.LEGAL_MOVES,
                    head=legal_moves_head,
                )

            self.logger.info(
                f"Stage 2 completed successfully: "
                f"accuracy={result.achieved_accuracy:.2%},"
                f"epochs={result.epochs_trained}"
            )

        # Stage 3: Policy + Value
        if stage3_config is not None:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 3: POLICY + VALUE LEARNING")
            self.logger.info("=" * 60)
            self.logger.info(
                "Stage 3 would use existing Learning.learn() implementation"
            )
            # Note: Actual Stage 3 implementation would delegate to
            # existing Learning class in dl.py

        return results

    def _save_stage_checkpoint(
        self,
        stage: TrainingStage,
        head: torch.nn.Module,
    ) -> None:
        """Save checkpoint for a stage.

        Args:
            stage: Training stage
            head: Trained head module
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if stage == TrainingStage.REACHABLE_SQUARES:
            head_filename = (
                f"stage1_reachable_head_{timestamp}.pt"
            )
            backbone_filename = (
                f"stage1_backbone_{timestamp}.pt"
            )
        elif stage == TrainingStage.LEGAL_MOVES:
            head_filename = (
                f"stage2_legal_moves_head_{timestamp}.pt"
            )
            backbone_filename = (
                f"stage2_backbone_{timestamp}.pt"
            )
        else:
            return

        # Save head
        head_path = self.model_dir / head_filename
        torch.save(head.state_dict(), head_path)
        self.logger.info(
            f"Saved {stage.name} head: {head_path}"
        )

        # Save backbone
        backbone_path = self.model_dir / backbone_filename
        torch.save(self.backbone.state_dict(), backbone_path)
        self.logger.info(
            f"Saved {stage.name} backbone: {backbone_path}"
        )
