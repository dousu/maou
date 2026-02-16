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
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import torch
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
    including data,optimization,and stopping criteria.
    """

    stage: TrainingStage
    max_epochs: int
    accuracy_threshold: float  # e.g.,0.99 for 99% accuracy
    dataloader: DataLoader
    loss_fn: torch.nn.Module
    optimizer: torch.optim.Optimizer
    learning_rate: Optional[float] = None


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

            self.logger.info(
                f"Stage {self.config.stage} Epoch {epoch + 1}/{self.config.max_epochs}: "
                f"Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2%}"
            )

            best_accuracy = max(best_accuracy, epoch_accuracy)
            final_loss = epoch_loss

            # Check if threshold met (early stopping)
            if epoch_accuracy >= self.config.accuracy_threshold:
                self.logger.info(
                    f"Stage {self.config.stage} threshold achieved! "
                    f"({epoch_accuracy:.2%} >= {self.config.accuracy_threshold:.2%})"
                )
                return StageResult(
                    stage=self.config.stage,
                    achieved_accuracy=epoch_accuracy,
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
        """Train one epoch and return (loss,accuracy).

        Args:
            epoch_idx: Current epoch index (0-based)

        Returns:
            Tuple of (average_loss,accuracy)
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

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
            self.config.optimizer.zero_grad()

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
                self.scaler.step(self.config.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.config.optimizer.step()

            # Compute accuracy (threshold at 0.5 for binary classification)
            with torch.no_grad():
                predictions = torch.sigmoid(logits) > 0.5
                correct = (
                    (predictions == targets.bool())
                    .float()
                    .sum()
                )
                total_correct += correct.item()
                total_samples += targets.numel()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.config.dataloader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy


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
                    f"  Hint: try increasing --stage1-max-epochs"
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

            legal_moves_head = LegalMovesHead(
                input_dim=self.backbone.embedding_dim
            )
            stage2_loop = SingleStageTrainingLoop(
                model=self.backbone,
                head=legal_moves_head,
                device=self.device,
                config=stage2_config,
            )
            result = stage2_loop.run()
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
                    f"  Hint: try increasing --stage2-max-epochs"
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
