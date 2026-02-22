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
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from maou.app.learning.network import (
    HeadlessNetwork,
    LegalMovesHead,
    ReachableSquaresHead,
)

if TYPE_CHECKING:
    from maou.app.learning.dataset import (
        Stage1Dataset,
        Stage2Dataset,
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


class Stage1ModelAdapter(torch.nn.Module):
    """Stage 1 用のモデルアダプタ．

    HeadlessNetwork と ReachableSquaresHead をラップし，
    TrainingLoop が期待する ``(policy, value)`` の2タプルを返す．
    ``value`` 出力はダミーゼロテンソルで，value loss は ``value_loss_ratio=0.0`` で無視される．

    Args:
        backbone: 共有バックボーンネットワーク
        head: Stage 1 用の ReachableSquaresHead
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


class Stage1DatasetAdapter(Dataset):
    """Stage1Dataset を TrainingLoop の入力形式に変換するアダプタ．

    Stage1Dataset は ((board, hand), reachable_squares) を返すが，
    TrainingLoop._unpack_batch() は
    ((board, hand), (labels_policy, labels_value, legal_move_mask))
    を期待する．

    PyTorch の default_collate は None を正しく処理する（全サンプルが
    None の場合，バッチレベルでも None を返す）ため，legal_move_mask=None
    はそのまま伝播する．

    Args:
        dataset: ラップする Stage1Dataset
    """

    def __init__(self, dataset: "Stage1Dataset") -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, None],
    ]:
        inputs, targets = self._dataset[idx]
        dummy_value = torch.zeros(1, dtype=torch.float32)
        return inputs, (targets, dummy_value, None)


class Stage2DatasetAdapter(Dataset):
    """Stage2Dataset を TrainingLoop の入力形式に変換するアダプタ．

    Stage2Dataset は ((board, hand), legal_moves) を返すが，
    TrainingLoop._unpack_batch() は
    ((board, hand), (labels_policy, labels_value, legal_move_mask))
    を期待する．

    Args:
        dataset: ラップする Stage2Dataset
    """

    def __init__(self, dataset: "Stage2Dataset") -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, None],
    ]:
        inputs, targets = self._dataset[idx]
        dummy_value = torch.zeros(1, dtype=torch.float32)
        return inputs, (targets, dummy_value, None)


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


class TruncatedStageModel(torch.nn.Module):
    """層分離時の Stage 1/2 用モデル．

    バックボーンの最初の ``(total_groups - trainable_layers)`` 個のグループのみを使い，
    後処理を経由してヘッドに接続する．Stage 3 で訓練する末尾グループは
    forward pass に含めない．

    ResNet の場合は Pool + Linear 投射を使用し，
    MLP-Mixer/ViT の場合は LayerNorm + mean pooling を使用する．

    Args:
        backbone: 共有バックボーンネットワーク
        head: Stage 1 or Stage 2 用のヘッドモジュール
        trainable_layers: Stage 3 で訓練する末尾グループ数
    """

    def __init__(
        self,
        backbone: HeadlessNetwork,
        head: torch.nn.Module,
        trainable_layers: int,
    ) -> None:
        super().__init__()

        groups = backbone.backbone.get_freezable_groups()
        total = len(groups)

        if trainable_layers >= total:
            msg = (
                f"trainable_layers ({trainable_layers}) must be less "
                f"than the total number of backbone groups ({total}). "
                f"No groups would remain for Stage 1/2 training."
            )
            raise ValueError(msg)

        n_use = total - trainable_layers

        # HeadlessNetwork の embedding/hand_projection を参照
        self.backbone = backbone
        self._is_resnet = backbone.architecture == "resnet"

        # 使用するグループの Sequential を構成 (元オブジェクトへの参照)
        self.partial_backbone = nn.Sequential(*groups[:n_use])

        if self._is_resnet:
            # ResNet: Pool + Linear 投射 (次元変化に対応)
            backbone_in_ch = (
                backbone._embedding_channels
                + backbone._hand_projection_dim
            )
            truncated_out_ch = self._compute_output_channels(
                self.partial_backbone,
                backbone_in_ch,
                backbone._board_size,
            )
            self.projection_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.projection_linear = nn.Linear(
                truncated_out_ch, backbone.embedding_dim
            )
        else:
            # MLP-Mixer/ViT: LayerNorm + mean pooling (次元不変)
            self.truncation_norm = nn.LayerNorm(
                backbone.embedding_dim
            )

        self.head = head

        # 除外グループのパラメータを凍結してメモリ浪費を防ぐ
        for group in groups[n_use:]:
            for param in group.parameters():
                param.requires_grad = False

    @staticmethod
    def _compute_output_channels(
        partial: nn.Module,
        input_channels: int,
        board_size: tuple[int, int],
    ) -> int:
        """ダミー入力を通して partial backbone の出力チャンネル数を推定する．"""
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *board_size)
            out = partial(dummy)
        return int(out.shape[1])

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """フォワードパスを実行し，(policy, dummy_value) を返す．

        HeadlessNetwork の embedding 処理を再利用し，
        preprocess → partial backbone → 後処理 → head の順に処理する．

        Args:
            inputs: (board, hand) のタプル

        Returns:
            (logits, dummy_value) のタプル
        """
        board_tensor, hand_tensor = (
            self.backbone._separate_inputs(inputs)
        )
        embedded = self.backbone._prepare_inputs(board_tensor)

        if self.backbone._hand_projection is not None:
            combined = self.backbone._combine_board_and_hand(
                embedded, hand_tensor
            )
        else:
            combined = embedded

        # 全アーキテクチャ共通: 前処理 + truncated blocks
        preprocessed = (
            self.backbone.backbone.preprocess_for_blocks(
                combined
            )
        )
        features = self.partial_backbone(preprocessed)

        # アーキテクチャ別の後処理
        if self._is_resnet:
            pooled = self.projection_pool(features)
            projected = torch.flatten(pooled, 1)
            projected = self.projection_linear(projected)
        else:
            # MLP-Mixer/ViT: norm + mean pool
            features = self.truncation_norm(features)
            projected = features.mean(dim=1)

        logits = self.head(projected)
        dummy_value = torch.zeros(
            logits.shape[0], 1, device=logits.device
        )
        return logits, dummy_value


def _compute_effective_lr(
    learning_rate: float,
    actual_batch_size: int,
    base_batch_size: int,
    logger: logging.Logger | None = None,
) -> float:
    """バッチサイズに基づく LR sqrt スケーリングを計算する．

    actual_batch_size > base_batch_size の場合，
    effective_lr = learning_rate * sqrt(actual / base) を返す．

    Args:
        learning_rate: ベース学習率
        actual_batch_size: 実際のバッチサイズ
        base_batch_size: 基準バッチサイズ(デフォルト 256)
        logger: ロガー(スケーリング情報の出力用)

    Returns:
        スケーリング後の学習率
    """
    if actual_batch_size > base_batch_size:
        scale = math.sqrt(actual_batch_size / base_batch_size)
        effective_lr = learning_rate * scale
        if logger is not None:
            logger.info(
                "LR sqrt scaling: base_lr=%.6f, scale=%.2f, "
                "effective_lr=%.6f",
                learning_rate,
                scale,
                effective_lr,
            )
        return effective_lr
    return learning_rate


def run_stage1_with_training_loop(
    *,
    backbone: HeadlessNetwork,
    config: StageConfig,
    device: torch.device,
    trainable_layers: int | None = None,
    logger: logging.Logger | None = None,
) -> tuple[StageResult, ReachableSquaresHead]:
    """TrainingLoop を使用して Stage 1 (Reachable Squares) を学習する．

    CUDA stream overlap，tqdm 進捗表示，コールバックアーキテクチャを活用し，
    高スループットな学習を実現する．

    Args:
        backbone: 共有バックボーンネットワーク
        config: Stage 1 の学習設定
        device: 学習デバイス (CPU or CUDA)
        logger: ロガー

    Returns:
        (StageResult, ReachableSquaresHead) のタプル．
        ヘッドはチェックポイント保存に使用される．
    """
    from maou.app.learning.callbacks import (
        LRSchedulerStepCallback,
        Stage1AccuracyCallback,
    )
    from maou.app.learning.setup import SchedulerFactory
    from maou.app.learning.training_loop import (
        Stage1TrainingLoop,
    )

    _logger = logger or logging.getLogger(__name__)

    # Head 作成
    reachable_head = ReachableSquaresHead(
        input_dim=backbone.embedding_dim,
        hidden_dim=config.head_hidden_dim,
    )

    # Model adapter (TrainingLoop 互換)
    if trainable_layers is not None and trainable_layers > 0:
        model: torch.nn.Module = TruncatedStageModel(
            backbone, reachable_head, trainable_layers
        )
    else:
        model = Stage1ModelAdapter(backbone, reachable_head)
    model.to(device)

    # torch.compile (オプション)
    if config.compilation:
        from maou.app.learning.compilation import (
            compile_module,
            warmup_compiled_model,
        )

        _logger.info(
            "Compiling Stage 1 model with torch.compile"
        )
        model = cast(Stage1ModelAdapter, compile_module(model))
        dummy_board = torch.zeros(
            config.actual_batch_size,
            9,
            9,
            dtype=torch.int64,
            device=device,
        )
        warmup_compiled_model(model, (dummy_board, None))

    # Sqrt scaling for larger batch sizes
    effective_lr = _compute_effective_lr(
        config.learning_rate,
        config.actual_batch_size,
        config.base_batch_size,
        logger=_logger,
    )

    # Optimizer (requires_grad のみ含める: TruncatedStageModel 時の除外グループを除く)
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_params, lr=effective_lr
    )

    # Callbacks
    accuracy_callback = Stage1AccuracyCallback()
    callbacks: list = [accuracy_callback]

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
    training_loop = Stage1TrainingLoop(
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
        "Starting Stage 1 (TrainingLoop): max_epochs=%d, "
        "threshold=%.1f%%",
        config.max_epochs,
        config.accuracy_threshold * 100,
    )

    best_accuracy = 0.0
    final_loss = 0.0

    for epoch in range(config.max_epochs):
        # IterableDataset のエポックシード更新
        ds = config.dataloader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

        accuracy_callback.reset()

        training_loop.run_epoch(
            dataloader=config.dataloader,
            epoch_idx=epoch,
            progress_bar=True,
            train_mode=True,
        )

        epoch_accuracy = accuracy_callback.get_epoch_accuracy()
        epoch_loss = accuracy_callback.get_average_loss()
        final_loss = epoch_loss

        _logger.info(
            "Stage 1 Epoch %d/%d: Loss=%.4f, Accuracy=%.2f%%",
            epoch + 1,
            config.max_epochs,
            epoch_loss,
            epoch_accuracy * 100,
        )

        if optimizer.param_groups:
            current_lr = optimizer.param_groups[0]["lr"]
            _logger.info(
                "Stage 1 Epoch %d: LR = %.6f",
                epoch + 1,
                current_lr,
            )

        best_accuracy = max(best_accuracy, epoch_accuracy)

        # Threshold check (early stopping)
        if epoch_accuracy >= config.accuracy_threshold:
            _logger.info(
                "Stage 1 Accuracy threshold achieved! "
                "(%.2f%% >= %.2f%%)",
                epoch_accuracy * 100,
                config.accuracy_threshold * 100,
            )
            return (
                StageResult(
                    stage=config.stage,
                    achieved_accuracy=epoch_accuracy,
                    final_loss=final_loss,
                    epochs_trained=epoch + 1,
                    threshold_met=True,
                ),
                reachable_head,
            )

    # Max epochs reached
    threshold_met = best_accuracy >= config.accuracy_threshold

    return (
        StageResult(
            stage=config.stage,
            achieved_accuracy=best_accuracy,
            final_loss=final_loss,
            epochs_trained=config.max_epochs,
            threshold_met=threshold_met,
        ),
        reachable_head,
    )


def run_stage2_with_training_loop(
    *,
    backbone: HeadlessNetwork,
    config: StageConfig,
    device: torch.device,
    trainable_layers: int | None = None,
    logger: logging.Logger | None = None,
) -> tuple[StageResult, LegalMovesHead]:
    """TrainingLoop を使用して Stage 2 (Legal Moves) を学習する．

    CUDA stream overlap，tqdm 進捗表示，コールバックアーキテクチャを活用し，
    高スループットな学習を実現する．

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
        RawLogitsTrainingLoop,
    )

    _logger = logger or logging.getLogger(__name__)

    # Head 作成
    legal_moves_head = LegalMovesHead(
        input_dim=backbone.embedding_dim,
        hidden_dim=config.head_hidden_dim,
        dropout=config.head_dropout,
    )

    # Model adapter (TrainingLoop 互換)
    if trainable_layers is not None and trainable_layers > 0:
        model: torch.nn.Module = TruncatedStageModel(
            backbone, legal_moves_head, trainable_layers
        )
    else:
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
    effective_lr = _compute_effective_lr(
        config.learning_rate,
        config.actual_batch_size,
        config.base_batch_size,
        logger=_logger,
    )

    # Optimizer (requires_grad のみ含める: TruncatedStageModel 時の除外グループを除く)
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_params, lr=effective_lr
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
    training_loop = RawLogitsTrainingLoop(
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

        if (
            trainable_layers is not None
            and trainable_layers > 0
        ):
            groups = backbone.backbone.get_freezable_groups()
            if trainable_layers >= len(groups):
                msg = (
                    f"trainable_layers ({trainable_layers}) must be "
                    f"less than total backbone groups "
                    f"({len(groups)}). No groups would remain "
                    f"for Stage 1/2 training."
                )
                raise ValueError(msg)

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

            result, reachable_head = (
                run_stage1_with_training_loop(
                    backbone=self.backbone,
                    config=stage1_config,
                    device=self.device,
                    trainable_layers=self.trainable_layers,
                    logger=self.logger,
                )
            )
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
                    trainable_layers=self.trainable_layers,
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
