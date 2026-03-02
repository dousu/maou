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
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from torch.utils.data import Dataset

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
    from maou.app.learning.network import Network
    from maou.app.learning.stage_component_factory import (
        StageComponents,
    )


class TrainingStage(IntEnum):
    """Training stage enumeration for multi-stage training."""

    REACHABLE_SQUARES = 1  # Stage 1: Learn reachable squares
    LEGAL_MOVES = 2  # Stage 2: Learn legal moves
    POLICY_VALUE = 3  # Stage 3: Learn policy + value (existing)


@dataclass(frozen=True)
class StageConfig:
    """学習ステージの制御パラメータ．

    学習ループの動作制御(エポック数，閾値)のみを保持する．
    モデル・データ・オプティマイザ等のコンポーネントは
    StageComponents で管理する．
    """

    stage: TrainingStage
    max_epochs: int
    accuracy_threshold: float


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


class Stage3ModelAdapter(torch.nn.Module):
    """Stage 3 用のモデルアダプタ．

    Network 本体をラップし，``torch.compile()`` のターゲットとする．
    アダプタをコンパイルすることで，Network 自体の ``state_dict()`` には
    ``_orig_mod.`` プレフィックスが付かず，保存時の除去処理が不要になる．

    Stage 1/2 と同じアダプタパターンにより，全ステージで一貫した
    コンパイル戦略を実現する．

    Args:
        network: Stage 3 学習対象の Network モデル
    """

    def __init__(self, network: "Network") -> None:
        super().__init__()
        self.network = network

    def forward(
        self,
        inputs: torch.Tensor
        | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """フォワードパスを Network に委譲する．

        Args:
            inputs: Network が受け取る入力 (board テンソルまたは (board, hand) タプル)

        Returns:
            (policy_logits, value_logit) のタプル
        """
        return self.network(inputs)


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

    PyTorch の default_collate は None を処理できないため，
    DataLoader 作成時には pre_stage_collate_fn を collate_fn に指定すること．

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

    PyTorch の default_collate は None を処理できないため，
    DataLoader 作成時には pre_stage_collate_fn を collate_fn に指定すること．

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


def pre_stage_collate_fn(
    batch: list[
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, None],
        ]
    ],
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, None],
]:
    """Stage1/Stage2 DatasetAdapter の出力をバッチに collate する．

    両アダプタは legal_move_mask=None を含むタプルを返すが，
    PyTorch の default_collate は None を処理できないため，
    None を手動で伝播させるカスタム collate 関数が必要．
    """
    inputs_list, labels_list = zip(*batch)
    boards = torch.stack([inp[0] for inp in inputs_list])
    hands = torch.stack([inp[1] for inp in inputs_list])
    targets = torch.stack([lbl[0] for lbl in labels_list])
    values = torch.stack([lbl[1] for lbl in labels_list])
    return (boards, hands), (targets, values, None)


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


def run_stage1_with_training_loop(
    *,
    components: StageComponents,
    config: StageConfig,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> tuple[StageResult, ReachableSquaresHead]:
    """TrainingLoop を使用して Stage 1 (Reachable Squares) を学習する．

    CUDA stream overlap，tqdm 進捗表示，コールバックアーキテクチャを活用し，
    高スループットな学習を実現する．

    Args:
        components: Stage 1 のコンポーネント一式．
        config: Stage 1 の学習制御パラメータ．
        device: 学習デバイス (CPU or CUDA)．
        logger: ロガー．

    Returns:
        (StageResult, ReachableSquaresHead) のタプル．
        ヘッドはチェックポイント保存に使用される．
    """
    from maou.app.learning.callbacks import (
        LRSchedulerStepCallback,
        Stage1AccuracyCallback,
    )
    from maou.app.learning.training_loop import (
        Stage1TrainingLoop,
    )

    _logger = logger or logging.getLogger(__name__)

    # model から head を取得
    model = components.model
    reachable_head = model.head  # type: ignore[union-attr]
    assert isinstance(reachable_head, ReachableSquaresHead), (
        f"Expected ReachableSquaresHead, got {type(reachable_head).__name__}"
    )

    # Callbacks
    accuracy_callback = Stage1AccuracyCallback()
    callbacks: list = [accuracy_callback]

    # LR Scheduler (optional)
    if components.lr_scheduler is not None:
        callbacks.append(
            LRSchedulerStepCallback(components.lr_scheduler)
        )

    # TrainingLoop 作成
    training_loop = Stage1TrainingLoop(
        model=model,
        device=device,
        optimizer=components.optimizer,
        loss_fn_policy=components.loss_fn,
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
        ds = components.train_dataloader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

        accuracy_callback.reset()

        training_loop.run_epoch(
            dataloader=components.train_dataloader,
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

        if components.optimizer.param_groups:
            current_lr = components.optimizer.param_groups[0][
                "lr"
            ]
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
    components: StageComponents,
    config: StageConfig,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> tuple[StageResult, LegalMovesHead]:
    """TrainingLoop を使用して Stage 2 (Legal Moves) を学習する．

    CUDA stream overlap，tqdm 進捗表示，コールバックアーキテクチャを活用し，
    高スループットな学習を実現する．

    Args:
        components: Stage 2 のコンポーネント一式．
        config: Stage 2 の学習制御パラメータ．
        device: 学習デバイス (CPU or CUDA)．
        logger: ロガー．

    Returns:
        (StageResult, LegalMovesHead) のタプル．
        ヘッドはチェックポイント保存に使用される．
    """
    from maou.app.learning.callbacks import (
        LRSchedulerStepCallback,
        Stage2F1Callback,
    )
    from maou.app.learning.training_loop import (
        RawLogitsTrainingLoop,
    )

    _logger = logger or logging.getLogger(__name__)

    # model から head を取得
    model = components.model
    legal_moves_head = model.head  # type: ignore[union-attr]
    assert isinstance(legal_moves_head, LegalMovesHead), (
        f"Expected LegalMovesHead, got {type(legal_moves_head).__name__}"
    )

    # Callbacks
    f1_callback = Stage2F1Callback()
    callbacks: list = [f1_callback]

    # LR Scheduler (optional)
    if components.lr_scheduler is not None:
        callbacks.append(
            LRSchedulerStepCallback(components.lr_scheduler)
        )

    # TrainingLoop 作成
    training_loop = RawLogitsTrainingLoop(
        model=model,
        device=device,
        optimizer=components.optimizer,
        loss_fn_policy=components.loss_fn,
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
        ds = components.train_dataloader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

        f1_callback.reset()

        training_loop.run_epoch(
            dataloader=components.train_dataloader,
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

        if components.optimizer.param_groups:
            current_lr = components.optimizer.param_groups[0][
                "lr"
            ]
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
        stage1_components: Optional[StageComponents] = None,
        stage1_config: Optional[StageConfig] = None,
        stage2_components: Optional[StageComponents] = None,
        stage2_config: Optional[StageConfig] = None,
        stage3_config: Optional[StageConfig] = None,
        save_checkpoints: bool = True,
    ) -> dict[TrainingStage, StageResult]:
        """Run all configured stages sequentially.

        Args:
            stage1_components: Stage 1 のコンポーネント一式．
            stage1_config: Stage 1 の学習制御パラメータ．
            stage2_components: Stage 2 のコンポーネント一式．
            stage2_config: Stage 2 の学習制御パラメータ．
            stage3_config: Configuration for Stage 3 (policy + value)
            save_checkpoints: Whether to save checkpoints after each stage

        Returns:
            Dictionary mapping TrainingStage to StageResult

        Raises:
            RuntimeError: If Stage 1 or 2 fails to meet threshold
        """
        results: dict[TrainingStage, StageResult] = {}

        # Stage 1: Reachable Squares
        if (
            stage1_components is not None
            and stage1_config is not None
        ):
            self.logger.info("=" * 60)
            self.logger.info(
                "STAGE 1: REACHABLE SQUARES LEARNING"
            )
            self.logger.info("=" * 60)

            result, reachable_head = (
                run_stage1_with_training_loop(
                    components=stage1_components,
                    config=stage1_config,
                    device=self.device,
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
        if (
            stage2_components is not None
            and stage2_config is not None
        ):
            self.logger.info("=" * 60)
            self.logger.info("STAGE 2: LEGAL MOVES LEARNING")
            self.logger.info("=" * 60)

            result, legal_moves_head = (
                run_stage2_with_training_loop(
                    components=stage2_components,
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
