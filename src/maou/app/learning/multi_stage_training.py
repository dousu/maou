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
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

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

    def __init__(self, network: Network) -> None:
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


class StageModelAdapter(torch.nn.Module):
    """Stage 1 / Stage 2 共通のモデルアダプタ．

    HeadlessNetwork と事前学習用 head をラップし，
    TrainingLoop が期待する ``(policy, value)`` の2タプルを返す．
    ``value`` 出力はダミーゼロテンソルで，value loss は ``value_loss_ratio=0.0`` で無視される．

    Stage 1 (``ReachableSquaresHead``) と Stage 2 (``LegalMovesHead``) は
    head を差し替えるだけで挙動が同じなので，1 クラスで両方を担う．
    ``Stage1ModelAdapter`` / ``Stage2ModelAdapter`` は本クラスの別名で，
    互換のために残してある．

    Args:
        backbone: 共有バックボーンネットワーク
        head: 事前学習用の head (Stage 1 なら ReachableSquaresHead，
            Stage 2 なら LegalMovesHead)
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


class StageDatasetAdapter(Dataset):
    """Stage1/Stage2Dataset を TrainingLoop の入力形式に変換するアダプタ．

    Stage1Dataset は ``((board, hand), reachable_squares)``，
    Stage2Dataset は ``((board, hand), legal_moves)`` を返すが，
    TrainingLoop._unpack_batch() は
    ((board, hand), (labels_policy, labels_value, move_win_rate))
    を期待する．変換は両者で同一なので 1 クラスで両方を担う．

    PyTorch の default_collate は None を処理できないため，
    DataLoader 作成時には pre_stage_collate_fn を collate_fn に指定すること．

    ``Stage1DatasetAdapter`` / ``Stage2DatasetAdapter`` は本クラスの
    別名で，互換のために残してある．

    Args:
        dataset: ラップする Stage1Dataset または Stage2Dataset
    """

    def __init__(
        self, dataset: Stage1Dataset | Stage2Dataset
    ) -> None:
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


Stage1DatasetAdapter = StageDatasetAdapter
"""``StageDatasetAdapter`` の別名 (Stage 1 用)．"""

Stage2DatasetAdapter = StageDatasetAdapter
"""``StageDatasetAdapter`` の別名 (Stage 2 用)．"""


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

    両アダプタは move_win_rate=None を含むタプルを返すが，
    PyTorch の default_collate は None を処理できないため，
    None を手動で伝播させるカスタム collate 関数が必要．
    """
    inputs_list, labels_list = zip(*batch)
    boards = torch.stack([inp[0] for inp in inputs_list])
    hands = torch.stack([inp[1] for inp in inputs_list])
    targets = torch.stack([lbl[0] for lbl in labels_list])
    values = torch.stack([lbl[1] for lbl in labels_list])
    return (boards, hands), (targets, values, None)


Stage1ModelAdapter = StageModelAdapter
"""``StageModelAdapter`` の別名 (Stage 1 用)．"""

Stage2ModelAdapter = StageModelAdapter
"""``StageModelAdapter`` の別名 (Stage 2 用)．"""


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
            truncated_out_ch = self._compute_output_channels(
                self.partial_backbone,
                backbone.backbone_input_channels,
                backbone.board_size,
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
        """ダミー入力を通して partial backbone の出力チャンネル数を推定する．

        ``partial`` は既にデバイスへ移動済みの backbone から切り出される
        (StageComponentFactory は ``backbone.to(device)`` 済みのものを渡す)
        ため，ダミー入力も同じデバイス・dtype で生成する．CPU 固定で
        生成すると resnet + ``--trainable-layers`` + ``--gpu`` の組み合わせが
        モデル構築時点で RuntimeError になる．
        """
        try:
            ref = next(partial.parameters())
            device = ref.device
            dtype = ref.dtype
        except StopIteration:
            # パラメータを持たない partial (n_use=0 等) は CPU で評価する
            device = torch.device("cpu")
            dtype = torch.float32
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                input_channels,
                *board_size,
                device=device,
                dtype=dtype,
            )
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
        combined = self.backbone.embed_inputs(inputs)

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


class _StageMetricCallback(Protocol):
    """Stage 1/2 の epoch メトリクスを蓄積するコールバックの構造．

    `Stage1AccuracyCallback` と `Stage2F1Callback` が満たす最小の
    インターフェース．epoch ごとのメトリクス取得はステージによって
    メソッド名が異なるため，`metric_getter` として外から与える．
    """

    def reset(self) -> None: ...

    def get_average_loss(self) -> float: ...


def _run_stage_with_training_loop[
    HeadT: nn.Module,
    CallbackT: _StageMetricCallback,
](
    *,
    components: StageComponents,
    config: StageConfig,
    device: torch.device,
    logger: logging.Logger | None,
    gradient_accumulation_steps: int,
    head_type: type[HeadT],
    callback_factory: Callable[[], CallbackT],
    metric_getter: Callable[[CallbackT], float],
    stage_label: str,
    metric_label: str,
) -> tuple[StageResult, HeadT]:
    """TrainingLoop を使用して Stage 1/2 を学習する共通ループ．

    Stage 1 と Stage 2 はヘッドの型・メトリクスコールバック・
    メトリクス取得メソッド・ログの見出し 2 種だけが異なり，
    学習ループ自体は同一である．TrainingLoop の実装も同一で，
    `Stage1TrainingLoop` は `RawLogitsTrainingLoop` の別名である
    (`training_loop.py`)．

    Args:
        components: 対象ステージのコンポーネント一式．
        config: 対象ステージの学習制御パラメータ．
        device: 学習デバイス (CPU or CUDA)．
        logger: ロガー．
        gradient_accumulation_steps: 勾配蓄積ステップ数．
        head_type: 期待するヘッドの型 (取り違え検出用)．
        callback_factory: メトリクスコールバックを生成する呼び出し可能．
        metric_getter: コールバックから epoch メトリクスを取り出す関数．
        stage_label: ログに出すステージ名 ("Stage 1" など)．
        metric_label: ログに出すメトリクス名 ("Accuracy" など)．

    Returns:
        (StageResult, head) のタプル．
        ヘッドはチェックポイント保存に使用される．
    """
    from maou.app.learning.callbacks import (
        LRSchedulerStepCallback,
    )
    from maou.app.learning.training_loop import (
        RawLogitsTrainingLoop,
    )

    _logger = logger or logging.getLogger(__name__)

    # model から head を取得
    model = components.model
    head = model.head  # type: ignore[union-attr]
    assert isinstance(head, head_type), (
        f"Expected {head_type.__name__}, got {type(head).__name__}"
    )

    # Callbacks
    metric_callback = callback_factory()
    callbacks: list = [metric_callback]

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
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Epoch loop
    _logger.info(
        "Starting %s (TrainingLoop): max_epochs=%d, "
        "threshold=%.1f%%",
        stage_label,
        config.max_epochs,
        config.accuracy_threshold * 100,
    )

    best_metric = 0.0
    final_loss = 0.0

    for epoch in range(config.max_epochs):
        # IterableDataset のエポックシード更新
        ds = components.train_dataloader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

        metric_callback.reset()

        training_loop.run_epoch(
            dataloader=components.train_dataloader,
            epoch_idx=epoch,
            progress_bar=True,
            train_mode=True,
        )

        epoch_metric = metric_getter(metric_callback)
        epoch_loss = metric_callback.get_average_loss()
        final_loss = epoch_loss

        _logger.info(
            "%s Epoch %d/%d: Loss=%.4f, %s=%.2f%%",
            stage_label,
            epoch + 1,
            config.max_epochs,
            epoch_loss,
            metric_label,
            epoch_metric * 100,
        )

        if components.optimizer.param_groups:
            current_lr = components.optimizer.param_groups[0][
                "lr"
            ]
            _logger.info(
                "%s Epoch %d: LR = %.6f",
                stage_label,
                epoch + 1,
                current_lr,
            )

        best_metric = max(best_metric, epoch_metric)

        # Threshold check (early stopping)
        if epoch_metric >= config.accuracy_threshold:
            _logger.info(
                "%s %s threshold achieved! (%.2f%% >= %.2f%%)",
                stage_label,
                metric_label,
                epoch_metric * 100,
                config.accuracy_threshold * 100,
            )
            return (
                StageResult(
                    stage=config.stage,
                    achieved_accuracy=epoch_metric,
                    final_loss=final_loss,
                    epochs_trained=epoch + 1,
                    threshold_met=True,
                ),
                head,
            )

    # Max epochs reached
    threshold_met = best_metric >= config.accuracy_threshold

    return (
        StageResult(
            stage=config.stage,
            achieved_accuracy=best_metric,
            final_loss=final_loss,
            epochs_trained=config.max_epochs,
            threshold_met=threshold_met,
        ),
        head,
    )


def run_stage1_with_training_loop(
    *,
    components: StageComponents,
    config: StageConfig,
    device: torch.device,
    logger: logging.Logger | None = None,
    gradient_accumulation_steps: int = 1,
) -> tuple[StageResult, ReachableSquaresHead]:
    """TrainingLoop を使用して Stage 1 (Reachable Squares) を学習する．

    CUDA stream overlap，tqdm 進捗表示，コールバックアーキテクチャを活用し，
    高スループットな学習を実現する．

    Args:
        components: Stage 1 のコンポーネント一式．
        config: Stage 1 の学習制御パラメータ．
        device: 学習デバイス (CPU or CUDA)．
        logger: ロガー．
        gradient_accumulation_steps: 勾配蓄積ステップ数(デフォルト: 1)．

    Returns:
        (StageResult, ReachableSquaresHead) のタプル．
        ヘッドはチェックポイント保存に使用される．
    """
    from maou.app.learning.callbacks import (
        Stage1AccuracyCallback,
    )

    return _run_stage_with_training_loop(
        components=components,
        config=config,
        device=device,
        logger=logger,
        gradient_accumulation_steps=gradient_accumulation_steps,
        head_type=ReachableSquaresHead,
        callback_factory=Stage1AccuracyCallback,
        metric_getter=lambda cb: cb.get_epoch_accuracy(),
        stage_label="Stage 1",
        metric_label="Accuracy",
    )


def run_stage2_with_training_loop(
    *,
    components: StageComponents,
    config: StageConfig,
    device: torch.device,
    logger: logging.Logger | None = None,
    gradient_accumulation_steps: int = 1,
) -> tuple[StageResult, LegalMovesHead]:
    """TrainingLoop を使用して Stage 2 (Legal Moves) を学習する．

    CUDA stream overlap，tqdm 進捗表示，コールバックアーキテクチャを活用し，
    高スループットな学習を実現する．

    Args:
        components: Stage 2 のコンポーネント一式．
        config: Stage 2 の学習制御パラメータ．
        device: 学習デバイス (CPU or CUDA)．
        logger: ロガー．
        gradient_accumulation_steps: 勾配蓄積ステップ数(デフォルト: 1)．

    Returns:
        (StageResult, LegalMovesHead) のタプル．
        ヘッドはチェックポイント保存に使用される．
    """
    from maou.app.learning.callbacks import Stage2F1Callback

    return _run_stage_with_training_loop(
        components=components,
        config=config,
        device=device,
        logger=logger,
        gradient_accumulation_steps=gradient_accumulation_steps,
        head_type=LegalMovesHead,
        callback_factory=Stage2F1Callback,
        metric_getter=lambda cb: cb.get_epoch_f1(),
        stage_label="Stage 2",
        metric_label="F1",
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
        trainable_layers: int | None = None,
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
        stage1_components: StageComponents | None = None,
        stage1_config: StageConfig | None = None,
        stage2_components: StageComponents | None = None,
        stage2_config: StageConfig | None = None,
        stage3_config: StageConfig | None = None,
        save_checkpoints: bool = True,
        gradient_accumulation_steps: int = 1,
    ) -> dict[TrainingStage, StageResult]:
        """Run all configured stages sequentially.

        Args:
            stage1_components: Stage 1 のコンポーネント一式．
            stage1_config: Stage 1 の学習制御パラメータ．
            stage2_components: Stage 2 のコンポーネント一式．
            stage2_config: Stage 2 の学習制御パラメータ．
            stage3_config: Configuration for Stage 3 (policy + value)
            save_checkpoints: Whether to save checkpoints after each stage
            gradient_accumulation_steps: Number of gradient accumulation
                steps. Effective batch size = batch_size × steps.

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
                    gradient_accumulation_steps=gradient_accumulation_steps,
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
                    gradient_accumulation_steps=gradient_accumulation_steps,
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

        timestamp = (
            datetime.now()
            .astimezone()
            .strftime("%Y%m%d_%H%M%S")
        )

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
