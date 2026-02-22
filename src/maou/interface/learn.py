import abc
import gc
import json
import logging
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from maou.app.learning.dataset import (
    Stage1Dataset,
    Stage2Dataset,
)
from maou.app.learning.dl import (
    CloudStorage,
    Learning,
    LearningDataSource,
)
from maou.app.learning.multi_stage_training import (
    MultiStageTrainingOrchestrator,
    Stage1DatasetAdapter,
    StageConfig,
    StageResult,
    TrainingStage,
)
from maou.app.learning.network import (
    BACKBONE_ARCHITECTURES,
    BackboneArchitecture,
)
from maou.app.learning.setup import (
    DataLoaderFactory,
    DeviceSetup,
    ModelFactory,
)
from maou.app.learning.streaming_dataset import (
    Stage1StreamingAdapter,
    Stage2StreamingAdapter,
    StreamingDataSource,
    StreamingStage1Dataset,
    StreamingStage2Dataset,
)
from maou.domain.loss.loss_fn import (
    LegalMovesLoss,
    ReachableSquaresLoss,
)

SUPPORTED_MODEL_ARCHITECTURES = BACKBONE_ARCHITECTURES


@dataclass(frozen=True)
class StageDataConfig:
    """Stage別データソース設定．遅延初期化用．

    DataSourceの直接初期化を避け，必要なStage実行直前にのみ
    DataSourceSpliterを生成するための設定を保持する．
    ``create_datasource`` にはDataSourceSpliterを生成するファクトリを渡す．
    """

    create_datasource: (
        "Callable[[], LearningDataSource.DataSourceSpliter]"
    )
    array_type: Literal["preprocessing", "stage1", "stage2"]


# Mapping from canonical scheduler keys to CLI display names.
SUPPORTED_LR_SCHEDULERS: Dict[str, str] = {
    "warmup_cosine_decay": "Warmup+CosineDecay",
    "cosine_annealing_lr": "CosineAnnealingLR",
}

# Cache for normalized lookup values to canonical scheduler keys.
_LR_SCHEDULER_ALIASES: Dict[str, str] = {}
for _canonical, _label in SUPPORTED_LR_SCHEDULERS.items():
    sanitized_canonical = "".join(
        filter(str.isalnum, _canonical.lower())
    )
    sanitized_label = "".join(
        filter(str.isalnum, _label.lower())
    )
    _LR_SCHEDULER_ALIASES[sanitized_canonical] = _canonical
    _LR_SCHEDULER_ALIASES[sanitized_label] = _canonical
    _LR_SCHEDULER_ALIASES[_label.lower()] = _canonical

logger: logging.Logger = logging.getLogger(__name__)


def normalize_lr_scheduler_name(
    scheduler_name: Optional[str],
) -> Optional[str]:
    """Normalize scheduler display names to canonical identifiers.

    Args:
        scheduler_name: User-provided scheduler name or alias.

    Returns:
        Canonical scheduler key when ``scheduler_name`` is provided, otherwise
        ``None`` when no scheduler is requested.

    Raises:
        ValueError: If ``scheduler_name`` does not match a supported scheduler.
    """

    if scheduler_name is None:
        return None

    sanitized = "".join(
        filter(str.isalnum, scheduler_name.lower())
    )
    if not sanitized:
        return None

    canonical = _LR_SCHEDULER_ALIASES.get(sanitized)
    if canonical is None:
        supported_labels = ", ".join(
            SUPPORTED_LR_SCHEDULERS.values()
        )
        raise ValueError(
            "Unsupported learning rate scheduler. "
            f"Supported options are: {supported_labels}"
        )

    return canonical


def _resolve_stage12_scheduler(
    scheduler_value: Optional[str],
    actual_batch_size: int,
    base_batch_size: int = 256,
) -> Optional[str]:
    """Stage 1/2 LRスケジューラの'auto'を解決する．

    Args:
        scheduler_value: CLIから渡されたスケジューラ値('auto', 'none', または具体名)．
        actual_batch_size: 実際のバッチサイズ．
        base_batch_size: 基準バッチサイズ(デフォルト256)．

    Returns:
        正規化されたスケジューラキー，またはNone(スケジューラなし)．
    """
    if (
        scheduler_value is None
        or scheduler_value.lower() == "none"
    ):
        return None
    if scheduler_value.lower() == "auto":
        if actual_batch_size > base_batch_size:
            return "warmup_cosine_decay"
        return None
    return normalize_lr_scheduler_name(scheduler_value)


class FileSystem(metaclass=abc.ABCMeta):
    """Abstract interface for file system operations.

    Provides an abstraction layer for file I/O operations
    for learning workflows.
    """

    @staticmethod
    @abc.abstractmethod
    def collect_files(
        p: Path, ext: Optional[str] = None
    ) -> list[Path]:
        pass


def file_validation(p: Path) -> None:
    """Validate that path is a file.

    Args:
        p: Path to validate

    Raises:
        ValueError: If path is not a file
    """
    if not p.is_file():
        raise ValueError(f"File `{p}` is not a file.")


def dir_init(d: Path) -> None:
    """Initialize directory, creating if it doesn't exist.

    Args:
        d: Directory path to initialize

    Raises:
        ValueError: If path exists but is not a directory
    """
    if not d.exists():
        d.mkdir()
    else:
        if not d.is_dir():
            raise ValueError(
                f"Directory `{d}` is not directory."
            )


def learn(
    datasource: Optional[LearningDataSource.DataSourceSpliter],
    *,
    gpu: Optional[str] = None,
    model_architecture: BackboneArchitecture = "resnet",
    compilation: bool = False,
    detect_anomaly: bool = False,
    test_ratio: Optional[float] = None,
    epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    dataloader_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    cache_transforms: Optional[bool] = None,
    gce_parameter: Optional[float] = None,
    policy_loss_ratio: Optional[float] = None,
    value_loss_ratio: Optional[float] = None,
    learning_ratio: Optional[float] = None,
    lr_scheduler: Optional[str] = None,
    momentum: Optional[float] = None,
    optimizer_name: Optional[str] = None,
    optimizer_beta1: Optional[float] = None,
    optimizer_beta2: Optional[float] = None,
    optimizer_eps: Optional[float] = None,
    resume_from: Optional[Path] = None,
    start_epoch: Optional[int] = None,
    resume_backbone_from: Optional[Path] = None,
    resume_policy_head_from: Optional[Path] = None,
    resume_value_head_from: Optional[Path] = None,
    freeze_backbone: bool = False,
    trainable_layers: Optional[int] = None,
    log_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    cloud_storage: Optional[CloudStorage] = None,
    input_cache_mode: Literal["file", "memory"] = "file",
    tensorboard_histogram_frequency: int = 0,
    tensorboard_histogram_modules: Optional[
        tuple[str, ...]
    ] = None,
    architecture_config: Optional[dict[str, Any]] = None,
    streaming: bool = False,
    streaming_train_source: Optional[
        StreamingDataSource
    ] = None,
    streaming_val_source: Optional[StreamingDataSource] = None,
) -> str:
    """Train neural network model on Shogi data.

    Args:
        datasource: Training data source
        gpu: GPU device to use for training
        model_architecture: Backbone architecture ('resnet', 'mlp-mixer', 'vit')
        compilation: Whether to compile the model
        detect_anomaly: Enable torch.autograd anomaly detection
        test_ratio: Ratio of data to use for testing
        epoch: Number of training epochs
        batch_size: Training batch size
        dataloader_workers: Number of data loader workers
        pin_memory: Enable pinned memory for faster GPU transfers
        prefetch_factor: Number of batches loaded in advance by each worker
        gce_parameter: GCE loss function parameter
        policy_loss_ratio: Policy loss weight
        value_loss_ratio: Value loss weight
        learning_ratio: Learning rate
        lr_scheduler: Learning rate scheduler selection
        momentum: SGD momentum parameter
        optimizer_name: Optimizer selection ('adamw' or 'sgd')
        optimizer_beta1: AdamW beta1 parameter
        optimizer_beta2: AdamW beta2 parameter
        optimizer_eps: AdamW epsilon parameter
        resume_from: Checkpoint file to resume from
        start_epoch: Starting epoch number for training
        resume_backbone_from: Backbone parameter file to resume from
        resume_policy_head_from: Policy head parameter file to resume from
        resume_value_head_from: Value head parameter file to resume from
        freeze_backbone: Freeze backbone parameters during training
        trainable_layers: Number of trailing backbone layer groups to keep
            trainable. None = no freezing, 0 = freeze all backbone layers.
        log_dir: Directory for training logs
        model_dir: Directory for saving trained model
        cloud_storage: Optional cloud storage for model uploads
        input_cache_mode: Strategy used by the input datasource cache
        tensorboard_histogram_frequency: Number of epochs between parameter
            histogram dumps (0 disables histogram logging)
        tensorboard_histogram_modules: Optional glob patterns to filter which
            module names emit histograms
        architecture_config: Optional architecture-specific configuration dict
            (e.g. ViT embed_dim, num_layers). Passed to ModelFactory.
        streaming: Use streaming IterableDataset instead of Map-style Dataset
        streaming_train_source: StreamingDataSource for training data
        streaming_val_source: StreamingDataSource for validation data

    Returns:
        JSON string with training results
    """
    if model_architecture not in BACKBONE_ARCHITECTURES:
        valid_options = ", ".join(BACKBONE_ARCHITECTURES)
        raise ValueError(
            f"model_architecture must be one of {valid_options}, "
            f"got {model_architecture}"
        )

    # テスト割合設定 (デフォルト0.2)
    if test_ratio is None:
        test_ratio = 0.2
    elif not 0.0 < test_ratio < 1.0:
        raise ValueError(
            f"test_ratio must be between 0 and 1, got {test_ratio}"
        )

    # エポック設定 (デフォルト10)
    if epoch is None:
        epoch = 10
    elif epoch <= 0:
        raise ValueError(f"epoch must be positive, got {epoch}")

    # バッチサイズ設定 (デフォルト1000)
    if batch_size is None:
        batch_size = 1000
    elif batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}"
        )

    # DataLoaderのワーカー数設定 (デフォルト0)
    if dataloader_workers is None:
        dataloader_workers = 0
    elif dataloader_workers < 0:
        raise ValueError(
            f"dataloader_workers must be non-negative, got {dataloader_workers}"
        )

    # pin_memory設定 (デフォルトFalse)
    if pin_memory is None:
        pin_memory = False

    # prefetch_factor設定 (デフォルト4)
    if prefetch_factor is None:
        prefetch_factor = 4
    elif prefetch_factor <= 0:
        raise ValueError(
            f"prefetch_factor must be positive, got {prefetch_factor}"
        )

    # 損失関数のパラメータ設定 (デフォルト0.7)
    if gce_parameter is None:
        gce_parameter = 0.7
    elif not 0.0 < gce_parameter <= 1.0:
        raise ValueError(
            f"gce_parameter must be between 0 and 1, got {gce_parameter}"
        )

    # policy損失関数のパラメータ設定 (デフォルト1)
    if policy_loss_ratio is None:
        policy_loss_ratio = 1
    elif policy_loss_ratio <= 0:
        raise ValueError(
            f"policy_loss_ratio must be positive, got {policy_loss_ratio}"
        )

    # value損失関数のパラメータ設定 (デフォルト1)
    if value_loss_ratio is None:
        value_loss_ratio = 1
    elif value_loss_ratio <= 0:
        raise ValueError(
            f"value_loss_ratio must be positive, got {value_loss_ratio}"
        )

    # オプティマイザのパラメータ設定 (デフォルト0.01)
    if learning_ratio is None:
        learning_ratio = 0.01
    elif learning_ratio <= 0:
        raise ValueError(
            f"learning_ratio must be positive, got {learning_ratio}"
        )

    lr_scheduler_key = normalize_lr_scheduler_name(lr_scheduler)
    if lr_scheduler_key is None:
        lr_scheduler_key = "warmup_cosine_decay"

    # オプティマイザのパラメータ設定momemtum (デフォルト0.9)
    if momentum is None:
        momentum = 0.9
    elif not 0.0 <= momentum <= 1.0:
        raise ValueError(
            f"momentum must be between 0 and 1, got {momentum}"
        )

    # Optimizer selection (default AdamW for stability)
    if optimizer_name is None:
        optimizer_name = "adamw"
    optimizer_key = optimizer_name.lower()
    if optimizer_key not in {"adamw", "sgd"}:
        raise ValueError(
            "optimizer_name must be 'adamw' or 'sgd', "
            f"got {optimizer_name}"
        )

    # AdamW beta1 parameter (default 0.9)
    if optimizer_beta1 is None:
        optimizer_beta1 = 0.9
    elif not 0.0 < optimizer_beta1 < 1.0:
        raise ValueError(
            "optimizer_beta1 must be between 0 and 1, "
            f"got {optimizer_beta1}"
        )

    # AdamW beta2 parameter (default 0.999)
    if optimizer_beta2 is None:
        optimizer_beta2 = 0.999
    elif not 0.0 < optimizer_beta2 < 1.0:
        raise ValueError(
            "optimizer_beta2 must be between 0 and 1, "
            f"got {optimizer_beta2}"
        )

    if optimizer_beta2 <= optimizer_beta1:
        raise ValueError(
            "optimizer_beta2 must be greater than optimizer_beta1 "
            f"(got {optimizer_beta1} and {optimizer_beta2})"
        )

    # AdamW epsilon parameter (default 1e-8)
    if optimizer_eps is None:
        optimizer_eps = 1e-8
    elif optimizer_eps <= 0:
        raise ValueError(
            f"optimizer_eps must be positive, got {optimizer_eps}"
        )

    # 学習開始に利用するチェックポイントファイル設定 (デフォルトNone)

    # 開始エポック数設定 (デフォルト0)
    if start_epoch is None:
        start_epoch = 0
    elif start_epoch < 0:
        raise ValueError(
            f"start_epoch must be non-negative, got {start_epoch}"
        )

    # SummaryWriterの書き込み先設定 (デフォルト./logs)
    if log_dir is None:
        log_dir = Path("./logs")

    # 学習後のモデルの書き込み先設定 (デフォルト./models)
    if model_dir is None:
        model_dir = Path("./models")

    # ファイル指定の場合は存在するかチェック
    if resume_from is not None:
        file_validation(resume_from)
    # 指定されたディレクトリが存在しない場合は自動で作成する
    if log_dir is not None:
        dir_init(log_dir)
    if model_dir is not None:
        dir_init(model_dir)
    logger.info(f"Input: {datasource}, Output: {model_dir}")

    if cache_transforms is None:
        cache_transforms = False
    normalized_cache_mode = input_cache_mode.lower()
    if normalized_cache_mode not in {"file", "memory"}:
        raise ValueError(
            "input_cache_mode must be either 'file' or 'memory', "
            f"got {input_cache_mode}"
        )
    if tensorboard_histogram_frequency < 0:
        raise ValueError(
            "tensorboard_histogram_frequency must be non-negative"
        )
    normalized_histogram_modules = (
        tensorboard_histogram_modules
        if tensorboard_histogram_modules
        else None
    )

    option = Learning.LearningOption(
        datasource=datasource,
        gpu=gpu,
        compilation=compilation,
        test_ratio=test_ratio,
        epoch=epoch,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        cache_transforms=cache_transforms,
        gce_parameter=gce_parameter,
        policy_loss_ratio=policy_loss_ratio,
        value_loss_ratio=value_loss_ratio,
        learning_ratio=learning_ratio,
        momentum=momentum,
        optimizer_name=optimizer_key,
        optimizer_beta1=optimizer_beta1,
        optimizer_beta2=optimizer_beta2,
        optimizer_eps=optimizer_eps,
        detect_anomaly=detect_anomaly,
        resume_from=resume_from,
        start_epoch=start_epoch,
        resume_backbone_from=resume_backbone_from,
        resume_policy_head_from=resume_policy_head_from,
        resume_value_head_from=resume_value_head_from,
        freeze_backbone=freeze_backbone,
        trainable_layers=trainable_layers,
        log_dir=log_dir,
        model_dir=model_dir,
        model_architecture=model_architecture,
        lr_scheduler_name=lr_scheduler_key,
        input_cache_mode=normalized_cache_mode,
        tensorboard_histogram_frequency=tensorboard_histogram_frequency,
        tensorboard_histogram_modules=normalized_histogram_modules,
        streaming=streaming,
        streaming_train_source=streaming_train_source,
        streaming_val_source=streaming_val_source,
    )

    learning_result = Learning(
        cloud_storage=cloud_storage
    ).learn(option, architecture_config=architecture_config)

    return json.dumps(learning_result)


def _find_latest_backbone_checkpoint(
    model_dir: Path,
) -> Optional[Path]:
    """Find the latest backbone checkpoint in model_dir.

    Looks for stage2_backbone_*.pt first, then stage1_backbone_*.pt.

    Args:
        model_dir: Directory containing checkpoint files.

    Returns:
        Path to latest backbone checkpoint, or None.
    """
    for prefix in ("stage2_backbone_", "stage1_backbone_"):
        candidates = sorted(
            model_dir.glob(f"{prefix}*.pt"), reverse=True
        )
        if candidates:
            return candidates[0]
    return None


def _run_stage1(
    *,
    data_config: StageDataConfig,
    orchestrator: MultiStageTrainingOrchestrator,
    backbone: "torch.nn.Module",
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    threshold: float,
    device: torch.device,
    lr_scheduler_name: Optional[str] = None,
    compilation: bool = False,
    stage1_pos_weight: float = 1.0,
) -> StageResult:
    """Stage 1 (Reachable Squares) を実行し結果を返す．

    DataSource/Dataset/DataLoaderはすべてこの関数のローカル変数．
    関数終了時にスコープから外れ，GCの解放対象になる．

    Args:
        data_config: Stage 1 データソース設定
        orchestrator: マルチステージオーケストレータ
        backbone: バックボーンネットワーク
        batch_size: バッチサイズ
        learning_rate: 学習率
        max_epochs: 最大エポック数
        threshold: 精度閾値
        device: 計算デバイス
        lr_scheduler_name: 学習率スケジューラ名
        compilation: torch.compileを有効化するか
        stage1_pos_weight: 正例の重み(デフォルト: 1.0)

    Returns:
        Stage 1 の訓練結果
    """
    datasource = data_config.create_datasource()
    train_ds, _ = datasource.train_test_split(test_ratio=0.0)

    raw_dataset = Stage1Dataset(datasource=train_ds)
    dataset = Stage1DatasetAdapter(raw_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    stage_config = StageConfig(
        stage=TrainingStage.REACHABLE_SQUARES,
        max_epochs=max_epochs,
        accuracy_threshold=threshold,
        dataloader=dataloader,
        loss_fn=ReachableSquaresLoss(
            pos_weight=stage1_pos_weight
        ),
        learning_rate=learning_rate,
        lr_scheduler_name=lr_scheduler_name,
        base_batch_size=256,
        actual_batch_size=batch_size,
        compilation=compilation,
    )

    results = orchestrator.run_all_stages(
        stage1_config=stage_config,
        stage2_config=None,
        stage3_config=None,
        save_checkpoints=True,
    )

    return results[TrainingStage.REACHABLE_SQUARES]


def _run_stage2(
    *,
    data_config: StageDataConfig,
    orchestrator: MultiStageTrainingOrchestrator,
    backbone: "torch.nn.Module",
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    threshold: float,
    device: torch.device,
    lr_scheduler_name: Optional[str] = None,
    compilation: bool = False,
    stage2_pos_weight: float = 1.0,
    stage2_gamma_pos: float = 0.0,
    stage2_gamma_neg: float = 0.0,
    stage2_clip: float = 0.0,
    stage2_head_hidden_dim: int | None = None,
    stage2_head_dropout: float = 0.0,
    stage2_test_ratio: float = 0.0,
) -> StageResult:
    """Stage 2 (Legal Moves) を実行し結果を返す．

    DataSource/Dataset/DataLoaderはすべてこの関数のローカル変数．
    関数終了時にスコープから外れ，GCの解放対象になる．

    Args:
        data_config: Stage 2 データソース設定
        orchestrator: マルチステージオーケストレータ
        backbone: バックボーンネットワーク
        batch_size: バッチサイズ
        learning_rate: 学習率
        max_epochs: 最大エポック数
        threshold: 精度閾値
        device: 計算デバイス
        lr_scheduler_name: 学習率スケジューラ名
        compilation: torch.compileを有効化するか
        stage2_pos_weight: 正例の重み(デフォルト: 1.0)
        stage2_gamma_pos: ASL正例フォーカシングパラメータ(デフォルト: 0.0)
        stage2_gamma_neg: ASL負例フォーカシングパラメータ(デフォルト: 0.0)
        stage2_clip: ASL負例クリッピングマージン(デフォルト: 0.0)
        stage2_head_hidden_dim: ヘッドの隠れ層次元(Noneで既定値)
        stage2_head_dropout: ヘッドのドロップアウト率(デフォルト: 0.0)
        stage2_test_ratio: 検証データ分割比率(デフォルト: 0.0で分割なし)

    Returns:
        Stage 2 の訓練結果
    """
    datasource = data_config.create_datasource()
    train_ds, val_ds = datasource.train_test_split(
        test_ratio=stage2_test_ratio
    )

    dataset = Stage2Dataset(datasource=train_ds)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_dataloader: Optional[DataLoader] = None
    if stage2_test_ratio > 0.0 and val_ds is not None:
        val_dataset = Stage2Dataset(datasource=val_ds)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    stage_config = StageConfig(
        stage=TrainingStage.LEGAL_MOVES,
        max_epochs=max_epochs,
        accuracy_threshold=threshold,
        dataloader=dataloader,
        loss_fn=LegalMovesLoss(
            pos_weight=stage2_pos_weight,
            gamma_pos=stage2_gamma_pos,
            gamma_neg=stage2_gamma_neg,
            clip=stage2_clip,
        ),
        learning_rate=learning_rate,
        lr_scheduler_name=lr_scheduler_name,
        base_batch_size=256,
        actual_batch_size=batch_size,
        compilation=compilation,
        head_hidden_dim=stage2_head_hidden_dim,
        head_dropout=stage2_head_dropout,
        val_dataloader=val_dataloader,
    )

    results = orchestrator.run_all_stages(
        stage1_config=None,
        stage2_config=stage_config,
        stage3_config=None,
        save_checkpoints=True,
    )

    return results[TrainingStage.LEGAL_MOVES]


def _run_stage1_streaming(
    *,
    streaming_source: StreamingDataSource,
    orchestrator: MultiStageTrainingOrchestrator,
    backbone: "torch.nn.Module",
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    threshold: float,
    device: torch.device,
    lr_scheduler_name: Optional[str] = None,
    compilation: bool = False,
    stage1_pos_weight: float = 1.0,
) -> StageResult:
    """Stage 1 (Reachable Squares) をストリーミングモードで実行する．

    Args:
        streaming_source: ストリーミングデータソース
        orchestrator: マルチステージオーケストレータ
        backbone: バックボーンネットワーク
        batch_size: バッチサイズ
        learning_rate: 学習率
        max_epochs: 最大エポック数
        threshold: 精度閾値
        device: 計算デバイス
        lr_scheduler_name: 学習率スケジューラ名
        compilation: torch.compileを有効化するか
        stage1_pos_weight: 正例の重み(デフォルト: 1.0)

    Returns:
        Stage 1 の訓練結果
    """
    raw_dataset = StreamingStage1Dataset(
        streaming_source=streaming_source,
        batch_size=batch_size,
        shuffle=True,
    )
    dataset = Stage1StreamingAdapter(raw_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    stage_config = StageConfig(
        stage=TrainingStage.REACHABLE_SQUARES,
        max_epochs=max_epochs,
        accuracy_threshold=threshold,
        dataloader=dataloader,
        loss_fn=ReachableSquaresLoss(
            pos_weight=stage1_pos_weight
        ),
        learning_rate=learning_rate,
        lr_scheduler_name=lr_scheduler_name,
        base_batch_size=256,
        actual_batch_size=batch_size,
        compilation=compilation,
    )

    results = orchestrator.run_all_stages(
        stage1_config=stage_config,
        stage2_config=None,
        stage3_config=None,
        save_checkpoints=True,
    )

    return results[TrainingStage.REACHABLE_SQUARES]


def _run_stage2_streaming(
    *,
    streaming_source: StreamingDataSource,
    orchestrator: MultiStageTrainingOrchestrator,
    backbone: "torch.nn.Module",
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    threshold: float,
    device: torch.device,
    lr_scheduler_name: Optional[str] = None,
    compilation: bool = False,
    stage2_pos_weight: float = 1.0,
    stage2_gamma_pos: float = 0.0,
    stage2_gamma_neg: float = 0.0,
    stage2_clip: float = 0.0,
    stage2_head_hidden_dim: int | None = None,
    stage2_head_dropout: float = 0.0,
    stage2_test_ratio: float = 0.0,
    dataloader_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
) -> StageResult:
    """Stage 2 (Legal Moves) をストリーミングモードで実行する．

    Args:
        streaming_source: ストリーミングデータソース
        orchestrator: マルチステージオーケストレータ
        backbone: バックボーンネットワーク
        batch_size: バッチサイズ
        learning_rate: 学習率
        max_epochs: 最大エポック数
        threshold: 精度閾値
        device: 計算デバイス
        lr_scheduler_name: 学習率スケジューラ名
        compilation: torch.compileを有効化するか
        stage2_pos_weight: 正例の重み(デフォルト: 1.0)
        stage2_gamma_pos: ASL正例フォーカシングパラメータ(デフォルト: 0.0)
        stage2_gamma_neg: ASL負例フォーカシングパラメータ(デフォルト: 0.0)
        stage2_clip: ASL負例クリッピングマージン(デフォルト: 0.0)
        stage2_head_hidden_dim: ヘッドの隠れ層次元(Noneで既定値)
        stage2_head_dropout: ヘッドのドロップアウト率(デフォルト: 0.0)
        stage2_test_ratio: 検証データ分割比率(ストリーミングでは未対応，デフォルト: 0.0)
        dataloader_workers: DataLoaderワーカー数(デフォルト: 0)
        pin_memory: pinned memoryを有効にするか(デフォルト: False)
        prefetch_factor: 各workerの先読みバッチ数(デフォルト: 2)

    Returns:
        Stage 2 の訓練結果
    """
    if stage2_test_ratio > 0.0:
        logger.warning(
            "stage2_test_ratio=%.2f is ignored in streaming mode. "
            "Streaming datasets do not support train/test split.",
            stage2_test_ratio,
        )

    raw_dataset = StreamingStage2Dataset(
        streaming_source=streaming_source,
        batch_size=batch_size,
        shuffle=True,
    )
    dataset = Stage2StreamingAdapter(raw_dataset)

    # Stage 2 streaming ではバリデーション分割を行わないため，
    # train のみの DataLoader を DataLoaderFactory 経由で作成する．
    # n_val_files=0 により val 側は num_workers=0 となる．
    _EmptyDataset = type(
        "_EmptyDataset",
        (IterableDataset,),
        {"__iter__": lambda self: iter([])},
    )
    dataloader, _ = (
        DataLoaderFactory.create_streaming_dataloaders(
            train_dataset=dataset,
            val_dataset=_EmptyDataset(),
            dataloader_workers=dataloader_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            n_train_files=len(streaming_source.file_paths),
            n_val_files=0,
            file_paths=streaming_source.file_paths,
        )
    )

    stage_config = StageConfig(
        stage=TrainingStage.LEGAL_MOVES,
        max_epochs=max_epochs,
        accuracy_threshold=threshold,
        dataloader=dataloader,
        loss_fn=LegalMovesLoss(
            pos_weight=stage2_pos_weight,
            gamma_pos=stage2_gamma_pos,
            gamma_neg=stage2_gamma_neg,
            clip=stage2_clip,
        ),
        learning_rate=learning_rate,
        lr_scheduler_name=lr_scheduler_name,
        base_batch_size=256,
        actual_batch_size=batch_size,
        compilation=compilation,
        head_hidden_dim=stage2_head_hidden_dim,
        head_dropout=stage2_head_dropout,
    )

    results = orchestrator.run_all_stages(
        stage1_config=None,
        stage2_config=stage_config,
        stage3_config=None,
        save_checkpoints=True,
    )

    return results[TrainingStage.LEGAL_MOVES]


def _log_memory_usage(context: str) -> None:
    """現在のメモリ使用量をログ出力する．

    RSS(ピーク値)とGPUメモリ使用量を出力する．
    外部依存なし(標準ライブラリ ``resource`` + ``torch.cuda``)．

    Args:
        context: ログのコンテキスト(例: "Stage 1 start")
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = usage.ru_maxrss / 1024  # Linux: KB → MB

    gpu_info = ""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        gpu_info = (
            f", GPU allocated={allocated:.0f}MB"
            f", GPU reserved={reserved:.0f}MB"
        )

    logger.info(
        "[Memory] %s: peak_RSS=%.0fMB%s",
        context,
        rss_mb,
        gpu_info,
    )


def _release_stage_memory(stage_name: str) -> None:
    """Stage遷移時のメモリ解放を実行する．

    ``gc.collect()`` でPythonオブジェクトを解放し，
    ``torch.cuda.empty_cache()`` でGPUキャッシュメモリをOSに返却する．

    Args:
        stage_name: 解放元のステージ名(ログ用)
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory released after %s", stage_name)


def learn_multi_stage(
    stage: str,
    *,
    stage1_data_config: Optional[StageDataConfig] = None,
    stage2_data_config: Optional[StageDataConfig] = None,
    stage3_data_config: Optional[StageDataConfig] = None,
    stage1_threshold: float = 0.99,
    stage2_threshold: float = 0.85,
    stage1_max_epochs: int = 10,
    stage2_max_epochs: int = 10,
    gpu: Optional[str] = None,
    model_architecture: BackboneArchitecture = "resnet",
    batch_size: int = 256,
    stage1_batch_size: Optional[int] = None,
    stage2_batch_size: Optional[int] = None,
    stage1_learning_rate: Optional[float] = None,
    stage2_learning_rate: Optional[float] = None,
    learning_rate: float = 0.001,
    model_dir: Optional[Path] = None,
    resume_backbone_from: Optional[Path] = None,
    resume_reachable_head_from: Optional[Path] = None,
    resume_legal_moves_head_from: Optional[Path] = None,
    # Stage 3 parameters
    compilation: bool = False,
    detect_anomaly: bool = False,
    test_ratio: Optional[float] = None,
    epoch: Optional[int] = None,
    dataloader_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    cache_transforms: Optional[bool] = None,
    gce_parameter: Optional[float] = None,
    policy_loss_ratio: Optional[float] = None,
    value_loss_ratio: Optional[float] = None,
    lr_scheduler: Optional[str] = None,
    momentum: Optional[float] = None,
    optimizer_name: Optional[str] = None,
    optimizer_beta1: Optional[float] = None,
    optimizer_beta2: Optional[float] = None,
    optimizer_eps: Optional[float] = None,
    freeze_backbone: bool = False,
    trainable_layers: Optional[int] = None,
    log_dir: Optional[Path] = None,
    cloud_storage: Optional[CloudStorage] = None,
    input_cache_mode: Literal["file", "memory"] = "file",
    architecture_config: Optional[dict[str, Any]] = None,
    stage12_lr_scheduler: Optional[str] = "auto",
    stage12_compilation: bool = False,
    stage1_pos_weight: float = 1.0,
    stage2_pos_weight: float = 1.0,
    stage2_gamma_pos: float = 0.0,
    stage2_gamma_neg: float = 0.0,
    stage2_clip: float = 0.0,
    stage2_head_hidden_dim: int | None = None,
    stage2_head_dropout: float = 0.0,
    stage2_test_ratio: float = 0.0,
    streaming: bool = False,
    stage1_streaming_source: Optional[
        StreamingDataSource
    ] = None,
    stage2_streaming_source: Optional[
        StreamingDataSource
    ] = None,
    stage3_streaming_train_source: Optional[
        StreamingDataSource
    ] = None,
    stage3_streaming_val_source: Optional[
        StreamingDataSource
    ] = None,
) -> str:
    """Execute multi-stage training workflow.

    Args:
        stage: Training stage to execute ("1", "2", "3", or "all")
        stage1_data_config: Data config for Stage 1 (reachable squares). Lazy init.
        stage2_data_config: Data config for Stage 2 (legal moves). Lazy init.
        stage3_data_config: Data config for Stage 3 (policy+value). Lazy init.
        stage1_threshold: Accuracy threshold for Stage 1 (default: 0.99)
        stage2_threshold: F1 threshold for Stage 2 (default: 0.85)
        stage1_max_epochs: Maximum epochs for Stage 1 (default: 10)
        stage2_max_epochs: Maximum epochs for Stage 2 (default: 10)
        gpu: GPU device to use
        model_architecture: Backbone architecture
        batch_size: Training batch size
        stage1_batch_size: Batch size for Stage 1 (default: inherits batch_size)
        stage2_batch_size: Batch size for Stage 2 (default: inherits batch_size)
        stage1_learning_rate: Stage 1の学習率．未指定時はlearning_rateにフォールバック．
        stage2_learning_rate: Stage 2の学習率．未指定時はlearning_rateにフォールバック．
        learning_rate: Learning rate
        model_dir: Model output directory
        resume_backbone_from: Backbone checkpoint to resume from
        resume_reachable_head_from: Reachable head checkpoint to resume from
        resume_legal_moves_head_from: Legal moves head checkpoint to resume from
        compilation: Enable PyTorch compilation for Stage 3
        detect_anomaly: Enable anomaly detection for Stage 3
        test_ratio: Test set ratio for Stage 3
        epoch: Number of training epochs for Stage 3
        dataloader_workers: DataLoader worker count for Stage 3
        pin_memory: Enable pinned memory for Stage 3
        prefetch_factor: Prefetch factor for Stage 3
        cache_transforms: Cache transforms for Stage 3
        gce_parameter: GCE parameter for Stage 3
        policy_loss_ratio: Policy loss weight for Stage 3
        value_loss_ratio: Value loss weight for Stage 3
        lr_scheduler: LR scheduler for Stage 3
        momentum: SGD momentum for Stage 3
        optimizer_name: Optimizer for Stage 3
        optimizer_beta1: AdamW beta1 for Stage 3
        optimizer_beta2: AdamW beta2 for Stage 3
        optimizer_eps: AdamW epsilon for Stage 3
        freeze_backbone: Freeze backbone in Stage 3
        trainable_layers: Number of trailing backbone groups to keep trainable
            in Stage 3. When set with multi-stage, enables layer separation:
            Stage 1/2 train only the first (total - N) groups via truncated model
        log_dir: Log directory for Stage 3
        cloud_storage: Cloud storage for Stage 3 model uploads
        input_cache_mode: Cache strategy for Stage 3 inputs
        architecture_config: Architecture-specific config dict for backbone
        stage12_lr_scheduler: Stage 1/2 LRスケジューラ設定．'auto'はbatch_size > 256で自動有効化．
        stage12_compilation: Stage 1/2でtorch.compileを有効化する．
        stage1_pos_weight: Stage 1の正例の重み(デフォルト: 1.0)
        stage2_pos_weight: Stage 2の正例の重み(デフォルト: 1.0)
        stage2_gamma_pos: Stage 2 ASL正例フォーカシングパラメータ(デフォルト: 0.0)
        stage2_gamma_neg: Stage 2 ASL負例フォーカシングパラメータ(デフォルト: 0.0)
        stage2_clip: Stage 2 ASL負例クリッピングマージン(デフォルト: 0.0)
        stage2_head_hidden_dim: Stage 2ヘッドの隠れ層次元(Noneで既定値)
        stage2_head_dropout: Stage 2ヘッドのドロップアウト率(デフォルト: 0.0)
        stage2_test_ratio: Stage 2検証データ分割比率(デフォルト: 0.0で分割なし)
        streaming: Use streaming IterableDataset for Stage 1/2/3
        stage1_streaming_source: StreamingDataSource for Stage 1
        stage2_streaming_source: StreamingDataSource for Stage 2
        stage3_streaming_train_source: StreamingDataSource for Stage 3 training
        stage3_streaming_val_source: StreamingDataSource for Stage 3 validation

    Returns:
        JSON string with training results

    Raises:
        ValueError: If stage parameter is invalid or required datasources missing
        RuntimeError: If Stage 1 or 2 fails to meet accuracy threshold
    """
    # Validate stage parameter
    if stage not in ("1", "2", "3", "all"):
        raise ValueError(
            f"Invalid stage: {stage}. Must be '1', '2', '3', or 'all'"
        )

    # Validate data configs based on stage
    if stage in ("1", "all") and stage1_data_config is None:
        raise ValueError(
            "stage1_data_config is required for stage 1 or all"
        )
    if stage in ("2", "all") and stage2_data_config is None:
        raise ValueError(
            "stage2_data_config is required for stage 2 or all"
        )
    if stage == "3" and stage3_data_config is None:
        raise ValueError(
            "stage3_data_config is required for stage 3"
        )

    # Set model directory default
    if model_dir is None:
        model_dir = Path("./models")
    dir_init(model_dir)

    # Setup device
    device_config = DeviceSetup.setup_device(gpu=gpu)
    device = device_config.device
    logger.info(f"Using device: {device}")

    # Create backbone
    backbone = ModelFactory.create_shogi_backbone(
        device=device,
        architecture=model_architecture,
        architecture_config=architecture_config,
    )

    # Load backbone if resuming
    if resume_backbone_from is not None:
        from maou.app.learning.model_io import ModelIO

        logger.info(
            f"Loading backbone from checkpoint: {resume_backbone_from}"
        )
        backbone_dict = ModelIO.load_backbone(
            resume_backbone_from, device
        )
        backbone.load_state_dict(backbone_dict)

    # Create orchestrator
    orchestrator = MultiStageTrainingOrchestrator(
        backbone=backbone,
        device=device,
        model_dir=model_dir,
        trainable_layers=trainable_layers,
    )

    # Format results as JSON
    results_dict: dict[str, Any] = {
        "stage": stage,
        "stages_completed": [],
    }

    # Resolve stage-specific batch sizes (fallback to global batch_size)
    effective_stage1_batch = stage1_batch_size or batch_size
    effective_stage2_batch = stage2_batch_size or batch_size

    # Warn about large Stage 2 batch sizes with ViT
    if (
        model_architecture == "vit"
        and effective_stage2_batch >= 2048
        and stage in ("2", "all")
    ):
        logger.warning(
            "Stage 2 batch size is %d, which may cause "
            "CUDA OOM with ViT architecture. Consider using "
            "--stage2-batch-size to set a smaller value, or "
            "--gradient-checkpointing to reduce activation memory.",
            effective_stage2_batch,
        )

    # Resolve Stage 1/2 LR scheduler ("auto" -> actual scheduler or None)
    resolved_stage1_scheduler = _resolve_stage12_scheduler(
        stage12_lr_scheduler, effective_stage1_batch
    )
    resolved_stage2_scheduler = _resolve_stage12_scheduler(
        stage12_lr_scheduler, effective_stage2_batch
    )

    # Resolve stage-specific learning rates (fallback to global learning_rate)
    effective_stage1_lr = stage1_learning_rate or learning_rate
    effective_stage2_lr = stage2_learning_rate or learning_rate

    logger.info(f"Starting multi-stage training: stage={stage}")

    # Stage 1: Reachable Squares
    # DataSource/Dataset/DataLoaderは_run_stage1内のローカル変数．
    # 関数終了時にスコープから外れ，GCの解放対象になる．
    _log_memory_usage("Stage 1 start")
    if stage in ("1", "all") and stage1_data_config is not None:
        if streaming and stage1_streaming_source is not None:
            stage1_result = _run_stage1_streaming(
                streaming_source=stage1_streaming_source,
                orchestrator=orchestrator,
                backbone=backbone,
                batch_size=effective_stage1_batch,
                learning_rate=effective_stage1_lr,
                max_epochs=stage1_max_epochs,
                threshold=stage1_threshold,
                device=device,
                lr_scheduler_name=resolved_stage1_scheduler,
                compilation=stage12_compilation,
                stage1_pos_weight=stage1_pos_weight,
            )
        else:
            stage1_result = _run_stage1(
                data_config=stage1_data_config,
                orchestrator=orchestrator,
                backbone=backbone,
                batch_size=effective_stage1_batch,
                learning_rate=effective_stage1_lr,
                max_epochs=stage1_max_epochs,
                threshold=stage1_threshold,
                device=device,
                lr_scheduler_name=resolved_stage1_scheduler,
                compilation=stage12_compilation,
                stage1_pos_weight=stage1_pos_weight,
            )
        results_dict["stages_completed"].append(
            {
                "stage": stage1_result.stage.value,
                "stage_name": stage1_result.stage.name,
                "achieved_accuracy": stage1_result.achieved_accuracy,
                "final_loss": stage1_result.final_loss,
                "epochs_trained": stage1_result.epochs_trained,
                "threshold_met": stage1_result.threshold_met,
            }
        )
        _release_stage_memory("Stage 1")

    # Stage 2: Legal Moves
    _log_memory_usage("Stage 2 start")
    if stage in ("2", "all") and stage2_data_config is not None:
        if streaming and stage2_streaming_source is not None:
            stage2_result = _run_stage2_streaming(
                streaming_source=stage2_streaming_source,
                orchestrator=orchestrator,
                backbone=backbone,
                batch_size=effective_stage2_batch,
                learning_rate=effective_stage2_lr,
                max_epochs=stage2_max_epochs,
                threshold=stage2_threshold,
                device=device,
                lr_scheduler_name=resolved_stage2_scheduler,
                compilation=stage12_compilation,
                stage2_pos_weight=stage2_pos_weight,
                stage2_gamma_pos=stage2_gamma_pos,
                stage2_gamma_neg=stage2_gamma_neg,
                stage2_clip=stage2_clip,
                stage2_head_hidden_dim=stage2_head_hidden_dim,
                stage2_head_dropout=stage2_head_dropout,
                stage2_test_ratio=stage2_test_ratio,
                dataloader_workers=dataloader_workers or 0,
                pin_memory=pin_memory or False,
                prefetch_factor=prefetch_factor or 2,
            )
        else:
            stage2_result = _run_stage2(
                data_config=stage2_data_config,
                orchestrator=orchestrator,
                backbone=backbone,
                batch_size=effective_stage2_batch,
                learning_rate=effective_stage2_lr,
                max_epochs=stage2_max_epochs,
                threshold=stage2_threshold,
                device=device,
                lr_scheduler_name=resolved_stage2_scheduler,
                compilation=stage12_compilation,
                stage2_pos_weight=stage2_pos_weight,
                stage2_gamma_pos=stage2_gamma_pos,
                stage2_gamma_neg=stage2_gamma_neg,
                stage2_clip=stage2_clip,
                stage2_head_hidden_dim=stage2_head_hidden_dim,
                stage2_head_dropout=stage2_head_dropout,
                stage2_test_ratio=stage2_test_ratio,
            )
        results_dict["stages_completed"].append(
            {
                "stage": stage2_result.stage.value,
                "stage_name": stage2_result.stage.name,
                "achieved_accuracy": stage2_result.achieved_accuracy,
                "final_loss": stage2_result.final_loss,
                "epochs_trained": stage2_result.epochs_trained,
                "threshold_met": stage2_result.threshold_met,
            }
        )
        _release_stage_memory("Stage 2")

    # Stage 3: Policy + Value (delegate to Learning.learn())
    # DataSourceはここで遅延初期化する．
    #
    # Stage 1/2で使用したオーケストレータとバックボーンを明示的に解放する．
    # Stage 3は learn() 内で新しいモデルを作成しチェックポイントから
    # backboneをロードするため，これらのオブジェクトは不要．
    # 解放しないとGPU/CPUメモリが残留し，Stage 3のDataLoaderワーカーが
    # メモリ枯渇でタイムアウトする原因となる．
    if stage == "all":
        del orchestrator
        del backbone
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        _release_stage_memory("pre-Stage 3 cleanup")

    _log_memory_usage("Stage 3 start")
    if stage in ("3", "all") and stage3_data_config is not None:
        logger.info("=" * 60)
        logger.info("STAGE 3: POLICY + VALUE LEARNING")
        logger.info("=" * 60)

        # Determine backbone checkpoint path
        stage3_resume_backbone = resume_backbone_from
        if stage == "all":
            saved_backbone = _find_latest_backbone_checkpoint(
                model_dir
            )
            if saved_backbone is not None:
                stage3_resume_backbone = saved_backbone
                logger.info(
                    f"Using backbone from Stage 1/2: {saved_backbone}"
                )

        # streaming mode では FileManager による全データロードをスキップする．
        # create_datasource() は FileDataSourceSpliter を生成し，
        # FileManager が全ファイルをメモリにロードする(Stage 3: ~123GB)．
        # streaming mode では StreamingKifDataset が遅延読み込みするため不要．
        # 不要なロードを行うと spawn ワーカー起動時に OOM kill される．
        stage3_datasource = (
            None
            if streaming
            else stage3_data_config.create_datasource()
        )
        stage3_result = learn(
            datasource=stage3_datasource,
            gpu=gpu,
            model_architecture=model_architecture,
            compilation=compilation,
            detect_anomaly=detect_anomaly,
            test_ratio=test_ratio,
            epoch=epoch,
            batch_size=batch_size,
            dataloader_workers=dataloader_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            cache_transforms=cache_transforms,
            gce_parameter=gce_parameter,
            policy_loss_ratio=policy_loss_ratio,
            value_loss_ratio=value_loss_ratio,
            learning_ratio=learning_rate,
            lr_scheduler=lr_scheduler,
            momentum=momentum,
            optimizer_name=optimizer_name,
            optimizer_beta1=optimizer_beta1,
            optimizer_beta2=optimizer_beta2,
            optimizer_eps=optimizer_eps,
            resume_backbone_from=stage3_resume_backbone,
            freeze_backbone=freeze_backbone,
            trainable_layers=trainable_layers,
            log_dir=log_dir,
            model_dir=model_dir,
            cloud_storage=cloud_storage,
            input_cache_mode=input_cache_mode,
            architecture_config=architecture_config,
            streaming=streaming,
            streaming_train_source=stage3_streaming_train_source,
            streaming_val_source=stage3_streaming_val_source,
        )
        results_dict["stages_completed"].append("stage3")
        results_dict["stage3_result"] = stage3_result
    elif stage == "all" and stage3_data_config is None:
        logger.warning(
            "Stage 3 skipped: no --stage3-data-path specified. "
            "Only Stage 1/2 were executed."
        )

    return json.dumps(results_dict)
