import abc
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from maou.app.learning.dl import (
    CloudStorage,
    Learning,
    LearningDataSource,
)
from maou.app.learning.network import (
    BACKBONE_ARCHITECTURES,
    BackboneArchitecture,
)

SUPPORTED_MODEL_ARCHITECTURES = BACKBONE_ARCHITECTURES

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
    datasource: LearningDataSource.DataSourceSpliter,
    datasource_type: str,
    *,
    gpu: Optional[str] = None,
    model_architecture: BackboneArchitecture = "resnet",
    compilation: bool = False,
    test_ratio: Optional[float] = None,
    epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    dataloader_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
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
    log_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    cloud_storage: Optional[CloudStorage] = None,
) -> str:
    """Train neural network model on Shogi data.

    Args:
        datasource: Training data source
        datasource_type: Type of data source ('hcpe' or 'preprocess')
        gpu: GPU device to use for training
        model_architecture: Backbone architecture ('resnet', 'mlp-mixer', 'vit')
        compilation: Whether to compile the model
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
        log_dir: Directory for training logs
        model_dir: Directory for saving trained model
        cloud_storage: Optional cloud storage for model uploads

    Returns:
        JSON string with training results
    """
    # データソースのtype確認 (hcpeかpreprocessのみ)
    if datasource_type not in ("hcpe", "preprocess"):
        raise ValueError(
            f"Data source type `{datasource_type}` is invalid."
        )

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

    # prefetch_factor設定 (デフォルト2)
    if prefetch_factor is None:
        prefetch_factor = 2
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
    option = Learning.LearningOption(
        datasource=datasource,
        datasource_type=datasource_type,
        gpu=gpu,
        compilation=compilation,
        test_ratio=test_ratio,
        epoch=epoch,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        gce_parameter=gce_parameter,
        policy_loss_ratio=policy_loss_ratio,
        value_loss_ratio=value_loss_ratio,
        learning_ratio=learning_ratio,
        momentum=momentum,
        optimizer_name=optimizer_key,
        optimizer_beta1=optimizer_beta1,
        optimizer_beta2=optimizer_beta2,
        optimizer_eps=optimizer_eps,
        resume_from=resume_from,
        start_epoch=start_epoch,
        log_dir=log_dir,
        model_dir=model_dir,
        model_architecture=model_architecture,
        lr_scheduler_name=lr_scheduler_key,
    )

    learning_result = Learning(
        cloud_storage=cloud_storage
    ).learn(option)

    return json.dumps(learning_result)
