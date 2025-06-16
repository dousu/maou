import abc
import json
import logging
from pathlib import Path
from typing import Optional

from maou.app.learning.dl import CloudStorage, Learning, LearningDataSource

logger: logging.Logger = logging.getLogger(__name__)


class FileSystem(metaclass=abc.ABCMeta):
    """Abstract interface for file system operations.

    Provides an abstraction layer for file I/O operations
    for learning workflows.
    """

    @staticmethod
    @abc.abstractmethod
    def collect_files(p: Path, ext: Optional[str] = None) -> list[Path]:
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
            raise ValueError(f"Directory `{d}` is not directory.")


def learn(
    datasource: LearningDataSource.DataSourceSpliter,
    datasource_type: str,
    *,
    gpu: Optional[str] = None,
    compilation: Optional[bool] = None,
    test_ratio: Optional[float] = None,
    epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    dataloader_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    enable_prefetch: Optional[bool] = None,
    gce_parameter: Optional[float] = None,
    policy_loss_ratio: Optional[float] = None,
    value_loss_ratio: Optional[float] = None,
    learning_ratio: Optional[float] = None,
    momentum: Optional[float] = None,
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
        compilation: Whether to compile the model
        test_ratio: Ratio of data to use for testing
        epoch: Number of training epochs
        batch_size: Training batch size
        dataloader_workers: Number of data loader workers
        pin_memory: Enable pinned memory for faster GPU transfers
        enable_prefetch: Enable background data prefetching for improved performance
        gce_parameter: GCE loss function parameter
        policy_loss_ratio: Policy loss weight
        value_loss_ratio: Value loss weight
        learning_ratio: Learning rate
        momentum: SGD momentum parameter
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
        raise ValueError(f"Data source type `{datasource_type}` is invalid.")

    # モデルをコンパイルするかどうか (デフォルトTrue)
    if compilation is None:
        compilation = True

    # テスト割合設定 (デフォルト0.25)
    if test_ratio is None:
        test_ratio = 0.25

    # エポック設定 (デフォルト10)
    if epoch is None:
        epoch = 10

    # バッチサイズ設定 (デフォルト1000)
    if batch_size is None:
        batch_size = 1000

    # DataLoaderのワーカー数設定 (デフォルト2)
    if dataloader_workers is None:
        dataloader_workers = 0

    # pin_memory設定 (デフォルトFalse)
    if pin_memory is None:
        pin_memory = False

    # enable_prefetch設定 (デフォルトFalse)
    if enable_prefetch is None:
        enable_prefetch = False

    # 損失関数のパラメータ設定 (デフォルト0.7)
    if gce_parameter is None:
        gce_parameter = 0.7

    # policy損失関数のパラメータ設定 (デフォルト1)
    if policy_loss_ratio is None:
        policy_loss_ratio = 1

    # value損失関数のパラメータ設定 (デフォルト1)
    if value_loss_ratio is None:
        value_loss_ratio = 1

    # オプティマイザのパラメータ設定 (デフォルト0.01)
    if learning_ratio is None:
        learning_ratio = 0.01

    # オプティマイザのパラメータ設定momemtum (デフォルト0.9)
    if momentum is None:
        momentum = 0.9

    # 学習開始に利用するチェックポイントファイル設定 (デフォルトNone)

    # 開始エポック数設定 (デフォルト0)
    if start_epoch is None:
        start_epoch = 0

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
        compilation=compilation,
        test_ratio=test_ratio,
        epoch=epoch,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        enable_prefetch=enable_prefetch,
        gce_parameter=gce_parameter,
        policy_loss_ratio=policy_loss_ratio,
        value_loss_ratio=value_loss_ratio,
        learning_ratio=learning_ratio,
        momentum=momentum,
        resume_from=resume_from,
        start_epoch=start_epoch,
        log_dir=log_dir,
        model_dir=model_dir,
    )

    learning_result = Learning(gpu=gpu, cloud_storage=cloud_storage).learn(option)

    return json.dumps(learning_result)
