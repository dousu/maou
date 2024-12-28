import abc
import json
import logging
from pathlib import Path
from typing import Optional

from maou.app.learning.dl import Learning

logger: logging.Logger = logging.getLogger(__name__)


class FileSystem(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def collect_files(p: Path, ext: Optional[str] = None) -> list[Path]:
        pass


def file_validation(p: Path) -> None:
    if not p.is_file():
        raise ValueError(f"File `{p}` is not a file.")


def dir_init(d: Path) -> None:
    if not d.exists():
        d.mkdir()
    else:
        if not d.is_dir():
            raise ValueError(f"Directory `{d}` is not directory.")


def learn(
    file_system: FileSystem,
    input_dir: Path,
    *,
    gpu: Optional[str] = None,
    compilation: Optional[bool] = None,
    test_ratio: Optional[float] = None,
    epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    dataloader_workers: Optional[int] = None,
    gce_parameter: Optional[float] = None,
    policy_loss_ratio: Optional[float] = None,
    value_loss_ratio: Optional[float] = None,
    learning_ratio: Optional[float] = None,
    momentum: Optional[float] = None,
    checkpoint_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> str:
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

    # チェックポイントの書き込み先設定 (デフォルトNone)

    # 学習開始に利用するチェックポイントファイル設定 (デフォルトNone)

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
    if checkpoint_dir is not None:
        dir_init(checkpoint_dir)
    if log_dir is not None:
        dir_init(log_dir)
    if model_dir is not None:
        dir_init(model_dir)
    logger.info(
        f"Input: {input_dir}, Output: {model_dir}, Checkpoint: {checkpoint_dir}"
    )
    option = Learning.LearningOption(
        input_paths=file_system.collect_files(input_dir),
        compilation=compilation,
        test_ratio=test_ratio,
        epoch=epoch,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        gce_parameter=gce_parameter,
        policy_loss_ratio=policy_loss_ratio,
        value_loss_ratio=value_loss_ratio,
        learning_ratio=learning_ratio,
        momentum=momentum,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        log_dir=log_dir,
        model_dir=model_dir,
    )
    learning_result = Learning(gpu).learn(option)

    return json.dumps(learning_result)
