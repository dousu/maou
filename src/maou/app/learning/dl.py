import abc
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchinfo import summary

from maou.app.learning.callbacks import (
    LoggingCallback,
    ValidationCallback,
)
from maou.app.learning.dataset import DataSource
from maou.app.learning.model_io import ModelIO
from maou.app.learning.network import Network
from maou.app.learning.setup import TrainingSetup
from maou.app.learning.training_loop import TrainingLoop
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.cloud_storage import CloudStorage


class LearningDataSource(DataSource):
    class DataSourceSpliter(metaclass=abc.ABCMeta):
        @abc.abstractmethod
        def train_test_split(
            self, test_ratio: float
        ) -> tuple[DataSource, DataSource]:
            pass


class Learning:
    """Learning.
    機械学習のトレーニングを行うユースケースを表現する．
    損失関数の設計では，
    レーティングにより重みづけしたり，
    ラベルノイズに強い損失関数を選ぶといった工夫が必要かも．
    """

    logger: logging.Logger = logging.getLogger(__name__)
    device: torch.device
    resume_from: Optional[Path]
    model: Network
    scaler: Optional[GradScaler]

    @dataclass(kw_only=True, frozen=True)
    class LearningOption:
        datasource: LearningDataSource.DataSourceSpliter
        datasource_type: str
        gpu: Optional[str] = None
        compilation: bool
        test_ratio: float
        epoch: int
        batch_size: int
        dataloader_workers: int
        pin_memory: bool
        prefetch_factor: int
        gce_parameter: float
        policy_loss_ratio: float
        value_loss_ratio: float
        learning_ratio: float
        momentum: float
        resume_from: Optional[Path] = None
        start_epoch: int = 0
        log_dir: Path
        model_dir: Path

    def __init__(
        self,
        *,
        cloud_storage: Optional[CloudStorage] = None,
    ):
        self.__cloud_storage = cloud_storage

    def learn(self, config: LearningOption) -> Dict[str, str]:
        """機械学習を行う."""
        self.logger.info("start learning")
        learning_result: Dict[str, str] = {}

        # 入力とテスト用のデータソース取得
        training_datasource, validation_datasource = (
            config.datasource.train_test_split(
                test_ratio=config.test_ratio
            )
        )

        device_config, dataloaders, model_components = (
            TrainingSetup.setup_training_components(
                training_datasource=training_datasource,
                validation_datasource=validation_datasource,
                datasource_type=config.datasource_type,
                gpu=config.gpu,
                compilation=config.compilation,
                batch_size=config.batch_size,
                dataloader_workers=config.dataloader_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                gce_parameter=config.gce_parameter,
                learning_ratio=config.learning_ratio,
                momentum=config.momentum,
            )
        )

        self.device = device_config.device
        self.training_loader, self.validation_loader = (
            dataloaders
        )
        self.model = model_components.model
        self.loss_fn_policy = model_components.loss_fn_policy
        self.loss_fn_value = model_components.loss_fn_value
        self.optimizer = model_components.optimizer
        self.policy_loss_ratio = config.policy_loss_ratio
        self.value_loss_ratio = config.value_loss_ratio
        self.log_dir = config.log_dir
        self.epoch = config.epoch
        self.model_dir = config.model_dir
        self.resume_from = config.resume_from
        self.start_epoch = config.start_epoch
        summary(
            self.model,
            input_size=(
                config.batch_size,
                FEATURES_NUM,
                9,
                9,
            ),
        )
        self.__train()

        learning_result["Data Samples"] = (
            f"Training: {len(self.training_loader.dataset)}, "  # type: ignore
            f"Test: {len(self.validation_loader.dataset)}"  # type: ignore
        )
        learning_result["Option"] = str(config)
        learning_result["Result"] = "Finish"

        return learning_result

    def __train_one_epoch(
        self, epoch_index: int, tb_writer: SummaryWriter
    ) -> float:
        # Create logging callback
        logging_callback = LoggingCallback(
            writer=tb_writer,
            dataloader_length=len(self.training_loader),
            logger=self.logger,
        )

        # Create training loop
        training_loop = TrainingLoop(
            model=self.model,
            device=self.device,
            optimizer=self.optimizer,
            loss_fn_policy=self.loss_fn_policy,
            loss_fn_value=self.loss_fn_value,
            policy_loss_ratio=self.policy_loss_ratio,
            value_loss_ratio=self.value_loss_ratio,
            callbacks=[logging_callback],
            logger=self.logger,
        )

        # Run training epoch
        training_loop.run_epoch(
            dataloader=self.training_loader,
            epoch_idx=epoch_index,
            progress_bar=True,
            train_mode=True,
        )

        return float(logging_callback.last_loss)

    def __train(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_writer_log_dir = (
            self.log_dir / "training_log_{}".format(timestamp)
        )
        writer = SummaryWriter(summary_writer_log_dir)
        epoch_number = 0

        EPOCHS = self.epoch

        best_vloss = 1_000_000.0

        # resume from checkpoint
        if self.resume_from is not None:
            self.model.load_state_dict(
                torch.load(
                    self.resume_from,
                    weights_only=True,
                    map_location=self.device,
                )
            )

        # start epoch設定
        epoch_number = self.start_epoch

        for _ in range(epoch_number, EPOCHS):
            self.logger.info(
                "EPOCH {}:".format(epoch_number + 1)
            )

            avg_loss = self.__train_one_epoch(
                epoch_number, writer
            )

            # 学習ごとに各層のパラメータを記録
            for name, param in self.model.named_parameters():
                try:
                    param_np = param.detach().cpu().numpy()
                    writer.add_histogram(
                        f"parameters/{name}",
                        param_np,
                        epoch_number + 1,
                    )
                    if param.grad is not None:
                        grad_np = (
                            param.grad.detach().cpu().numpy()
                        )
                        writer.add_histogram(
                            f"gradients/{name}",
                            grad_np,
                            epoch_number + 1,
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to log histogram for {name}: {e}"
                    )

            # Create validation callback
            validation_callback = ValidationCallback(
                logger=self.logger
            )

            # Create validation training loop
            validation_loop = TrainingLoop(
                model=self.model,
                device=self.device,
                optimizer=self.optimizer,
                loss_fn_policy=self.loss_fn_policy,
                loss_fn_value=self.loss_fn_value,
                policy_loss_ratio=self.policy_loss_ratio,
                value_loss_ratio=self.value_loss_ratio,
                callbacks=[validation_callback],
                logger=self.logger,
            )

            # Run validation epoch
            validation_loop.run_epoch(
                dataloader=self.validation_loader,
                epoch_idx=epoch_number,
                train_mode=False,
                progress_bar=True,
            )

            # Get validation metrics
            avg_vloss = validation_callback.get_average_loss()
            avg_accuracy_policy, avg_accuracy_value = (
                validation_callback.get_average_accuracies()
            )

            # Reset callback for next epoch
            validation_callback.reset()

            self.logger.info(
                "LOSS train {} valid {}".format(
                    avg_loss, avg_vloss
                )
            )
            self.logger.info(
                "ACCURACY policy {} value {}".format(
                    avg_accuracy_policy, avg_accuracy_value
                )
            )
            writer.add_scalars(
                "Accuracy",
                {
                    "Policy": avg_accuracy_policy,
                    "Value": avg_accuracy_value,
                },
                epoch_number + 1,
            )

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number + 1,
            )
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                ModelIO.save_model(
                    model=self.model,
                    dir=self.model_dir,
                    id=timestamp,
                    epoch=epoch_number + 1,
                    device=self.device,
                    cloud_storage=self.__cloud_storage,
                )

            # SummaryWriterのイベントをGCSに保存する
            if self.__cloud_storage is not None:
                self.logger.info(
                    "Uploading tensorboard logs to cloud storage"
                )
                self.__cloud_storage.upload_folder_from_local(
                    local_folder=summary_writer_log_dir,
                    cloud_folder="tensorboard",
                )

            epoch_number += 1

        writer.close()
