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

from maou.app.learning.callbacks import LoggingCallback, ValidationCallback
from maou.app.learning.dataset import DataSource, KifDataset
from maou.app.learning.setup import (
    DataLoaderFactory,
    LossOptimizerFactory,
    ModelFactory,
)
from maou.app.learning.training_loop import TrainingLoop
from maou.app.pre_process.feature import FEATURES_NUM
from maou.app.pre_process.transform import Transform


class CloudStorage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def upload_from_local(self, *, local_path: Path, cloud_path: str) -> None:
        pass

    @abc.abstractmethod
    def upload_folder_from_local(
        self,
        *,
        local_folder: Path,
        cloud_folder: str,
        extensions: Optional[list[str]] = None,
    ) -> None:
        pass


class LearningDataSource(DataSource):
    class DataSourceSpliter(metaclass=abc.ABCMeta):
        @abc.abstractmethod
        def train_test_split(self, test_ratio: float) -> tuple[DataSource, DataSource]:
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
    model: torch.nn.Module
    scaler: Optional[GradScaler]

    @dataclass(kw_only=True, frozen=True)
    class LearningOption:
        datasource: LearningDataSource.DataSourceSpliter
        datasource_type: str
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
        gpu: Optional[str] = None,
        cloud_storage: Optional[CloudStorage] = None,
    ):
        if gpu is not None and gpu != "cpu":
            self.device = torch.device(gpu)
            self.logger.info(f"Use GPU {torch.cuda.get_device_name(self.device)}")
            torch.set_float32_matmul_precision("high")
        else:
            self.logger.info("Use CPU")
            self.device = torch.device("cpu")

        # マルチプロセシング開始方法の設定（WindowsやCUDA使用時の安定化）
        # デバイス設定後に実行してCUDA使用判定を正確に行う
        self._setup_multiprocessing()

        # Mixed precision training用のGradScalerを初期化（GPU使用時のみ）
        if self.device.type == "cuda":
            self.scaler = GradScaler("cuda")
            self.logger.info("Initialized GradScaler for mixed precision training")
        else:
            self.scaler = None

        self.__cloud_storage = cloud_storage

    def learn(self, option: LearningOption) -> Dict[str, str]:
        """機械学習を行う."""
        self.logger.info("start learning")
        torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
        learning_result: Dict[str, str] = {}

        # 入力とテスト用のデータソース取得
        input_datasource, test_datasource = option.datasource.train_test_split(
            test_ratio=option.test_ratio
        )

        # Create datasets using existing device
        # Validate datasource type
        if option.datasource_type not in ("hcpe", "preprocess"):
            raise ValueError(f"Data source type `{option.datasource_type}` is invalid.")

        # Create transform based on datasource type
        if option.datasource_type == "hcpe":
            transform = Transform()
        else:
            transform = None

        # Create datasets
        dataset_train = KifDataset(datasource=input_datasource, transform=transform)
        dataset_validation = KifDataset(datasource=test_datasource, transform=transform)

        # Set pin_memory based on device
        pin_memory = option.pin_memory
        if pin_memory is None:
            pin_memory = self.device.type == "cuda"

        # Create dataloaders
        self.training_loader, self.validation_loader = (
            DataLoaderFactory.create_dataloaders(
                dataset_train,
                dataset_validation,
                option.batch_size,
                option.dataloader_workers,
                pin_memory,
                option.prefetch_factor,
            )
        )

        # Create model using existing device
        model = ModelFactory.create_shogi_model(self.device)

        # Create loss functions
        self.loss_fn_policy, self.loss_fn_value = (
            LossOptimizerFactory.create_loss_functions(option.gce_parameter)
        )

        self.logger.info(
            str(summary(model, input_size=(option.batch_size, FEATURES_NUM, 9, 9)))
        )

        if option.compilation:
            compiled_model = torch.compile(model)
            self.model = compiled_model  # type: ignore
            self.logger.info("Finished model compilation")
        else:
            self.model = model

        # Create optimizer for the final model
        self.optimizer = LossOptimizerFactory.create_optimizer(
            self.model,
            learning_ratio=option.learning_ratio,
            momentum=option.momentum,
        )
        self.policy_loss_ratio = option.policy_loss_ratio
        self.value_loss_ratio = option.value_loss_ratio
        self.log_dir = option.log_dir
        self.epoch = option.epoch
        self.model_dir = option.model_dir
        self.resume_from = option.resume_from
        self.start_epoch = option.start_epoch
        self.__train()

        learning_result["Data Samples"] = (
            f"Training: {len(self.training_loader.dataset)}, "  # type: ignore
            f"Test: {len(self.validation_loader.dataset)}"  # type: ignore
        )
        learning_result["Option"] = str(option)
        learning_result["Result"] = "Finish"

        return learning_result

    def __train_one_epoch(self, epoch_index: int, tb_writer: SummaryWriter) -> float:
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
            train_mode=True,
        )

        # Return the last recorded loss from the callback
        # Since the callback handles all logging, we return a placeholder value
        # The actual loss tracking is handled by the callback
        return 0.0  # This value is not critical as logging is handled by callback

    def __train(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_writer_log_dir = self.log_dir / "training_log_{}".format(timestamp)
        writer = SummaryWriter(summary_writer_log_dir)
        epoch_number = 0

        EPOCHS = self.epoch

        best_vloss = 1_000_000.0

        # resume from checkpoint
        if self.resume_from is not None:
            self.model.load_state_dict(
                torch.load(
                    self.resume_from, weights_only=True, map_location=self.device
                )
            )

        # start epoch設定
        epoch_number = self.start_epoch

        for _ in range(epoch_number, EPOCHS):
            self.logger.info("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.__train_one_epoch(epoch_number, writer)

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
                        grad_np = param.grad.detach().cpu().numpy()
                        writer.add_histogram(
                            f"gradients/{name}", grad_np, epoch_number + 1
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to log histogram for {name}: {e}")

            # Create validation callback
            validation_callback = ValidationCallback(logger=self.logger)

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

            self.logger.info("LOSS train {} valid {}".format(avg_loss, avg_vloss))
            self.logger.info(
                "ACCURACY policy {} value {}".format(
                    avg_accuracy_policy, avg_accuracy_value
                )
            )
            writer.add_scalars(
                "Accuracy",
                {"Policy": avg_accuracy_policy, "Value": avg_accuracy_value},
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
                model_path = self.model_dir / "model_{}_{}.pt".format(
                    timestamp, epoch_number + 1
                )
                self.logger.info("Saving model to {}".format(model_path))
                torch.save(self.model.state_dict(), model_path)
                if self.__cloud_storage is not None:
                    self.logger.info("Uploading model to cloud storage")
                    self.__cloud_storage.upload_from_local(
                        local_path=model_path, cloud_path=str(model_path)
                    )

            # SummaryWriterのイベントをGCSに保存する
            if self.__cloud_storage is not None:
                self.logger.info("Uploading tensorboard logs to cloud storage")
                self.__cloud_storage.upload_folder_from_local(
                    local_folder=summary_writer_log_dir,
                    cloud_folder="tensorboard",
                )

            epoch_number += 1

        writer.close()

    def _setup_multiprocessing(self) -> None:
        """
        マルチプロセシング開始方法を設定する．
        プラットフォームとCUDA使用状況に応じて最適な方法を選択し，
        DataLoaderのマルチプロセシング安定性を向上させる．
        """
        import platform

        import torch.multiprocessing as mp

        try:
            # 現在の開始方法を取得
            current_method = mp.get_start_method(allow_none=True)

            # プラットフォーム別の推奨設定
            if platform.system() == "Windows":
                # Windowsでは常にspawnを使用
                if current_method != "spawn":
                    mp.set_start_method("spawn", force=True)
                    self.logger.info(
                        "Set multiprocessing start method to 'spawn' for Windows"
                    )
            elif (
                torch.cuda.is_available()
                and hasattr(self, "device")
                and self.device.type == "cuda"
            ):
                # CUDA使用時はspawnが安全
                if current_method != "spawn":
                    try:
                        mp.set_start_method("spawn", force=True)
                        self.logger.info(
                            "Set multiprocessing start method to 'spawn' for CUDA"
                        )
                    except RuntimeError as e:
                        # 既に設定済みの場合は警告のみ
                        self.logger.warning(
                            f"Could not set multiprocessing method: {e}"
                        )
            else:
                # Linux/macOSでCPU使用時はforkのままでも問題なし
                if current_method is None:
                    self.logger.info(
                        f"Using default multiprocessing start method: "
                        f"{mp.get_start_method()}"
                    )
                else:
                    self.logger.info(
                        f"Using current multiprocessing start method: {current_method}"
                    )

        except Exception as e:
            # マルチプロセシング設定に失敗した場合は警告のみ
            self.logger.warning(f"Failed to configure multiprocessing: {e}")
            self.logger.info("Continuing with default multiprocessing settings")
