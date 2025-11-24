import abc
import fnmatch
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, MutableMapping, Optional, cast

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import (
    SummaryWriter,  # type: ignore
)
from torchinfo import summary

from maou.app.learning.callbacks import (
    LoggingCallback,
    ValidationCallback,
    ValidationMetrics,
)
from maou.app.learning.dataset import DataSource
from maou.app.learning.model_io import ModelIO
from maou.app.learning.network import (
    BackboneArchitecture,
    Network,
)
from maou.app.learning.compilation import compile_module
from maou.app.learning.setup import TrainingSetup
from maou.app.learning.training_loop import TrainingLoop
from maou.domain.cloud_storage import CloudStorage

try:
    from torch.optim.lr_scheduler import LRScheduler
except (
    ImportError
):  # pragma: no cover - PyTorch < 2.0 compatibility
    from torch.optim.lr_scheduler import (  # type: ignore
        _LRScheduler as LRScheduler,
    )


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
    model_architecture: BackboneArchitecture

    @dataclass(kw_only=True, frozen=True)
    class LearningOption:
        datasource: LearningDataSource.DataSourceSpliter
        datasource_type: str
        gpu: Optional[str] = None
        model_architecture: BackboneArchitecture = "resnet"
        compilation: bool
        test_ratio: float
        epoch: int
        batch_size: int
        dataloader_workers: int
        pin_memory: bool
        prefetch_factor: int
        cache_transforms: bool
        gce_parameter: float
        policy_loss_ratio: float
        value_loss_ratio: float
        learning_ratio: float
        momentum: float
        optimizer_name: str
        optimizer_beta1: float
        optimizer_beta2: float
        optimizer_eps: float
        detect_anomaly: bool = False
        resume_from: Optional[Path] = None
        start_epoch: int = 0
        log_dir: Path
        model_dir: Path
        lr_scheduler_name: Optional[str] = None
        input_cache_mode: Literal["mmap", "memory"] = "mmap"
        tensorboard_histogram_frequency: int = 0
        tensorboard_histogram_modules: (
            tuple[str, ...] | None
        ) = None

    def __init__(
        self,
        *,
        cloud_storage: Optional[CloudStorage] = None,
    ):
        self.__cloud_storage = cloud_storage

    @staticmethod
    def _format_parameter_count(parameter_count: int) -> str:
        """Return a compact, human-friendly parameter count label."""

        def _format(value: float) -> str:
            formatted = f"{value:.1f}"
            if formatted.endswith(".0"):
                return formatted[:-2]
            return formatted

        if parameter_count >= 1_000_000:
            return f"{_format(parameter_count / 1_000_000)}m"
        if parameter_count >= 1_000:
            return f"{_format(parameter_count / 1_000)}k"
        return str(parameter_count)

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
                cache_transforms=config.cache_transforms,
                gpu=config.gpu,
                model_architecture=config.model_architecture,
                batch_size=config.batch_size,
                dataloader_workers=config.dataloader_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                gce_parameter=config.gce_parameter,
                learning_ratio=config.learning_ratio,
                momentum=config.momentum,
                optimizer_name=config.optimizer_name,
                optimizer_beta1=config.optimizer_beta1,
                optimizer_beta2=config.optimizer_beta2,
                optimizer_eps=config.optimizer_eps,
                lr_scheduler_name=config.lr_scheduler_name,
                max_epochs=config.epoch,
                detect_anomaly=config.detect_anomaly,
            )
        )

        self.device = device_config.device
        self.training_loader, self.validation_loader = (
            dataloaders
        )
        self.model = model_components.model
        self.model_architecture = config.model_architecture
        self.loss_fn_policy = model_components.loss_fn_policy
        self.loss_fn_value = model_components.loss_fn_value
        self.optimizer = model_components.optimizer
        self.lr_scheduler: Optional[LRScheduler] = (
            model_components.lr_scheduler
        )
        self.policy_loss_ratio = config.policy_loss_ratio
        self.value_loss_ratio = config.value_loss_ratio
        self.log_dir = config.log_dir
        self.epoch = config.epoch
        self.model_dir = config.model_dir
        self.resume_from = config.resume_from
        self.start_epoch = config.start_epoch
        self.tensorboard_histogram_frequency = (
            config.tensorboard_histogram_frequency
        )
        self.tensorboard_histogram_modules = (
            config.tensorboard_histogram_modules
        )
        summary(
            self.model,
            input_size=(
                config.batch_size,
                9,
                9,
            ),
            dtypes=[torch.int64],
        )

        if config.compilation:
            self.logger.info(
                "Compiling model with torch.compile (dynamic shapes disabled)"
            )
            self.model = cast(
                Network, compile_module(self.model)
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
        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )
        parameter_label = self._format_parameter_count(
            parameter_count
        )
        model_tag = (
            f"{self.model_architecture}-{parameter_label}"
        )
        summary_writer_log_dir = (
            self.log_dir
            / f"{model_tag}_training_log_{timestamp}"
        )
        self.logger.info(
            "TensorBoard log directory: %s (parameters ≈ %s)",
            summary_writer_log_dir,
            parameter_label,
        )
        writer = SummaryWriter(summary_writer_log_dir)
        epoch_number = 0

        EPOCHS = self.epoch

        best_vloss = 1_000_000.0

        # resume from checkpoint
        if self.resume_from is not None:
            state_dict: MutableMapping[str, torch.Tensor] = (
                torch.load(
                    self.resume_from,
                    weights_only=True,
                    map_location=self.device,
                )
            )

            self._load_resume_state_dict(state_dict)

        # start epoch設定
        epoch_number = self.start_epoch

        # 学習率スケジューラをstart_epoch分だけ進める
        if (
            self.lr_scheduler is not None
            and self.start_epoch > 0
        ):
            self.logger.info(
                f"Advancing LR scheduler to epoch {self.start_epoch}"
            )
            for _ in range(self.start_epoch):
                self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"LR scheduler advanced: current learning rate = {current_lr}"
            )

        for _ in range(epoch_number, EPOCHS):
            self.logger.info(
                "EPOCH {}:".format(epoch_number + 1)
            )

            avg_loss = self.__train_one_epoch(
                epoch_number, writer
            )

            self._log_parameter_histograms(
                writer=writer, epoch_number=epoch_number
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
            metrics = validation_callback.get_average_metrics()

            # Reset callback for next epoch
            validation_callback.reset()

            self.logger.info(
                "LOSS train {} valid {}".format(
                    avg_loss, avg_vloss
                )
            )
            self.logger.info(
                (
                    "METRICS policy_cross_entropy {} value_brier_score {} "
                    "policy_top5_accuracy {} value_high_confidence_rate {}"
                ).format(
                    metrics.policy_cross_entropy,
                    metrics.value_brier_score,
                    metrics.policy_top5_accuracy,
                    metrics.value_high_confidence_rate,
                )
            )
            current_lr = self.optimizer.param_groups[0]["lr"]
            self._log_epoch_metrics(
                writer=writer,
                metrics=metrics,
                avg_loss=avg_loss,
                avg_vloss=avg_vloss,
                epoch_number=epoch_number,
                learning_rate=current_lr,
            )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                ModelIO.save_model(
                    trained_model=self.model,
                    dir=self.model_dir,
                    id=timestamp,
                    epoch=epoch_number + 1,
                    device=self.device,
                    architecture=self.model_architecture,
                    cloud_storage=self.__cloud_storage,
                )

            # SummaryWriterのイベントをGCSに保存する
            if self.__cloud_storage is not None:
                self.logger.info(
                    "Uploading tensorboard logs to cloud storage"
                )
                cloud_tensorboard_folder = (
                    f"tensorboard/{summary_writer_log_dir.name}"
                )
                self.__cloud_storage.upload_folder_from_local(
                    local_folder=summary_writer_log_dir,
                    cloud_folder=cloud_tensorboard_folder,
                )

            epoch_number += 1

        writer.close()

    def _log_parameter_histograms(
        self, writer: SummaryWriter, epoch_number: int
    ) -> None:
        """Optionally log model parameter and gradient histograms."""

        frequency = getattr(
            self, "tensorboard_histogram_frequency", 0
        )
        if frequency <= 0:
            return

        if (epoch_number + 1) % frequency != 0:
            return

        module_filters = getattr(
            self, "tensorboard_histogram_modules", None
        )

        for name, param in self.model.named_parameters():
            if module_filters is not None and not any(
                fnmatch.fnmatch(name, pattern)
                for pattern in module_filters
            ):
                continue

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
                        f"gradients/{name}",
                        grad_np,
                        epoch_number + 1,
                    )
            except (
                Exception
            ) as exc:  # pragma: no cover - log-only path
                self.logger.warning(
                    "Failed to log histogram for %s: %s",
                    name,
                    exc,
                )

    def _log_epoch_metrics(
        self,
        *,
        writer: SummaryWriter,
        metrics: ValidationMetrics,
        avg_loss: float,
        avg_vloss: float,
        epoch_number: int,
        learning_rate: float,
    ) -> None:
        """Log scalar metrics that should always be emitted."""

        writer.add_scalars(
            "Validation Loss Metrics",
            {
                "Policy Cross Entropy": metrics.policy_cross_entropy,
                "Value Brier Score": metrics.value_brier_score,
            },
            epoch_number + 1,
        )
        writer.add_scalars(
            "Validation Accuracy Metrics",
            {
                "Policy Top-5 Accuracy": metrics.policy_top5_accuracy,
                "Value ≥0.8 Precision": metrics.value_high_confidence_rate,
            },
            epoch_number + 1,
        )
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.add_scalar(
            "Learning Rate",
            learning_rate,
            epoch_number + 1,
        )

    def _load_resume_state_dict(
        self, state_dict: MutableMapping[str, torch.Tensor]
    ) -> None:
        """Load a checkpoint while remaining compatible with compiled models."""

        if hasattr(self.model, "_orig_mod"):
            needs_prefix = any(
                not key.startswith("_orig_mod.")
                for key in state_dict.keys()
            )
            if needs_prefix:
                state_dict = {
                    f"_orig_mod.{key}": value
                    for key, value in state_dict.items()
                }

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            self.logger.error(
                "Failed to load checkpoint into model: %s",
                exc,
            )
            raise
