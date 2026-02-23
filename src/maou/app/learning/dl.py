import abc
import fnmatch
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Literal,
    MutableMapping,
    Optional,
    cast,
)

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import (
    SummaryWriter,  # type: ignore
)
from torchinfo import summary

from maou.app.learning.callbacks import (
    LoggingCallback,
    LRSchedulerStepCallback,
    Stage3LossCallback,
    ValidationCallback,
    ValidationMetrics,
)
from maou.app.learning.compilation import (
    compile_module,
    warmup_compiled_model,
)
from maou.app.learning.dataset import DataSource
from maou.app.learning.model_io import ModelIO
from maou.app.learning.network import (
    BackboneArchitecture,
    Network,
)
from maou.app.learning.setup import (
    DataLoaderFactory,
    DeviceConfig,
    DeviceSetup,
    LossOptimizerFactory,
    ModelComponents,
    ModelFactory,
    SchedulerFactory,
    TrainingSetup,
)
from maou.app.learning.streaming_dataset import (
    StreamingDataSource,
    StreamingKifDataset,
)
from maou.app.learning.training_loop import TrainingLoop
from maou.domain.cloud_storage import CloudStorage

try:
    from torch.optim.lr_scheduler import LRScheduler
except (
    ImportError
):  # pragma: no cover - PyTorch < 2.0 compatibility
    from torch.optim.lr_scheduler import (
        _LRScheduler as LRScheduler,  # type: ignore
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
    model: Network
    scaler: Optional[GradScaler]
    model_architecture: BackboneArchitecture

    @dataclass(kw_only=True, frozen=True)
    class LearningOption:
        datasource: Optional[
            LearningDataSource.DataSourceSpliter
        ]
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
        resume_backbone_from: Optional[Path] = None
        resume_policy_head_from: Optional[Path] = None
        resume_value_head_from: Optional[Path] = None
        freeze_backbone: bool = False
        trainable_layers: Optional[int] = None
        start_epoch: int = 0
        log_dir: Path
        model_dir: Path
        lr_scheduler_name: Optional[str] = None
        input_cache_mode: Literal["file", "memory"] = "file"
        streaming: bool = False
        streaming_train_source: Optional[
            StreamingDataSource
        ] = None
        streaming_val_source: Optional[StreamingDataSource] = (
            None
        )
        tensorboard_histogram_frequency: int = 0
        tensorboard_histogram_modules: (
            tuple[str, ...] | None
        ) = None
        save_split_params: bool = False

    def __init__(
        self,
        *,
        cloud_storage: Optional[CloudStorage] = None,
    ):
        self.__cloud_storage = cloud_storage

    def learn(
        self,
        config: LearningOption,
        *,
        architecture_config: dict[str, Any] | None = None,
    ) -> Dict[str, str]:
        """機械学習を行う.

        Args:
            config: 学習設定．
            architecture_config: アーキテクチャ固有の設定dict．
                ModelFactoryに渡される．
        """
        self.logger.info("start learning")
        learning_result: Dict[str, str] = {}

        if config.streaming:
            device_config, dataloaders, model_components = (
                self._setup_streaming_components(
                    config, architecture_config
                )
            )
        else:
            # 入力とテスト用のデータソース取得
            if config.datasource is None:
                raise ValueError(
                    "datasource is required when streaming=False"
                )
            training_datasource, validation_datasource = (
                config.datasource.train_test_split(
                    test_ratio=config.test_ratio
                )
            )

            device_config, dataloaders, model_components = (
                TrainingSetup.setup_training_components(
                    training_datasource=training_datasource,
                    validation_datasource=validation_datasource,
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
                    architecture_config=architecture_config,
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
        self.policy_loss_ratio = config.policy_loss_ratio
        self.value_loss_ratio = config.value_loss_ratio
        self.log_dir = config.log_dir
        self.epoch = config.epoch
        self.model_dir = config.model_dir
        self.start_epoch = config.start_epoch
        self.freeze_backbone = config.freeze_backbone
        self.trainable_layers = config.trainable_layers
        self.tensorboard_histogram_frequency = (
            config.tensorboard_histogram_frequency
        )
        self.tensorboard_histogram_modules = (
            config.tensorboard_histogram_modules
        )
        self.architecture_config = architecture_config
        self.config = config

        # Load checkpoints before freeze (moved from __train)
        self._load_checkpoints(config)

        # Apply freeze if requested, then recreate optimizer
        trainable_layers = self._resolve_trainable_layers()
        if trainable_layers is not None:
            self._freeze_backbone(trainable_layers)
            # Recreate optimizer with only trainable parameters
            self.optimizer = (
                LossOptimizerFactory.create_optimizer(
                    self.model,
                    config.learning_ratio,
                    config.momentum,
                    optimizer_name=config.optimizer_name,
                    betas=(
                        config.optimizer_beta1,
                        config.optimizer_beta2,
                    ),
                    eps=config.optimizer_eps,
                )
            )
            self.lr_scheduler: Optional[LRScheduler] = (
                SchedulerFactory.create_scheduler(
                    self.optimizer,
                    lr_scheduler_name=config.lr_scheduler_name,
                    max_epochs=config.epoch,
                    steps_per_epoch=len(self.training_loader),
                )
            )
        else:
            self.optimizer = model_components.optimizer
            self.lr_scheduler = model_components.lr_scheduler

        # Generate model tag (includes -tlN suffix when freezing)
        self.model_tag = ModelIO.generate_model_tag(
            self.model,
            self.model_architecture,
            trainable_layers=trainable_layers,
        )

        # Log training configuration summary
        self._log_training_config(
            config, architecture_config, self.model_tag
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

        # Compile after freeze + optimizer setup
        if config.compilation:
            self.logger.info(
                "Compiling model with torch.compile "
                "(dynamic shapes disabled)"
            )
            self.model = cast(
                Network, compile_module(self.model)
            )
            dummy = torch.zeros(
                config.batch_size,
                9,
                9,
                dtype=torch.int64,
                device=self.device,
            )
            warmup_compiled_model(self.model, dummy)
        self.__train()

        train_ds = self.training_loader.dataset
        val_ds = self.validation_loader.dataset
        train_count = (
            len(train_ds)  # type: ignore[arg-type]
            if hasattr(train_ds, "__len__")
            else "streaming"
        )
        val_count = (
            len(val_ds)  # type: ignore[arg-type]
            if hasattr(val_ds, "__len__")
            else "streaming"
        )
        learning_result["Data Samples"] = (
            f"Training: {train_count}, Test: {val_count}"
        )
        learning_result["Option"] = str(config)
        learning_result["Result"] = "Finish"

        return learning_result

    def _setup_streaming_components(
        self,
        config: LearningOption,
        architecture_config: dict[str, Any] | None,
    ) -> tuple[
        DeviceConfig,
        tuple[DataLoader, DataLoader],
        ModelComponents,
    ]:
        """Streaming用の学習コンポーネントをセットアップする．

        Map-style TrainingSetup.setup_training_components() の代わりに，
        StreamingKifDataset + create_streaming_dataloaders() を使用する．

        Args:
            config: 学習設定
            architecture_config: アーキテクチャ固有の設定dict

        Returns:
            (DeviceConfig, (train_loader, val_loader), ModelComponents)

        Raises:
            ValueError: streaming_train_source / streaming_val_source が未設定
        """
        if config.streaming_train_source is None:
            raise ValueError(
                "streaming_train_source is required "
                "when streaming=True"
            )
        if config.streaming_val_source is None:
            raise ValueError(
                "streaming_val_source is required "
                "when streaming=True"
            )

        # Torch config
        if config.detect_anomaly:
            torch.autograd.set_detect_anomaly(
                mode=True, check_nan=True
            )

        # Device setup
        device_config = DeviceSetup.setup_device(
            config.gpu, config.pin_memory
        )

        # Create streaming datasets
        train_dataset = StreamingKifDataset(
            streaming_source=config.streaming_train_source,
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_dataset = StreamingKifDataset(
            streaming_source=config.streaming_val_source,
            batch_size=config.batch_size,
            shuffle=False,
        )

        # Create streaming dataloaders
        n_train_files = len(
            config.streaming_train_source.file_paths
        )
        n_val_files = len(
            config.streaming_val_source.file_paths
        )

        training_loader, validation_loader = (
            DataLoaderFactory.create_streaming_dataloaders(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                dataloader_workers=config.dataloader_workers,
                pin_memory=device_config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                n_train_files=n_train_files,
                n_val_files=n_val_files,
                file_paths=config.streaming_train_source.file_paths,
            )
        )

        # Model creation
        model = ModelFactory.create_shogi_model(
            device_config.device,
            architecture=config.model_architecture,
            architecture_config=architecture_config,
        )

        # Loss functions and optimizer
        loss_fn_policy, loss_fn_value = (
            LossOptimizerFactory.create_loss_functions(
                config.gce_parameter
            )
        )
        optimizer = LossOptimizerFactory.create_optimizer(
            model,
            config.learning_ratio,
            config.momentum,
            optimizer_name=config.optimizer_name,
            betas=(
                config.optimizer_beta1,
                config.optimizer_beta2,
            ),
            eps=config.optimizer_eps,
        )
        lr_scheduler = SchedulerFactory.create_scheduler(
            optimizer,
            lr_scheduler_name=config.lr_scheduler_name,
            max_epochs=config.epoch,
            steps_per_epoch=len(training_loader),
        )

        model_components = ModelComponents(
            model=model,
            loss_fn_policy=loss_fn_policy,
            loss_fn_value=loss_fn_value,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        self.logger.info(
            "Streaming training components setup completed"
        )

        return (
            device_config,
            (training_loader, validation_loader),
            model_components,
        )

    def __train_one_epoch(
        self, epoch_index: int, tb_writer: SummaryWriter
    ) -> float:
        # Create logging callback
        logging_callback = LoggingCallback(
            writer=tb_writer,
            dataloader_length=len(self.training_loader),
            logger=self.logger,
        )
        loss_callback = Stage3LossCallback()

        # Build callback list
        callbacks: list[
            LoggingCallback
            | LRSchedulerStepCallback
            | Stage3LossCallback
        ] = [logging_callback, loss_callback]
        if self.lr_scheduler is not None:
            callbacks.append(
                LRSchedulerStepCallback(self.lr_scheduler)
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
            callbacks=callbacks,
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
        model_tag = self.model_tag
        summary_writer_log_dir = (
            self.log_dir
            / f"{model_tag}_training_log_{timestamp}"
        )
        self.logger.info(
            "TensorBoard log directory: %s (model: %s)",
            summary_writer_log_dir,
            model_tag,
        )
        writer = SummaryWriter(summary_writer_log_dir)
        epoch_number = 0

        EPOCHS = self.epoch

        best_vloss = 1_000_000.0
        last_metrics: Optional[ValidationMetrics] = None

        # Checkpoint loading and freeze are handled in learn()
        # before optimizer creation.

        # start epoch設定
        epoch_number = self.start_epoch

        # 学習率スケジューラをstart_epochのステップ分だけ進める
        if (
            self.lr_scheduler is not None
            and self.start_epoch > 0
        ):
            steps_to_advance = self.start_epoch * len(
                self.training_loader
            )
            self.logger.info(
                "Advancing LR scheduler by %d steps (epoch %d)",
                steps_to_advance,
                self.start_epoch,
            )
            for _ in range(steps_to_advance):
                self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"LR scheduler advanced: current learning rate = {current_lr}"
            )

        for _ in range(epoch_number, EPOCHS):
            self.logger.info(
                "EPOCH {}:".format(epoch_number + 1)
            )

            # Streaming IterableDatasetのエポックシード更新
            for loader in (
                self.training_loader,
                self.validation_loader,
            ):
                ds = loader.dataset
                if hasattr(ds, "set_epoch"):
                    ds.set_epoch(epoch_number)

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
            validation_loss_callback = Stage3LossCallback()

            # Create validation training loop
            validation_loop = TrainingLoop(
                model=self.model,
                device=self.device,
                optimizer=self.optimizer,
                loss_fn_policy=self.loss_fn_policy,
                loss_fn_value=self.loss_fn_value,
                policy_loss_ratio=self.policy_loss_ratio,
                value_loss_ratio=self.value_loss_ratio,
                callbacks=[
                    validation_callback,
                    validation_loss_callback,
                ],
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
            last_metrics = metrics

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

            # LR scheduler is stepped per-batch via LRSchedulerStepCallback

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
                    architecture_config=self.architecture_config,
                    hand_projection_dim=self.model.hand_projection_dim,
                    save_split_params=self.config.save_split_params,
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

        # Record hyperparameters for TensorBoard HParams dashboard
        if last_metrics is not None:
            self._log_hparams(
                writer=writer,
                model_tag=model_tag,
                best_vloss=best_vloss,
                final_metrics=last_metrics,
            )

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
                "Policy F1 Score": metrics.policy_f1_score,
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

    def _log_hparams(
        self,
        *,
        writer: SummaryWriter,
        model_tag: str,
        best_vloss: float,
        final_metrics: ValidationMetrics,
    ) -> None:
        """Record hyperparameters and final metrics to TensorBoard."""
        config = self.config
        hparam_dict: dict[str, bool | int | float | str] = {
            # Freeze
            "trainable_layers": config.trainable_layers
            if config.trainable_layers is not None
            else -1,
            "freeze_backbone": config.freeze_backbone,
            # Model
            "model_architecture": config.model_architecture,
            "model_tag": model_tag,
            # Training
            "learning_ratio": config.learning_ratio,
            "optimizer_name": config.optimizer_name,
            "lr_scheduler_name": config.lr_scheduler_name
            or "none",
            "batch_size": config.batch_size,
            "epoch": config.epoch,
            # Loss
            "gce_parameter": config.gce_parameter,
            "policy_loss_ratio": config.policy_loss_ratio,
            "value_loss_ratio": config.value_loss_ratio,
        }

        # ViT固有パラメータ
        if (
            self.architecture_config
            and config.model_architecture == "vit"
        ):
            for key in ("embed_dim", "num_layers", "num_heads"):
                if key in self.architecture_config:
                    hparam_dict[f"vit_{key}"] = (
                        self.architecture_config[key]
                    )

        metric_dict: dict[str, float] = {
            "hparam/best_vloss": best_vloss,
            "hparam/policy_top5_accuracy": final_metrics.policy_top5_accuracy,
            "hparam/policy_cross_entropy": final_metrics.policy_cross_entropy,
            "hparam/value_brier_score": final_metrics.value_brier_score,
        }

        writer.add_hparams(
            hparam_dict,
            metric_dict,
            run_name=".",
        )

    def _log_training_config(
        self,
        config: LearningOption,
        architecture_config: dict[str, Any] | None,
        model_tag: str,
    ) -> None:
        """Log training configuration summary at learning start."""
        # Model description
        model_desc = model_tag
        if (
            architecture_config
            and config.model_architecture == "vit"
        ):
            vit_parts = []
            for key in ("embed_dim", "num_layers", "num_heads"):
                if key in architecture_config:
                    vit_parts.append(
                        f"{key}={architecture_config[key]}"
                    )
            if vit_parts:
                model_desc += f" ({', '.join(vit_parts)})"

        # Freeze description
        resolved = self._resolve_trainable_layers()
        if resolved is not None:
            freeze_desc = f"trainable_layers={resolved}"
        else:
            freeze_desc = "none"

        # Scheduler description
        scheduler_desc = config.lr_scheduler_name or "none"

        lines = [
            "=== Training Configuration ===",
            f"Model: {model_desc}",
            f"Freeze: {freeze_desc}",
            f"Optimizer: {config.optimizer_name} "
            f"(lr={config.learning_ratio}, "
            f"beta1={config.optimizer_beta1}, "
            f"beta2={config.optimizer_beta2})",
            f"Scheduler: {scheduler_desc}",
            f"Batch: {config.batch_size}, "
            f"Epoch: {config.epoch}, "
            f"Workers: {config.dataloader_workers}",
            f"Loss: GCE(q={config.gce_parameter}), "
            f"policy_ratio={config.policy_loss_ratio}, "
            f"value_ratio={config.value_loss_ratio}",
            f"Data: preprocessing, "
            f"cache={config.input_cache_mode}",
            "==============================",
        ]
        self.logger.info("\n".join(lines))

    def _load_component_state_dict(
        self, state_dict: MutableMapping[str, torch.Tensor]
    ) -> None:
        """Load component state_dict with partial loading support.

        Args:
            state_dict: Component state_dict (backbone，policy head，or value head)
        """
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
            # strict=Falseで部分読み込みを許可
            incompatible_keys = self.model.load_state_dict(
                state_dict, strict=False
            )
            if incompatible_keys.missing_keys:
                self.logger.info(
                    f"Missing keys (expected for partial load): {incompatible_keys.missing_keys}"
                )
            if incompatible_keys.unexpected_keys:
                self.logger.warning(
                    f"Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}"
                )
        except RuntimeError as exc:
            self.logger.error(
                "Failed to load component state_dict into model: %s",
                exc,
            )
            raise

    def _load_checkpoints(self, config: LearningOption) -> None:
        """Load model state from checkpoint or component files.

        Args:
            config: Learning configuration with resume paths.
        """
        if config.resume_backbone_from is not None:
            backbone_dict = ModelIO.load_backbone(
                config.resume_backbone_from, self.device
            )
            self._load_component_state_dict(backbone_dict)

        if config.resume_policy_head_from is not None:
            policy_dict = ModelIO.load_policy_head(
                config.resume_policy_head_from, self.device
            )
            self._load_component_state_dict(policy_dict)

        if config.resume_value_head_from is not None:
            value_dict = ModelIO.load_value_head(
                config.resume_value_head_from, self.device
            )
            self._load_component_state_dict(value_dict)

    def _resolve_trainable_layers(self) -> Optional[int]:
        """Resolve effective trainable_layers from config options.

        Returns:
            int if freezing should be applied, None otherwise.
        """
        if (
            self.trainable_layers is not None
            and self.freeze_backbone
        ):
            self.logger.warning(
                "Both freeze_backbone and trainable_layers specified. "
                "Using trainable_layers=%d.",
                self.trainable_layers,
            )
            return self.trainable_layers

        if self.trainable_layers is not None:
            return self.trainable_layers

        if self.freeze_backbone:
            return 0

        return None

    def _freeze_backbone(
        self, trainable_layers: int = 0
    ) -> None:
        """Freeze backbone parameters with optional partial unfreezing.

        Args:
            trainable_layers: Number of trailing backbone groups
                to keep trainable.
                0 = freeze all backbone groups (backward-compatible).
        """
        frozen_count = self.model.freeze_except_last_n(
            trainable_layers
        )

        # Also freeze _hand_projection (Network-level attribute)
        if hasattr(self.model, "_hand_projection"):
            for (
                param
            ) in self.model._hand_projection.parameters():
                param.requires_grad = False
                frozen_count += 1

        self.logger.info(
            f"Frozen {frozen_count} backbone parameters "
            f"(trainable_layers={trainable_layers}). "
            f"Policy and value heads remain trainable."
        )
