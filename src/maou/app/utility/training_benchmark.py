import json
import logging
from dataclasses import dataclass, replace
from typing import Any, Optional, cast

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from maou.app.learning.callbacks import (
    ResourceMonitoringCallback,
    TimingCallback,
)
from maou.app.learning.compilation import compile_module
from maou.app.learning.dl import LearningDataSource
from maou.app.learning.network import (
    BackboneArchitecture,
    Network,
)
from maou.app.learning.resource_monitor import ResourceUsage
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
from maou.app.learning.stage_component_factory import (
    StageComponentFactory,
)
from maou.app.learning.streaming_dataset import (
    StreamingDataSource,
    StreamingKifDataset,
)
from maou.app.learning.training_loop import TrainingLoop


@dataclass(frozen=True)
class BenchmarkResult:
    """単一エポックベンチマーク結果を格納するデータクラス．"""

    # 全体的なタイミング
    total_epoch_time: float
    average_batch_time: float
    actual_average_batch_time: float
    total_batches: int

    # ウォームアップ情報
    warmup_time: float
    warmup_batches: int
    measured_time: float
    measured_batches: int

    # 詳細なタイミング分析
    data_loading_time: float
    gpu_transfer_time: float
    forward_pass_time: float
    loss_computation_time: float
    backward_pass_time: float
    optimizer_step_time: float

    # 損失情報
    final_loss: float
    average_loss: float

    # パフォーマンス指標
    samples_per_second: float
    batches_per_second: float

    # リソース使用率情報
    resource_usage: Optional[ResourceUsage] = None

    data_load_method: str = "map-style"

    def to_dict(self) -> dict[str, object]:
        """ベンチマーク結果を辞書形式で返す．"""
        result = {
            "total_epoch_time": self.total_epoch_time,
            "average_batch_time": self.average_batch_time,
            "actual_average_batch_time": self.actual_average_batch_time,
            "total_batches": float(self.total_batches),
            "warmup_time": self.warmup_time,
            "warmup_batches": float(self.warmup_batches),
            "measured_time": self.measured_time,
            "measured_batches": float(self.measured_batches),
            "data_loading_time": self.data_loading_time,
            "gpu_transfer_time": self.gpu_transfer_time,
            "forward_pass_time": self.forward_pass_time,
            "loss_computation_time": self.loss_computation_time,
            "backward_pass_time": self.backward_pass_time,
            "optimizer_step_time": self.optimizer_step_time,
            "final_loss": self.final_loss,
            "average_loss": self.average_loss,
            "samples_per_second": self.samples_per_second,
            "batches_per_second": self.batches_per_second,
            "data_load_method": self.data_load_method,
        }

        # リソース使用率情報があれば追加
        if self.resource_usage is not None:
            resource_dict = self.resource_usage.to_dict()
            # float型に変換できるもののみ追加
            for key, value in resource_dict.items():
                if value is not None:
                    result[f"resource_{key}"] = float(value)

        return result


class SingleEpochBenchmark:
    """Single epoch benchmark for training performance measurement.

    Args:
        model: 学習対象のネットワークモデル
        device: 学習に使用するデバイス
        optimizer: オプティマイザ
        loss_fn_policy: 方策損失関数
        loss_fn_value: 価値損失関数
        policy_loss_ratio: 方策損失の重み係数
        value_loss_ratio: 価値損失の重み係数
        enable_resource_monitoring: リソース監視を有効にするか
        training_loop_class: 使用するTrainingLoopクラス（デフォルト: TrainingLoop）
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        model: Network,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn_policy: torch.nn.Module,
        loss_fn_value: torch.nn.Module,
        policy_loss_ratio: float,
        value_loss_ratio: float,
        enable_resource_monitoring: bool = False,
        training_loop_class: type[TrainingLoop] = TrainingLoop,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn_policy = loss_fn_policy
        self.loss_fn_value = loss_fn_value
        self.policy_loss_ratio = policy_loss_ratio
        self.value_loss_ratio = value_loss_ratio
        self.enable_resource_monitoring = (
            enable_resource_monitoring
        )
        self.training_loop_class = training_loop_class

        # Mixed precision training用のGradScalerを初期化（GPU使用時のみ）
        if self.device.type == "cuda":
            self.scaler: Optional[GradScaler] = GradScaler(
                "cuda"
            )
        else:
            self.scaler = None

    def benchmark_epoch(
        self,
        dataloader: DataLoader,
        *,
        warmup_batches: int = 5,
        max_batches: Optional[int] = None,
        enable_profiling: bool = False,
    ) -> BenchmarkResult:
        """
        単一エポックのベンチマークを実行する．

        Args:
            dataloader: 学習用データローダー
            warmup_batches: ウォームアップバッチ数（タイミング測定から除外）
            max_batches: 最大処理バッチ数（Noneの場合は全バッチ）
            enable_profiling: PyTorchプロファイラを使用するか

        Returns:
            詳細なタイミング情報を含むベンチマーク結果
        """
        self.logger.info("Starting single epoch benchmark")

        # Create timing callback
        timing_callback = TimingCallback(
            warmup_batches=warmup_batches
        )

        # Create callbacks list
        callbacks = [timing_callback]

        # Add resource monitoring callback if enabled
        resource_callback = None
        if self.enable_resource_monitoring:
            resource_callback = ResourceMonitoringCallback(
                device=self.device,
                logger=self.logger,
            )
            callbacks.append(
                cast("TimingCallback", resource_callback)
            )

        # Create training loop
        training_loop = self.training_loop_class(
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

        # Run training epoch with timing
        training_loop.run_epoch(
            dataloader=dataloader,
            epoch_idx=0,  # Epoch index for benchmarking
            max_batches=max_batches,
            enable_profiling=enable_profiling,
            progress_bar=True,
            train_mode=True,
        )

        # Get timing statistics
        timing_stats = timing_callback.get_timing_statistics()
        # Use measured batches + warmup for total count
        actual_batches = (
            timing_callback.measured_batches + warmup_batches
        )
        performance_metrics = (
            timing_callback.get_performance_metrics(
                actual_batches
            )
        )
        loss_metrics = timing_callback.get_loss_metrics(
            actual_batches
        )

        # Get resource usage if monitoring was enabled
        resource_usage = None
        if resource_callback is not None:
            resource_usage = (
                resource_callback.get_resource_usage()
            )

        # Log summary
        self.logger.info(
            f"Benchmark completed: {performance_metrics['total_batches']} batches in "
            f"{performance_metrics['total_epoch_time']:.2f}s"
        )
        self.logger.info(
            f"Average batch time: {timing_stats['average_batch_time']:.4f}s"
        )
        self.logger.info(
            f"Samples per second: {performance_metrics['samples_per_second']:.1f}"
        )

        return BenchmarkResult(
            total_epoch_time=performance_metrics[
                "total_epoch_time"
            ],
            average_batch_time=timing_stats[
                "average_batch_time"
            ],
            actual_average_batch_time=performance_metrics[
                "actual_average_batch_time"
            ],
            total_batches=int(
                performance_metrics["total_batches"]
            ),
            warmup_time=performance_metrics["warmup_time"],
            warmup_batches=warmup_batches,
            measured_time=performance_metrics["measured_time"],
            measured_batches=timing_callback.measured_batches,
            data_loading_time=timing_stats["data_loading_time"],
            gpu_transfer_time=timing_stats["gpu_transfer_time"],
            forward_pass_time=timing_stats["forward_pass_time"],
            loss_computation_time=timing_stats[
                "loss_computation_time"
            ],
            backward_pass_time=timing_stats[
                "backward_pass_time"
            ],
            optimizer_step_time=timing_stats[
                "optimizer_step_time"
            ],
            final_loss=loss_metrics[
                "total_loss"
            ],  # Use total as final for now
            average_loss=loss_metrics["average_loss"],
            samples_per_second=performance_metrics[
                "samples_per_second"
            ],
            batches_per_second=performance_metrics[
                "batches_per_second"
            ],
            resource_usage=resource_usage,
        )

    def benchmark_validation(
        self,
        dataloader: DataLoader,
        *,
        max_batches: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        バリデーション（推論のみ）のベンチマークを実行する．

        Args:
            dataloader: バリデーション用データローダー
            max_batches: 最大処理バッチ数（Noneの場合は全バッチ）

        Returns:
            推論処理のタイミング情報を含むベンチマーク結果
        """
        self.logger.info("Starting validation benchmark")

        # Create timing callback for validation (no warmup for validation)
        timing_callback = TimingCallback(warmup_batches=0)

        # Create callbacks list
        callbacks = [timing_callback]

        # Add resource monitoring callback if enabled
        resource_callback = None
        if self.enable_resource_monitoring:
            resource_callback = ResourceMonitoringCallback(
                device=self.device,
                logger=self.logger,
            )
            callbacks.append(
                cast("TimingCallback", resource_callback)
            )

        # Create training loop in evaluation mode
        training_loop = self.training_loop_class(
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

        # Run validation epoch with timing
        training_loop.run_epoch(
            dataloader=dataloader,
            epoch_idx=0,  # Epoch index for benchmarking
            max_batches=max_batches,
            enable_profiling=False,  # No profiling for validation
            progress_bar=True,
            train_mode=False,  # Evaluation mode
        )

        # Get timing statistics
        timing_stats = timing_callback.get_timing_statistics()
        # For validation, use all processed batches (no warmup)
        actual_batches = timing_callback.measured_batches
        performance_metrics = (
            timing_callback.get_performance_metrics(
                actual_batches
            )
        )
        loss_metrics = timing_callback.get_loss_metrics(
            actual_batches
        )

        # Get resource usage if monitoring was enabled
        resource_usage = None
        if resource_callback is not None:
            resource_usage = (
                resource_callback.get_resource_usage()
            )

        # Log summary
        self.logger.info(
            f"Validation benchmark completed: {performance_metrics['total_batches']} "
            f"batches in {performance_metrics['total_epoch_time']:.2f}s"
        )
        self.logger.info(
            f"Average batch time: {timing_stats['average_batch_time']:.4f}s"
        )
        self.logger.info(
            f"Samples per second: {performance_metrics['samples_per_second']:.1f}"
        )

        return BenchmarkResult(
            total_epoch_time=performance_metrics[
                "total_epoch_time"
            ],
            average_batch_time=timing_stats[
                "average_batch_time"
            ],
            actual_average_batch_time=performance_metrics[
                "actual_average_batch_time"
            ],
            total_batches=int(
                performance_metrics["total_batches"]
            ),
            warmup_time=performance_metrics["warmup_time"],
            warmup_batches=0,
            measured_time=performance_metrics["measured_time"],
            measured_batches=timing_callback.measured_batches,
            data_loading_time=timing_stats["data_loading_time"],
            gpu_transfer_time=timing_stats["gpu_transfer_time"],
            forward_pass_time=timing_stats["forward_pass_time"],
            loss_computation_time=timing_stats[
                "loss_computation_time"
            ],
            backward_pass_time=0.0,  # バリデーションでは逆伝播なし
            optimizer_step_time=0.0,  # バリデーションではオプティマイザステップなし
            final_loss=loss_metrics[
                "total_loss"
            ],  # Use total as final for now
            average_loss=loss_metrics["average_loss"],
            samples_per_second=performance_metrics[
                "samples_per_second"
            ],
            batches_per_second=performance_metrics[
                "batches_per_second"
            ],
            resource_usage=resource_usage,
        )


@dataclass(frozen=True)
class TrainingBenchmarkConfig:
    """Configuration for training benchmark."""

    datasource: Optional[
        LearningDataSource.DataSourceSpliter
    ] = None
    gpu: Optional[str] = None
    compilation: bool = False
    detect_anomaly: bool = False
    batch_size: int = 256
    dataloader_workers: int = 4
    pin_memory: Optional[bool] = None
    prefetch_factor: int = 2
    cache_transforms: Optional[bool] = None
    gce_parameter: float = 0.1
    policy_loss_ratio: float = 1.0
    value_loss_ratio: float = 1.0
    learning_ratio: float = 0.01
    momentum: float = 0.9
    optimizer_name: str = "adamw"
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    lr_scheduler_name: Optional[str] = None
    warmup_batches: int = 10
    max_batches: Optional[int] = None
    enable_profiling: bool = False
    test_ratio: float = 0.2
    run_validation: bool = False
    sample_ratio: Optional[float] = None
    enable_resource_monitoring: bool = False
    model_architecture: BackboneArchitecture = "resnet"
    streaming: bool = False
    streaming_train_source: Optional[StreamingDataSource] = None
    streaming_val_source: Optional[StreamingDataSource] = None

    # Stage 関連
    stage: int = 3
    stage1_datasource: Optional[
        LearningDataSource.DataSourceSpliter
    ] = None
    stage2_datasource: Optional[
        LearningDataSource.DataSourceSpliter
    ] = None
    stage2_streaming_train_source: Optional[
        StreamingDataSource
    ] = None
    stage2_streaming_val_source: Optional[
        StreamingDataSource
    ] = None
    stage12_lr_scheduler_name: Optional[str] = None
    stage12_compilation: bool = False

    # Stage 1/2 Head パラメータ
    stage1_pos_weight: float = 1.0
    stage2_pos_weight: float = 1.0
    stage2_gamma_pos: float = 0.0
    stage2_gamma_neg: float = 0.0
    stage2_clip: float = 0.0
    stage2_hidden_dim: int = 128
    stage2_head_dropout: float = 0.0
    stage2_test_ratio: float = 0.2

    # ViT architecture config
    architecture_config: dict[str, Any] | None = None

    # Layer freezing
    freeze_backbone: bool = False
    trainable_layers: Optional[int] = None

    # Stage-specific batch sizes
    stage1_batch_size: Optional[int] = None
    stage2_batch_size: Optional[int] = None


class TrainingBenchmarkUseCase:
    """Use case for training performance benchmarking."""

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        pass

    def _setup_streaming_components(
        self,
        config: TrainingBenchmarkConfig,
    ) -> tuple[
        DeviceConfig,
        tuple[DataLoader, DataLoader],
        ModelComponents,
    ]:
        """Streaming用のベンチマークコンポーネントをセットアップする．

        Args:
            config: ベンチマーク設定

        Returns:
            (DeviceConfig, (train_loader, val_loader), ModelComponents)

        Raises:
            ValueError: streaming_train_source が未設定の場合
        """
        if config.streaming_train_source is None:
            raise ValueError(
                "streaming_train_source is required "
                "when streaming=True"
            )

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

        if config.streaming_val_source is not None:
            val_dataset: torch.utils.data.IterableDataset = StreamingKifDataset(
                streaming_source=config.streaming_val_source,
                batch_size=config.batch_size,
                shuffle=False,
            )
            n_val_files = len(
                config.streaming_val_source.file_paths
            )
        else:
            _EmptyDataset = type(
                "_EmptyDataset",
                (torch.utils.data.IterableDataset,),
                {"__iter__": lambda self: iter([])},
            )
            val_dataset = _EmptyDataset()
            n_val_files = 0

        # Create streaming dataloaders
        n_train_files = len(
            config.streaming_train_source.file_paths
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
            architecture_config=config.architecture_config,
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
            max_epochs=1,
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
            "Streaming benchmark components setup completed"
        )

        return (
            device_config,
            (training_loader, validation_loader),
            model_components,
        )

    def _setup_stage1_components(
        self,
        config: TrainingBenchmarkConfig,
    ) -> tuple[
        DeviceConfig,
        tuple[DataLoader, DataLoader],
        ModelComponents,
    ]:
        """Stage 1 (Reachable Squares) 用のベンチマークコンポーネントをセットアップする．

        Args:
            config: ベンチマーク設定

        Returns:
            (DeviceConfig, (train_loader, val_loader), ModelComponents)

        Raises:
            ValueError: stage1_datasource が未設定の場合
        """
        if config.stage1_datasource is None:
            raise ValueError(
                "stage1_datasource is required when stage=1"
            )

        if config.detect_anomaly:
            torch.autograd.set_detect_anomaly(
                mode=True, check_nan=True
            )

        device_config = DeviceSetup.setup_device(
            config.gpu, config.pin_memory
        )
        device = device_config.device

        # Backbone + Head
        backbone = ModelFactory.create_shogi_backbone(
            device,
            architecture=config.model_architecture,
            architecture_config=config.architecture_config,
        )

        components = (
            StageComponentFactory.create_stage1_components(
                datasource=config.stage1_datasource,
                backbone=backbone,
                device=device,
                batch_size=(
                    config.stage1_batch_size
                    if config.stage1_batch_size is not None
                    else config.batch_size
                ),
                learning_rate=config.learning_ratio,
                pos_weight=config.stage1_pos_weight,
                lr_scheduler_name=(
                    config.stage12_lr_scheduler_name
                    or config.lr_scheduler_name
                ),
                optimizer_name=config.optimizer_name,
                momentum=config.momentum,
                num_workers=config.dataloader_workers,
                pin_memory=device_config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                total_epochs=1,
            )
        )

        loss_fn_value: torch.nn.Module = torch.nn.MSELoss()

        model_components = ModelComponents(
            model=components.model,
            loss_fn_policy=components.loss_fn,
            loss_fn_value=loss_fn_value,
            optimizer=components.optimizer,
            lr_scheduler=components.lr_scheduler,
        )

        # Stage 1 has no validation - provide empty loader
        # for benchmark interface compatibility
        _EmptyDataset = type(
            "_EmptyDataset",
            (torch.utils.data.IterableDataset,),
            {"__iter__": lambda self: iter([])},
        )
        empty_loader: DataLoader = DataLoader(
            _EmptyDataset(), batch_size=1
        )

        self.logger.info(
            "Stage 1 benchmark components setup completed"
        )

        return (
            device_config,
            (components.train_dataloader, empty_loader),
            model_components,
        )

    def _setup_stage2_components(
        self,
        config: TrainingBenchmarkConfig,
    ) -> tuple[
        DeviceConfig,
        tuple[DataLoader, DataLoader],
        ModelComponents,
    ]:
        """Stage 2 (Legal Moves) map-style 用のベンチマークコンポーネントをセットアップする．

        Args:
            config: ベンチマーク設定

        Returns:
            (DeviceConfig, (train_loader, val_loader), ModelComponents)

        Raises:
            ValueError: stage2_datasource が未設定の場合
        """
        if config.stage2_datasource is None:
            raise ValueError(
                "stage2_datasource is required "
                "when stage=2 and streaming=False"
            )

        if config.detect_anomaly:
            torch.autograd.set_detect_anomaly(
                mode=True, check_nan=True
            )

        device_config = DeviceSetup.setup_device(
            config.gpu, config.pin_memory
        )
        device = device_config.device

        # Backbone + Head
        backbone = ModelFactory.create_shogi_backbone(
            device,
            architecture=config.model_architecture,
            architecture_config=config.architecture_config,
        )

        components = (
            StageComponentFactory.create_stage2_components(
                datasource=config.stage2_datasource,
                backbone=backbone,
                device=device,
                batch_size=(
                    config.stage2_batch_size
                    if config.stage2_batch_size is not None
                    else config.batch_size
                ),
                learning_rate=config.learning_ratio,
                pos_weight=config.stage2_pos_weight,
                gamma_pos=config.stage2_gamma_pos,
                gamma_neg=config.stage2_gamma_neg,
                clip=config.stage2_clip,
                head_hidden_dim=config.stage2_hidden_dim,
                head_dropout=config.stage2_head_dropout,
                test_ratio=config.stage2_test_ratio,
                lr_scheduler_name=(
                    config.stage12_lr_scheduler_name
                    or config.lr_scheduler_name
                ),
                optimizer_name=config.optimizer_name,
                momentum=config.momentum,
                num_workers=config.dataloader_workers,
                pin_memory=device_config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                total_epochs=1,
            )
        )

        loss_fn_value: torch.nn.Module = torch.nn.MSELoss()

        model_components = ModelComponents(
            model=components.model,
            loss_fn_policy=components.loss_fn,
            loss_fn_value=loss_fn_value,
            optimizer=components.optimizer,
            lr_scheduler=components.lr_scheduler,
        )

        validation_loader = components.val_dataloader
        if validation_loader is None:
            _EmptyDataset = type(
                "_EmptyDataset",
                (torch.utils.data.IterableDataset,),
                {"__iter__": lambda self: iter([])},
            )
            empty_loader: DataLoader = DataLoader(
                _EmptyDataset(), batch_size=1
            )
            validation_loader = empty_loader

        self.logger.info(
            "Stage 2 benchmark components setup completed"
        )

        return (
            device_config,
            (components.train_dataloader, validation_loader),
            model_components,
        )

    def _setup_stage2_streaming_components(
        self,
        config: TrainingBenchmarkConfig,
    ) -> tuple[
        DeviceConfig,
        tuple[DataLoader, DataLoader],
        ModelComponents,
    ]:
        """Stage 2 (Legal Moves) streaming 用のベンチマークコンポーネントをセットアップする．

        Args:
            config: ベンチマーク設定

        Returns:
            (DeviceConfig, (train_loader, val_loader), ModelComponents)

        Raises:
            ValueError: stage2_streaming_train_source が未設定の場合
        """
        if config.stage2_streaming_train_source is None:
            raise ValueError(
                "stage2_streaming_train_source is required "
                "when stage=2 and streaming=True"
            )

        if config.detect_anomaly:
            torch.autograd.set_detect_anomaly(
                mode=True, check_nan=True
            )

        device_config = DeviceSetup.setup_device(
            config.gpu, config.pin_memory
        )
        device = device_config.device

        # Backbone + Head
        backbone = ModelFactory.create_shogi_backbone(
            device,
            architecture=config.model_architecture,
            architecture_config=config.architecture_config,
        )

        # Data pipeline from factory
        pipeline = StageComponentFactory.create_stage2_streaming_data_pipeline(
            streaming_source=config.stage2_streaming_train_source,
            batch_size=(
                config.stage2_batch_size
                if config.stage2_batch_size is not None
                else config.batch_size
            ),
            pos_weight=config.stage2_pos_weight,
            gamma_pos=config.stage2_gamma_pos,
            gamma_neg=config.stage2_gamma_neg,
            clip=config.stage2_clip,
            dataloader_workers=config.dataloader_workers,
            pin_memory=device_config.pin_memory,
            prefetch_factor=config.prefetch_factor,
        )

        # Model (keep existing model creation - not in streaming data pipeline)
        from maou.app.learning.multi_stage_training import (
            Stage2ModelAdapter,
        )
        from maou.app.learning.network import LegalMovesHead

        legal_moves_head = LegalMovesHead(
            input_dim=backbone.embedding_dim,
            hidden_dim=config.stage2_hidden_dim
            if config.stage2_hidden_dim > 0
            else None,
            dropout=config.stage2_head_dropout,
        )
        model: torch.nn.Module = Stage2ModelAdapter(
            backbone, legal_moves_head
        )
        model.to(device)

        # Loss, optimizer, scheduler
        loss_fn_value: torch.nn.Module = torch.nn.MSELoss()
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
            lr_scheduler_name=(
                config.stage12_lr_scheduler_name
                or config.lr_scheduler_name
            ),
            max_epochs=1,
            steps_per_epoch=len(pipeline.train_dataloader),
        )

        model_components = ModelComponents(
            model=model,
            loss_fn_policy=pipeline.loss_fn,
            loss_fn_value=loss_fn_value,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        # Streaming validation loader
        if config.stage2_streaming_val_source is not None:
            from maou.app.learning.streaming_dataset import (
                Stage2StreamingAdapter,
                StreamingStage2Dataset,
            )

            val_batch_size = (
                config.stage2_batch_size
                if config.stage2_batch_size is not None
                else config.batch_size
            )
            val_dataset_raw: torch.utils.data.IterableDataset = Stage2StreamingAdapter(
                StreamingStage2Dataset(
                    streaming_source=config.stage2_streaming_val_source,
                    batch_size=val_batch_size,
                    shuffle=False,
                )
            )
            _, validation_loader = (
                DataLoaderFactory.create_streaming_dataloaders(
                    train_dataset=pipeline.train_dataloader.dataset,
                    val_dataset=val_dataset_raw,
                    dataloader_workers=config.dataloader_workers,
                    pin_memory=device_config.pin_memory,
                    prefetch_factor=config.prefetch_factor,
                    n_train_files=len(
                        config.stage2_streaming_train_source.file_paths
                    ),
                    n_val_files=len(
                        config.stage2_streaming_val_source.file_paths
                    ),
                    file_paths=config.stage2_streaming_train_source.file_paths,
                )
            )
        else:
            _EmptyDataset = type(
                "_EmptyDataset",
                (torch.utils.data.IterableDataset,),
                {"__iter__": lambda self: iter([])},
            )
            validation_loader = DataLoader(
                _EmptyDataset(), batch_size=1
            )

        self.logger.info(
            "Stage 2 streaming benchmark components setup completed"
        )

        return (
            device_config,
            (pipeline.train_dataloader, validation_loader),
            model_components,
        )

    def _resolve_trainable_layers(
        self, config: TrainingBenchmarkConfig
    ) -> Optional[int]:
        """Resolve effective trainable_layers from config options.

        Returns:
            int if freezing should be applied, None otherwise.
        """
        if (
            config.trainable_layers is not None
            and config.freeze_backbone
        ):
            self.logger.warning(
                "Both freeze_backbone and trainable_layers specified. "
                "Using trainable_layers=%d.",
                config.trainable_layers,
            )
            return config.trainable_layers

        if config.trainable_layers is not None:
            return config.trainable_layers

        if config.freeze_backbone:
            return 0

        return None

    def _apply_layer_freezing(
        self,
        model: torch.nn.Module,
        trainable_layers: int,
    ) -> None:
        """Freeze backbone parameters with optional partial unfreezing.

        Args:
            model: The model (Network or stage adapter) to freeze.
            trainable_layers: Number of trailing backbone groups
                to keep trainable.
        """
        frozen_count = 0
        # Network (Stage 3) has freeze_except_last_n directly
        if hasattr(model, "freeze_except_last_n"):
            frozen_count = model.freeze_except_last_n(  # type: ignore[operator]
                trainable_layers
            )
            if hasattr(model, "_hand_projection"):
                hand_proj = cast(
                    torch.nn.Module,
                    model._hand_projection,  # type: ignore[union-attr]
                )
                for param in hand_proj.parameters():
                    param.requires_grad = False
                    frozen_count += 1
        # Stage1/2 adapters wrap backbone
        elif hasattr(model, "backbone"):
            backbone = model.backbone
            if hasattr(backbone, "freeze_except_last_n"):
                frozen_count = backbone.freeze_except_last_n(  # type: ignore[operator]
                    trainable_layers
                )
                if hasattr(backbone, "_hand_projection"):
                    bb_hand_proj = cast(
                        torch.nn.Module,
                        backbone._hand_projection,  # type: ignore[union-attr]
                    )
                    for param in bb_hand_proj.parameters():
                        param.requires_grad = False
                        frozen_count += 1
            else:
                self.logger.warning(
                    "Model backbone does not support "
                    "freeze_except_last_n; "
                    "skipping layer freezing."
                )
                return
        else:
            self.logger.warning(
                "Model does not support layer freezing; "
                "skipping."
            )
            return

        self.logger.info(
            "Frozen %d backbone parameters "
            "(trainable_layers=%d).",
            frozen_count,
            trainable_layers,
        )

    def execute(self, config: TrainingBenchmarkConfig) -> str:
        """Execute training benchmark and return JSON results."""
        self.logger.info(
            "Starting training benchmark use case (stage=%d)",
            config.stage,
        )

        # Stage 別コンポーネントセットアップ
        from maou.app.learning.training_loop import (
            RawLogitsTrainingLoop,
        )

        training_loop_class: type[TrainingLoop] = TrainingLoop

        if config.stage == 1:
            device_config, dataloaders, model_components = (
                self._setup_stage1_components(config)
            )
            training_loop_class = RawLogitsTrainingLoop
            data_load_method = "map-style"
        elif config.stage == 2:
            if (
                config.streaming
                and config.stage2_streaming_train_source
                is not None
            ):
                device_config, dataloaders, model_components = (
                    self._setup_stage2_streaming_components(
                        config
                    )
                )
                data_load_method = "streaming"
            else:
                device_config, dataloaders, model_components = (
                    self._setup_stage2_components(config)
                )
                data_load_method = "map-style"
            training_loop_class = RawLogitsTrainingLoop
        else:
            # Stage 3: 既存のロジック
            if (
                config.streaming
                and config.streaming_train_source is not None
            ):
                device_config, dataloaders, model_components = (
                    self._setup_streaming_components(config)
                )
                data_load_method = "streaming"
            else:
                if config.datasource is None:
                    raise ValueError(
                        "datasource is required "
                        "when streaming=False"
                    )

                training_datasource, validation_datasource = (
                    config.datasource.train_test_split(
                        test_ratio=config.test_ratio
                    )
                )

                cache_transforms_enabled = (
                    config.cache_transforms
                    if config.cache_transforms is not None
                    else False
                )
                device_config, dataloaders, model_components = (
                    TrainingSetup.setup_training_components(
                        training_datasource=training_datasource,
                        validation_datasource=validation_datasource,
                        cache_transforms=cache_transforms_enabled,
                        gpu=config.gpu,
                        model_architecture=config.model_architecture,
                        architecture_config=config.architecture_config,
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
                        max_epochs=1,
                        detect_anomaly=config.detect_anomaly,
                    )
                )
                data_load_method = "map-style"

        training_loader, validation_loader = dataloaders
        device = device_config.device

        # Apply layer freezing (before compilation)
        effective_trainable_layers = (
            self._resolve_trainable_layers(config)
        )
        if effective_trainable_layers is not None:
            self._apply_layer_freezing(
                model_components.model,
                effective_trainable_layers,
            )

            # Recreate optimizer with only trainable parameters
            model_components.optimizer = (
                LossOptimizerFactory.create_optimizer(
                    model_components.model,
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

            # Recreate lr_scheduler bound to the new optimizer
            lr_scheduler_name = config.lr_scheduler_name
            if (
                config.stage in (1, 2)
                and config.stage12_lr_scheduler_name
            ):
                lr_scheduler_name = (
                    config.stage12_lr_scheduler_name
                )
            model_components.lr_scheduler = (
                SchedulerFactory.create_scheduler(
                    model_components.optimizer,
                    lr_scheduler_name=lr_scheduler_name,
                    max_epochs=1,
                    steps_per_epoch=len(training_loader),
                )
            )

        should_compile = config.compilation
        if (
            config.stage in (1, 2)
            and config.stage12_compilation
        ):
            should_compile = True
        if should_compile:
            self.logger.info(
                "Compiling model with torch.compile for benchmarking (dynamic shapes disabled)"
            )
            model_components.model = compile_module(
                model_components.model
            )

        # warmup バッチ数の自動調整（全ステージ共通）
        estimated_batches = len(training_loader)
        if config.max_batches is not None:
            estimated_batches = min(
                estimated_batches, config.max_batches
            )
        effective_warmup = min(
            config.warmup_batches, estimated_batches - 2
        )
        effective_warmup = max(0, effective_warmup)
        if effective_warmup != config.warmup_batches:
            self.logger.warning(
                "warmup_batches を %d → %d に自動調整"
                "(estimated_batches=%d)",
                config.warmup_batches,
                effective_warmup,
                estimated_batches,
            )

        # Get total number of batches in the full dataset
        total_batches_in_dataset = len(training_loader)

        # Create benchmark instance
        # Stage 1/2: value_loss_ratio=0.0 でダミー value loss を無視
        actual_value_loss_ratio = (
            0.0
            if config.stage in (1, 2)
            else config.value_loss_ratio
        )

        benchmark = SingleEpochBenchmark(
            model=model_components.model,
            device=device,
            optimizer=model_components.optimizer,
            loss_fn_policy=model_components.loss_fn_policy,
            loss_fn_value=model_components.loss_fn_value,
            policy_loss_ratio=config.policy_loss_ratio,
            value_loss_ratio=actual_value_loss_ratio,
            enable_resource_monitoring=config.enable_resource_monitoring,
            training_loop_class=training_loop_class,
        )

        # Run training benchmark
        self.logger.info("Starting training benchmark...")
        training_result = benchmark.benchmark_epoch(
            training_loader,
            warmup_batches=effective_warmup,
            max_batches=config.max_batches,
            enable_profiling=config.enable_profiling,
        )
        training_result = replace(
            training_result, data_load_method=data_load_method
        )

        # Run validation benchmark if requested
        validation_result = None
        validation_skipped_reason: Optional[str] = None

        if config.run_validation:
            if config.stage == 1:
                validation_skipped_reason = (
                    "Stage 1 does not support validation "
                    "(accuracy threshold is used instead). "
                    "--run-validation is ignored."
                )
                self.logger.warning(validation_skipped_reason)
            elif (
                config.stage == 2
                and config.stage2_test_ratio == 0.0
            ):
                validation_skipped_reason = (
                    "--run-validation requires "
                    "--stage2-test-ratio > 0. "
                    "Skipping validation."
                )
                self.logger.warning(validation_skipped_reason)
            else:
                self.logger.info(
                    "Starting validation benchmark..."
                )
                validation_result = (
                    benchmark.benchmark_validation(
                        validation_loader,
                        max_batches=config.max_batches,
                    )
                )
                validation_result = replace(
                    validation_result,
                    data_load_method=data_load_method,
                )

        # Calculate sample ratio estimation if provided
        estimation_results = {}
        if config.sample_ratio is not None:
            estimated_full_epoch_time = (
                training_result.total_epoch_time
                / config.sample_ratio
            )
            estimated_total_batches = int(
                float(training_result.total_batches)
                / config.sample_ratio
            )
            estimation_results = {
                "sample_ratio": config.sample_ratio,
                "estimated_full_epoch_time_seconds": estimated_full_epoch_time,
                "estimated_full_epoch_time_minutes": estimated_full_epoch_time
                / 60.0,
                "estimated_total_batches": float(
                    estimated_total_batches
                ),
                "actual_batches_processed": float(
                    training_result.total_batches
                ),
            }
            self.logger.info(
                f"Sample ratio: {config.sample_ratio:.1%}"
            )
            self.logger.info(
                f"Estimated full epoch time: "
                f"{estimated_full_epoch_time / 60:.1f} minutes"
            )
            self.logger.info(
                f"Estimated total batches: {estimated_total_batches:,}"
            )

        # Format results for display
        def format_timing_summary(
            result: BenchmarkResult,
            label: str,
            total_batches_in_dataset: int,
        ) -> str:
            # Calculate estimated full epoch time
            estimated_full_epoch_time_seconds = (
                result.actual_average_batch_time
                * total_batches_in_dataset
            )
            estimated_full_epoch_time_minutes = (
                estimated_full_epoch_time_seconds / 60.0
            )

            # Pre-calculate percentages based on actual average batch time
            # (includes all overhead: data loading, processing, etc.)
            data_pct = (
                result.data_loading_time
                / result.actual_average_batch_time
                * 100
            )
            gpu_pct = (
                result.gpu_transfer_time
                / result.actual_average_batch_time
                * 100
            )
            forward_pct = (
                result.forward_pass_time
                / result.actual_average_batch_time
                * 100
            )
            loss_pct = (
                result.loss_computation_time
                / result.actual_average_batch_time
                * 100
            )
            backward_pct = (
                result.backward_pass_time
                / result.actual_average_batch_time
                * 100
            )
            opt_pct = (
                result.optimizer_step_time
                / result.actual_average_batch_time
                * 100
            )
            # Format resource usage summary if available
            resource_summary = ""
            if result.resource_usage is not None:
                ru = result.resource_usage
                resource_summary = f"""

  Resource Usage Summary:
  - CPU Max Usage: {ru.cpu_max_percent:.1f}%
  - Memory Max Usage: {ru.memory_max_bytes / 1024**3:.1f}GB ({ru.memory_max_percent:.1f}%)"""

                if ru.gpu_max_percent is not None:
                    resource_summary += f"""
  - GPU Max Usage: {ru.gpu_max_percent:.1f}%"""

                if (
                    ru.gpu_memory_max_bytes is not None
                    and ru.gpu_memory_total_bytes is not None
                    and ru.gpu_memory_max_percent is not None
                ):
                    resource_summary += f"""
  - GPU Memory Max Usage: {ru.gpu_memory_max_bytes / 1024**3:.1f}GB / {ru.gpu_memory_total_bytes / 1024**3:.1f}GB ({ru.gpu_memory_max_percent:.1f}%)"""

            return f"""{label} Performance Summary:
  Processed Batches: {result.total_batches} / {total_batches_in_dataset}
  Total Time (Processed): {result.total_epoch_time:.2f}s
  Warmup: {result.warmup_batches} batches in {result.warmup_time:.2f}s
  Measured: {result.measured_batches} batches in {result.measured_time:.2f}s
  Estimated Full Epoch Time: {estimated_full_epoch_time_seconds:.2f}s ({estimated_full_epoch_time_minutes:.2f} minutes)
  Actual Average Time per Batch: {result.actual_average_batch_time:.4f}s
  Processing Time per Batch (excl. data loading): {result.average_batch_time:.4f}s
  Samples per Second: {result.samples_per_second:.1f}
  Batches per Second: {result.batches_per_second:.2f}

  Detailed Timing Breakdown (per batch, warmup excluded):
  - Data Loading: {result.data_loading_time:.4f}s ({data_pct:.1f}%)
  - GPU Transfer: {result.gpu_transfer_time:.4f}s ({gpu_pct:.1f}%)
  - Forward Pass: {result.forward_pass_time:.4f}s ({forward_pct:.1f}%)
  - Loss Computation: {result.loss_computation_time:.4f}s ({loss_pct:.1f}%)
  - Backward Pass: {result.backward_pass_time:.4f}s ({backward_pct:.1f}%)
  - Optimizer Step: {result.optimizer_step_time:.4f}s ({opt_pct:.1f}%)

  Loss Information:
  - Final Loss: {result.final_loss:.6f}
  - Average Loss: {result.average_loss:.6f}{resource_summary}"""

        training_summary = format_timing_summary(
            training_result,
            "Training",
            total_batches_in_dataset,
        )

        # Create recommendations based on timing analysis
        recommendations = []

        # Batch time analysis
        if training_result.average_batch_time > 0.1:
            recommendations.append(
                "Consider increasing batch size for better GPU utilization"
            )
        elif training_result.average_batch_time < 0.01:
            recommendations.append(
                "Batch size might be too large, consider reducing for memory efficiency"
            )

        # Data loading analysis
        data_loading_percentage = (
            training_result.data_loading_time
            / training_result.average_batch_time
        ) * 100
        if data_loading_percentage > 20:
            recommendations.append(
                "Data loading is a bottleneck - consider increasing DataLoader "
                "workers or enabling prefetch"
            )

        # GPU transfer analysis
        if device.type == "cuda":
            gpu_transfer_percentage = (
                training_result.gpu_transfer_time
                / training_result.average_batch_time
            ) * 100
            if gpu_transfer_percentage > 10:
                recommendations.append(
                    "GPU transfer is slow - ensure pin_memory=True and consider "
                    "larger batch sizes"
                )

        # Throughput analysis
        if (
            training_result.samples_per_second < 1000
            and device.type == "cuda"
        ):
            recommendations.append(
                "Low throughput detected - consider optimizing batch size, "
                "DataLoader workers, or model compilation"
            )

        recommendations_text = (
            "Performance Recommendations:\n"
            + "\n".join(f"- {rec}" for rec in recommendations)
        )

        # Create output
        output = {
            "benchmark_results": {
                "Summary": training_summary,
                "Recommendations": recommendations_text,
            },
            "training_metrics": training_result.to_dict(),
            "estimation": estimation_results,
            "configuration": {
                "device": str(device),
                "batch_size": config.batch_size,
                "dataloader_workers": config.dataloader_workers,
                "pin_memory": device_config.pin_memory,
                "prefetch_factor": config.prefetch_factor,
                "warmup_batches": config.warmup_batches,
                "max_batches": config.max_batches,
                "enable_profiling": config.enable_profiling,
                "sample_ratio": config.sample_ratio,
            },
        }

        # Add validation results if available
        if validation_result is not None:
            total_validation_batches = len(validation_loader)
            validation_summary = format_timing_summary(
                validation_result,
                "Validation",
                total_validation_batches,
            )
            # Dynamic dict key assignment for validation summary
            output["benchmark_results"]["ValidationSummary"] = (  # type: ignore[index]
                validation_summary
            )
            output["validation_metrics"] = (
                validation_result.to_dict()
            )

        if validation_skipped_reason is not None:
            output["benchmark_results"][
                "validation_skipped"
            ] = (  # type: ignore[index]
                validation_skipped_reason
            )

        return json.dumps(output, indent=2)
