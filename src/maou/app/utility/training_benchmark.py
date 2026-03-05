import json
import logging
import math
from collections.abc import Iterator
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
from maou.app.learning.policy_targets import PolicyTargetMode
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


class _EmptyIterableDataset(torch.utils.data.IterableDataset):
    """空のIterableDataset．ベンチマーク時のダミーバリデーション用．"""

    def __iter__(self) -> Iterator[object]:
        return iter([])


@dataclass(frozen=True)
class ModelInfo:
    """モデルの基本情報を格納するデータクラス．"""

    total_parameters: int
    trainable_parameters: int
    model_memory_bytes: int

    def to_dict(self) -> dict[str, object]:
        """辞書形式で返す．"""
        return {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "model_memory_bytes": self.model_memory_bytes,
        }


@dataclass(frozen=True)
class GPUMemoryBreakdown:
    """GPUメモリ内訳を格納するデータクラス．"""

    model_parameters_bytes: int
    optimizer_state_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    total_gpu_memory_bytes: int

    def to_dict(self) -> dict[str, object]:
        """辞書形式で返す．"""
        return {
            "model_parameters_bytes": self.model_parameters_bytes,
            "optimizer_state_bytes": self.optimizer_state_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "total_gpu_memory_bytes": self.total_gpu_memory_bytes,
            "activation_estimate_bytes": max(
                0,
                self.peak_allocated_bytes
                - self.model_parameters_bytes
                - self.optimizer_state_bytes,
            ),
        }


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

    # タイミング分布統計
    timing_distribution: Optional[
        dict[str, dict[str, float]]
    ] = None

    # モデル情報
    model_info: Optional[ModelInfo] = None

    # GPUメモリ内訳
    gpu_memory_breakdown: Optional[GPUMemoryBreakdown] = None

    @property
    def unaccounted_time(self) -> float:
        """計上されていない時間を算出する．"""
        accounted = (
            self.data_loading_time
            + self.gpu_transfer_time
            + self.forward_pass_time
            + self.loss_computation_time
            + self.backward_pass_time
            + self.optimizer_step_time
        )
        return self.actual_average_batch_time - accounted

    def to_dict(self) -> dict[str, object]:
        """ベンチマーク結果を辞書形式で返す．"""
        result: dict[str, object] = {
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

        result["unaccounted_time"] = self.unaccounted_time

        # リソース使用率情報があれば追加
        if self.resource_usage is not None:
            resource_dict = self.resource_usage.to_dict()
            # float型に変換できるもののみ追加
            for key, value in resource_dict.items():
                if value is not None:
                    result[f"resource_{key}"] = float(value)

        # タイミング分布統計
        if self.timing_distribution is not None:
            result["timing_distribution"] = (
                self.timing_distribution
            )

        # モデル情報
        if self.model_info is not None:
            result["model_info"] = self.model_info.to_dict()

        # GPUメモリ内訳
        if self.gpu_memory_breakdown is not None:
            result["gpu_memory_breakdown"] = (
                self.gpu_memory_breakdown.to_dict()
            )

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
        policy_target_mode: Policy教師信号モード
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
        policy_target_mode: PolicyTargetMode = PolicyTargetMode.WIN_RATE,
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
        self.policy_target_mode = policy_target_mode
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

    def _collect_model_info(self) -> ModelInfo:
        """モデルの基本情報を収集する．"""
        total_params = sum(
            p.numel() for p in self.model.parameters()
        )
        trainable_params = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )
        model_memory = sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        )
        return ModelInfo(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_memory_bytes=model_memory,
        )

    def _collect_gpu_memory_breakdown(
        self,
        model_param_bytes: int,
    ) -> Optional[GPUMemoryBreakdown]:
        """GPUメモリ内訳を収集する．"""
        if self.device.type != "cuda":
            return None
        optimizer_state_bytes = 0
        for state in self.optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    optimizer_state_bytes += (
                        v.numel() * v.element_size()
                    )

        peak_allocated = torch.cuda.max_memory_allocated(
            self.device
        )
        peak_reserved = torch.cuda.max_memory_reserved(
            self.device
        )
        total_memory = torch.cuda.get_device_properties(
            self.device
        ).total_mem

        return GPUMemoryBreakdown(
            model_parameters_bytes=model_param_bytes,
            optimizer_state_bytes=optimizer_state_bytes,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            total_gpu_memory_bytes=total_memory,
        )

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
            policy_target_mode=self.policy_target_mode,
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
        timing_distribution = (
            timing_callback.get_timing_distribution()
        )
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

        # Collect model info and GPU memory breakdown
        model_info = self._collect_model_info()
        gpu_memory_breakdown = (
            self._collect_gpu_memory_breakdown(
                model_info.model_memory_bytes
            )
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
            timing_distribution=timing_distribution,
            model_info=model_info,
            gpu_memory_breakdown=gpu_memory_breakdown,
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
            policy_target_mode=self.policy_target_mode,
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
    policy_target_mode: PolicyTargetMode = (
        PolicyTargetMode.WIN_RATE
    )
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

    # Sweep parameters
    batch_sizes: Optional[list[int]] = None
    learning_rates: Optional[list[float]] = None
    estimate_cbs: bool = False


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
            val_dataset = _EmptyIterableDataset()
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
        empty_loader: DataLoader = DataLoader(
            _EmptyIterableDataset(), batch_size=1
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
            validation_loader = DataLoader(
                _EmptyIterableDataset(), batch_size=1
            )

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
            validation_loader = DataLoader(
                _EmptyIterableDataset(), batch_size=1
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
            policy_target_mode=config.policy_target_mode,
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
        training_summary = _format_timing_summary(
            training_result,
            "Training",
            total_batches_in_dataset,
        )

        # Create recommendations
        recommendations = _generate_recommendations(
            training_result, device
        )
        recommendations_text = (
            "Performance Recommendations:\n"
            + "\n".join(f"- {rec}" for rec in recommendations)
        )

        # Create output with complete configuration
        output: dict[str, object] = {
            "benchmark_results": {
                "Summary": training_summary,
                "Recommendations": recommendations_text,
            },
            "training_metrics": training_result.to_dict(),
            "estimation": estimation_results,
            "configuration": {
                "device": str(device),
                "stage": config.stage,
                "batch_size": config.batch_size,
                "dataloader_workers": config.dataloader_workers,
                "pin_memory": device_config.pin_memory,
                "prefetch_factor": config.prefetch_factor,
                "warmup_batches": config.warmup_batches,
                "max_batches": config.max_batches,
                "enable_profiling": config.enable_profiling,
                "sample_ratio": config.sample_ratio,
                "model_architecture": config.model_architecture,
                "streaming": config.streaming,
                "data_load_method": data_load_method,
                "optimizer": config.optimizer_name,
                "learning_rate": config.learning_ratio,
                "momentum": config.momentum,
                "policy_loss_ratio": config.policy_loss_ratio,
                "value_loss_ratio": config.value_loss_ratio,
                "compilation": config.compilation,
                "freeze_backbone": config.freeze_backbone,
                "trainable_layers": config.trainable_layers,
            },
        }

        # Add validation results if available
        if validation_result is not None:
            total_validation_batches = len(validation_loader)
            validation_summary = _format_timing_summary(
                validation_result,
                "Validation",
                total_validation_batches,
            )
            benchmark_results = cast(
                dict[str, str], output["benchmark_results"]
            )
            benchmark_results["ValidationSummary"] = (
                validation_summary
            )
            output["validation_metrics"] = (
                validation_result.to_dict()
            )

        if validation_skipped_reason is not None:
            benchmark_results_v = cast(
                dict[str, str], output["benchmark_results"]
            )
            benchmark_results_v["validation_skipped"] = (
                validation_skipped_reason
            )

        return json.dumps(output, indent=2)

    def execute_batch_size_sweep(
        self,
        config: TrainingBenchmarkConfig,
    ) -> str:
        """複数バッチサイズでベンチマークを実行し比較結果を返す．

        Args:
            config: ベンチマーク設定（batch_sizes フィールドが必須）

        Returns:
            JSON形式の比較結果
        """
        if not config.batch_sizes:
            raise ValueError(
                "batch_sizes is required for sweep"
            )

        self.logger.info(
            "Starting batch size sweep: %s",
            config.batch_sizes,
        )

        results: list[dict[str, Any]] = []
        for bs in config.batch_sizes:
            self.logger.info(
                "Running benchmark with batch_size=%d", bs
            )
            # Reset CUDA memory stats before each run
            if (
                config.gpu is not None
                or torch.cuda.is_available()
            ):
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception as e:
                    self.logger.debug(
                        "Failed to reset CUDA memory stats: %s",
                        e,
                    )

            sweep_config = replace(config, batch_size=bs)
            try:
                result_json = self.execute(sweep_config)
            except torch.cuda.OutOfMemoryError:
                self.logger.warning(
                    "CUDA OOM at batch_size=%d, skipping",
                    bs,
                )
                torch.cuda.empty_cache()
                results.append(
                    {
                        "sweep_batch_size": bs,
                        "oom": True,
                    }
                )
                continue
            result: dict[str, Any] = json.loads(result_json)
            result["sweep_batch_size"] = bs
            results.append(result)

        # Build comparison table (skip OOM results)
        comparison: list[dict[str, Any]] = []
        oom_batch_sizes: list[int] = []
        for r in results:
            bs = r["sweep_batch_size"]
            if r.get("oom"):
                oom_batch_sizes.append(bs)
                continue
            tm = r["training_metrics"]
            entry: dict[str, Any] = {
                "batch_size": bs,
                "samples_per_second": tm["samples_per_second"],
                "avg_batch_time": tm[
                    "actual_average_batch_time"
                ],
                "data_loading_time": tm["data_loading_time"],
                "forward_pass_time": tm["forward_pass_time"],
                "backward_pass_time": tm["backward_pass_time"],
                "average_loss": tm["average_loss"],
            }
            # GPU memory if available
            if "gpu_memory_breakdown" in tm:
                gm = tm["gpu_memory_breakdown"]
                entry["peak_gpu_memory_mb"] = (
                    gm["peak_allocated_bytes"] / 1024**2
                )
            comparison.append(entry)

        # Filter out OOM results for analysis
        successful_results = [
            r for r in results if not r.get("oom")
        ]

        # GPU memory-based batch size recommendation
        gpu_recommendation = self._recommend_max_batch_size(
            successful_results
        )

        # CBS estimation if requested
        cbs_estimation = None
        if config.estimate_cbs and len(successful_results) >= 2:
            cbs_estimation = self._estimate_cbs_from_sweep(
                successful_results,
                config_batch_size=config.batch_size,
            )

        # Format comparison summary
        summary_lines = ["=== Batch Size Sweep Results ==="]
        summary_lines.append(
            f"{'BS':>6} | {'samp/s':>8} | {'batch_t':>8} | {'data_t':>8} | {'fwd_t':>8} | {'bwd_t':>8} | {'GPU MB':>8} | {'loss':>10}"
        )
        summary_lines.append("-" * 85)
        for c in comparison:
            gpu_mb = c.get("peak_gpu_memory_mb", "N/A")
            gpu_str = (
                f"{gpu_mb:>8.0f}"
                if isinstance(gpu_mb, (int, float))
                else f"{gpu_mb:>8}"
            )
            summary_lines.append(
                f"{c['batch_size']:>6} | "
                f"{c['samples_per_second']:>8.1f} | "
                f"{c['avg_batch_time']:>8.4f} | "
                f"{c['data_loading_time']:>8.4f} | "
                f"{c['forward_pass_time']:>8.4f} | "
                f"{c['backward_pass_time']:>8.4f} | "
                f"{gpu_str} | "
                f"{c['average_loss']:>10.6f}"
            )

        if oom_batch_sizes:
            summary_lines.append("")
            summary_lines.append(
                f"CUDA OOM at batch sizes: {oom_batch_sizes}"
            )

        if gpu_recommendation:
            summary_lines.append("")
            summary_lines.append(
                f"GPU Memory-based Max Batch Size Estimate: "
                f"{gpu_recommendation['max_batch_size']}"
            )
            summary_lines.append(
                f"  (based on {gpu_recommendation['available_memory_mb']:.0f}MB available, "
                f"{gpu_recommendation['per_sample_activation_mb']:.3f}MB/sample)"
            )

        if cbs_estimation:
            summary_lines.append("")
            summary_lines.append(
                f"Estimated Critical Batch Size (CBS): "
                f"{cbs_estimation['estimated_cbs']}"
            )
            summary_lines.append(
                f"  Gradient Noise Scale (B_noise): "
                f"{cbs_estimation['gradient_noise_scale']:.1f}"
            )
            summary_lines.append(
                f"  Recommendation: {cbs_estimation['recommendation']}"
            )

        output: dict[str, Any] = {
            "sweep_type": "batch_size",
            "comparison": comparison,
            "summary": "\n".join(summary_lines),
            "detailed_results": results,
            "oom_batch_sizes": oom_batch_sizes,
        }
        if gpu_recommendation:
            output["gpu_batch_size_recommendation"] = (
                gpu_recommendation
            )
        if cbs_estimation:
            output["cbs_estimation"] = cbs_estimation

        return json.dumps(output, indent=2)

    def execute_learning_rate_sweep(
        self,
        config: TrainingBenchmarkConfig,
    ) -> str:
        """複数学習率でベンチマークを実行し比較結果を返す．

        Args:
            config: ベンチマーク設定（learning_rates フィールドが必須）

        Returns:
            JSON形式の比較結果
        """
        if not config.learning_rates:
            raise ValueError(
                "learning_rates is required for sweep"
            )

        self.logger.info(
            "Starting learning rate sweep: %s",
            config.learning_rates,
        )

        # Note: Unlike batch size sweep, OOM is not caught here because
        # learning rate changes do not affect memory usage. OOM during
        # LR sweep indicates a config/batch_size issue, not an LR issue.
        results: list[dict[str, Any]] = []
        for lr in config.learning_rates:
            self.logger.info(
                "Running benchmark with learning_rate=%g", lr
            )
            sweep_config = replace(config, learning_ratio=lr)
            result_json = self.execute(sweep_config)
            result_dict: dict[str, Any] = json.loads(
                result_json
            )
            result_dict["sweep_learning_rate"] = lr
            results.append(result_dict)

        # Build comparison table
        comparison: list[dict[str, Any]] = []
        for r in results:
            lr = r["sweep_learning_rate"]
            tm = r["training_metrics"]
            comparison.append(
                {
                    "learning_rate": lr,
                    "average_loss": tm["average_loss"],
                    "final_loss": tm["final_loss"],
                    "samples_per_second": tm[
                        "samples_per_second"
                    ],
                }
            )

        # Format summary
        summary_lines = ["=== Learning Rate Sweep Results ==="]
        summary_lines.append(
            f"{'LR':>12} | {'avg_loss':>10} | {'final_loss':>10} | {'samp/s':>8}"
        )
        summary_lines.append("-" * 50)
        for c in comparison:
            summary_lines.append(
                f"{c['learning_rate']:>12.6f} | "
                f"{c['average_loss']:>10.6f} | "
                f"{c['final_loss']:>10.6f} | "
                f"{c['samples_per_second']:>8.1f}"
            )

        # Find best learning rate
        best = min(comparison, key=lambda x: x["average_loss"])
        summary_lines.append("")
        summary_lines.append(
            f"Best Learning Rate: {best['learning_rate']} "
            f"(avg_loss: {best['average_loss']:.6f})"
        )

        output: dict[str, Any] = {
            "sweep_type": "learning_rate",
            "comparison": comparison,
            "summary": "\n".join(summary_lines),
            "detailed_results": results,
        }

        return json.dumps(output, indent=2)

    @staticmethod
    def _recommend_max_batch_size(
        results: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """GPU メモリ使用量からバッチサイズの最大推奨値を推定する．"""
        # Need at least 2 data points to extrapolate
        points: list[tuple[int, int, int, int]] = []
        for r in results:
            bs = int(r["sweep_batch_size"])
            tm = r.get("training_metrics", {})
            if not isinstance(tm, dict):
                continue
            gm = tm.get("gpu_memory_breakdown")
            if not isinstance(gm, dict):
                continue
            points.append(
                (
                    bs,
                    int(gm["peak_allocated_bytes"]),
                    int(gm["model_parameters_bytes"]),
                    int(gm["total_gpu_memory_bytes"]),
                )
            )

        if len(points) < 2:
            return None

        # Estimate per-sample activation memory using least-squares
        # peak_memory = fixed_cost + per_sample * batch_size
        n = len(points)
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        sum_xy = sum(p[0] * p[1] for p in points)
        sum_xx = sum(p[0] * p[0] for p in points)

        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return None

        per_sample_bytes = (n * sum_xy - sum_x * sum_y) / denom
        if per_sample_bytes <= 0:
            return None

        fixed_cost = (sum_y - per_sample_bytes * sum_x) / n
        # Clamp negative intercept to 0 (regression artifact from
        # outliers/few data points) to avoid overestimating capacity.
        fixed_cost = max(0, fixed_cost)
        total_memory = points[0][3]

        # Use 85% safety margin
        usable_memory = total_memory * 0.85
        max_batch_size = int(
            (usable_memory - fixed_cost) / per_sample_bytes
        )
        # Cap at 4x the largest tested batch size to bound extrapolation
        max_tested_bs = max(p[0] for p in points)
        max_batch_size = min(max_batch_size, max_tested_bs * 4)
        max_batch_size = max(1, max_batch_size)

        return {
            "max_batch_size": max_batch_size,
            "per_sample_activation_mb": per_sample_bytes
            / 1024**2,
            "fixed_cost_mb": fixed_cost / 1024**2,
            "available_memory_mb": usable_memory / 1024**2,
            "total_gpu_memory_mb": total_memory / 1024**2,
            "safety_margin": 0.85,
        }

    @staticmethod
    def _estimate_cbs_from_sweep(
        results: list[dict[str, Any]],
        config_batch_size: int = 256,
    ) -> Optional[dict[str, Any]]:
        """複数バッチサイズの結果からCBSを推定する．

        CBS推定手法: 効率スケーリング分析
        バッチサイズを2倍にしたときのスループット向上率から
        Gradient Noise Scale (B_noise) を推定する．

        B_noise ≈ CBS の近似として使用．
        """
        points: list[tuple[int, float]] = []
        for r in results:
            bs = int(r["sweep_batch_size"])
            tm = r.get("training_metrics", {})
            if not isinstance(tm, dict):
                continue
            sps = float(tm.get("samples_per_second", 0))
            if sps > 0:
                points.append((bs, sps))

        if len(points) < 2:
            return None

        points.sort(key=lambda x: x[0])

        # Compute efficiency at each batch size
        # Efficiency = throughput / batch_size (proportional to gradient updates per second)
        # At B << CBS: efficiency ≈ constant
        # At B >> CBS: efficiency ∝ 1/B
        # CBS ≈ B where efficiency drops to ~50% of small-B efficiency
        base_bs, base_sps = points[0]
        base_efficiency = base_sps / base_bs

        # Find B_noise using pairwise CBS estimates from all
        # pairs to reduce baseline bias.
        # From sps(B) = sps_max * B * CBS / (CBS + B):
        #   eff(B) = sps(B)/B = sps_max * CBS / (CBS + B)
        # For pair (B_i, B_j) where eff_i > eff_j:
        #   eff_i * (CBS + B_i) = eff_j * (CBS + B_j)
        #   CBS = (eff_j * B_j - eff_i * B_i) / (eff_i - eff_j)
        # Equivalently:
        #   CBS = (sps_j - sps_i) * B_i * B_j
        #       / (sps_i * B_j - sps_j * B_i)
        cbs_estimates: list[float] = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                bs_i, sps_i = points[i]
                bs_j, sps_j = points[j]
                eff_i = sps_i / bs_i
                eff_j = sps_j / bs_j
                if eff_i > eff_j > 0:
                    # CBS from this pair
                    cbs_est = (bs_j * eff_j - bs_i * eff_i) / (
                        eff_i - eff_j
                    )
                    if cbs_est > 0:
                        cbs_estimates.append(cbs_est)

        if not cbs_estimates:
            # All batch sizes show linear scaling - CBS is larger
            # than all tested batch sizes
            max_tested = max(bs for bs, _ in points)
            return {
                "estimated_cbs": max_tested,
                "cbs_exceeds_tested": True,
                "gradient_noise_scale": None,
                "recommendation": (
                    f"CBS exceeds all tested batch sizes (max: {max_tested}). "
                    f"All tested sizes are in the linear scaling regime. "
                    f"Current batch size is efficient."
                ),
                "scaling_efficiency": {
                    str(bs): round(
                        (sps / bs) / base_efficiency, 3
                    )
                    for bs, sps in points
                },
            }

        sorted_estimates = sorted(cbs_estimates)
        mid = len(sorted_estimates) // 2
        if len(sorted_estimates) % 2 == 0:
            median_cbs = (
                sorted_estimates[mid - 1]
                + sorted_estimates[mid]
            ) / 2
        else:
            median_cbs = sorted_estimates[mid]
        cbs_rounded = int(
            2 ** round(math.log2(max(1, median_cbs)))
        )

        # Generate recommendation
        config_bs = config_batch_size
        rec: str
        if config_bs < cbs_rounded:
            rec = (
                f"Current batch size ({config_bs}) is below CBS ({cbs_rounded}). "
                f"Increasing batch size up to {cbs_rounded} will improve "
                f"training efficiency with proportional speedup."
            )
        elif config_bs > cbs_rounded * 2:
            rec = (
                f"Current batch size ({config_bs}) is well above CBS ({cbs_rounded}). "
                f"Diminishing returns on larger batches. "
                f"Consider reducing to ~{cbs_rounded} for better resource efficiency."
            )
        else:
            rec = (
                f"Current batch size ({config_bs}) is near CBS ({cbs_rounded}). "
                f"Good balance between speed and efficiency."
            )

        return {
            "estimated_cbs": cbs_rounded,
            "cbs_exceeds_tested": False,
            "gradient_noise_scale": round(median_cbs, 1),
            "recommendation": rec,
            "scaling_efficiency": {
                str(bs): round((sps / bs) / base_efficiency, 3)
                for bs, sps in points
            },
        }


def _format_timing_summary(
    result: BenchmarkResult,
    label: str,
    total_batches_in_dataset: int,
) -> str:
    """ベンチマーク結果をフォーマットされた文字列として返す．"""
    # Calculate estimated full epoch time
    estimated_full_epoch_time_seconds = (
        result.actual_average_batch_time
        * total_batches_in_dataset
    )
    estimated_full_epoch_time_minutes = (
        estimated_full_epoch_time_seconds / 60.0
    )

    avg_batch = result.actual_average_batch_time
    if avg_batch == 0:
        avg_batch = 1e-9  # avoid division by zero

    # Pre-calculate percentages
    data_pct = result.data_loading_time / avg_batch * 100
    gpu_pct = result.gpu_transfer_time / avg_batch * 100
    forward_pct = result.forward_pass_time / avg_batch * 100
    loss_pct = result.loss_computation_time / avg_batch * 100
    backward_pct = result.backward_pass_time / avg_batch * 100
    opt_pct = result.optimizer_step_time / avg_batch * 100

    # Unaccounted time
    unaccounted = result.unaccounted_time
    unaccounted_pct = unaccounted / avg_batch * 100

    lines = [
        f"{label} Performance Summary:",
        f"  Processed Batches: {result.total_batches} / {total_batches_in_dataset}",
        f"  Total Time (Processed): {result.total_epoch_time:.2f}s",
        f"  Warmup: {result.warmup_batches} batches in {result.warmup_time:.2f}s",
        f"  Measured: {result.measured_batches} batches in {result.measured_time:.2f}s",
        f"  Estimated Full Epoch Time: {estimated_full_epoch_time_seconds:.2f}s ({estimated_full_epoch_time_minutes:.2f} minutes)",
        f"  Actual Average Time per Batch: {result.actual_average_batch_time:.4f}s",
        f"  Processing Time per Batch (excl. data loading): {result.average_batch_time:.4f}s",
        f"  Samples per Second: {result.samples_per_second:.1f}",
        f"  Batches per Second: {result.batches_per_second:.2f}",
        "",
        "  Detailed Timing Breakdown (per batch, warmup excluded):",
        f"  - Data Loading: {result.data_loading_time:.4f}s ({data_pct:.1f}%)",
        f"  - GPU Transfer: {result.gpu_transfer_time:.4f}s ({gpu_pct:.1f}%)",
        f"  - Forward Pass: {result.forward_pass_time:.4f}s ({forward_pct:.1f}%)",
        f"  - Loss Computation: {result.loss_computation_time:.4f}s ({loss_pct:.1f}%)",
        f"  - Backward Pass: {result.backward_pass_time:.4f}s ({backward_pct:.1f}%)",
        f"  - Optimizer Step: {result.optimizer_step_time:.4f}s ({opt_pct:.1f}%)",
        f"  - Unaccounted: {unaccounted:.4f}s ({unaccounted_pct:.1f}%)",
    ]

    # Timing distribution (p50, p95, p99)
    if result.timing_distribution is not None:
        tb = result.timing_distribution.get("total_batch")
        if tb:
            lines.append("")
            lines.append("  Batch Time Distribution:")
            lines.append(
                f"  - p50: {tb['p50']:.4f}s  p95: {tb['p95']:.4f}s  p99: {tb['p99']:.4f}s"
            )
            lines.append(
                f"  - min: {tb['min']:.4f}s  max: {tb['max']:.4f}s  std: {tb['std']:.4f}s"
            )

    lines.append("")
    lines.append("  Loss Information:")
    lines.append(f"  - Final Loss: {result.final_loss:.6f}")
    lines.append(f"  - Average Loss: {result.average_loss:.6f}")

    # Model info
    if result.model_info is not None:
        mi = result.model_info
        lines.append("")
        lines.append("  Model Information:")
        lines.append(
            f"  - Total Parameters: {mi.total_parameters:,}"
        )
        lines.append(
            f"  - Trainable Parameters: {mi.trainable_parameters:,}"
        )
        lines.append(
            f"  - Model Memory: {mi.model_memory_bytes / 1024**2:.1f}MB"
        )

    # GPU memory breakdown
    if result.gpu_memory_breakdown is not None:
        gm = result.gpu_memory_breakdown
        activation_est = max(
            0,
            gm.peak_allocated_bytes
            - gm.model_parameters_bytes
            - gm.optimizer_state_bytes,
        )
        lines.append("")
        lines.append("  GPU Memory Breakdown:")
        lines.append(
            f"  - Model Parameters: {gm.model_parameters_bytes / 1024**2:.1f}MB"
        )
        lines.append(
            f"  - Optimizer State: {gm.optimizer_state_bytes / 1024**2:.1f}MB"
        )
        lines.append(
            f"  - Activations (est.): {activation_est / 1024**2:.1f}MB"
        )
        lines.append(
            f"  - Peak Allocated: {gm.peak_allocated_bytes / 1024**2:.1f}MB"
        )
        lines.append(
            f"  - Peak Reserved: {gm.peak_reserved_bytes / 1024**2:.1f}MB"
        )
        lines.append(
            f"  - Total GPU Memory: {gm.total_gpu_memory_bytes / 1024**2:.1f}MB"
        )
        if gm.total_gpu_memory_bytes > 0:
            usage_pct = (
                gm.peak_allocated_bytes
                / gm.total_gpu_memory_bytes
                * 100
            )
            lines.append(f"  - Usage: {usage_pct:.1f}%")

    # Resource usage summary
    if result.resource_usage is not None:
        ru = result.resource_usage
        lines.append("")
        lines.append("  Resource Usage Summary:")
        lines.append(
            f"  - CPU Max Usage: {ru.cpu_max_percent:.1f}%"
        )
        lines.append(
            f"  - Memory Max Usage: {ru.memory_max_bytes / 1024**3:.1f}GB ({ru.memory_max_percent:.1f}%)"
        )
        if ru.gpu_max_percent is not None:
            avg_str = ""
            if ru.gpu_avg_percent is not None:
                avg_str = f", avg: {ru.gpu_avg_percent:.1f}%"
            lines.append(
                f"  - GPU Usage: max {ru.gpu_max_percent:.1f}%{avg_str}"
            )
        if (
            ru.gpu_memory_max_bytes is not None
            and ru.gpu_memory_total_bytes is not None
            and ru.gpu_memory_max_percent is not None
        ):
            lines.append(
                f"  - GPU Memory Max Usage: "
                f"{ru.gpu_memory_max_bytes / 1024**3:.1f}GB / "
                f"{ru.gpu_memory_total_bytes / 1024**3:.1f}GB "
                f"({ru.gpu_memory_max_percent:.1f}%)"
            )

    return "\n".join(lines)


def _generate_recommendations(
    result: BenchmarkResult,
    device: torch.device,
) -> list[str]:
    """タイミング分析に基づく推奨事項を生成する．"""
    recommendations: list[str] = []

    # Batch time analysis
    if result.average_batch_time > 0.1:
        recommendations.append(
            "Consider increasing batch size for better GPU utilization"
        )
    elif result.average_batch_time < 0.01:
        recommendations.append(
            "Batch size might be too large, consider reducing "
            "for memory efficiency"
        )

    # Data loading analysis
    if result.average_batch_time > 0:
        data_loading_pct = (
            result.data_loading_time
            / result.average_batch_time
            * 100
        )
        if data_loading_pct > 20:
            recommendations.append(
                "Data loading is a bottleneck "
                f"({data_loading_pct:.0f}% of batch time) - "
                "consider increasing DataLoader workers or "
                "enabling prefetch"
            )

    # GPU transfer analysis
    if device.type == "cuda" and result.average_batch_time > 0:
        gpu_transfer_pct = (
            result.gpu_transfer_time
            / result.average_batch_time
            * 100
        )
        if gpu_transfer_pct > 10:
            recommendations.append(
                "GPU transfer is slow - ensure pin_memory=True "
                "and consider larger batch sizes"
            )

    # Throughput analysis
    if (
        result.samples_per_second < 1000
        and device.type == "cuda"
    ):
        recommendations.append(
            "Low throughput detected - consider optimizing "
            "batch size, DataLoader workers, or model compilation"
        )

    # Timing variability analysis
    if result.timing_distribution is not None:
        tb = result.timing_distribution.get("total_batch")
        if tb and tb["mean"] > 0:
            cv = tb["std"] / tb["mean"]
            if cv > 0.3:
                recommendations.append(
                    f"High batch time variability (CV={cv:.2f}) - "
                    "possible I/O stalls or GC interference"
                )

    # GPU memory utilization analysis
    if result.gpu_memory_breakdown is not None:
        gm = result.gpu_memory_breakdown
        if gm.total_gpu_memory_bytes > 0:
            usage_pct = (
                gm.peak_allocated_bytes
                / gm.total_gpu_memory_bytes
                * 100
            )
            remaining_mb = (
                gm.total_gpu_memory_bytes
                - gm.peak_allocated_bytes
            ) / 1024**2
            if usage_pct < 30:
                recommendations.append(
                    f"GPU memory underutilized ({usage_pct:.0f}%) - "
                    f"{remaining_mb:.0f}MB available, "
                    "consider increasing batch size"
                )

    # GPU utilization analysis
    if (
        result.resource_usage is not None
        and result.resource_usage.gpu_avg_percent is not None
    ):
        avg_gpu = result.resource_usage.gpu_avg_percent
        if avg_gpu < 50:
            recommendations.append(
                f"Low average GPU utilization ({avg_gpu:.0f}%) - "
                "GPU is idle for significant time, "
                "likely data loading bottleneck"
            )

    if not recommendations:
        recommendations.append(
            "No significant bottlenecks detected"
        )

    return recommendations
