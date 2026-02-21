import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, cast

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from maou.app.learning.callbacks import (
    ResourceMonitoringCallback,
    TimingCallback,
)
from maou.app.learning.compilation import compile_module
from maou.app.learning.dl import LearningDataSource
from maou.app.learning.gpu_prefetcher import (
    calculate_recommended_buffer_size,
)
from maou.app.learning.network import (
    BackboneArchitecture,
    Network,
)
from maou.app.learning.resource_monitor import ResourceUsage
from maou.app.learning.setup import TrainingSetup
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

    # GPU Prefetch情報
    gpu_prefetch_enabled: bool
    gpu_prefetch_buffer_size: int

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

    def to_dict(self) -> Dict[str, float]:
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
            "gpu_prefetch_enabled": float(
                self.gpu_prefetch_enabled
            ),
            "gpu_prefetch_buffer_size": float(
                self.gpu_prefetch_buffer_size
            ),
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
    """Single epoch benchmark for training performance measurement."""

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
        enable_gpu_prefetch: bool = True,
        gpu_prefetch_buffer_size: int = 0,
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
        self.enable_gpu_prefetch = enable_gpu_prefetch
        self.gpu_prefetch_buffer_size = gpu_prefetch_buffer_size

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
            enable_gpu_prefetch=self.enable_gpu_prefetch,
            gpu_prefetch_buffer_size=self.gpu_prefetch_buffer_size,
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

        # GPU Prefetchの実効バッファサイズを決定
        effective_gpu_prefetch = (
            self.enable_gpu_prefetch
            and self.device.type == "cuda"
        )
        if effective_gpu_prefetch:
            if self.gpu_prefetch_buffer_size <= 0:
                effective_batch_size = (
                    dataloader.batch_size
                    if dataloader.batch_size is not None
                    else 256
                )
                effective_buffer_size = (
                    calculate_recommended_buffer_size(
                        effective_batch_size
                    )
                )
            else:
                effective_buffer_size = (
                    self.gpu_prefetch_buffer_size
                )
        else:
            effective_buffer_size = 0

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
            gpu_prefetch_enabled=effective_gpu_prefetch,
            gpu_prefetch_buffer_size=effective_buffer_size,
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
            enable_gpu_prefetch=self.enable_gpu_prefetch,
            gpu_prefetch_buffer_size=self.gpu_prefetch_buffer_size,
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

        # GPU Prefetchの実効バッファサイズを決定
        effective_gpu_prefetch = (
            self.enable_gpu_prefetch
            and self.device.type == "cuda"
        )
        if effective_gpu_prefetch:
            if self.gpu_prefetch_buffer_size <= 0:
                effective_batch_size = (
                    dataloader.batch_size
                    if dataloader.batch_size is not None
                    else 256
                )
                effective_buffer_size = (
                    calculate_recommended_buffer_size(
                        effective_batch_size
                    )
                )
            else:
                effective_buffer_size = (
                    self.gpu_prefetch_buffer_size
                )
        else:
            effective_buffer_size = 0

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
            gpu_prefetch_enabled=effective_gpu_prefetch,
            gpu_prefetch_buffer_size=effective_buffer_size,
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

    datasource: LearningDataSource.DataSourceSpliter
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
    warmup_batches: int = 5
    max_batches: Optional[int] = None
    enable_profiling: bool = False
    test_ratio: float = 0.2
    run_validation: bool = False
    sample_ratio: Optional[float] = None
    enable_resource_monitoring: bool = False
    model_architecture: BackboneArchitecture = "resnet"


class TrainingBenchmarkUseCase:
    """Use case for training performance benchmarking."""

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        pass

    def execute(self, config: TrainingBenchmarkConfig) -> str:
        """Execute training benchmark and return JSON results."""
        self.logger.info("Starting training benchmark use case")

        # Split data into training and validation sets
        training_datasource, validation_datasource = (
            config.datasource.train_test_split(
                test_ratio=config.test_ratio
            )
        )

        # Setup all training components using shared setup module
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

        if config.compilation:
            self.logger.info(
                "Compiling model with torch.compile for benchmarking (dynamic shapes disabled)"
            )
            model_components.model = compile_module(
                model_components.model
            )

        training_loader, validation_loader = dataloaders
        device = device_config.device

        # Get total number of batches in the full dataset
        total_batches_in_dataset = len(training_loader)

        # Create benchmark instance
        benchmark = SingleEpochBenchmark(
            model=model_components.model,
            device=device,
            optimizer=model_components.optimizer,
            loss_fn_policy=model_components.loss_fn_policy,
            loss_fn_value=model_components.loss_fn_value,
            policy_loss_ratio=config.policy_loss_ratio,
            value_loss_ratio=config.value_loss_ratio,
            enable_resource_monitoring=config.enable_resource_monitoring,
        )

        # Run training benchmark
        self.logger.info("Starting training benchmark...")
        training_result = benchmark.benchmark_epoch(
            training_loader,
            warmup_batches=config.warmup_batches,
            max_batches=config.max_batches,
            enable_profiling=config.enable_profiling,
        )

        # Run validation benchmark if requested
        validation_result = None
        if config.run_validation:
            self.logger.info("Starting validation benchmark...")
            validation_result = benchmark.benchmark_validation(
                validation_loader,
                max_batches=config.max_batches,
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
            # GPU Prefetch情報
            if result.gpu_prefetch_enabled:
                gpu_prefetch_info = f"  GPU Prefetch: enabled (buffer={result.gpu_prefetch_buffer_size})"
            else:
                gpu_prefetch_info = "  GPU Prefetch: disabled"

            data_label = (
                "Prefetch Wait"
                if result.gpu_prefetch_enabled
                else "Data Loading"
            )

            prefetch_note = ""
            if result.gpu_prefetch_enabled:
                prefetch_note = (
                    "\n\n  Note: GPU Prefetcher active"
                    " - 'Prefetch Wait' shows buffer fetch time,"
                    " not actual disk I/O time."
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
{gpu_prefetch_info}
  Estimated Full Epoch Time: {estimated_full_epoch_time_seconds:.2f}s ({estimated_full_epoch_time_minutes:.2f} minutes)
  Actual Average Time per Batch: {result.actual_average_batch_time:.4f}s
  Processing Time per Batch (excl. data loading): {result.average_batch_time:.4f}s
  Samples per Second: {result.samples_per_second:.1f}
  Batches per Second: {result.batches_per_second:.2f}

  Detailed Timing Breakdown (per batch, warmup excluded):
  - {data_label}: {result.data_loading_time:.4f}s ({data_pct:.1f}%)
  - GPU Transfer: {result.gpu_transfer_time:.4f}s ({gpu_pct:.1f}%)
  - Forward Pass: {result.forward_pass_time:.4f}s ({forward_pct:.1f}%)
  - Loss Computation: {result.loss_computation_time:.4f}s ({loss_pct:.1f}%)
  - Backward Pass: {result.backward_pass_time:.4f}s ({backward_pct:.1f}%)
  - Optimizer Step: {result.optimizer_step_time:.4f}s ({opt_pct:.1f}%)

  Loss Information:
  - Final Loss: {result.final_loss:.6f}
  - Average Loss: {result.average_loss:.6f}{resource_summary}{prefetch_note}"""

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

        return json.dumps(output, indent=2)
