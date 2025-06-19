import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from maou.app.learning.dl import LearningDataSource
from maou.app.learning.setup import TrainingSetup
from maou.domain.loss.loss_fn import MaskedGCELoss


@dataclass(frozen=True)
class BenchmarkResult:
    """単一エポックベンチマーク結果を格納するデータクラス．"""

    # 全体的なタイミング
    total_epoch_time: float
    average_batch_time: float
    total_batches: int

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

    def to_dict(self) -> Dict[str, float]:
        """ベンチマーク結果を辞書形式で返す．"""
        return {
            "total_epoch_time": self.total_epoch_time,
            "average_batch_time": self.average_batch_time,
            "total_batches": self.total_batches,
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


class SingleEpochBenchmark:
    """
    単一エポックの学習プロセスをベンチマークするクラス．

    Learning クラスの __train_one_epoch メソッドをベースに，
    詳細なタイミング測定機能を追加した実装．
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn_policy: MaskedGCELoss,
        loss_fn_value: torch.nn.Module,
        policy_loss_ratio: float,
        value_loss_ratio: float,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn_policy = loss_fn_policy
        self.loss_fn_value = loss_fn_value
        self.policy_loss_ratio = policy_loss_ratio
        self.value_loss_ratio = value_loss_ratio

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

        # モデルを学習モードに設定
        self.model.train(True)

        # タイミング測定用変数
        timing_stats: Dict[str, List[float]] = {
            "data_loading": [],
            "gpu_transfer": [],
            "forward_pass": [],
            "loss_computation": [],
            "backward_pass": [],
            "optimizer_step": [],
            "total_batch": [],
        }

        running_loss = 0.0
        total_loss = 0.0
        measured_batches = 0
        total_samples = 0
        previous_batch_end_time = None

        epoch_start_time = time.perf_counter()

        # PyTorchプロファイラーの設定
        profiler = None
        if enable_profiling:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./profiler_logs"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler.start()

        try:
            for batch_idx, data in tqdm(enumerate(dataloader), desc="Benchmarking"):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch_start_time = time.perf_counter()

                # データローディング時間の測定
                if previous_batch_end_time is not None:
                    data_load_time = batch_start_time - previous_batch_end_time
                else:
                    data_load_time = 0.0  # 最初のバッチでは測定しない

                # データの展開
                inputs, (labels_policy, labels_value, legal_move_mask) = data
                batch_size = inputs.size(0)
                total_samples += batch_size

                # GPU転送時間の測定
                gpu_transfer_start = time.perf_counter()
                inputs = inputs.to(self.device, non_blocking=True)
                labels_policy = labels_policy.to(self.device, non_blocking=True)
                labels_value = labels_value.to(self.device, non_blocking=True)
                legal_move_mask = legal_move_mask.to(self.device, non_blocking=True)

                # GPU同期（正確な転送時間測定のため）
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                gpu_transfer_time = time.perf_counter() - gpu_transfer_start

                # 前のバッチの勾配をクリア
                self.optimizer.zero_grad()

                # 順伝播時間の測定
                forward_start = time.perf_counter()
                outputs_policy, outputs_value = self.model(inputs)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                forward_time = time.perf_counter() - forward_start

                # 損失計算時間の測定
                loss_start = time.perf_counter()
                loss = self.policy_loss_ratio * self.loss_fn_policy(
                    outputs_policy, labels_policy, legal_move_mask
                ) + self.value_loss_ratio * self.loss_fn_value(
                    outputs_value, labels_value
                )
                loss_time = time.perf_counter() - loss_start

                # 逆伝播時間の測定
                backward_start = time.perf_counter()
                loss.backward()

                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                backward_time = time.perf_counter() - backward_start

                # オプティマイザステップ時間の測定
                step_start = time.perf_counter()
                self.optimizer.step()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                step_time = time.perf_counter() - step_start

                batch_total_time = time.perf_counter() - batch_start_time
                batch_end_time = time.perf_counter()

                # 損失の累積
                loss_value = loss.item()
                running_loss += loss_value
                total_loss += loss_value

                # ウォームアップ期間後のタイミング統計を記録
                if batch_idx >= warmup_batches:
                    timing_stats["data_loading"].append(data_load_time)
                    timing_stats["gpu_transfer"].append(gpu_transfer_time)
                    timing_stats["forward_pass"].append(forward_time)
                    timing_stats["loss_computation"].append(loss_time)
                    timing_stats["backward_pass"].append(backward_time)
                    timing_stats["optimizer_step"].append(step_time)
                    timing_stats["total_batch"].append(batch_total_time)
                    measured_batches += 1

                # 次のバッチのデータローディング時間測定のために記録
                previous_batch_end_time = batch_end_time

                # プロファイラーのステップ
                if profiler is not None:
                    profiler.step()

                # 定期的な進捗報告
                if batch_idx > 0 and batch_idx % max(1, len(dataloader) // 10) == 0:
                    avg_loss = running_loss / max(1, batch_idx + 1 - warmup_batches)
                    self.logger.info(
                        f"  Batch {batch_idx + 1}: avg_loss={avg_loss:.6f}"
                    )

        finally:
            if profiler is not None:
                profiler.stop()

        epoch_total_time = time.perf_counter() - epoch_start_time

        # 統計の計算
        if not timing_stats["total_batch"]:
            raise RuntimeError("No batches were processed for timing measurement")

        avg_data_loading = sum(timing_stats["data_loading"]) / len(
            timing_stats["data_loading"]
        )
        avg_gpu_transfer = sum(timing_stats["gpu_transfer"]) / len(
            timing_stats["gpu_transfer"]
        )
        avg_forward = sum(timing_stats["forward_pass"]) / len(
            timing_stats["forward_pass"]
        )
        avg_loss_comp = sum(timing_stats["loss_computation"]) / len(
            timing_stats["loss_computation"]
        )
        avg_backward = sum(timing_stats["backward_pass"]) / len(
            timing_stats["backward_pass"]
        )
        avg_step = sum(timing_stats["optimizer_step"]) / len(
            timing_stats["optimizer_step"]
        )
        avg_batch = sum(timing_stats["total_batch"]) / len(timing_stats["total_batch"])

        final_loss = loss_value if "loss_value" in locals() else 0.0
        average_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0.0

        # パフォーマンス指標の計算
        samples_per_second = total_samples / epoch_total_time
        batches_per_second = (batch_idx + 1) / epoch_total_time

        self.logger.info(
            f"Benchmark completed: {batch_idx + 1} batches in {epoch_total_time:.2f}s"
        )
        self.logger.info(f"Average batch time: {avg_batch:.4f}s")
        self.logger.info(f"Samples per second: {samples_per_second:.1f}")

        return BenchmarkResult(
            total_epoch_time=epoch_total_time,
            average_batch_time=avg_batch,
            total_batches=batch_idx + 1,
            data_loading_time=avg_data_loading,
            gpu_transfer_time=avg_gpu_transfer,
            forward_pass_time=avg_forward,
            loss_computation_time=avg_loss_comp,
            backward_pass_time=avg_backward,
            optimizer_step_time=avg_step,
            final_loss=final_loss,
            average_loss=average_loss,
            samples_per_second=samples_per_second,
            batches_per_second=batches_per_second,
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

        # モデルを評価モードに設定
        self.model.eval()

        timing_stats: Dict[str, List[float]] = {
            "data_loading": [],
            "gpu_transfer": [],
            "forward_pass": [],
            "loss_computation": [],
            "total_batch": [],
        }

        total_loss = 0.0
        total_samples = 0

        epoch_start_time = time.perf_counter()

        # 勾配計算を無効化して推論のみ実行
        with torch.no_grad():
            for batch_idx, data in tqdm(
                enumerate(dataloader), desc="Validation Benchmark"
            ):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch_start_time = time.perf_counter()

                # データの展開
                inputs, (labels_policy, labels_value, legal_move_mask) = data
                batch_size = inputs.size(0)
                total_samples += batch_size

                # GPU転送時間の測定
                gpu_transfer_start = time.perf_counter()
                inputs = inputs.to(self.device, non_blocking=True)
                labels_policy = labels_policy.to(self.device, non_blocking=True)
                labels_value = labels_value.to(self.device, non_blocking=True)
                legal_move_mask = legal_move_mask.to(self.device, non_blocking=True)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                gpu_transfer_time = time.perf_counter() - gpu_transfer_start

                # 順伝播時間の測定
                forward_start = time.perf_counter()
                outputs_policy, outputs_value = self.model(inputs)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                forward_time = time.perf_counter() - forward_start

                # 損失計算時間の測定
                loss_start = time.perf_counter()
                loss = self.policy_loss_ratio * self.loss_fn_policy(
                    outputs_policy, labels_policy, legal_move_mask
                ) + self.value_loss_ratio * self.loss_fn_value(
                    outputs_value, labels_value
                )
                loss_time = time.perf_counter() - loss_start

                batch_total_time = time.perf_counter() - batch_start_time

                # 損失の累積
                loss_value = loss.item()
                total_loss += loss_value

                # タイミング統計を記録
                data_load_time = 0.0  # バリデーションでは詳細な測定をスキップ
                timing_stats["data_loading"].append(data_load_time)
                timing_stats["gpu_transfer"].append(gpu_transfer_time)
                timing_stats["forward_pass"].append(forward_time)
                timing_stats["loss_computation"].append(loss_time)
                timing_stats["total_batch"].append(batch_total_time)

        epoch_total_time = time.perf_counter() - epoch_start_time

        # 統計の計算
        if not timing_stats["total_batch"]:
            raise RuntimeError("No batches were processed for timing measurement")

        avg_data_loading = sum(timing_stats["data_loading"]) / len(
            timing_stats["data_loading"]
        )
        avg_gpu_transfer = sum(timing_stats["gpu_transfer"]) / len(
            timing_stats["gpu_transfer"]
        )
        avg_forward = sum(timing_stats["forward_pass"]) / len(
            timing_stats["forward_pass"]
        )
        avg_loss_comp = sum(timing_stats["loss_computation"]) / len(
            timing_stats["loss_computation"]
        )
        avg_batch = sum(timing_stats["total_batch"]) / len(timing_stats["total_batch"])

        final_loss = loss_value if "loss_value" in locals() else 0.0
        average_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0.0

        # パフォーマンス指標の計算
        samples_per_second = total_samples / epoch_total_time
        batches_per_second = (batch_idx + 1) / epoch_total_time

        self.logger.info(
            f"Validation benchmark completed: {batch_idx + 1} batches in "
            f"{epoch_total_time:.2f}s"
        )
        self.logger.info(f"Average batch time: {avg_batch:.4f}s")
        self.logger.info(f"Samples per second: {samples_per_second:.1f}")

        return BenchmarkResult(
            total_epoch_time=epoch_total_time,
            average_batch_time=avg_batch,
            total_batches=batch_idx + 1,
            data_loading_time=avg_data_loading,
            gpu_transfer_time=avg_gpu_transfer,
            forward_pass_time=avg_forward,
            loss_computation_time=avg_loss_comp,
            backward_pass_time=0.0,  # バリデーションでは逆伝播なし
            optimizer_step_time=0.0,  # バリデーションではオプティマイザステップなし
            final_loss=final_loss,
            average_loss=average_loss,
            samples_per_second=samples_per_second,
            batches_per_second=batches_per_second,
        )


@dataclass(frozen=True)
class TrainingBenchmarkConfig:
    """Configuration for training benchmark."""

    datasource: LearningDataSource.DataSourceSpliter
    datasource_type: str
    gpu: Optional[str] = None
    batch_size: int = 256
    dataloader_workers: int = 4
    pin_memory: Optional[bool] = None
    enable_prefetch: bool = False
    prefetch_factor: int = 2
    gce_parameter: float = 0.1
    policy_loss_ratio: float = 1.0
    value_loss_ratio: float = 1.0
    learning_ratio: float = 0.01
    momentum: float = 0.9
    warmup_batches: int = 5
    max_batches: Optional[int] = None
    enable_profiling: bool = False
    test_ratio: float = 0.2
    run_validation: bool = False
    sample_ratio: Optional[float] = None


class TrainingBenchmarkUseCase:
    """Use case for training performance benchmarking."""

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        pass

    def execute(self, config: TrainingBenchmarkConfig) -> str:
        """Execute training benchmark and return JSON results."""
        self.logger.info("Starting training benchmark use case")

        # Split data into training and validation sets
        training_datasource, validation_datasource = config.datasource.train_test_split(
            test_ratio=config.test_ratio
        )

        # Setup all training components using shared setup module
        device_config, dataloaders, model_components = (
            TrainingSetup.setup_training_components(
                training_datasource=training_datasource,
                validation_datasource=validation_datasource,
                datasource_type=config.datasource_type,
                gpu=config.gpu,
                batch_size=config.batch_size,
                dataloader_workers=config.dataloader_workers,
                pin_memory=config.pin_memory,
                enable_prefetch=config.enable_prefetch,
                prefetch_factor=config.prefetch_factor,
                gce_parameter=config.gce_parameter,
                learning_ratio=config.learning_ratio,
                momentum=config.momentum,
            )
        )

        training_loader, validation_loader = dataloaders
        device = device_config.device

        # Create benchmark instance
        benchmark = SingleEpochBenchmark(
            model=model_components.model,
            device=device,
            optimizer=model_components.optimizer,
            loss_fn_policy=model_components.loss_fn_policy,
            loss_fn_value=model_components.loss_fn_value,
            policy_loss_ratio=config.policy_loss_ratio,
            value_loss_ratio=config.value_loss_ratio,
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
                training_result.total_epoch_time / config.sample_ratio
            )
            estimated_total_batches = int(
                training_result.total_batches / config.sample_ratio
            )
            estimation_results = {
                "sample_ratio": config.sample_ratio,
                "estimated_full_epoch_time_seconds": estimated_full_epoch_time,
                "estimated_full_epoch_time_minutes": estimated_full_epoch_time / 60,
                "estimated_total_batches": estimated_total_batches,
                "actual_batches_processed": training_result.total_batches,
            }
            self.logger.info(f"Sample ratio: {config.sample_ratio:.1%}")
            self.logger.info(
                f"Estimated full epoch time: "
                f"{estimated_full_epoch_time / 60:.1f} minutes"
            )
            self.logger.info(f"Estimated total batches: {estimated_total_batches:,}")

        # Format results for display
        def format_timing_summary(result: BenchmarkResult, label: str) -> str:
            # Pre-calculate percentages to avoid long lines
            data_pct = result.data_loading_time / result.average_batch_time * 100
            gpu_pct = result.gpu_transfer_time / result.average_batch_time * 100
            forward_pct = result.forward_pass_time / result.average_batch_time * 100
            loss_pct = result.loss_computation_time / result.average_batch_time * 100
            backward_pct = result.backward_pass_time / result.average_batch_time * 100
            opt_pct = result.optimizer_step_time / result.average_batch_time * 100
            return f"""{label} Performance Summary:
  Total Time: {result.total_epoch_time:.2f}s
  Average Batch Time: {result.average_batch_time:.4f}s
  Samples per Second: {result.samples_per_second:.1f}
  Batches per Second: {result.batches_per_second:.2f}

  Detailed Timing Breakdown:
  - Data Loading: {result.data_loading_time:.4f}s ({data_pct:.1f}%)
  - GPU Transfer: {result.gpu_transfer_time:.4f}s ({gpu_pct:.1f}%)
  - Forward Pass: {result.forward_pass_time:.4f}s ({forward_pct:.1f}%)
  - Loss Computation: {result.loss_computation_time:.4f}s ({loss_pct:.1f}%)
  - Backward Pass: {result.backward_pass_time:.4f}s ({backward_pct:.1f}%)
  - Optimizer Step: {result.optimizer_step_time:.4f}s ({opt_pct:.1f}%)

  Loss Information:
  - Final Loss: {result.final_loss:.6f}
  - Average Loss: {result.average_loss:.6f}"""

        training_summary = format_timing_summary(training_result, "Training")

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
            training_result.data_loading_time / training_result.average_batch_time
        ) * 100
        if data_loading_percentage > 20:
            recommendations.append(
                "Data loading is a bottleneck - consider increasing DataLoader "
                "workers or enabling prefetch"
            )

        # GPU transfer analysis
        if device.type == "cuda":
            gpu_transfer_percentage = (
                training_result.gpu_transfer_time / training_result.average_batch_time
            ) * 100
            if gpu_transfer_percentage > 10:
                recommendations.append(
                    "GPU transfer is slow - ensure pin_memory=True and consider "
                    "larger batch sizes"
                )

        # Throughput analysis
        if training_result.samples_per_second < 1000 and device.type == "cuda":
            recommendations.append(
                "Low throughput detected - consider optimizing batch size, "
                "DataLoader workers, or model compilation"
            )

        recommendations_text = "Performance Recommendations:\n" + "\n".join(
            f"- {rec}" for rec in recommendations
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
                "enable_prefetch": config.enable_prefetch,
                "prefetch_factor": config.prefetch_factor,
                "warmup_batches": config.warmup_batches,
                "max_batches": config.max_batches,
                "enable_profiling": config.enable_profiling,
                "sample_ratio": config.sample_ratio,
            },
        }

        # Add validation results if available
        if validation_result is not None:
            validation_summary = format_timing_summary(validation_result, "Validation")
            # Dynamic dict key assignment for validation summary
            output["benchmark_results"]["ValidationSummary"] = (  # type: ignore
                validation_summary
            )
            output["validation_metrics"] = validation_result.to_dict()

        return json.dumps(output, indent=2)
