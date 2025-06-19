import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from maou.app.learning.dataset import DataSource, KifDataset
from maou.app.pre_process.transform import Transform


@dataclass(frozen=True)
class BenchmarkResult:
    """DataLoader benchmark result for a specific configuration."""

    num_workers: int
    prefetch_factor: Optional[int]
    pin_memory: bool
    time_taken: float
    batches_processed: int
    avg_batch_time: float


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for DataLoader benchmarking."""

    datasource: DataSource
    datasource_type: str
    batch_size: int
    device: torch.device
    pin_memory: bool
    num_batches: int = 100
    num_workers_to_test: Optional[List[int]] = None
    prefetch_factors_to_test: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.num_workers_to_test is None:
            object.__setattr__(self, "num_workers_to_test", [0, 2, 4, 8, 16])
        if self.prefetch_factors_to_test is None:
            object.__setattr__(self, "prefetch_factors_to_test", [1, 2, 4, 8])


class DataLoaderBenchmark:
    """Benchmark utility for finding optimal DataLoader parameters."""

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.logger.info(
            f"Initializing DataLoader benchmark on device: {config.device}"
        )

    def benchmark_configuration(
        self, num_workers: int, prefetch_factor: Optional[int]
    ) -> BenchmarkResult:
        """Benchmark a specific DataLoader configuration."""
        self.logger.info(
            f"Benchmarking num_workers={num_workers}, "
            f"prefetch_factor={prefetch_factor}, "
            f"pin_memory={self.config.pin_memory}"
        )

        # Create dataset
        dataset: KifDataset
        if self.config.datasource_type == "hcpe":
            transform = Transform()
        elif self.config.datasource_type == "preprocess":
            transform = None
        else:
            raise ValueError(
                f"Data source type `{self.config.datasource_type}` is invalid."
            )

        dataset = KifDataset(
            datasource=self.config.datasource,
            transform=transform,
        )

        # Setup worker initialization for reproducibility
        def worker_init_fn(worker_id: int) -> None:
            import random

            import numpy as np

            # Use worker_id to avoid unused variable warning
            worker_seed = (torch.initial_seed() + worker_id) % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        # Create DataLoader with specified configuration
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=True,
            timeout=120 if num_workers > 0 else 0,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
        )

        # Warmup phase to initialize workers and CUDA contexts
        warmup_batches = min(5, len(dataloader))
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break
            inputs, _ = batch
            if self.config.device.type == "cuda":
                inputs = inputs.to(self.config.device, non_blocking=True)
                torch.cuda.synchronize()  # Ensure CUDA operations complete

        # Benchmark phase
        start_time = time.time()
        batches_processed = 0

        for i, batch in enumerate(dataloader):
            if i >= self.config.num_batches:
                break

            inputs, (labels_policy, labels_value, legal_move_mask) = batch

            # Simulate GPU transfer if using CUDA
            if self.config.device.type == "cuda":
                inputs = inputs.to(self.config.device, non_blocking=True)
                labels_policy = labels_policy.to(self.config.device, non_blocking=True)
                labels_value = labels_value.to(self.config.device, non_blocking=True)
                legal_move_mask = legal_move_mask.to(
                    self.config.device, non_blocking=True
                )
                # Synchronize to ensure all transfers complete
                torch.cuda.synchronize()

            batches_processed += 1

        end_time = time.time()
        time_taken = end_time - start_time
        avg_batch_time = time_taken / batches_processed if batches_processed > 0 else 0

        # Clean up dataloader workers
        del dataloader

        return BenchmarkResult(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=self.config.pin_memory,
            time_taken=time_taken,
            batches_processed=batches_processed,
            avg_batch_time=avg_batch_time,
        )

    def run_benchmark(self) -> Tuple[List[BenchmarkResult], BenchmarkResult]:
        """Run comprehensive DataLoader benchmark.

        Returns:
            Tuple of (all results, optimal result)
        """
        self.logger.info("Starting DataLoader benchmark")

        # Ensure we have valid test configurations
        num_workers_list = self.config.num_workers_to_test or [0, 2, 4, 8, 16]
        prefetch_factors_list = self.config.prefetch_factors_to_test or [1, 2, 4, 8]

        self.logger.info(f"Testing {len(num_workers_list)} worker configurations")
        self.logger.info(
            f"Processing {self.config.num_batches} batches per configuration"
        )

        results: List[BenchmarkResult] = []

        # Test different num_workers configurations
        for num_workers in num_workers_list:
            if num_workers == 0:
                # No prefetch_factor for single-threaded loading
                result = self.benchmark_configuration(
                    num_workers=num_workers, prefetch_factor=None
                )
                results.append(result)
            else:
                # Test different prefetch_factors for multi-threaded loading
                for prefetch_factor in prefetch_factors_list:
                    result = self.benchmark_configuration(
                        num_workers=num_workers, prefetch_factor=prefetch_factor
                    )
                    results.append(result)

        # Find optimal configuration (lowest average batch time)
        optimal_result = min(results, key=lambda r: r.avg_batch_time)

        self.logger.info(
            f"Benchmark complete. Optimal configuration: "
            f"num_workers={optimal_result.num_workers}, "
            f"prefetch_factor={optimal_result.prefetch_factor}, "
            f"avg_batch_time={optimal_result.avg_batch_time:.4f}s"
        )

        return results, optimal_result

    def format_results(
        self, results: List[BenchmarkResult], optimal: BenchmarkResult
    ) -> Dict[str, str]:
        """Format benchmark results for display.

        Returns:
            Dictionary with formatted results
        """
        output = {}

        # Summary table
        summary_lines = ["DataLoader Benchmark Results:"]
        for result in sorted(results, key=lambda r: r.avg_batch_time):
            marker = " ← Optimal" if result == optimal else ""
            summary_lines.append(
                f"num_workers={result.num_workers}, "
                f"prefetch_factor={result.prefetch_factor}, "
                f"time={result.time_taken:.2f}s, "
                f"avg_batch={result.avg_batch_time:.4f}s{marker}"
            )

        output["Summary"] = "\n".join(summary_lines)

        # Recommendations
        rec_lines = ["Recommended configuration for maou learn-model:"]
        rec_lines.append(f"--dataloader-workers {optimal.num_workers}")
        if optimal.pin_memory:
            rec_lines.append("--pin-memory")
        if optimal.prefetch_factor is not None:
            rec_lines.append(f"--prefetch-factor {optimal.prefetch_factor}")

        output["Recommendations"] = "\n".join(rec_lines)

        # Performance insights
        insights = []
        if optimal.num_workers == 0:
            insights.append(
                "Single-threaded loading performed best. "
                "Consider CPU limitations or small dataset size."
            )
        elif optimal.num_workers >= 8:
            insights.append(
                "High worker count optimal. Ensure sufficient CPU cores and memory."
            )
        else:
            insights.append(
                f"Moderate worker count ({optimal.num_workers}) optimal. "
                "Good balance of parallelism and overhead."
            )

        fastest_time = min(r.avg_batch_time for r in results)
        slowest_time = max(r.avg_batch_time for r in results)
        speedup = slowest_time / fastest_time
        insights.append(
            f"Performance improvement: {speedup:.1f}x faster than worst configuration"
        )

        output["Insights"] = "\n".join(insights)

        return output

