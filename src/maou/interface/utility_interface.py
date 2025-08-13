import json
import logging
from typing import Optional

import torch

from maou.app.learning.dl import LearningDataSource
from maou.app.utility.dataloader_benchmark import (
    BenchmarkConfig,
    DataLoaderBenchmark,
)
from maou.app.utility.training_benchmark import (
    TrainingBenchmarkConfig,
    TrainingBenchmarkUseCase,
)

logger: logging.Logger = logging.getLogger(__name__)


def benchmark_dataloader(
    datasource: LearningDataSource.DataSourceSpliter,
    datasource_type: str,
    *,
    gpu: Optional[str] = None,
    batch_size: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    num_batches: Optional[int] = None,
    sample_ratio: Optional[float] = None,
) -> str:
    """Benchmark DataLoader configurations to find optimal parameters.

    Args:
        datasource: Training data source splitter
        datasource_type: Type of data source ('hcpe' or 'preprocess')
        gpu: GPU device to use for benchmarking
        batch_size: Batch size for benchmarking
        pin_memory: Enable pinned memory for GPU transfers
        num_batches: Number of batches to process per configuration
        sample_ratio: Ratio of data to sample for cloud sources (0.01-1.0)

    Returns:
        JSON string with benchmark results and recommendations
    """
    # Validate datasource type
    if datasource_type not in ("hcpe", "preprocess"):
        raise ValueError(
            f"Data source type `{datasource_type}` is invalid."
        )

    # Set device
    if gpu is not None and gpu != "cpu":
        device = torch.device(gpu)
        logger.info(
            f"Benchmarking on GPU: {torch.cuda.get_device_name(device)}"
        )
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")
        logger.info("Benchmarking on CPU")

    # Set batch size (default: 256 for benchmarking)
    if batch_size is None:
        batch_size = 256
    elif batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}"
        )

    # Set pin_memory (default: True for GPU, False for CPU)
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    # Set number of batches for benchmark (default: 100)
    if num_batches is None:
        num_batches = 100
    elif num_batches <= 0:
        raise ValueError(
            f"num_batches must be positive, got {num_batches}"
        )

    # Set sample ratio for cloud sources (default: None, means use full data)
    if sample_ratio is not None:
        if not 0.01 <= sample_ratio <= 1.0:
            raise ValueError(
                f"sample_ratio must be between 0.01 and 1.0, got {sample_ratio}"
            )

    logger.info("Benchmark configuration:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Pin memory: {pin_memory}")
    logger.info(f"  Batches per test: {num_batches}")
    if sample_ratio is not None:
        logger.info(f"  Sample ratio: {sample_ratio:.1%}")

    # Get training datasource (ignore test split for benchmarking)
    training_datasource, _ = datasource.train_test_split(
        test_ratio=0.1
    )

    # Create benchmark configuration
    config = BenchmarkConfig(
        datasource=training_datasource,
        datasource_type=datasource_type,
        batch_size=batch_size,
        device=device,
        pin_memory=pin_memory,
        num_batches=num_batches,
    )

    # Run benchmark
    benchmark = DataLoaderBenchmark(config)
    results, optimal = benchmark.run_benchmark()

    # Format results
    formatted_results = benchmark.format_results(
        results, optimal
    )

    # Create comprehensive output
    output = {
        "benchmark_results": formatted_results,
        "optimal_config": {
            "num_workers": optimal.num_workers,
            "prefetch_factor": optimal.prefetch_factor,
            "pin_memory": optimal.pin_memory,
            "avg_batch_time": optimal.avg_batch_time,
            "total_time": optimal.time_taken,
        },
        "all_results": [
            {
                "num_workers": r.num_workers,
                "prefetch_factor": r.prefetch_factor,
                "pin_memory": r.pin_memory,
                "time_taken": r.time_taken,
                "avg_batch_time": r.avg_batch_time,
                "batches_processed": r.batches_processed,
            }
            for r in results
        ],
        "device": str(device),
        "batch_size": batch_size,
        "num_batches": num_batches,
    }

    return json.dumps(output, indent=2)


def benchmark_training(
    datasource: LearningDataSource.DataSourceSpliter,
    datasource_type: str,
    *,
    gpu: Optional[str] = None,
    compilation: bool = False,
    test_ratio: Optional[float] = None,
    batch_size: Optional[int] = None,
    dataloader_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    gce_parameter: Optional[float] = None,
    policy_loss_ratio: Optional[float] = None,
    value_loss_ratio: Optional[float] = None,
    learning_ratio: Optional[float] = None,
    momentum: Optional[float] = None,
    warmup_batches: Optional[int] = None,
    max_batches: Optional[int] = None,
    enable_profiling: Optional[bool] = None,
    run_validation: Optional[bool] = None,
    sample_ratio: Optional[float] = None,
) -> str:
    """
    Benchmark single epoch training performance with detailed timing analysis.

    Args:
        datasource: Training data source splitter
        datasource_type: Type of data source ('hcpe' or 'preprocess')
        gpu: GPU device to use for benchmarking
        compilation: Whether to compile the model
        test_ratio: Test set ratio for validation benchmark
        batch_size: Training batch size
        dataloader_workers: Number of DataLoader workers
        pin_memory: Enable pinned memory for GPU transfers
        prefetch_factor: Number of batches loaded in advance by each worker
        gce_parameter: GCE loss hyperparameter
        policy_loss_ratio: Policy loss weight
        value_loss_ratio: Value loss weight
        learning_ratio: Learning rate
        momentum: Optimizer momentum
        warmup_batches: Number of warmup batches to exclude from timing
        max_batches: Maximum number of batches to process
        enable_profiling: Enable PyTorch profiler for detailed analysis
        run_validation: Also run validation benchmark (inference only)
        sample_ratio: Ratio of data to sample for cloud sources (0.01-1.0)

    Returns:
        JSON string with benchmark results and recommendations
    """
    # Create configuration with defaults
    config = TrainingBenchmarkConfig(
        datasource=datasource,
        datasource_type=datasource_type,
        gpu=gpu,
        compilation=compilation,
        test_ratio=test_ratio or 0.2,
        batch_size=batch_size or 1000,
        dataloader_workers=dataloader_workers or 0,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor or 2,
        gce_parameter=gce_parameter or 0.7,
        policy_loss_ratio=policy_loss_ratio or 1.0,
        value_loss_ratio=value_loss_ratio or 1.0,
        learning_ratio=learning_ratio or 0.01,
        momentum=momentum or 0.9,
        warmup_batches=warmup_batches or 5,
        max_batches=max_batches or 100,
        enable_profiling=enable_profiling or False,
        run_validation=run_validation or False,
        sample_ratio=sample_ratio,
    )

    # Validate sample_ratio
    if (
        sample_ratio is not None
        and not 0.01 <= sample_ratio <= 1.0
    ):
        raise ValueError(
            f"sample_ratio must be between 0.01 and 1.0, got {sample_ratio}"
        )

    # Execute use case
    use_case = TrainingBenchmarkUseCase()
    return use_case.execute(config)
