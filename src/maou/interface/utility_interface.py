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
    cache_transforms: Optional[bool] = None,
    gce_parameter: Optional[float] = None,
    policy_loss_ratio: Optional[float] = None,
    value_loss_ratio: Optional[float] = None,
    learning_ratio: Optional[float] = None,
    momentum: Optional[float] = None,
    optimizer_name: Optional[str] = None,
    optimizer_beta1: Optional[float] = None,
    optimizer_beta2: Optional[float] = None,
    optimizer_eps: Optional[float] = None,
    warmup_batches: Optional[int] = None,
    max_batches: Optional[int] = None,
    enable_profiling: Optional[bool] = None,
    run_validation: Optional[bool] = None,
    sample_ratio: Optional[float] = None,
    enable_resource_monitoring: Optional[bool] = None,
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
        cache_transforms: Enable in-memory caching of dataset transforms
        gce_parameter: GCE loss hyperparameter
        policy_loss_ratio: Policy loss weight
        value_loss_ratio: Value loss weight
        learning_ratio: Learning rate
        momentum: Optimizer momentum
        optimizer_name: Optimizer selection ('adamw' or 'sgd')
        optimizer_beta1: AdamW beta1 parameter
        optimizer_beta2: AdamW beta2 parameter
        optimizer_eps: AdamW epsilon parameter
        warmup_batches: Number of warmup batches to exclude from timing
        max_batches: Maximum number of batches to process
        enable_profiling: Enable PyTorch profiler for detailed analysis
        run_validation: Also run validation benchmark (inference only)
        sample_ratio: Ratio of data to sample for cloud sources (0.01-1.0)
        enable_resource_monitoring: Enable CPU, memory, and GPU usage monitoring

    Returns:
        JSON string with benchmark results and recommendations
    """

    if datasource_type not in ("hcpe", "preprocess"):
        raise ValueError(
            f"Data source type `{datasource_type}` is invalid."
        )

    if test_ratio is None:
        test_ratio = 0.2
    elif not 0.0 < test_ratio < 1.0:
        raise ValueError(
            f"test_ratio must be between 0 and 1, got {test_ratio}"
        )

    if batch_size is None:
        batch_size = 1000
    elif batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}"
        )

    if dataloader_workers is None:
        dataloader_workers = 0
    elif dataloader_workers < 0:
        raise ValueError(
            "dataloader_workers must be non-negative, "
            f"got {dataloader_workers}"
        )

    if pin_memory is None:
        pin_memory = False

    if prefetch_factor is None:
        prefetch_factor = 2
    if cache_transforms is None:
        cache_transforms_enabled = datasource_type == "hcpe"
    else:
        cache_transforms_enabled = cache_transforms
    elif prefetch_factor <= 0:
        raise ValueError(
            f"prefetch_factor must be positive, got {prefetch_factor}"
        )

    if gce_parameter is None:
        gce_parameter = 0.7
    elif not 0.0 < gce_parameter <= 1.0:
        raise ValueError(
            f"gce_parameter must be between 0 and 1, got {gce_parameter}"
        )

    if policy_loss_ratio is None:
        policy_loss_ratio = 1.0
    elif policy_loss_ratio <= 0:
        raise ValueError(
            "policy_loss_ratio must be positive, "
            f"got {policy_loss_ratio}"
        )

    if value_loss_ratio is None:
        value_loss_ratio = 1.0
    elif value_loss_ratio <= 0:
        raise ValueError(
            "value_loss_ratio must be positive, "
            f"got {value_loss_ratio}"
        )

    if learning_ratio is None:
        learning_ratio = 0.01
    elif learning_ratio <= 0:
        raise ValueError(
            f"learning_ratio must be positive, got {learning_ratio}"
        )

    if momentum is None:
        momentum = 0.9
    elif not 0.0 <= momentum <= 1.0:
        raise ValueError(
            f"momentum must be between 0 and 1, got {momentum}"
        )

    if optimizer_name is None:
        optimizer_name = "adamw"
    optimizer_key = optimizer_name.lower()
    if optimizer_key not in {"adamw", "sgd"}:
        raise ValueError(
            "optimizer_name must be 'adamw' or 'sgd', "
            f"got {optimizer_name}"
        )

    if optimizer_beta1 is None:
        optimizer_beta1 = 0.9
    elif not 0.0 < optimizer_beta1 < 1.0:
        raise ValueError(
            "optimizer_beta1 must be between 0 and 1, "
            f"got {optimizer_beta1}"
        )

    if optimizer_beta2 is None:
        optimizer_beta2 = 0.999
    elif not 0.0 < optimizer_beta2 < 1.0:
        raise ValueError(
            "optimizer_beta2 must be between 0 and 1, "
            f"got {optimizer_beta2}"
        )

    if optimizer_beta2 <= optimizer_beta1:
        raise ValueError(
            "optimizer_beta2 must be greater than optimizer_beta1 "
            f"(got {optimizer_beta1} and {optimizer_beta2})"
        )

    if optimizer_eps is None:
        optimizer_eps = 1e-8
    elif optimizer_eps <= 0:
        raise ValueError(
            f"optimizer_eps must be positive, got {optimizer_eps}"
        )

    if warmup_batches is None:
        warmup_batches = 5
    elif warmup_batches < 0:
        raise ValueError(
            f"warmup_batches must be non-negative, got {warmup_batches}"
        )

    if max_batches is None:
        max_batches = 100
    elif max_batches <= 0:
        raise ValueError(
            f"max_batches must be positive, got {max_batches}"
        )

    if enable_profiling is None:
        enable_profiling = False

    if run_validation is None:
        run_validation = False

    if enable_resource_monitoring is None:
        enable_resource_monitoring = False

    if (
        sample_ratio is not None
        and not 0.01 <= sample_ratio <= 1.0
    ):
        raise ValueError(
            f"sample_ratio must be between 0.01 and 1.0, got {sample_ratio}"
        )

    config = TrainingBenchmarkConfig(
        datasource=datasource,
        datasource_type=datasource_type,
        gpu=gpu,
        compilation=compilation,
        test_ratio=test_ratio,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        cache_transforms=cache_transforms_enabled,
        gce_parameter=gce_parameter,
        policy_loss_ratio=policy_loss_ratio,
        value_loss_ratio=value_loss_ratio,
        learning_ratio=learning_ratio,
        momentum=momentum,
        optimizer_name=optimizer_key,
        optimizer_beta1=optimizer_beta1,
        optimizer_beta2=optimizer_beta2,
        optimizer_eps=optimizer_eps,
        warmup_batches=warmup_batches,
        max_batches=max_batches,
        enable_profiling=enable_profiling,
        run_validation=run_validation,
        sample_ratio=sample_ratio,
        enable_resource_monitoring=enable_resource_monitoring,
    )

    use_case = TrainingBenchmarkUseCase()
    return use_case.execute(config)
