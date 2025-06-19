import json
import logging
from typing import Optional

import torch

from maou.app.learning.dl import LearningDataSource
from maou.app.utility.dataloader_benchmark import BenchmarkConfig, DataLoaderBenchmark

logger: logging.Logger = logging.getLogger(__name__)


def benchmark_dataloader(
    datasource: LearningDataSource.DataSourceSpliter,
    datasource_type: str,
    *,
    gpu: Optional[str] = None,
    batch_size: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    num_batches: Optional[int] = None,
) -> str:
    """Benchmark DataLoader configurations to find optimal parameters.

    Args:
        datasource: Training data source splitter
        datasource_type: Type of data source ('hcpe' or 'preprocess')
        gpu: GPU device to use for benchmarking
        batch_size: Batch size for benchmarking
        pin_memory: Enable pinned memory for GPU transfers
        num_batches: Number of batches to process per configuration

    Returns:
        JSON string with benchmark results and recommendations
    """
    # Validate datasource type
    if datasource_type not in ("hcpe", "preprocess"):
        raise ValueError(f"Data source type `{datasource_type}` is invalid.")

    # Set device
    if gpu is not None and gpu != "cpu":
        device = torch.device(gpu)
        logger.info(f"Benchmarking on GPU: {torch.cuda.get_device_name(device)}")
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")
        logger.info("Benchmarking on CPU")

    # Set batch size (default: 256 for benchmarking)
    if batch_size is None:
        batch_size = 256
    elif batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Set pin_memory (default: True for GPU, False for CPU)
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    # Set number of batches for benchmark (default: 100)
    if num_batches is None:
        num_batches = 100
    elif num_batches <= 0:
        raise ValueError(f"num_batches must be positive, got {num_batches}")

    logger.info("Benchmark configuration:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Pin memory: {pin_memory}")
    logger.info(f"  Batches per test: {num_batches}")

    # Get training datasource (ignore test split for benchmarking)
    training_datasource, _ = datasource.train_test_split(test_ratio=0.1)

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
    formatted_results = benchmark.format_results(results, optimal)

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

