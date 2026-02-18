import json
from pathlib import Path
from typing import Optional

import click

import maou.interface.learn as learn
import maou.interface.utility_interface as utility_interface
from maou.infra.console.common import (
    HAS_AWS,
    HAS_BIGQUERY,
    HAS_GCS,
    BigQueryDataSource,
    FileDataSource,
    FileSystem,
    GCSDataSource,
    S3DataSource,
    app_logger,
    handle_exception,
)


@click.command("benchmark-dataloader")
@click.option(
    "--input-path",
    help="Input file or directory path.",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--input-file-packed",
    type=bool,
    is_flag=True,
    help="Enable unpacking local numpy file.",
    default=False,
    required=False,
)
@click.option(
    "--input-dataset-id",
    help="BigQuery dataset ID for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-table-name",
    help="BigQuery table name for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-format",
    help="Input format: 'hcpe' or 'preprocess'.",
    type=str,
    default="hcpe",
    required=False,
)
@click.option(
    "--input-batch-size",
    help="Batch size for reading from BigQuery.",
    type=int,
    default=10000,
    required=False,
)
@click.option(
    "--input-max-cached-bytes",
    help="Max cache size in bytes for input (default: 500MB).",
    type=int,
    default=500 * 1024 * 1024,
    required=False,
)
@click.option(
    "--input-cache-mode",
    type=click.Choice(
        ["file", "memory", "mmap"], case_sensitive=False
    ),
    help="Cache strategy for local inputs (default: file). 'mmap' is deprecated, use 'file' instead.",
    default="file",
    required=False,
    show_default=True,
)
@click.option(
    "--input-clustering-key",
    help="BigQuery clustering key.",
    type=str,
    required=False,
)
@click.option(
    "--input-partitioning-key-date",
    help="BigQuery date partitioning key.",
    type=str,
    required=False,
)
@click.option(
    "--input-local-cache",
    type=bool,
    is_flag=True,
    help="Enable local caching of cloud data.",
    default=False,
    required=False,
)
@click.option(
    "--input-local-cache-dir",
    type=str,
    help="Directory path for storing the local cache of cloud data.",
    required=False,
)
@click.option(
    "--input-enable-bundling",
    type=bool,
    is_flag=True,
    help="Enable bundling of arrays for efficient local caching (1GB chunks).",
    default=False,
    required=False,
)
@click.option(
    "--input-bundle-size-gb",
    type=float,
    help="Target bundle size in GB for array bundling (default: 1.0).",
    default=1.0,
    required=False,
)
@click.option(
    "--input-gcs",
    type=bool,
    is_flag=True,
    help="Use GCS as input data source.",
    required=False,
)
@click.option(
    "--input-s3",
    type=bool,
    is_flag=True,
    help="Use S3 as input data source.",
    required=False,
)
@click.option(
    "--input-bucket-name",
    help="S3/GCS bucket name for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-prefix",
    help="S3/GCS prefix path for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-data-name",
    help="Name to identify the data in S3/GCS for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-max-workers",
    help="Number of parallel download threads for S3/GCS input (default: 8).",
    type=int,
    required=False,
    default=8,
)
@click.option(
    "--gpu",
    type=str,
    help="PyTorch device (e.g., 'cuda:0' or 'cpu').",
    required=False,
)
@click.option(
    "--batch-size",
    type=int,
    help="Batch size for benchmarking (default: 256).",
    required=False,
    default=256,
)
@click.option(
    "--pin-memory",
    type=bool,
    is_flag=True,
    help="Enable pinned memory for faster GPU transfers.",
    required=False,
)
@click.option(
    "--num-batches",
    type=int,
    help="Number of batches to process per configuration (default: 100).",
    required=False,
    default=100,
)
@click.option(
    "--sample-ratio",
    type=float,
    help="Ratio of data to sample for cloud sources (0.01-1.0, default: full data).",
    required=False,
)
@handle_exception
def benchmark_dataloader(
    input_path: Optional[Path],
    input_file_packed: bool,
    input_dataset_id: Optional[str],
    input_table_name: Optional[str],
    input_format: str,
    input_batch_size: int,
    input_max_cached_bytes: int,
    input_cache_mode: str,
    input_clustering_key: Optional[str],
    input_partitioning_key_date: Optional[str],
    input_local_cache: bool,
    input_local_cache_dir: Optional[str],
    input_enable_bundling: bool,
    input_bundle_size_gb: float,
    input_gcs: Optional[bool],
    input_s3: Optional[bool],
    input_bucket_name: Optional[str],
    input_prefix: Optional[str],
    input_data_name: Optional[str],
    input_max_workers: int,
    gpu: Optional[str],
    batch_size: int,
    pin_memory: Optional[bool],
    num_batches: int,
    sample_ratio: Optional[float],
) -> None:
    """Benchmark DataLoader configurations to find optimal parameters."""
    # Normalize cache_mode: "mmap" is deprecated, convert to "file"
    if input_cache_mode.lower() == "mmap":
        import warnings

        warnings.warn(
            "--input-cache-mode 'mmap' is deprecated. Use 'file' instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        input_cache_mode = "file"

    # Validate input_format early
    if input_format not in ("hcpe", "preprocess"):
        raise ValueError(
            "Please specify a valid input_format ('hcpe' or 'preprocess')."
        )
    # Convert input_format to array_type for data sources
    array_type = (
        "preprocessing"
        if input_format == "preprocess"
        else "hcpe"
    )

    # Check for mixing cloud providers for input
    cloud_input_count = sum(
        [
            bool(
                input_dataset_id is not None
                or input_table_name is not None
            ),
            bool(input_gcs),
            bool(input_s3),
        ]
    )
    if cloud_input_count > 1:
        error_msg = (
            "Cannot use multiple cloud providers for input simultaneously. "
            "Please choose only one: BigQuery, GCS, or S3."
        )
        app_logger.error(error_msg)
        raise ValueError(error_msg)

    # Initialize datasource (similar to learn_model command)
    if input_path is not None:
        if sample_ratio is not None:
            app_logger.warning(
                "sample_ratio is ignored for local file data source."
            )
        datasource = FileDataSource.FileDataSourceSpliter(
            file_paths=FileSystem.collect_files(input_path),
            array_type=array_type,
            bit_pack=input_file_packed,
            cache_mode=input_cache_mode,
        )
    elif (
        input_dataset_id is not None
        and input_table_name is not None
    ):
        if sample_ratio is not None:
            app_logger.info(
                f"Using BigQuery TABLESAMPLE with {sample_ratio:.1%} sampling ratio."
            )
        if HAS_BIGQUERY:
            try:
                if hasattr(
                    BigQueryDataSource,
                    "BigQueryDataSourceSpliter",
                ):
                    datasource = BigQueryDataSource.BigQueryDataSourceSpliter(
                        array_type=array_type,
                        dataset_id=input_dataset_id,
                        table_name=input_table_name,
                        batch_size=input_batch_size,
                        max_cached_bytes=input_max_cached_bytes,
                        clustering_key=input_clustering_key,
                        partitioning_key_date=input_partitioning_key_date,
                        use_local_cache=input_local_cache,
                        local_cache_dir=input_local_cache_dir,
                        sample_ratio=sample_ratio,
                    )
                else:
                    app_logger.error(
                        "BigQueryDataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "BigQueryDataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize BigQueryDataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "BigQuery input requested but required packages are not installed. "
                "Install with 'uv sync --extra gcp'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    elif (
        input_gcs
        and input_bucket_name is not None
        and input_prefix is not None
        and input_data_name is not None
        and input_local_cache_dir is not None
    ):
        if HAS_GCS:
            try:
                if hasattr(GCSDataSource, "DataSourceSpliter"):
                    datasource = GCSDataSource.DataSourceSpliter(
                        cls_ref=GCSDataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        local_cache_dir=input_local_cache_dir,
                        array_type=array_type,
                        max_workers=input_max_workers,
                        max_cached_bytes=input_max_cached_bytes,
                        sample_ratio=sample_ratio,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "DataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "DataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize DataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "GCS input requested but required packages are not installed. "
                "Install with 'uv sync --extra gcp'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    elif (
        input_s3
        and input_bucket_name is not None
        and input_prefix is not None
        and input_data_name is not None
        and input_local_cache_dir is not None
    ):
        if HAS_AWS:
            try:
                if hasattr(S3DataSource, "DataSourceSpliter"):
                    datasource = S3DataSource.DataSourceSpliter(
                        cls_ref=S3DataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=input_max_workers,
                        max_cached_bytes=input_max_cached_bytes,
                        sample_ratio=sample_ratio,
                        array_type=array_type,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "DataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "DataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize DataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "S3 input requested but required packages are not installed. "
                "Install with 'uv sync --extra aws'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    else:
        raise Exception(
            "Please specify an input directory, a BigQuery table, "
            "a GCS bucket, or an S3 bucket."
        )

    # Run benchmark
    result_json = utility_interface.benchmark_dataloader(
        datasource=datasource,
        datasource_type=input_format,
        gpu=gpu,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_batches=num_batches,
        sample_ratio=sample_ratio,
    )

    # Parse and display results
    result = json.loads(result_json)

    click.echo(result["benchmark_results"]["Summary"])
    click.echo()
    click.echo(result["benchmark_results"]["Recommendations"])
    click.echo()
    click.echo(result["benchmark_results"]["Insights"])


@click.command("benchmark-training")
@click.option(
    "--input-path",
    help="Input file or directory path.",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--input-file-packed",
    type=bool,
    is_flag=True,
    help="Enable unpacking local numpy file.",
    default=False,
    required=False,
)
@click.option(
    "--input-dataset-id",
    help="BigQuery dataset ID for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-table-name",
    help="BigQuery table name for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-format",
    help="Input format: 'hcpe' or 'preprocess'.",
    type=str,
    default="hcpe",
    required=False,
)
@click.option(
    "--input-batch-size",
    help="Batch size for reading from BigQuery.",
    type=int,
    default=10000,
    required=False,
)
@click.option(
    "--input-max-cached-bytes",
    help="Max cache size in bytes for input (default: 500MB).",
    type=int,
    default=500 * 1024 * 1024,
    required=False,
)
@click.option(
    "--input-cache-mode",
    type=click.Choice(
        ["file", "memory", "mmap"], case_sensitive=False
    ),
    help="Cache strategy for local inputs (default: file). 'mmap' is deprecated, use 'file' instead.",
    default="file",
    required=False,
    show_default=True,
)
@click.option(
    "--input-clustering-key",
    help="BigQuery clustering key.",
    type=str,
    required=False,
)
@click.option(
    "--input-partitioning-key-date",
    help="BigQuery date partitioning key.",
    type=str,
    required=False,
)
@click.option(
    "--input-local-cache",
    type=bool,
    is_flag=True,
    help="Enable local caching of cloud data.",
    default=False,
    required=False,
)
@click.option(
    "--input-local-cache-dir",
    type=str,
    help="Directory path for storing the local cache of cloud data.",
    required=False,
)
@click.option(
    "--input-enable-bundling",
    type=bool,
    is_flag=True,
    help="Enable bundling of arrays for efficient local caching (1GB chunks).",
    default=False,
    required=False,
)
@click.option(
    "--input-bundle-size-gb",
    type=float,
    help="Target bundle size in GB for array bundling (default: 1.0).",
    default=1.0,
    required=False,
)
@click.option(
    "--input-gcs",
    type=bool,
    is_flag=True,
    help="Use GCS as input data source.",
    required=False,
)
@click.option(
    "--input-s3",
    type=bool,
    is_flag=True,
    help="Use S3 as input data source.",
    required=False,
)
@click.option(
    "--input-bucket-name",
    help="S3/GCS bucket name for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-prefix",
    help="S3/GCS prefix path for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-data-name",
    help="Name to identify the data in S3/GCS for input.",
    type=str,
    required=False,
)
@click.option(
    "--input-max-workers",
    help="Number of parallel download threads for S3/GCS input (default: 8).",
    type=int,
    required=False,
    default=8,
)
@click.option(
    "--gpu",
    type=str,
    help="PyTorch device (e.g., 'cuda:0' or 'cpu').",
    required=False,
)
@click.option(
    "--model-architecture",
    type=click.Choice(
        list(learn.SUPPORTED_MODEL_ARCHITECTURES),
        case_sensitive=False,
    ),
    help=(
        "Backbone architecture to use. Supported values: "
        + ", ".join(learn.SUPPORTED_MODEL_ARCHITECTURES)
    ),
    required=False,
    default="resnet",
    show_default=True,
)
@click.option(
    "--compilation",
    type=bool,
    help="Enable PyTorch compilation.",
    required=False,
    default=False,
)
@click.option(
    "--detect-anomaly",
    type=bool,
    is_flag=True,
    help="Enable torch.autograd anomaly detection (default: disabled).",
    required=False,
    default=False,
)
@click.option(
    "--test-ratio",
    type=float,
    help="Test set ratio for validation benchmark (default: 0.2).",
    required=False,
    default=0.2,
)
@click.option(
    "--batch-size",
    type=int,
    help="Training batch size for benchmark (default: 256).",
    required=False,
    default=256,
)
@click.option(
    "--dataloader-workers",
    type=int,
    help="Number of DataLoader workers (default: 4).",
    required=False,
    default=4,
)
@click.option(
    "--pin-memory",
    type=bool,
    is_flag=True,
    help="Enable pinned memory for faster GPU transfers.",
    required=False,
)
@click.option(
    "--prefetch-factor",
    type=int,
    help="Number of batches loaded in advance by each worker (default: 2).",
    required=False,
    default=2,
)
@click.option(
    "--cache-transforms/--no-cache-transforms",
    default=None,
    help=(
        "Enable in-memory caching of dataset transforms when supported by the "
        "input pipeline."
    ),
    required=False,
)
@click.option(
    "--gce-parameter",
    type=float,
    help="GCE loss hyperparameter (default: 0.1).",
    required=False,
    default=0.1,
)
@click.option(
    "--policy-loss-ratio",
    type=float,
    help="Policy loss weight (default: 1.0).",
    required=False,
    default=1.0,
)
@click.option(
    "--value-loss-ratio",
    type=float,
    help="Value loss weight (default: 1.0).",
    required=False,
    default=1.0,
)
@click.option(
    "--learning-ratio",
    type=float,
    help="Learning rate (default: 0.01).",
    required=False,
    default=0.01,
)
@click.option(
    "--momentum",
    type=float,
    help="Optimizer momentum (default: 0.9).",
    required=False,
    default=0.9,
)
@click.option(
    "--lr-scheduler",
    type=click.Choice(
        list(learn.SUPPORTED_LR_SCHEDULERS.values()),
        case_sensitive=False,
    ),
    help="Learning rate scheduler to apply.",
    required=False,
    default=learn.SUPPORTED_LR_SCHEDULERS[
        "warmup_cosine_decay"
    ],
    show_default=True,
)
@click.option(
    "--optimizer",
    type=click.Choice(["adamw", "sgd"], case_sensitive=False),
    help="Optimizer to use (default: adamw).",
    required=False,
    default="adamw",
    show_default=True,
)
@click.option(
    "--optimizer-beta1",
    type=float,
    help="AdamW beta1 parameter (default: 0.9).",
    required=False,
    default=0.9,
    show_default=True,
)
@click.option(
    "--optimizer-beta2",
    type=float,
    help="AdamW beta2 parameter (default: 0.999).",
    required=False,
    default=0.999,
    show_default=True,
)
@click.option(
    "--optimizer-eps",
    type=float,
    help="AdamW epsilon parameter (default: 1e-08).",
    required=False,
    default=1e-8,
    show_default=True,
)
@click.option(
    "--stage1-pos-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Positive class weight for Stage 1 loss.",
)
@click.option(
    "--stage2-pos-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Positive class weight for Stage 2 BCE loss. Usually 1.0 when ASL is enabled.",
)
@click.option(
    "--stage2-gamma-pos",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "ASL positive focusing parameter for Stage 2."
        " 0.0 = no down-weighting of positive loss (recommended)."
    ),
)
@click.option(
    "--stage2-gamma-neg",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "ASL negative focusing parameter for Stage 2."
        " 0.0 = standard BCE, 2.0 = recommended for imbalanced data."
    ),
)
@click.option(
    "--stage2-clip",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "ASL negative probability clipping margin for Stage 2."
        " 0.0 = disabled, 0.02 = recommended."
    ),
)
@click.option(
    "--stage2-hidden-dim",
    type=int,
    default=None,
    show_default=True,
    help="Hidden layer dimension for Stage 2 head. None = single linear layer.",
)
@click.option(
    "--stage2-head-dropout",
    type=float,
    default=0.0,
    show_default=True,
    help="Dropout rate for Stage 2 head (requires --stage2-hidden-dim).",
)
@click.option(
    "--stage2-test-ratio",
    type=float,
    default=0.0,
    show_default=True,
    help="Validation split ratio for Stage 2 (0.0 = no split, e.g. 0.1 = 10%% validation).",
)
@click.option(
    "--no-streaming",
    is_flag=True,
    default=False,
    help="Disable streaming mode for file input (use map-style dataset instead).",
)
@click.option(
    "--warmup-batches",
    type=int,
    help="Number of warmup batches to exclude from timing (default: 5).",
    required=False,
    default=5,
)
@click.option(
    "--max-batches",
    type=int,
    help="Maximum number of batches to process (default: 100).",
    required=False,
    default=100,
)
@click.option(
    "--enable-profiling",
    type=bool,
    is_flag=True,
    help="Enable PyTorch profiler for detailed analysis.",
    required=False,
)
@click.option(
    "--run-validation",
    type=bool,
    is_flag=True,
    help="Also run validation benchmark (inference only).",
    required=False,
)
@click.option(
    "--sample-ratio",
    type=float,
    help="Ratio of data to sample for cloud sources (0.01-1.0, default: full data).",
    required=False,
)
@click.option(
    "--enable-resource-monitoring",
    type=bool,
    is_flag=True,
    help="Enable CPU, memory, and GPU usage monitoring during training.",
    required=False,
)
@handle_exception
def benchmark_training(
    input_path: Optional[Path],
    input_file_packed: bool,
    input_dataset_id: Optional[str],
    input_table_name: Optional[str],
    input_format: str,
    input_batch_size: int,
    input_max_cached_bytes: int,
    input_cache_mode: str,
    input_clustering_key: Optional[str],
    input_partitioning_key_date: Optional[str],
    input_local_cache: bool,
    input_local_cache_dir: Optional[str],
    input_enable_bundling: bool,
    input_bundle_size_gb: float,
    input_gcs: Optional[bool],
    input_s3: Optional[bool],
    input_bucket_name: Optional[str],
    input_prefix: Optional[str],
    input_data_name: Optional[str],
    input_max_workers: int,
    gpu: Optional[str],
    model_architecture: utility_interface.BackboneArchitecture,
    compilation: bool,
    detect_anomaly: bool,
    test_ratio: float,
    batch_size: int,
    dataloader_workers: int,
    pin_memory: Optional[bool],
    prefetch_factor: int,
    cache_transforms: Optional[bool],
    gce_parameter: float,
    policy_loss_ratio: float,
    value_loss_ratio: float,
    learning_ratio: float,
    momentum: float,
    lr_scheduler: str,
    optimizer: str,
    optimizer_beta1: float,
    optimizer_beta2: float,
    optimizer_eps: float,
    stage1_pos_weight: float,
    stage2_pos_weight: float,
    stage2_gamma_pos: float,
    stage2_gamma_neg: float,
    stage2_clip: float,
    stage2_hidden_dim: Optional[int],
    stage2_head_dropout: float,
    stage2_test_ratio: float,
    no_streaming: bool,
    warmup_batches: int,
    max_batches: int,
    enable_profiling: Optional[bool],
    run_validation: Optional[bool],
    sample_ratio: Optional[float],
    enable_resource_monitoring: Optional[bool],
) -> None:
    """Benchmark single epoch training performance with detailed timing analysis."""
    # Normalize cache_mode: "mmap" is deprecated, convert to "file"
    if input_cache_mode.lower() == "mmap":
        import warnings

        warnings.warn(
            "--input-cache-mode 'mmap' is deprecated. Use 'file' instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        input_cache_mode = "file"

    # Validate input_format early
    if input_format not in ("hcpe", "preprocess"):
        raise ValueError(
            "Please specify a valid input_format ('hcpe' or 'preprocess')."
        )
    # Convert input_format to array_type for data sources
    array_type = (
        "preprocessing"
        if input_format == "preprocess"
        else "hcpe"
    )

    # Check for mixing cloud providers for input
    cloud_input_count = sum(
        [
            bool(
                input_dataset_id is not None
                or input_table_name is not None
            ),
            bool(input_gcs),
            bool(input_s3),
        ]
    )
    if cloud_input_count > 1:
        error_msg = (
            "Cannot use multiple cloud providers for input simultaneously. "
            "Please choose only one: BigQuery, GCS, or S3."
        )
        app_logger.error(error_msg)
        raise ValueError(error_msg)

    # Initialize datasource (similar to learn_model command)
    if input_path is not None:
        if sample_ratio is not None:
            app_logger.warning(
                "sample_ratio is ignored for local file data source."
            )
        if (
            input_format != "hcpe"
            and input_format != "preprocess"
        ):
            raise Exception(
                "Please specify a valid input_format ('hcpe' or 'preprocess')."
            )
        datasource = FileDataSource.FileDataSourceSpliter(
            file_paths=FileSystem.collect_files(input_path),
            array_type=array_type,
            bit_pack=input_file_packed,
            cache_mode=input_cache_mode,
        )
    elif (
        input_dataset_id is not None
        and input_table_name is not None
    ):
        if sample_ratio is not None:
            app_logger.info(
                f"Using BigQuery TABLESAMPLE with {sample_ratio:.1%} sampling ratio."
            )
        if HAS_BIGQUERY:
            try:
                if hasattr(
                    BigQueryDataSource,
                    "BigQueryDataSourceSpliter",
                ):
                    datasource = BigQueryDataSource.BigQueryDataSourceSpliter(
                        array_type=array_type,
                        dataset_id=input_dataset_id,
                        table_name=input_table_name,
                        batch_size=input_batch_size,
                        max_cached_bytes=input_max_cached_bytes,
                        clustering_key=input_clustering_key,
                        partitioning_key_date=input_partitioning_key_date,
                        use_local_cache=input_local_cache,
                        local_cache_dir=input_local_cache_dir,
                        sample_ratio=sample_ratio,
                    )
                else:
                    app_logger.error(
                        "BigQueryDataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "BigQueryDataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize BigQueryDataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "BigQuery input requested but required packages are not installed. "
                "Install with 'uv sync --extra gcp'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    elif (
        input_gcs
        and input_bucket_name is not None
        and input_prefix is not None
        and input_data_name is not None
        and input_local_cache_dir is not None
    ):
        if HAS_GCS:
            try:
                if hasattr(GCSDataSource, "DataSourceSpliter"):
                    datasource = GCSDataSource.DataSourceSpliter(
                        cls_ref=GCSDataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        array_type=array_type,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=input_max_workers,
                        max_cached_bytes=input_max_cached_bytes,
                        sample_ratio=sample_ratio,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "DataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "DataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize DataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "GCS input requested but required packages are not installed. "
                "Install with 'uv sync --extra gcp'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    elif (
        input_s3
        and input_bucket_name is not None
        and input_prefix is not None
        and input_data_name is not None
        and input_local_cache_dir is not None
    ):
        if HAS_AWS:
            try:
                if hasattr(S3DataSource, "DataSourceSpliter"):
                    datasource = S3DataSource.DataSourceSpliter(
                        cls_ref=S3DataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        array_type=array_type,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=input_max_workers,
                        max_cached_bytes=input_max_cached_bytes,
                        sample_ratio=sample_ratio,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "DataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "DataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize DataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "S3 input requested but required packages are not installed. "
                "Install with 'uv sync --extra aws'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    else:
        raise Exception(
            "Please specify an input directory, a BigQuery table, "
            "a GCS bucket, or an S3 bucket."
        )

    # Run benchmark
    result_json = utility_interface.benchmark_training(
        datasource=datasource,
        datasource_type=input_format,
        gpu=gpu,
        compilation=compilation,
        detect_anomaly=detect_anomaly,
        test_ratio=test_ratio,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        cache_transforms=cache_transforms,
        model_architecture=model_architecture,
        gce_parameter=gce_parameter,
        policy_loss_ratio=policy_loss_ratio,
        value_loss_ratio=value_loss_ratio,
        learning_ratio=learning_ratio,
        momentum=momentum,
        lr_scheduler=lr_scheduler,
        optimizer_name=optimizer,
        optimizer_beta1=optimizer_beta1,
        optimizer_beta2=optimizer_beta2,
        optimizer_eps=optimizer_eps,
        stage1_pos_weight=stage1_pos_weight,
        stage2_pos_weight=stage2_pos_weight,
        stage2_gamma_pos=stage2_gamma_pos,
        stage2_gamma_neg=stage2_gamma_neg,
        stage2_clip=stage2_clip,
        stage2_hidden_dim=stage2_hidden_dim,
        stage2_head_dropout=stage2_head_dropout,
        stage2_test_ratio=stage2_test_ratio,
        warmup_batches=warmup_batches,
        max_batches=max_batches,
        enable_profiling=enable_profiling,
        run_validation=run_validation,
        sample_ratio=sample_ratio,
        enable_resource_monitoring=enable_resource_monitoring,
    )

    # Parse and display results
    result = json.loads(result_json)

    click.echo("=== Training Performance Benchmark Results ===")
    click.echo()
    click.echo(result["benchmark_results"]["Summary"])
    if "ValidationSummary" in result["benchmark_results"]:
        click.echo()
        click.echo(
            result["benchmark_results"]["ValidationSummary"]
        )

    # Display estimation results if sample_ratio was used
    if "estimation" in result and result["estimation"]:
        click.echo()
        click.echo("=== Data Sampling Estimation ===")
        estimation = result["estimation"]
        click.echo(
            f"Sample Ratio Used: {estimation['sample_ratio']:.1%}"
        )
        click.echo(
            f"Actual Batches Processed: {estimation['actual_batches_processed']:,}"
        )
        click.echo(
            f"Estimated Total Batches: {estimation['estimated_total_batches']:,}"
        )
        click.echo(
            f"Estimated Full Epoch Time: "
            f"{estimation['estimated_full_epoch_time_minutes']:.1f} minutes"
        )

    click.echo()
    click.echo(result["benchmark_results"]["Recommendations"])


@click.command("generate-stage1-data")
@click.option(
    "--output-dir",
    help="Output directory for Stage 1 training data.",
    type=click.Path(path_type=Path),
    required=True,
)
@handle_exception
def generate_stage1_data(output_dir: Path) -> None:
    """Generate Stage 1 training data for learning piece movement rules.

    Creates minimal board positions with single pieces to learn basic movement:

    \b
    - Board patterns: 1 piece on board (normal or promoted)
    - Hand patterns: 1 piece in hand (normal pieces only)

    \b
    Total patterns: ~1,105
    Output format: Arrow IPC (.feather) with LZ4 compression

    \b
    Example:
        maou utility generate-stage1-data --output-dir ./stage1_data/
    """
    app_logger.info("Generating Stage 1 training data...")

    result_json = utility_interface.generate_stage1_data(
        output_dir=output_dir
    )
    result = json.loads(result_json)

    click.echo(
        f"✓ Generated {result['total_patterns']} patterns"
    )
    click.echo(f"✓ Saved to: {result['output_file']}")
    click.echo()
    click.echo("Stage 1 data generation complete!")


@click.command("generate-stage2-data")
@click.option(
    "--input-path",
    help="Input directory containing HCPE feather files.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    help="Directory for output files.",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--output-gcs",
    type=bool,
    is_flag=True,
    help="Output features to Google Cloud Storage.",
    required=False,
)
@click.option(
    "--output-bucket-name",
    help="GCS bucket name for output.",
    type=str,
    required=False,
)
@click.option(
    "--output-prefix",
    help="GCS prefix path for output.",
    type=str,
    required=False,
)
@click.option(
    "--output-data-name",
    help="Name to identify the data in GCS for output.",
    type=str,
    default="stage2",
    required=False,
)
@click.option(
    "--chunk-size",
    help="Positions per output chunk (default: 100000).",
    type=int,
    default=100_000,
    required=False,
)
@click.option(
    "--intermediate-cache-dir",
    help="Directory for intermediate data cache (default: temporary directory).",
    type=click.Path(path_type=Path),
    required=False,
)
@handle_exception
def generate_stage2_data(
    input_path: Path,
    output_dir: Path,
    output_gcs: bool,
    output_bucket_name: Optional[str],
    output_prefix: Optional[str],
    output_data_name: str,
    chunk_size: int,
    intermediate_cache_dir: Optional[Path],
) -> None:
    """Generate Stage 2 training data for legal moves prediction.

    \b
    Creates training data for the legal moves prediction head from HCPE data:

    \b
    Phase 1: Collect unique positions (deduplication via board hash)
    Phase 2: Generate legal move labels for each unique position

    \b
    Output format: Arrow IPC (.feather) with LZ4 compression

    \b
    Example:
        maou utility generate-stage2-data --input-path ./converted_hcpe/ --output-dir ./stage2_data/
    """
    import json

    app_logger.info("Generating Stage 2 training data...")

    result_json = utility_interface.generate_stage2_data(
        input_dir=input_path,
        output_dir=output_dir,
        output_data_name=output_data_name,
        chunk_size=chunk_size,
        cache_dir=intermediate_cache_dir,
    )
    result = json.loads(result_json)

    click.echo(
        f"✓ Input: {result['total_input_positions']} positions"
    )
    click.echo(
        f"✓ Unique: {result['total_unique_positions']} positions"
    )
    click.echo(f"✓ Output: {len(result['output_files'])} files")
    for f in result["output_files"]:
        click.echo(f"  - {f}")

    # GCS upload if requested
    if (
        output_gcs
        and output_bucket_name is not None
        and output_prefix is not None
    ):
        if HAS_GCS:
            try:
                from maou.infra.gcs.gcs import GCS

                gcs = GCS(bucket_name=output_bucket_name)
                for file_path_str in result["output_files"]:
                    file_path = Path(file_path_str)
                    gcs_path = (
                        f"{output_prefix}{file_path.name}"
                    )
                    gcs.upload(
                        local_path=file_path,
                        remote_path=gcs_path,
                    )
                    click.echo(f"✓ Uploaded to GCS: {gcs_path}")
            except Exception as e:
                app_logger.error(f"GCS upload failed: {e}")
        else:
            app_logger.warning(
                "GCS support not available. Install google-cloud-storage."
            )

    click.echo()
    click.echo("Stage 2 data generation complete!")


@click.group()
def utility() -> None:
    """Utility commands for ML development experiments."""
    pass


utility.add_command(benchmark_dataloader)
utility.add_command(benchmark_training)
utility.add_command(generate_stage1_data)
utility.add_command(generate_stage2_data)
