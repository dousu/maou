import json
from pathlib import Path
from typing import Optional

import click

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
from maou.interface import utility as utility_interface


@click.command("benchmark-dataloader")
@click.option(
    "--input-dir",
    help="Input data directory.",
    type=click.Path(exists=True, path_type=Path),
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
    input_dir: Optional[Path],
    input_dataset_id: Optional[str],
    input_table_name: Optional[str],
    input_format: str,
    input_batch_size: int,
    input_max_cached_bytes: int,
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
    if input_dir is not None:
        if sample_ratio is not None:
            app_logger.warning(
                "sample_ratio is ignored for local file data source."
            )
        datasource = FileDataSource.FileDataSourceSpliter(
            file_paths=FileSystem.collect_files(input_dir),
            array_type=array_type,
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
                "Install with 'poetry install -E gcp'"
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
                if hasattr(
                    GCSDataSource, "GCSDataSourceSpliter"
                ):
                    datasource = GCSDataSource.DataSourceSpliter(
                        cls_ref=GCSDataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        local_cache_dir=input_local_cache_dir,
                        array_type=array_type,
                        max_workers=input_max_workers,
                        sample_ratio=sample_ratio,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "GCSDataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "GCSDataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize GCSDataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "GCS input requested but required packages are not installed. "
                "Install with 'poetry install -E gcp'"
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
                if hasattr(S3DataSource, "S3DataSourceSpliter"):
                    datasource = S3DataSource.DataSourceSpliter(
                        cls_ref=S3DataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=input_max_workers,
                        sample_ratio=sample_ratio,
                        array_type=array_type,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "S3DataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "S3DataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize S3DataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "S3 input requested but required packages are not installed. "
                "Install with 'poetry install -E aws'"
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
    "--input-dir",
    help="Input data directory.",
    type=click.Path(exists=True, path_type=Path),
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
    "--test-ratio",
    type=float,
    help="Test set ratio for validation benchmark (default: 0.2).",
    required=False,
    default=0.2,
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
@handle_exception
def benchmark_training(
    input_dir: Optional[Path],
    input_dataset_id: Optional[str],
    input_table_name: Optional[str],
    input_format: str,
    input_batch_size: int,
    input_max_cached_bytes: int,
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
    dataloader_workers: int,
    pin_memory: Optional[bool],
    prefetch_factor: int,
    gce_parameter: float,
    policy_loss_ratio: float,
    value_loss_ratio: float,
    learning_ratio: float,
    momentum: float,
    warmup_batches: int,
    max_batches: int,
    enable_profiling: Optional[bool],
    test_ratio: float,
    run_validation: Optional[bool],
    sample_ratio: Optional[float],
) -> None:
    """Benchmark single epoch training performance with detailed timing analysis."""
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
    if input_dir is not None:
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
            file_paths=FileSystem.collect_files(input_dir),
            array_type=array_type,
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
                "Install with 'poetry install -E gcp'"
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
                if hasattr(
                    GCSDataSource, "GCSDataSourceSpliter"
                ):
                    datasource = GCSDataSource.DataSourceSpliter(
                        cls_ref=GCSDataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        array_type=array_type,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=input_max_workers,
                        sample_ratio=sample_ratio,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "GCSDataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "GCSDataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize GCSDataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "GCS input requested but required packages are not installed. "
                "Install with 'poetry install -E gcp'"
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
                if hasattr(S3DataSource, "S3DataSourceSpliter"):
                    datasource = S3DataSource.DataSourceSpliter(
                        cls_ref=S3DataSource,
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        array_type=array_type,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=input_max_workers,
                        sample_ratio=sample_ratio,
                        enable_bundling=input_enable_bundling,
                        bundle_size_gb=input_bundle_size_gb,
                    )
                else:
                    app_logger.error(
                        "S3DataSourceSpliter not available"
                    )
                    raise AttributeError(
                        "S3DataSourceSpliter not available"
                    )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize S3DataSourceSpliter: {e}"
                )
                raise
        else:
            error_msg = (
                "S3 input requested but required packages are not installed. "
                "Install with 'poetry install -E aws'"
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
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        gce_parameter=gce_parameter,
        policy_loss_ratio=policy_loss_ratio,
        value_loss_ratio=value_loss_ratio,
        learning_ratio=learning_ratio,
        momentum=momentum,
        warmup_batches=warmup_batches,
        max_batches=max_batches,
        enable_profiling=enable_profiling,
        test_ratio=test_ratio,
        run_validation=run_validation,
        sample_ratio=sample_ratio,
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


@click.group()
def utility() -> None:
    """Utility commands for ML development experiments."""
    pass


utility.add_command(benchmark_dataloader)
utility.add_command(benchmark_training)
