from pathlib import Path
from typing import Optional

import click

from maou.infra.console.common import (
    GCS,
    HAS_AWS,
    HAS_BIGQUERY,
    HAS_GCS,
    S3,
    BigQueryDataSource,
    FileDataSource,
    FileSystem,
    GCSDataSource,
    S3DataSource,
    app_logger,
    handle_exception,
)
from maou.interface import learn


@click.command("learn-model")
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
    "--compilation",
    type=bool,
    help="Enable PyTorch compilation.",
    required=False,
)
@click.option(
    "--test-ratio",
    type=float,
    help="Test set ratio.",
    required=False,
)
@click.option(
    "--epoch",
    type=int,
    help="Number of training epochs.",
    required=False,
)
@click.option(
    "--batch-size",
    type=int,
    help="Training batch size.",
    required=False,
)
@click.option(
    "--dataloader-workers",
    type=int,
    help="Number of DataLoader workers.",
    required=False,
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
)
@click.option(
    "--gce-parameter",
    type=float,
    help="GCE loss hyperparameter.",
    required=False,
)
@click.option(
    "--policy-loss-ratio",
    type=float,
    help="Policy loss weight.",
    required=False,
)
@click.option(
    "--value-loss-ratio",
    type=float,
    help="Value loss weight.",
    required=False,
)
@click.option(
    "--learning-ratio",
    type=float,
    help="Learning rate.",
    required=False,
)
@click.option(
    "--momentum",
    type=float,
    help="Optimizer momentum.",
    required=False,
)
@click.option(
    "--resume-from",
    type=click.Path(path_type=Path),
    help="Checkpoint file to resume training.",
    required=False,
)
@click.option(
    "--start-epoch",
    type=int,
    help=(
        "Starting epoch number (0-indexed, default: 0). "
        "If set to 1, training starts from epoch 2."
    ),
    required=False,
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    help="Log directory for SummaryWriter.",
    required=False,
)
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    help="Model output directory.",
    required=False,
)
@click.option(
    "--output-gcs",
    type=bool,
    is_flag=True,
    help="Upload output to Google Cloud Storage.",
    required=False,
)
@click.option(
    "--gcs-bucket-name",
    help="GCS bucket name.",
    type=str,
    required=False,
)
@click.option(
    "--gcs-base-path",
    help="Base path in GCS bucket.",
    type=str,
    required=False,
)
@click.option(
    "--output-s3",
    type=bool,
    is_flag=True,
    help="Upload output to AWS S3.",
    required=False,
)
@click.option(
    "--s3-bucket-name",
    help="S3 bucket name.",
    type=str,
    required=False,
)
@click.option(
    "--s3-base-path",
    help="Base path in S3 bucket.",
    type=str,
    required=False,
)
@handle_exception
def learn_model(
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
    compilation: Optional[bool],
    test_ratio: Optional[float],
    epoch: Optional[int],
    batch_size: Optional[int],
    dataloader_workers: Optional[int],
    pin_memory: Optional[bool],
    prefetch_factor: Optional[int],
    gce_parameter: Optional[float],
    policy_loss_ratio: Optional[float],
    value_loss_ratio: Optional[float],
    learning_ratio: Optional[float],
    momentum: Optional[float],
    resume_from: Optional[Path],
    start_epoch: Optional[int],
    log_dir: Optional[Path],
    model_dir: Optional[Path],
    output_gcs: Optional[bool],
    gcs_bucket_name: Optional[str],
    gcs_base_path: Optional[str],
    output_s3: Optional[bool],
    s3_bucket_name: Optional[str],
    s3_base_path: Optional[str],
) -> None:
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

    # Check for mixing cloud providers for output
    if output_gcs and output_s3:
        error_msg = (
            "Cannot use both GCS and S3 as cloud storage simultaneously. "
            "Please choose only one cloud provider for output."
        )
        app_logger.error(error_msg)
        raise ValueError(error_msg)

    # Initialize cloud storage
    cloud_storage = None
    if (
        output_gcs
        and gcs_bucket_name is not None
        and gcs_base_path is not None
    ):
        if HAS_GCS:
            try:
                cloud_storage = GCS(
                    bucket_name=gcs_bucket_name,
                    base_path=gcs_base_path,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize GCS: {e}"
                )
        else:
            app_logger.warning(
                "GCS output requested but required packages are not installed. "
                "Install with 'poetry install -E gcp'"
            )
    elif (
        output_s3
        and s3_bucket_name is not None
        and s3_base_path is not None
    ):
        if HAS_AWS:
            try:
                cloud_storage = S3(
                    bucket_name=s3_bucket_name,
                    base_path=s3_base_path,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize S3: {e}"
                )
        else:
            app_logger.warning(
                "S3 output requested but required packages are not installed. "
                "Install with 'poetry install -E aws'"
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

    # Initialize datasource
    if input_dir is not None:
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
        if HAS_BIGQUERY:
            try:
                # BigQueryDataSourceSpliterを使用
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
                        sample_ratio=None,
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
                # GCSDataSourceSpliterを使用
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
                        sample_ratio=None,
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
                # S3DataSourceSpliterを使用
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
                        sample_ratio=None,
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
    click.echo(
        learn.learn(
            datasource=datasource,
            datasource_type=input_format,
            gpu=gpu,
            compilation=compilation,
            test_ratio=test_ratio,
            epoch=epoch,
            batch_size=batch_size,
            dataloader_workers=dataloader_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            gce_parameter=gce_parameter,
            policy_loss_ratio=policy_loss_ratio,
            value_loss_ratio=value_loss_ratio,
            learning_ratio=learning_ratio,
            momentum=momentum,
            resume_from=resume_from,
            start_epoch=start_epoch,
            log_dir=log_dir,
            model_dir=model_dir,
            cloud_storage=cloud_storage,
        )
    )
