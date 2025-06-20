import logging
from pathlib import Path
from typing import Optional

import click

from maou.infra.app_logging import app_logger, get_log_level_from_env
from maou.infra.file_system.file_data_source import FileDataSource
from maou.infra.file_system.file_system import FileSystem
from maou.interface import converter, learn, preprocess
from maou.interface import utility as utility_interface

# 必要なライブラリが利用可能かどうかをチェックする変数
HAS_BIGQUERY = False
HAS_GCS = False
HAS_AWS = False

# BigQuery関連のライブラリのインポートを試みる
try:
    from maou.infra.bigquery.bq_data_source import BigQueryDataSource
    from maou.infra.bigquery.bq_feature_store import BigQueryFeatureStore

    HAS_BIGQUERY = True
except ImportError:
    app_logger.debug(
        "BigQuery dependencies not available. Some features will be disabled."
    )

# GCS関連のライブラリのインポートを試みる
try:
    from maou.infra.gcs.gcs import GCS
    from maou.infra.gcs.gcs_data_source import GCSDataSource
    from maou.infra.gcs.gcs_feature_store import GCSFeatureStore

    HAS_GCS = True
except ImportError:
    app_logger.debug("GCS dependencies not available. Some features will be disabled.")

# AWS S3関連のライブラリのインポートを試みる
try:
    from maou.infra.s3.s3 import S3
    from maou.infra.s3.s3_data_source import S3DataSource
    from maou.infra.s3.s3_feature_store import S3FeatureStore

    HAS_AWS = True
except ImportError:
    app_logger.debug(
        "AWS S3 dependencies not available. Some features will be disabled."
    )


@click.group()
@click.option("--debug-mode", "-d", is_flag=True, help="Enable debug logging.")
def main(debug_mode: bool) -> None:
    if debug_mode:
        app_logger.setLevel(logging.DEBUG)
    else:
        # 環境変数MAOU_LOG_LEVELからログレベルを取得
        # デバッグモードが指定されていない場合のみ環境変数を参照
        app_logger.setLevel(get_log_level_from_env())


@click.command()
@click.option(
    "--input-path",
    help="Input file or directory path.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--input-format",
    type=str,
    help="Input format: 'kif' or 'csa'.",
    required=True,
)
@click.option(
    "--output-dir",
    help="Directory for output files.",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--min-rating",
    help="Minimum game rating.",
    type=int,
    required=False,
)
@click.option(
    "--min-moves",
    help="Minimum moves threshold.",
    type=int,
    required=False,
)
@click.option(
    "--max-moves",
    help="Maximum moves threshold.",
    type=int,
    required=False,
)
@click.option(
    "--allowed-endgame-status",
    help="Allowed endgame statuses (e.g., '%TORYO', '%SENNICHITE'). Repeatable.",
    type=str,
    required=False,
    multiple=True,
)
@click.option(
    "--exclude-moves",
    help="Exclude specific move numbers. Repeatable.",
    type=int,
    required=False,
    multiple=True,
)
@click.option(
    "--output-bigquery",
    type=bool,
    is_flag=True,
    help="Output features to BigQuery.",
    required=False,
)
@click.option(
    "--dataset-id",
    help="BigQuery dataset ID for output.",
    type=str,
    required=False,
)
@click.option(
    "--table-name",
    help="BigQuery table name for output.",
    type=str,
    required=False,
)
@click.option(
    "--output-gcs",
    type=bool,
    is_flag=True,
    help="Output features to Google Cloud Storage.",
    required=False,
)
@click.option(
    "--output-s3",
    type=bool,
    is_flag=True,
    help="Output features to AWS S3.",
    required=False,
)
@click.option(
    "--bucket-name",
    help="S3/GCS bucket name for output.",
    type=str,
    required=False,
)
@click.option(
    "--prefix",
    help="S3/GCS prefix path for output.",
    type=str,
    required=False,
)
@click.option(
    "--data-name",
    help="Name to identify the data in S3/GCS.",
    type=str,
    required=False,
)
@click.option(
    "--max-cached-bytes",
    help="Max cache size in bytes for output (default: 500MB).",
    type=int,
    required=False,
    default=500 * 1024 * 1024,
)
@click.option(
    "--max-workers",
    help="Number of parallel upload threads for S3/GCS (default: 4).",
    type=int,
    required=False,
    default=4,
)
@click.option(
    "--cpu-workers",
    help="Number of parallel CPU workers for file processing (default: auto-detect).",
    type=int,
    required=False,
)
def hcpe_convert(
    input_path: Path,
    input_format: str,
    output_dir: Path,
    min_rating: Optional[int],
    min_moves: Optional[int],
    max_moves: Optional[int],
    allowed_endgame_status: Optional[list[str]],
    exclude_moves: Optional[list[int]],
    output_bigquery: Optional[bool],
    dataset_id: Optional[str],
    table_name: Optional[str],
    output_gcs: Optional[bool],
    output_s3: Optional[bool],
    bucket_name: Optional[str],
    prefix: Optional[str],
    data_name: Optional[str],
    max_cached_bytes: int,
    max_workers: int,
    cpu_workers: Optional[int],
) -> None:
    try:
        feature_store = None

        # Check for mixing cloud providers
        cloud_provider_count = sum(
            [bool(output_bigquery), bool(output_gcs), bool(output_s3)]
        )
        if cloud_provider_count > 1:
            error_msg = (
                "Cannot use multiple cloud providers simultaneously. "
                "Please choose only one: BigQuery, GCS, or S3."
            )
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize BigQuery feature store if requested
        if output_bigquery and dataset_id is not None and table_name is not None:
            if HAS_BIGQUERY:
                try:
                    feature_store = BigQueryFeatureStore(
                        dataset_id=dataset_id,
                        table_name=table_name,
                        max_cached_bytes=max_cached_bytes,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize BigQuery: {e}")
            else:
                app_logger.warning(
                    "BigQuery output requested "
                    "but required packages are not installed. "
                    "Install with 'poetry install -E gcp'"
                )
        # Initialize GCS feature store if requested
        elif (
            output_gcs
            and bucket_name is not None
            and prefix is not None
            and data_name is not None
        ):
            if HAS_GCS:
                try:
                    feature_store = GCSFeatureStore(
                        bucket_name=bucket_name,
                        prefix=prefix,
                        data_name=data_name,
                        max_cached_bytes=max_cached_bytes,
                        max_workers=max_workers,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize GCSFeatureStore: {e}")
            else:
                app_logger.warning(
                    "GCS output requested "
                    "but required packages are not installed. "
                    "Install with 'poetry install -E gcp'"
                )
        # Initialize S3 feature store if requested
        elif (
            output_s3
            and bucket_name is not None
            and prefix is not None
            and data_name is not None
        ):
            if HAS_AWS:
                try:
                    feature_store = S3FeatureStore(
                        bucket_name=bucket_name,
                        prefix=prefix,
                        data_name=data_name,
                        max_cached_bytes=max_cached_bytes,
                        max_workers=max_workers,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize S3FeatureStore: {e}")
            else:
                app_logger.warning(
                    "S3 output requested "
                    "but required packages are not installed. "
                    "Install with 'poetry install -E aws'"
                )

        click.echo(
            converter.transform(
                FileSystem(),
                input_path,
                input_format,
                output_dir,
                min_rating=min_rating,
                min_moves=min_moves,
                max_moves=max_moves,
                allowed_endgame_status=allowed_endgame_status,
                exclude_moves=exclude_moves,
                feature_store=feature_store,
                max_workers=cpu_workers,
            )
        )
    except Exception:
        app_logger.exception("Error occurred", stack_info=True)


@click.command()
@click.option(
    "--input-path",
    help="Input file or directory path.",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--output-dir",
    help="Directory for output files.",
    type=click.Path(path_type=Path),
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
    "--output-bigquery",
    type=bool,
    is_flag=True,
    help="Store output features in BigQuery.",
    required=False,
)
@click.option(
    "--dataset-id",
    help="BigQuery dataset ID for output.",
    type=str,
    required=False,
)
@click.option(
    "--table-name",
    help="BigQuery table name for output.",
    type=str,
    required=False,
)
@click.option(
    "--output-max-cached-bytes",
    help="Max cache size in bytes for output (default: 500MB).",
    type=int,
    required=False,
    default=500 * 1024 * 1024,
)
@click.option(
    "--output-gcs",
    type=bool,
    is_flag=True,
    help="Output features to Google Cloud Storage.",
    required=False,
)
@click.option(
    "--output-s3",
    type=bool,
    is_flag=True,
    help="Output features to AWS S3.",
    required=False,
)
@click.option(
    "--output-bucket-name",
    help="S3/GCS bucket name for output.",
    type=str,
    required=False,
)
@click.option(
    "--output-prefix",
    help="S3/GCS prefix path for output.",
    type=str,
    required=False,
)
@click.option(
    "--output-data-name",
    help="Name to identify the data in S3/GCS for output.",
    type=str,
    required=False,
)
@click.option(
    "--max-workers",
    help="Number of parallel upload threads for S3 (default: 4).",
    type=int,
    required=False,
    default=4,
)
@click.option(
    "--cpu-workers",
    help="Number of parallel CPU workers for record processing (default: auto-detect).",
    type=int,
    required=False,
)
def pre_process(
    input_path: Optional[Path],
    output_dir: Optional[Path],
    input_dataset_id: Optional[str],
    input_table_name: Optional[str],
    input_batch_size: int,
    input_max_cached_bytes: int,
    input_clustering_key: Optional[str],
    input_partitioning_key_date: Optional[str],
    input_local_cache: bool,
    input_local_cache_dir: Optional[str],
    input_gcs: Optional[bool],
    input_s3: Optional[bool],
    input_bucket_name: Optional[str],
    input_prefix: Optional[str],
    input_data_name: Optional[str],
    output_bigquery: Optional[bool],
    dataset_id: Optional[str],
    table_name: Optional[str],
    output_max_cached_bytes: int,
    output_gcs: Optional[bool],
    output_s3: Optional[bool],
    output_bucket_name: Optional[str],
    output_prefix: Optional[str],
    output_data_name: Optional[str],
    max_workers: int,
    cpu_workers: Optional[int],
) -> None:
    try:
        # Check for mixing cloud providers for input
        cloud_input_count = sum(
            [
                bool(input_dataset_id is not None or input_table_name is not None),
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
        datasource = None
        if input_dataset_id is not None and input_table_name is not None:
            if HAS_BIGQUERY:
                try:
                    datasource = BigQueryDataSource(
                        dataset_id=input_dataset_id,
                        table_name=input_table_name,
                        batch_size=input_batch_size,
                        max_cached_bytes=input_max_cached_bytes,
                        clustering_key=input_clustering_key,
                        partitioning_key_date=input_partitioning_key_date,
                        use_local_cache=input_local_cache,
                        local_cache_dir=input_local_cache_dir,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize BigQueryDataSource: {e}")
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
                    datasource = GCSDataSource(
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=max_workers,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize GCSDataSource: {e}")
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
                    datasource = S3DataSource(
                        bucket_name=input_bucket_name,
                        prefix=input_prefix,
                        data_name=input_data_name,
                        local_cache_dir=input_local_cache_dir,
                        max_workers=max_workers,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize S3DataSource: {e}")
                    raise
            else:
                error_msg = (
                    "S3 input requested but required packages are not installed. "
                    "Install with 'poetry install -E aws'"
                )
                app_logger.error(error_msg)
                raise ImportError(error_msg)
        elif input_path is not None:
            input_paths = FileSystem.collect_files(input_path)
            datasource = FileDataSource(file_paths=input_paths)
        else:
            error_msg = (
                "Please specify an input source "
                "(file path, BigQuery table, GCS bucket, or S3 bucket)."
            )
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for mixing cloud providers for output
        cloud_output_count = sum(
            [bool(output_bigquery), bool(output_gcs), bool(output_s3)]
        )
        if cloud_output_count > 1:
            error_msg = (
                "Cannot use multiple cloud providers for output simultaneously. "
                "Please choose only one: BigQuery, GCS, or S3."
            )
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize feature store
        feature_store = None
        if output_bigquery and dataset_id is not None and table_name is not None:
            if HAS_BIGQUERY:
                try:
                    feature_store = BigQueryFeatureStore(
                        dataset_id=dataset_id,
                        table_name=table_name,
                        max_cached_bytes=output_max_cached_bytes,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize BigQuery: {e}")
            else:
                app_logger.warning(
                    "BigQuery output requested "
                    "but required packages are not installed. "
                    "Install with 'poetry install -E gcp'"
                )
        elif (
            output_gcs
            and output_bucket_name is not None
            and output_prefix is not None
            and output_data_name is not None
        ):
            if HAS_GCS:
                try:
                    feature_store = GCSFeatureStore(
                        bucket_name=output_bucket_name,
                        prefix=output_prefix,
                        data_name=output_data_name,
                        max_cached_bytes=output_max_cached_bytes,
                        max_workers=max_workers,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize GCSFeatureStore: {e}")
            else:
                app_logger.warning(
                    "GCS output requested "
                    "but required packages are not installed. "
                    "Install with 'poetry install -E gcp'"
                )
        elif (
            output_s3
            and output_bucket_name is not None
            and output_prefix is not None
            and output_data_name is not None
        ):
            if HAS_AWS:
                try:
                    feature_store = S3FeatureStore(
                        bucket_name=output_bucket_name,
                        prefix=output_prefix,
                        data_name=output_data_name,
                        max_cached_bytes=output_max_cached_bytes,
                        max_workers=max_workers,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize S3FeatureStore: {e}")
            else:
                app_logger.warning(
                    "S3 output requested "
                    "but required packages are not installed. "
                    "Install with 'poetry install -E aws'"
                )
        click.echo(
            preprocess.transform(
                datasource=datasource,
                output_dir=output_dir,
                feature_store=feature_store,
                max_workers=cpu_workers,
            )
        )
    except Exception:
        app_logger.exception("Error occurred", stack_info=True)


@click.command()
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
    try:
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
        if output_gcs and gcs_bucket_name is not None and gcs_base_path is not None:
            if HAS_GCS:
                try:
                    cloud_storage = GCS(
                        bucket_name=gcs_bucket_name, base_path=gcs_base_path
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize GCS: {e}")
            else:
                app_logger.warning(
                    "GCS output requested but required packages are not installed. "
                    "Install with 'poetry install -E gcp'"
                )
        elif output_s3 and s3_bucket_name is not None and s3_base_path is not None:
            if HAS_AWS:
                try:
                    cloud_storage = S3(
                        bucket_name=s3_bucket_name,
                        base_path=s3_base_path,
                    )
                except Exception as e:
                    app_logger.error(f"Failed to initialize S3: {e}")
            else:
                app_logger.warning(
                    "S3 output requested but required packages are not installed. "
                    "Install with 'poetry install -E aws'"
                )

        # Check for mixing cloud providers for input
        cloud_input_count = sum(
            [
                bool(input_dataset_id is not None or input_table_name is not None),
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
            if input_format != "hcpe" and input_format != "preprocess":
                raise Exception(
                    "Please specify a valid input_format ('hcpe' or 'preprocess')."
                )
            datasource = FileDataSource.FileDataSourceSpliter(
                file_paths=FileSystem.collect_files(input_dir),
            )
        elif input_dataset_id is not None and input_table_name is not None:
            if HAS_BIGQUERY:
                try:
                    # BigQueryDataSourceSpliterを使用
                    if hasattr(BigQueryDataSource, "BigQueryDataSourceSpliter"):
                        datasource = BigQueryDataSource.BigQueryDataSourceSpliter(
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
                        app_logger.error("BigQueryDataSourceSpliter not available")
                        raise AttributeError("BigQueryDataSourceSpliter not available")
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
                    if hasattr(GCSDataSource, "GCSDataSourceSpliter"):
                        datasource = GCSDataSource.GCSDataSourceSpliter(
                            bucket_name=input_bucket_name,
                            prefix=input_prefix,
                            data_name=input_data_name,
                            local_cache_dir=input_local_cache_dir,
                            max_workers=input_max_workers,
                            sample_ratio=None,
                        )
                    else:
                        app_logger.error("GCSDataSourceSpliter not available")
                        raise AttributeError("GCSDataSourceSpliter not available")
                except Exception as e:
                    app_logger.error(f"Failed to initialize GCSDataSourceSpliter: {e}")
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
                    if hasattr(S3DataSource, "S3DataSourceSpliter"):
                        datasource = S3DataSource.S3DataSourceSpliter(
                            bucket_name=input_bucket_name,
                            prefix=input_prefix,
                            data_name=input_data_name,
                            local_cache_dir=input_local_cache_dir,
                            max_workers=input_max_workers,
                            sample_ratio=None,
                        )
                    else:
                        app_logger.error("S3DataSourceSpliter not available")
                        raise AttributeError("S3DataSourceSpliter not available")
                except Exception as e:
                    app_logger.error(f"Failed to initialize S3DataSourceSpliter: {e}")
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
    except Exception:
        app_logger.exception("Error occurred", stack_info=True)


@click.command()
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
    try:
        # Check for mixing cloud providers for input
        cloud_input_count = sum(
            [
                bool(input_dataset_id is not None or input_table_name is not None),
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
            if input_format != "hcpe" and input_format != "preprocess":
                raise Exception(
                    "Please specify a valid input_format ('hcpe' or 'preprocess')."
                )
            datasource = FileDataSource.FileDataSourceSpliter(
                file_paths=FileSystem.collect_files(input_dir),
            )
        elif input_dataset_id is not None and input_table_name is not None:
            if sample_ratio is not None:
                app_logger.info(
                    f"Using BigQuery TABLESAMPLE with "
                    f"{sample_ratio:.1%} sampling ratio."
                )
            if HAS_BIGQUERY:
                try:
                    if hasattr(BigQueryDataSource, "BigQueryDataSourceSpliter"):
                        datasource = BigQueryDataSource.BigQueryDataSourceSpliter(
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
                        app_logger.error("BigQueryDataSourceSpliter not available")
                        raise AttributeError("BigQueryDataSourceSpliter not available")
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
                    if hasattr(GCSDataSource, "GCSDataSourceSpliter"):
                        datasource = GCSDataSource.GCSDataSourceSpliter(
                            bucket_name=input_bucket_name,
                            prefix=input_prefix,
                            data_name=input_data_name,
                            local_cache_dir=input_local_cache_dir,
                            max_workers=input_max_workers,
                            sample_ratio=sample_ratio,
                        )
                    else:
                        app_logger.error("GCSDataSourceSpliter not available")
                        raise AttributeError("GCSDataSourceSpliter not available")
                except Exception as e:
                    app_logger.error(f"Failed to initialize GCSDataSourceSpliter: {e}")
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
                        datasource = S3DataSource.S3DataSourceSpliter(
                            bucket_name=input_bucket_name,
                            prefix=input_prefix,
                            data_name=input_data_name,
                            local_cache_dir=input_local_cache_dir,
                            max_workers=input_max_workers,
                            sample_ratio=sample_ratio,
                        )
                    else:
                        app_logger.error("S3DataSourceSpliter not available")
                        raise AttributeError("S3DataSourceSpliter not available")
                except Exception as e:
                    app_logger.error(f"Failed to initialize S3DataSourceSpliter: {e}")
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
        import json

        result = json.loads(result_json)

        click.echo(result["benchmark_results"]["Summary"])
        click.echo()
        click.echo(result["benchmark_results"]["Recommendations"])
        click.echo()
        click.echo(result["benchmark_results"]["Insights"])

    except Exception:
        app_logger.exception("Error occurred", stack_info=True)


@click.command()
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
    try:
        # Check for mixing cloud providers for input
        cloud_input_count = sum(
            [
                bool(input_dataset_id is not None or input_table_name is not None),
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
            if input_format != "hcpe" and input_format != "preprocess":
                raise Exception(
                    "Please specify a valid input_format ('hcpe' or 'preprocess')."
                )
            datasource = FileDataSource.FileDataSourceSpliter(
                file_paths=FileSystem.collect_files(input_dir),
            )
        elif input_dataset_id is not None and input_table_name is not None:
            if sample_ratio is not None:
                app_logger.info(
                    f"Using BigQuery TABLESAMPLE with "
                    f"{sample_ratio:.1%} sampling ratio."
                )
            if HAS_BIGQUERY:
                try:
                    if hasattr(BigQueryDataSource, "BigQueryDataSourceSpliter"):
                        datasource = BigQueryDataSource.BigQueryDataSourceSpliter(
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
                        app_logger.error("BigQueryDataSourceSpliter not available")
                        raise AttributeError("BigQueryDataSourceSpliter not available")
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
                    if hasattr(GCSDataSource, "GCSDataSourceSpliter"):
                        datasource = GCSDataSource.GCSDataSourceSpliter(
                            bucket_name=input_bucket_name,
                            prefix=input_prefix,
                            data_name=input_data_name,
                            local_cache_dir=input_local_cache_dir,
                            max_workers=input_max_workers,
                            sample_ratio=sample_ratio,
                        )
                    else:
                        app_logger.error("GCSDataSourceSpliter not available")
                        raise AttributeError("GCSDataSourceSpliter not available")
                except Exception as e:
                    app_logger.error(f"Failed to initialize GCSDataSourceSpliter: {e}")
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
                        datasource = S3DataSource.S3DataSourceSpliter(
                            bucket_name=input_bucket_name,
                            prefix=input_prefix,
                            data_name=input_data_name,
                            local_cache_dir=input_local_cache_dir,
                            max_workers=input_max_workers,
                            sample_ratio=sample_ratio,
                        )
                    else:
                        app_logger.error("S3DataSourceSpliter not available")
                        raise AttributeError("S3DataSourceSpliter not available")
                except Exception as e:
                    app_logger.error(f"Failed to initialize S3DataSourceSpliter: {e}")
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
        import json

        result = json.loads(result_json)

        click.echo("=== Training Performance Benchmark Results ===")
        click.echo()
        click.echo(result["benchmark_results"]["Summary"])
        if "ValidationSummary" in result["benchmark_results"]:
            click.echo()
            click.echo(result["benchmark_results"]["ValidationSummary"])
        click.echo()
        click.echo(result["benchmark_results"]["Recommendations"])

    except Exception:
        app_logger.exception("Error occurred", stack_info=True)


@click.group()
def utility() -> None:
    """Utility commands for ML development experiments."""
    pass


utility.add_command(benchmark_dataloader)
utility.add_command(benchmark_training)
main.add_command(hcpe_convert)
main.add_command(pre_process)
main.add_command(learn_model)
main.add_command(utility)
