from pathlib import Path
from typing import Optional

import click

from maou.infra.console.common import (
    HAS_AWS,
    HAS_BIGQUERY,
    HAS_GCS,
    BigQueryDataSource,
    BigQueryFeatureStore,
    FileDataSource,
    FileSystem,
    GCSDataSource,
    GCSFeatureStore,
    S3DataSource,
    S3FeatureStore,
    app_logger,
    handle_exception,
)
from maou.interface import preprocess


@click.command("pre-process")
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
@handle_exception
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
    input_enable_bundling: bool,
    input_bundle_size_gb: float,
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
) -> None:
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
    datasource = None
    if (
        input_dataset_id is not None
        and input_table_name is not None
    ):
        if HAS_BIGQUERY:
            try:
                datasource = BigQueryDataSource(
                    array_type="hcpe",
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
                app_logger.error(
                    f"Failed to initialize BigQueryDataSource: {e}"
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
                datasource = GCSDataSource(
                    bucket_name=input_bucket_name,
                    prefix=input_prefix,
                    data_name=input_data_name,
                    local_cache_dir=input_local_cache_dir,
                    max_workers=max_workers,
                    array_type="hcpe",
                    enable_bundling=input_enable_bundling,
                    bundle_size_gb=input_bundle_size_gb,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize GCSDataSource: {e}"
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
                datasource = S3DataSource(
                    bucket_name=input_bucket_name,
                    prefix=input_prefix,
                    data_name=input_data_name,
                    local_cache_dir=input_local_cache_dir,
                    max_workers=max_workers,
                    array_type="hcpe",
                    enable_bundling=input_enable_bundling,
                    bundle_size_gb=input_bundle_size_gb,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize S3DataSource: {e}"
                )
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
        datasource = FileDataSource(
            file_paths=input_paths,
            array_type="hcpe",
        )
    else:
        error_msg = (
            "Please specify an input source "
            "(file path, BigQuery table, GCS bucket, or S3 bucket)."
        )
        app_logger.error(error_msg)
        raise ValueError(error_msg)

    # Check for mixing cloud providers for output
    cloud_output_count = sum(
        [
            bool(output_bigquery),
            bool(output_gcs),
            bool(output_s3),
        ]
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
    if (
        output_bigquery
        and dataset_id is not None
        and table_name is not None
    ):
        if HAS_BIGQUERY:
            try:
                feature_store = BigQueryFeatureStore(
                    dataset_id=dataset_id,
                    table_name=table_name,
                    max_cached_bytes=output_max_cached_bytes,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize BigQuery: {e}"
                )
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
                    array_type="preprocessing",
                    max_cached_bytes=output_max_cached_bytes,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize GCSFeatureStore: {e}"
                )
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
                    array_type="preprocessing",
                    max_cached_bytes=output_max_cached_bytes,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize S3FeatureStore: {e}"
                )
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
            max_workers=max_workers,
        )
    )
