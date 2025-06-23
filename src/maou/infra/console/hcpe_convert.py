from pathlib import Path
from typing import Optional

import click

from maou.infra.console.common import (
    HAS_AWS,
    HAS_BIGQUERY,
    HAS_GCS,
    BigQueryFeatureStore,
    FileSystem,
    GCSFeatureStore,
    S3FeatureStore,
    app_logger,
    handle_exception,
)
from maou.interface import converter


@click.command("hcpe-convert")
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
@handle_exception
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
