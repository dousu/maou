from pathlib import Path

import click

from maou.infra.console.common import (
    HAS_AWS,
    HAS_BIGQUERY,
    HAS_GCS,
    BigQueryDataSource,
    BigQueryFeatureStore,
    FileSystem,
    GCSDataSource,
    GCSFeatureStore,
    S3DataSource,
    S3FeatureStore,
    app_logger,
    handle_exception,
)
from maou.interface import preprocess

_NO_INPUT_SOURCE_MSG = (
    "Please specify an input source "
    "(file path, BigQuery table, GCS bucket, or S3 bucket)."
)


def describe_missing_input_options(
    *,
    input_dataset_id: str | None,
    input_table_name: str | None,
    input_gcs: bool | None,
    input_s3: bool | None,
    input_bucket_name: str | None,
    input_prefix: str | None,
    input_data_name: str | None,
    input_local_cache_dir: str | None,
) -> str:
    """Explain why no input datasource could be built.

    Every input branch needs its own set of companion options,
    so a cloud input that is missing one of them falls through
    to the same place as an invocation with no input at all.
    Reporting "please specify an input source" in that case
    points at the wrong option; this names the ones that are
    actually missing.

    Args:
        input_dataset_id: ``--input-dataset-id`` value.
        input_table_name: ``--input-table-name`` value.
        input_gcs: ``--input-gcs`` flag.
        input_s3: ``--input-s3`` flag.
        input_bucket_name: ``--input-bucket-name`` value.
        input_prefix: ``--input-prefix`` value.
        input_data_name: ``--input-data-name`` value.
        input_local_cache_dir: ``--input-local-cache-dir`` value.

    Returns:
        The error message to log and raise.
    """
    if input_gcs or input_s3:
        source = "--input-gcs" if input_gcs else "--input-s3"
        required = {
            "--input-bucket-name": input_bucket_name,
            "--input-prefix": input_prefix,
            "--input-data-name": input_data_name,
            "--input-local-cache-dir": input_local_cache_dir,
        }
    elif (
        input_dataset_id is not None
        or input_table_name is not None
    ):
        source = "BigQuery input"
        required = {
            "--input-dataset-id": input_dataset_id,
            "--input-table-name": input_table_name,
        }
    else:
        return _NO_INPUT_SOURCE_MSG

    missing = [
        option
        for option, value in required.items()
        if value is None
    ]
    if not missing:
        # 呼び出し側の分岐条件と required が食い違ったときだけ
        # ここに来る．原因を隠すより元の文言で落とす方がよい．
        return _NO_INPUT_SOURCE_MSG
    return (
        f"{source} was specified but these required options "
        f"are missing: {', '.join(missing)}."
    )


@click.command("pre-process")
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
    help="[Deprecated] Enable unpacking local numpy file. This option has no effect.",
    default=False,
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
    "--input-max-workers",
    help="Number of parallel download processes for S3/GCS (default: 4).",
    type=int,
    required=False,
    default=4,
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
    "--output-max-workers",
    help="Number of parallel upload processes for S3/GCS (default: 4).",
    type=int,
    required=False,
    default=4,
)
@click.option(
    "--output-max-queue-size",
    help="Number of max queue size (default: 4).",
    type=int,
    required=False,
    default=4,
)
@click.option(
    "--process-max-workers",
    help="Number of parallel processing processes (default: 3).",
    type=int,
    required=False,
    default=3,
)
@click.option(
    "--intermediate-cache-dir",
    help="Directory for intermediate data cache (default: temporary directory).",
    type=click.Path(path_type=Path),
    required=False,
)
@click.option(
    "--intermediate-batch-size",
    help="Records to accumulate before flushing to DuckDB (default: 50000, optimized for A100 high-memory).",
    type=int,
    required=False,
    default=50_000,
)
@click.option(
    "--input-split-rows",
    help="Resize input files to approximately this many rows for better parallelism. "
    "Large files are split and small files are merged. "
    "(default: 500000, optimized for A100 high-memory). Set to 0 to disable.",
    type=int,
    required=False,
    default=500_000,
)
@click.option(
    "--position-count-threshold",
    help="Minimum position occurrence count for per-move win rate calculation. "
    "Positions with count below this threshold use the fallback selected by "
    "--win-rate-fallback. (default: 2, recommended range: 2-4)",
    type=int,
    required=False,
    default=2,
)
@click.option(
    "--prior-strength",
    help="Beta prior strength for win rate smoothing. "
    "Applies (wins + prior) / (total + 2 * prior) to shrink "
    "low-count move win rates toward 50%%. "
    "0.0 disables smoothing. (default: 5.0)",
    type=click.FloatRange(min=0.0),
    required=False,
    default=5.0,
)
@click.option(
    "--win-rate-fallback",
    help="How to fill moveWinRate/bestMoveWinRate for positions below "
    "--position-count-threshold. 'neutral' records a neutral 0.5 "
    "(default). 'raw-outcome' records the unsmoothed actual win/loss "
    "outcome instead (0.0/1.0, or 0.5 for a draw, when the position "
    "occurred once). Ignored when --drop-below-threshold is set.",
    type=click.Choice(["neutral", "raw-outcome"]),
    required=False,
    default="neutral",
)
@click.option(
    "--drop-below-threshold",
    help="Exclude positions with occurrence count below "
    "--position-count-threshold from the output entirely, instead of "
    "recording a fallback value for them. (default: False)",
    is_flag=True,
    default=False,
    required=False,
)
@click.option(
    "--search-value-path",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=False,
    help="Output of `maou utility search-values`: the shard directory, a "
    ".feather file, or any mix. Repeat the option to union several runs "
    "(*.feather/*.arrow under a directory are all unioned; duplicate ids "
    "resolve last-wins in the given order). Replaces resultValue with the "
    "per-position search value for the positions it covers. A game result "
    "is the same for every position of a game, so recalling which game a "
    "position came from fits the training set without transferring to "
    "unseen games; search values differ position by position. Positions "
    "absent from the input keep their game-result value. Unreadable "
    "files, or files without a usable id/searchWinRate column, fail the "
    "run before any input is downloaded or resized.",
)
@handle_exception
def pre_process(
    input_path: Path | None,
    input_file_packed: bool,
    output_dir: Path | None,
    input_dataset_id: str | None,
    input_table_name: str | None,
    input_batch_size: int,
    input_max_cached_bytes: int,
    input_clustering_key: str | None,
    input_partitioning_key_date: str | None,
    input_max_workers: int,
    input_local_cache: bool,
    input_local_cache_dir: str | None,
    input_enable_bundling: bool,
    input_bundle_size_gb: float,
    input_gcs: bool | None,
    input_s3: bool | None,
    input_bucket_name: str | None,
    input_prefix: str | None,
    input_data_name: str | None,
    output_bigquery: bool | None,
    dataset_id: str | None,
    table_name: str | None,
    output_max_cached_bytes: int,
    output_gcs: bool | None,
    output_s3: bool | None,
    output_bucket_name: str | None,
    output_prefix: str | None,
    output_data_name: str | None,
    output_max_workers: int,
    output_max_queue_size: int,
    process_max_workers: int,
    intermediate_cache_dir: Path | None,
    intermediate_batch_size: int,
    input_split_rows: int,
    position_count_threshold: int,
    prior_strength: float,
    win_rate_fallback: str,
    drop_below_threshold: bool,
    search_value_path: tuple[Path, ...],
) -> None:
    import warnings

    if input_file_packed:
        warnings.warn(
            "--input-file-packed is deprecated and has no effect. "
            "It will be removed in a future version.",
            DeprecationWarning,
            stacklevel=1,
        )

    # 探索値の検査はここで済ませる．データソースの構築 (クラウド入力は
    # 全ダウンロード) と resize_input_files (入力コーパス全体の書き直し) の
    # どちらより手前でなければ，パスの打ち間違い 1 つでその 1 往復を捨てる．
    # フッタからスキーマを読むだけなので，ここに置いても実質無料．
    preprocess.validate_search_value_paths(search_value_path)

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
                datasource = GCSDataSource(
                    bucket_name=input_bucket_name,
                    prefix=input_prefix,
                    data_name=input_data_name,
                    local_cache_dir=input_local_cache_dir,
                    max_workers=input_max_workers,
                    max_cached_bytes=input_max_cached_bytes,
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
                datasource = S3DataSource(
                    bucket_name=input_bucket_name,
                    prefix=input_prefix,
                    data_name=input_data_name,
                    local_cache_dir=input_local_cache_dir,
                    max_workers=input_max_workers,
                    max_cached_bytes=input_max_cached_bytes,
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
                "Install with 'uv sync --extra aws'"
            )
            app_logger.error(error_msg)
            raise ImportError(error_msg)
    elif input_path is not None:
        input_paths = FileSystem.collect_files(input_path)

        # 入力ファイルのリサイズ（大きなファイルは分割，小さなファイルはチャンク）
        if input_split_rows > 0:
            input_paths = preprocess.resize_input_files(
                file_paths=input_paths,
                rows_per_file=input_split_rows,
                work_dir=intermediate_cache_dir,
            )

        # ストリーミングモード: 1ファイルずつ遅延ロードしてメモリ使用量を削減
        from maou.infra.file_system.streaming_hcpe_source import (
            StreamingHcpeDataSource,
        )

        datasource = StreamingHcpeDataSource(
            file_paths=input_paths,
        )
    else:
        error_msg = describe_missing_input_options(
            input_dataset_id=input_dataset_id,
            input_table_name=input_table_name,
            input_gcs=input_gcs,
            input_s3=input_s3,
            input_bucket_name=input_bucket_name,
            input_prefix=input_prefix,
            input_data_name=input_data_name,
            input_local_cache_dir=input_local_cache_dir,
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
                "Install with 'uv sync --extra gcp'"
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
                    max_workers=output_max_workers,
                    max_cached_bytes=output_max_cached_bytes,
                    max_queue_size=output_max_queue_size,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize GCSFeatureStore: {e}"
                )
        else:
            app_logger.warning(
                "GCS output requested "
                "but required packages are not installed. "
                "Install with 'uv sync --extra gcp'"
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
                    max_workers=output_max_workers,
                    max_cached_bytes=output_max_cached_bytes,
                    max_queue_size=output_max_queue_size,
                )
            except Exception as e:
                app_logger.error(
                    f"Failed to initialize S3FeatureStore: {e}"
                )
        else:
            app_logger.warning(
                "S3 output requested "
                "but required packages are not installed. "
                "Install with 'uv sync --extra aws'"
            )
    click.echo(
        preprocess.transform(
            datasource=datasource,
            output_dir=output_dir,
            feature_store=feature_store,
            max_workers=process_max_workers,
            intermediate_cache_dir=intermediate_cache_dir,
            intermediate_batch_size=intermediate_batch_size,
            position_count_threshold=position_count_threshold,
            prior_strength=prior_strength,
            win_rate_fallback=win_rate_fallback,
            drop_below_threshold=drop_below_threshold,
            search_value_paths=search_value_path,
        )
    )
