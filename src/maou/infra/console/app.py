import logging
from pathlib import Path
from typing import Optional

import click

from maou.infra.app_logging import app_logger
from maou.infra.bigquery.bigquery import BigQuery
from maou.infra.bigquery.bq_data_source import BigQueryDataSource
from maou.infra.file_system.file_data_source import FileDataSource
from maou.infra.file_system.file_system import FileSystem
from maou.infra.gcs.gcs import GCS
from maou.interface import converter, learn, preprocess


@click.group()
@click.option("--debug-mode", "-d", is_flag=True, help="Enable debug logging.")
def main(debug_mode: bool) -> None:
    if debug_mode:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(logging.INFO)


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
    type=click.Path(exists=True, path_type=Path),
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
    "--max-cached-bytes",
    help="Max cache size in bytes for output (default: 500MB).",
    type=int,
    required=False,
    default=500 * 1024 * 1024,
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
    max_cached_bytes: int,
) -> None:
    try:
        feature_store = (
            BigQuery(
                dataset_id=dataset_id,
                table_name=table_name,
                max_cached_bytes=max_cached_bytes,
            )
            if output_bigquery and dataset_id is not None and table_name is not None
            else None
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
def pre_process(
    input_path: Optional[Path],
    output_dir: Optional[Path],
    input_dataset_id: Optional[str],
    input_table_name: Optional[str],
    input_batch_size: int,
    input_max_cached_bytes: int,
    output_bigquery: Optional[bool],
    dataset_id: Optional[str],
    table_name: Optional[str],
    output_max_cached_bytes: int,
) -> None:
    try:
        if input_dataset_id is not None and input_table_name is not None:
            datasource = BigQueryDataSource(
                dataset_id=input_dataset_id,
                table_name=input_table_name,
                batch_size=input_batch_size,
                max_cached_bytes=input_max_cached_bytes,
            )
        elif input_path is not None:
            schema_datasource = {
                "hcp": "hcp",
                "bestMove16": "bestMove16",
                "gameResult": "gameResult",
                "eval": "eval",
            }
            input_paths = FileSystem.collect_files(input_path)
            datasource = FileDataSource(
                file_paths=input_paths, schema=schema_datasource
            )
        feature_store = (
            BigQuery(
                dataset_id=dataset_id,
                table_name=table_name,
                max_cached_bytes=output_max_cached_bytes,
            )
            if output_bigquery and dataset_id is not None and table_name is not None
            else None
        )
        click.echo(
            preprocess.transform(
                datasource=datasource,
                output_dir=output_dir,
                feature_store=feature_store,
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
    "--checkpoint-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Checkpoint directory.",
    required=False,
)
@click.option(
    "--resume-from",
    type=click.Path(exists=True, path_type=Path),
    help="Checkpoint file to resume training.",
    required=False,
)
@click.option(
    "--log-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Log directory for SummaryWriter.",
    required=False,
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True, path_type=Path),
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
    "--bucket-name",
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
def learn_model(
    input_dir: Optional[Path],
    input_dataset_id: Optional[str],
    input_table_name: Optional[str],
    input_format: str,
    input_batch_size: int,
    input_max_cached_bytes: int,
    input_clustering_key: Optional[str],
    input_partitioning_key_date: Optional[str],
    gpu: Optional[str],
    compilation: Optional[bool],
    test_ratio: Optional[float],
    epoch: Optional[int],
    batch_size: Optional[int],
    dataloader_workers: Optional[int],
    gce_parameter: Optional[float],
    policy_loss_ratio: Optional[float],
    value_loss_ratio: Optional[float],
    learning_ratio: Optional[float],
    momentum: Optional[float],
    checkpoint_dir: Optional[Path],
    resume_from: Optional[Path],
    log_dir: Optional[Path],
    model_dir: Optional[Path],
    output_gcs: Optional[bool],
    bucket_name: Optional[str],
    gcs_base_path: Optional[str],
) -> None:
    try:
        cloud_storage = (
            GCS(bucket_name=bucket_name, base_path=gcs_base_path)
            if output_gcs and bucket_name is not None and gcs_base_path is not None
            else None
        )
        if input_dir is not None:
            if input_format == "hcpe":
                schema = {
                    "hcp": "hcp",
                    "bestMove16": "bestMove16",
                    "gameResult": "gameResult",
                    "eval": "eval",
                }
            elif input_format == "preprocess":
                schema = {
                    "id": "id",
                    "eval": "eval",
                    "features": "features",
                    "moveLabel": "moveLabel",
                    "resultValue": "resultValue",
                    "legalMoveMask": "legalMoveMask",
                }
            else:
                raise Exception(
                    "Please specify a valid input_format ('hcpe' or 'preprocess')."
                )
            datasource = FileDataSource.FileDataSourceSpliter(
                file_paths=FileSystem.collect_files(input_dir),
                schema=schema,
            )
        elif input_dataset_id is not None and input_table_name is not None:
            datasource = BigQueryDataSource.BigQueryDataSourceSpliter(
                dataset_id=input_dataset_id,
                table_name=input_table_name,
                batch_size=input_batch_size,
                max_cached_bytes=input_max_cached_bytes,
                clustering_key=input_clustering_key,
                partitioning_key_date=input_partitioning_key_date,
            )
        else:
            raise Exception("Please specify an input directory or a BigQuery table.")
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
                gce_parameter=gce_parameter,
                policy_loss_ratio=policy_loss_ratio,
                value_loss_ratio=value_loss_ratio,
                learning_ratio=learning_ratio,
                momentum=momentum,
                checkpoint_dir=checkpoint_dir,
                resume_from=resume_from,
                log_dir=log_dir,
                model_dir=model_dir,
                cloud_storage=cloud_storage,
            )
        )
    except Exception:
        app_logger.exception("Error occurred", stack_info=True)


main.add_command(hcpe_convert)
main.add_command(pre_process)
main.add_command(learn_model)
