import logging
from pathlib import Path
from typing import Optional

import click

from maou.infra.app_logging import app_logger
from maou.infra.bigquery import BigQuery
from maou.infra.file_system import FileSystem
from maou.infra.gcs import GCS
from maou.interface import converter, learn, preprocess


@click.group()
@click.option("--debug_mode", "-d", is_flag=True, help="Enable debug logging.")
def main(debug_mode: bool) -> None:
    if debug_mode is True:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(logging.INFO)


@click.command()
@click.option(
    "--input-path",
    help="Path to the input file or directory.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--input-format",
    type=str,
    help='Specify the input format. Supported formats: "kif" or "csa".',
    required=True,
)
@click.option(
    "--output-dir",
    help="Directory to save the output files.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--min-rating",
    help="Minimum rating of games to include.",
    type=int,
    required=False,
)
@click.option(
    "--min-moves",
    help="Minimum number of moves required for a game to be included.",
    type=int,
    required=False,
)
@click.option(
    "--max-moves",
    help="Maximum number of moves allowed for a game to be included.",
    type=int,
    required=False,
)
@click.option(
    "--allowed-endgame-status",
    help=(
        "Allowed endgame statuses (e.g., '%TORYO', '%SENNICHITE', '%KACHI')."
        " Can be specified multiple times."
    ),
    type=str,
    required=False,
    multiple=True,
)
@click.option(
    "--exclude-moves",
    help=(
        "Moves to exclude from the game (integer move numbers)."
        " Can be specified multiple times."
    ),
    type=int,
    required=False,
    multiple=True,
)
@click.option(
    "--bigquery",
    type=bool,
    is_flag=True,
    help="Store features in BigQuery.",
    required=False,
)
@click.option(
    "--dataset-id",
    help="BigQuery dataset ID for storing features.",
    type=str,
    required=False,
)
@click.option(
    "--table-name",
    help="BigQuery table name for storing features.",
    type=str,
    required=False,
)
@click.option(
    "--max-buffer-bytes",
    help="Maximum buffer size in bytes (default: 50MB).",
    type=int,
    required=False,
    default=50 * 1024 * 1024,
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
    bigquery: Optional[bool],
    dataset_id: Optional[str],
    table_name: Optional[str],
    max_buffer_bytes: int,
) -> None:
    try:
        feature_store = (
            BigQuery(
                dataset_id=dataset_id,
                table_name=table_name,
                max_buffer_size=max_buffer_bytes,
            )
            if bigquery
            and dataset_id is not None
            and table_name is not None
            and max_buffer_bytes is not None
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
        app_logger.exception("Error Occured", stack_info=True)


@click.command()
@click.option(
    "--input-path",
    help="Path to the input file or directory.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    help="Directory to save the output files.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--bigquery",
    type=bool,
    is_flag=True,
    help="Use BigQuery as feature store",
    required=False,
)
@click.option(
    "--dataset-id",
    help="BigQuery dataset ID for storing features.",
    type=str,
    required=False,
)
@click.option(
    "--table-name",
    help="BigQuery table name for storing features.",
    type=str,
    required=False,
)
@click.option(
    "--max-buffer-bytes",
    help="Maximum buffer size in bytes (default: 50MB).",
    type=int,
    required=False,
    default=50 * 1024 * 1024,
)
def pre_process(
    input_path: Path,
    output_dir: Path,
    bigquery: Optional[bool],
    dataset_id: Optional[str],
    table_name: Optional[str],
    max_buffer_bytes: int,
) -> None:
    try:
        feature_store = (
            BigQuery(
                dataset_id=dataset_id,
                table_name=table_name,
                max_buffer_size=max_buffer_bytes,
            )
            if bigquery
            and dataset_id is not None
            and table_name is not None
            and max_buffer_bytes is not None
            else None
        )
        click.echo(
            preprocess.transform(
                file_system=FileSystem(),
                input_path=input_path,
                output_dir=output_dir,
                feature_store=feature_store,
            )
        )
    except Exception:
        app_logger.exception("Error Occured", stack_info=True)


@click.command()
@click.option(
    "--input-dir",
    help="Specify the directory where the input data is located.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--gpu",
    type=str,
    help='Specify the PyTorch device (e.g., "cuda:0" or "cpu").',
    required=False,
)
@click.option(
    "--compilation",
    type=bool,
    help="Enable PyTorch compilation (True/False).",
    required=False,
)
@click.option(
    "--test-ratio",
    type=float,
    help="Ratio of test data in train-test split.",
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
    help="Batch size for training.",
    required=False,
)
@click.option(
    "--dataloader-workers",
    type=int,
    help="Number of workers for DataLoader.",
    required=False,
)
@click.option(
    "--gce-parameter",
    type=float,
    help="Hyperparameter for GCE loss.",
    required=False,
)
@click.option(
    "--policy-loss-ratio",
    type=float,
    help="Loss coefficient for policy loss.",
    required=False,
)
@click.option(
    "--value-loss-ratio",
    type=float,
    help="Loss coefficient for value loss.",
    required=False,
)
@click.option(
    "--learning-ratio",
    type=float,
    help="Learning rate for optimizer.",
    required=False,
)
@click.option(
    "--momentum",
    type=float,
    help="Momentum parameter for optimizer.",
    required=False,
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory to save model checkpoints.",
    required=False,
)
@click.option(
    "--resume-from",
    type=click.Path(exists=True, path_type=Path),
    help="Path to a checkpoint file to resume training.",
    required=False,
)
@click.option(
    "--log-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory to save log files for SummaryWriter.",
    required=False,
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory to save trained models.",
    required=False,
)
@click.option(
    "--gcs",
    type=bool,
    is_flag=True,
    help="Save files to Google Cloud Storage (GCS).",
    required=False,
)
@click.option(
    "--bucket-name",
    help="Google Cloud Storage bucket name.",
    type=str,
    required=False,
)
@click.option(
    "--gcs-base-path",
    help="Base path within the GCS bucket.",
    type=str,
    required=False,
)
def learn_model(
    input_dir: Path,
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
    gcs: Optional[bool],
    bucket_name: Optional[str],
    gcs_base_path: Optional[str],
) -> None:
    try:
        cloud_storage = (
            GCS(bucket_name=bucket_name, base_path=gcs_base_path)
            if gcs and bucket_name is not None and gcs_base_path is not None
            else None
        )
        click.echo(
            learn.learn(
                FileSystem(),
                input_dir=input_dir,
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
        app_logger.exception("Error Occured", stack_info=True)


main.add_command(hcpe_convert)
main.add_command(pre_process)
main.add_command(learn_model)
