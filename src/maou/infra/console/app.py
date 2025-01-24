import logging
from pathlib import Path
from typing import Optional

import click

from maou.infra.app_logging import app_logger
from maou.infra.bigquery import BigQuery
from maou.infra.file_system import FileSystem
from maou.infra.gcs import GCS
from maou.interface import converter, learn


@click.group()
@click.option("--debug_mode", "-d", is_flag=True, help="Show debug log")
def main(debug_mode: bool) -> None:
    if debug_mode is True:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(logging.INFO)


@click.command()
@click.option(
    "--input-path",
    help="Specify the file or directory where the input is located.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--input-format",
    type=str,
    help='This command supports kif or csa. Input "kif" or "csa".',
    required=True,
)
@click.option(
    "--output-dir",
    help="Specify the directory where the output files is saved.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--min-rating",
    help="Specify the minimum rating of the game to include in the conversion.",
    type=int,
    required=False,
)
@click.option(
    "--min-moves",
    help="Specify the minimum number of moves required to include the game.",
    type=int,
    required=False,
)
@click.option(
    "--max-moves",
    help="Specify the maximum number of moves allowed to include the game.",
    type=int,
    required=False,
)
@click.option(
    "--allowed-endgame-status",
    help=(
        "Specify allowed endgame statuses (e.g., '%TORYO', '%SENNICHITE', '%KACHI')."
        " Multiple values can be provided."
    ),
    type=str,
    required=False,
    multiple=True,
)
@click.option(
    "--exclude-moves",
    help=(
        "Specify moves to exclude from the game."
        " Provide move nuymbers as integers."
        " Multiple values can be provided."
    ),
    type=int,
    required=False,
    multiple=True,
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
    help="Specify the BigQuery dataset ID to use as the feature store.",
    type=str,
    required=False,
)
@click.option(
    "--table-name",
    help="Specify the BigQuery table name to use as the feature store.",
    type=str,
    required=False,
)
@click.option(
    "--max-buffer-bytes",
    help="max buffer bytes",
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
    "--input-dir",
    help="Specify the directory where the input data is located.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--gpu",
    type=str,
    help="Configure pytorch device",
    required=False,
)
@click.option(
    "--compilation",
    type=bool,
    help="Configure pytorch device",
    required=False,
)
@click.option(
    "--test-ratio",
    type=float,
    help="test_size in train_test_split",
    required=False,
)
@click.option(
    "--epoch",
    type=int,
    help="number of epochs",
    required=False,
)
@click.option(
    "--batch-size",
    type=int,
    help="batch size",
    required=False,
)
@click.option(
    "--dataloader-workers",
    type=int,
    help="number of workers for DataLoader",
    required=False,
)
@click.option(
    "--gce-parameter",
    type=float,
    help="Hyper parameter for GCE loss",
    required=False,
)
@click.option(
    "--policy-loss-ratio",
    type=float,
    help="Coefficient value for loss of policy",
    required=False,
)
@click.option(
    "--value-loss-ratio",
    type=float,
    help="Coefficient value for loss of value",
    required=False,
)
@click.option(
    "--learning-ratio",
    type=float,
    help="learning ratio for optimizer",
    required=False,
)
@click.option(
    "--momentum",
    type=float,
    help="momentum value for optimizer",
    required=False,
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Specify the directory where the checkpoint files is saved.",
    required=False,
)
@click.option(
    "--resume-from",
    type=click.Path(exists=True, path_type=Path),
    help="Specify the checkpoint file. Learning will start at checkpoint's epoch",
    required=False,
)
@click.option(
    "--log-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Specify the directory where the log files for SummeryaWriter is saved.",
    required=False,
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Specify the directory where the models is saved.",
    required=False,
)
@click.option(
    "--gcs",
    type=bool,
    is_flag=True,
    help="Use GCS object storage to save files",
    required=False,
)
@click.option(
    "--bucket-name",
    help="GCS bucket name",
    type=str,
    required=False,
)
@click.option(
    "--gcs-base-path",
    help="base path for GCS bucket",
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
main.add_command(learn_model)
