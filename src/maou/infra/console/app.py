import logging
from pathlib import Path
from typing import Optional

import click

from maou.infra.app_logging import app_logger
from maou.infra.file_system import FileSystem
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
    help="",
    type=int,
    required=False,
)
@click.option(
    "--min-moves",
    help="",
    type=int,
    required=False,
)
@click.option(
    "--max-moves",
    help="",
    type=int,
    required=False,
)
@click.option(
    "--allowed-endgame-status",
    help="",
    type=str,
    required=False,
    multiple=True,
)
def hcpe_convert(
    input_path: Path,
    input_format: str,
    output_dir: Path,
    min_rating: int,
    min_moves: int,
    max_moves: int,
    allowed_endgame_status: list[str],
) -> None:
    try:
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
) -> None:
    try:
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
            )
        )
    except Exception:
        app_logger.exception("Error Occured", stack_info=True)


main.add_command(hcpe_convert)
main.add_command(learn_model)
