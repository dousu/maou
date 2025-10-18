from pathlib import Path
from typing import Optional

import click

from maou.infra.console.common import (
    app_logger,
    handle_exception,
)
from maou.interface.pretrain import (
    pretrain as pretrain_interface,
)


@click.command("pretrain")
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path, exists=True),
    required=False,
    help="Input directory containing training data.",
)
@click.option(
    "--config-path",
    type=click.Path(path_type=Path, exists=True),
    required=False,
    help="Optional configuration file for pretraining.",
)
@handle_exception
def pretrain(
    input_dir: Optional[Path],
    config_path: Optional[Path],
) -> None:
    """CLI entry point for masked autoencoder pretraining (stub)."""
    app_logger.info(
        "Invoking masked autoencoder pretraining command."
    )
    message = pretrain_interface(
        input_dir=input_dir,
        config_path=config_path,
    )
    click.echo(message)
