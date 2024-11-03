import logging
from pathlib import Path

import click

from maou.infra.app_logging import app_logger
from maou.infra.file_system import FileSystem
from maou.interface import hcpe_converter_interface


@click.group()
@click.option("--debug_mode", "-d", is_flag=True, help="Show debug log")
def main(debug_mode: bool) -> None:
    if debug_mode is True:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(logging.INFO)


@click.command()
def status() -> None:
    click.echo("Good")


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
def hcpe_convert(input_path: Path, input_format: str, output_dir: Path) -> None:
    try:
        click.echo(
            hcpe_converter_interface.transform(
                app_logger, FileSystem(), input_path, input_format, output_dir
            )
        )
    except Exception:
        app_logger.exception("Error Occured", stack_info=True)


main.add_command(status)
main.add_command(hcpe_convert)
