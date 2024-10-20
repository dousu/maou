from pathlib import Path

import click

from maou.interface import hcpe_converter_interface


@click.group()
def main() -> None:
    pass


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
            hcpe_converter_interface.transform(input_path, input_format, output_dir)
        )
    except Exception as e:
        print(f"An Error Occured: {e}")


main.add_command(status)
main.add_command(hcpe_convert)


def hoge() -> str:
    return "hoge"
