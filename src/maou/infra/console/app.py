import click

from maou.infra import file_system
from maou.interface import hcpe_converter_interface


@click.group()
def main() -> None:
    pass


@click.command()
def status() -> None:
    click.echo("Good")


@click.command()
@click.option("--file", type=str, required=True)
@click.option(
    "--input-format",
    type=str,
    help='This command supports kif or csa. Input "kif" or "csa".',
)
def hcpe_convert(file: str, input_format: str) -> None:
    try:
        click.echo(
            hcpe_converter_interface.transform(
                file, input_format, file_system.FileSystem()
            )
        )
    except Exception as e:
        print(f"An Error Occured: {e}")


main.add_command(status)
main.add_command(hcpe_convert)


def hoge() -> str:
    return "hoge"
