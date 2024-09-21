import io

import click

from maou.interface import pre_process


@click.group()
def main() -> None:
    pass


@click.command()
def status() -> None:
    click.echo("Good")


@click.command()
@click.argument("input", type=click.File(mode="r"))
def process(input: io.TextIOBase) -> None:
    click.echo("Good")
    # typeはTextIOWrapper
    # click.echo(type(input))
    # readでEOFまで読める
    # click.echo(type(input.read()))
    click.echo(pre_process.pre_process(input))


main.add_command(status)
main.add_command(process)


def hoge() -> str:
    return "hoge"
