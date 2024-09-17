import click


@click.group()
def main():
    pass


@click.command()
def status():
    click.echo("Good")


@click.command()
def process():
    click.echo("Good")


main.add_command(status)
main.add_command(process)


def hoge():
    return "hoge"
