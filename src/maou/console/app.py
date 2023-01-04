import click


@click.group()
def main():
    pass


@click.command()
def status():
    click.echo("Good")


main.add_command(status)


def hoge():
    return "hoge"
