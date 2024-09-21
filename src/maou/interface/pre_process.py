import io


def pre_process(content: io.TextIOBase) -> str:
    print(content.read())
    return "fin"
