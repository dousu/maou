import abc
import enum
from pathlib import Path

from maou.app.converter.hcpe_converter import HCPEConverter


# 特定の文字列しか入力されないようにする
@enum.unique
class InputFormat(enum.Enum):
    CSA = "csa"
    KIF = "kif"


class FileLoader(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def get_text(filename: str) -> str:
        pass


def input_format_validation(input_format: str) -> None:
    try:
        InputFormat(input_format)
    except ValueError:
        raise ValueError('Input "kif" or "csa".')


def output_dir_validation(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"Output Dir `{output_dir}` is not directory.")


def collect_files(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    elif p.is_dir():
        return [f for f in p.glob("**/*") if f.is_file()]
    else:
        raise ValueError(f"Path `{p}` is neither a file nor a directory.")


def transform(input_path: Path, input_format: str, output_dir: Path) -> str:
    input_format_validation(input_format)
    output_dir_validation(output_dir)
    print(f"Input: {input_path}\nOutput: {output_dir}")
    HCPEConverter.convert(collect_files(input_path), input_format, output_dir)

    return "fin"
