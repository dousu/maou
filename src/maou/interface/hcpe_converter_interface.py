import abc
import enum

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


def transform(file_name: str, input_format: str, file_loader: FileLoader) -> str:
    input_format_validation(input_format)
    print(file_name)
    HCPEConverter.convert(file_loader.get_text(file_name), input_format)

    return "fin"
