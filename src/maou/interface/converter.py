import abc
import enum
import json
import logging
from pathlib import Path
from typing import Optional

from maou.app.converter.hcpe_converter import FeatureStore, HCPEConverter

logger: logging.Logger = logging.getLogger(__name__)


# 特定の文字列しか入力されないようにする
@enum.unique
class InputFormat(enum.Enum):
    CSA = "csa"
    KIF = "kif"


class FileSystem(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def get_text(filename: str) -> str:
        pass

    @staticmethod
    @abc.abstractmethod
    def collect_files(p: Path, ext: Optional[str] = None) -> list[Path]:
        pass


def input_format_validation(input_format: str) -> None:
    try:
        InputFormat(input_format)
    except ValueError:
        raise ValueError('Input "kif" or "csa".')


def output_dir_validation(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"Output Dir `{output_dir}` is not directory.")


def transform(
    file_system: FileSystem,
    input_path: Path,
    input_format: str,
    output_dir: Path,
    *,
    min_rating: Optional[int] = None,
    min_moves: Optional[int] = None,
    max_moves: Optional[int] = None,
    allowed_endgame_status: Optional[list[str]] = None,
    exclude_moves: Optional[list[int]] = None,
    feature_store: Optional[FeatureStore] = None,
) -> str:
    input_format_validation(input_format)
    output_dir_validation(output_dir)
    logger.info(f"Input: {input_path}, Output: {output_dir}")

    option = HCPEConverter.ConvertOption(
        input_paths=file_system.collect_files(input_path),
        input_format=input_format,
        output_dir=output_dir,
        min_rating=min_rating,
        min_moves=min_moves,
        max_moves=max_moves,
        allowed_endgame_status=allowed_endgame_status,
        exclude_moves=exclude_moves,
    )
    conversion_result = HCPEConverter(
        feature_store=feature_store,
    ).convert(option)

    return json.dumps(conversion_result)
