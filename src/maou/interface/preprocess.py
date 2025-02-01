import abc
import json
import logging
from pathlib import Path
from typing import Optional

from maou.app.pre_process.hcpe_transform import FeatureStore, PreProcess

logger: logging.Logger = logging.getLogger(__name__)


class FileSystem(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def collect_files(p: Path, ext: Optional[str] = None) -> list[Path]:
        pass


def output_dir_validation(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"Output Dir `{output_dir}` is not directory.")


def transform(
    *,
    file_system: FileSystem,
    input_path: Path,
    output_dir: Path,
    feature_store: Optional[FeatureStore] = None,
) -> str:
    output_dir_validation(output_dir)
    logger.info(f"Input: {input_path}, Output: {output_dir}")

    option = PreProcess.PreProcessOption(
        input_paths=file_system.collect_files(input_path),
        output_dir=output_dir,
    )
    pre_process_result = PreProcess(
        feature_store=feature_store,
    ).transform(option)

    return json.dumps(pre_process_result)
