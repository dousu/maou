import abc
import json
import logging
from pathlib import Path
from typing import Optional

from maou.app.pre_process.hcpe_transform import DataSource, FeatureStore, PreProcess

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
    datasource: DataSource,
    output_dir: Optional[Path],
    feature_store: Optional[FeatureStore] = None,
) -> str:
    if output_dir is not None:
        output_dir_validation(output_dir)

    option = PreProcess.PreProcessOption(
        output_dir=output_dir,
    )
    pre_process_result = PreProcess(
        datasource=datasource,
        feature_store=feature_store,
    ).transform(option)

    return json.dumps(pre_process_result)
