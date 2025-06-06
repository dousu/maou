import abc
import json
import logging
from pathlib import Path
from typing import Optional

from maou.app.pre_process.hcpe_transform import DataSource, FeatureStore, PreProcess

logger: logging.Logger = logging.getLogger(__name__)


class FileSystem(metaclass=abc.ABCMeta):
    """Abstract interface for file system operations.

    Provides an abstraction layer for file I/O operations
    for pre-processing workflows.
    """

    @staticmethod
    @abc.abstractmethod
    def collect_files(p: Path, ext: Optional[str] = None) -> list[Path]:
        pass


def output_dir_validation(output_dir: Path) -> None:
    """Validate output directory exists.

    Args:
        output_dir: Directory path to validate

    Raises:
        ValueError: If path is not a directory
    """
    if not output_dir.is_dir():
        raise ValueError(f"Output Dir `{output_dir}` is not directory.")


def transform(
    *,
    datasource: DataSource,
    output_dir: Optional[Path],
    feature_store: Optional[FeatureStore] = None,
) -> str:
    """Transform HCPE data into neural network training features.

    Args:
        datasource: Source of HCPE data to process
        output_dir: Optional directory for output files
        feature_store: Optional storage backend for features

    Returns:
        JSON string with processing results
    """
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
