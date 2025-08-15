import abc
import json
import logging
import os
from pathlib import Path
from typing import Optional

from maou.app.pre_process.hcpe_transform import (
    DataSource,
    FeatureStore,
    PreProcess,
)

logger: logging.Logger = logging.getLogger(__name__)


class FileSystem(metaclass=abc.ABCMeta):
    """Abstract interface for file system operations.

    Provides an abstraction layer for file I/O operations
    for pre-processing workflows.
    """

    @staticmethod
    @abc.abstractmethod
    def collect_files(
        p: Path, ext: Optional[str] = None
    ) -> list[Path]:
        pass


def output_dir_init(output_dir: Path) -> None:
    """Initialize output directory, creating if it doesn't exist.

    Args:
        output_dir: Directory path to initialize

    Raises:
        ValueError: If path exists but is not a directory
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        raise ValueError(
            f"Output Dir `{output_dir}` is not directory."
        )


def transform(
    *,
    datasource: DataSource,
    output_dir: Optional[Path],
    feature_store: Optional[FeatureStore] = None,
    max_workers: Optional[int] = None,
) -> str:
    """Transform HCPE data into neural network training features.

    Args:
        datasource: Source of HCPE data to process
        output_dir: Optional directory for output files
        feature_store: Optional storage backend for features
        max_workers: Number of parallel workers for CPU processing

    Returns:
        JSON string with processing results
    """
    if output_dir is not None:
        output_dir_init(output_dir)

    # 並列処理数 (デフォルトCPU数か4の小さい方)
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)
    elif max_workers < 0:
        raise ValueError(
            f"max_workers must be non-negative, got {max_workers}"
        )

    option = PreProcess.PreProcessOption(
        output_dir=output_dir,
        max_workers=max_workers,
    )
    pre_process_result = PreProcess(
        datasource=datasource,
        feature_store=feature_store,
    ).transform(option)

    return json.dumps(pre_process_result)
