import abc
import json
import logging
import os
import tempfile
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


def split_input_files(
    file_paths: list[Path],
    rows_per_file: int,
    split_dir: Optional[Path] = None,
) -> list[Path]:
    """入力HCPEファイルを指定行数ごとに事前分割する．

    大きなファイルを複数の小さなファイルに分割し，
    並列処理時のワーカー間負荷分散を改善する．
    Rustバックエンドを使用してLZ4圧縮を維持したまま高速に分割する．

    Args:
        file_paths: 入力ファイルパスのリスト
        rows_per_file: 1ファイルあたりの最大行数
        split_dir: 分割ファイルの出力ディレクトリ（Noneの場合は一時ディレクトリ）

    Returns:
        分割後のファイルパスのリスト（分割不要なファイルは元のパスをそのまま含む）
    """
    from maou.domain.data.rust_io import split_hcpe_feather

    if split_dir is None:
        split_dir = Path(tempfile.mkdtemp(prefix="maou_split_"))

    result_paths: list[Path] = []
    for fp in file_paths:
        split_paths = split_hcpe_feather(
            fp, split_dir, rows_per_file
        )
        result_paths.extend(split_paths)

    split_count = len(result_paths) - len(file_paths)
    if split_count > 0:
        logger.info(
            "Split %d files into %d files (rows_per_file=%d)",
            len(file_paths),
            len(result_paths),
            rows_per_file,
        )

    return result_paths


def chunk_input_files(
    file_paths: list[Path],
    rows_per_chunk: int,
    chunk_dir: Optional[Path] = None,
) -> list[Path]:
    """複数の小さな入力HCPEファイルをチャンクにまとめる．

    細かいファイルが大量にある場合に，指定行数ごとにまとめて
    並列処理に適したファイル粒度に統合する．
    Rustバックエンドを使用してLZ4圧縮を維持したまま高速に結合する．

    Args:
        file_paths: 入力ファイルパスのリスト
        rows_per_chunk: 1チャンクあたりの目標行数
        chunk_dir: チャンクファイルの出力ディレクトリ（Noneの場合は一時ディレクトリ）

    Returns:
        チャンク後のファイルパスのリスト
    """
    from maou.domain.data.rust_io import (
        merge_hcpe_feather_files,
    )

    if len(file_paths) <= 1:
        return file_paths

    if chunk_dir is None:
        chunk_dir = Path(tempfile.mkdtemp(prefix="maou_chunk_"))

    result_paths = merge_hcpe_feather_files(
        file_paths=file_paths,
        output_dir=chunk_dir,
        rows_per_chunk=rows_per_chunk,
        output_prefix="chunked",
    )

    if len(result_paths) < len(file_paths):
        logger.info(
            "Chunked %d files into %d files "
            "(rows_per_chunk=%d)",
            len(file_paths),
            len(result_paths),
            rows_per_chunk,
        )

    return result_paths


def resize_input_files(
    file_paths: list[Path],
    rows_per_file: int,
    work_dir: Optional[Path] = None,
) -> list[Path]:
    """入力HCPEファイルのサイズを最適化する．

    大きなファイルは分割し，小さなファイルはチャンクにまとめて，
    並列処理に適したファイル粒度に調整する．

    1. まず大きなファイルを分割（split）
    2. 次に小さなファイルをチャンク（merge）

    Args:
        file_paths: 入力ファイルパスのリスト
        rows_per_file: 1ファイルあたりの目標行数
        work_dir: 作業ディレクトリ（Noneの場合は一時ディレクトリ）

    Returns:
        リサイズ後のファイルパスのリスト
    """
    if not file_paths:
        return file_paths

    # Step 1: 大きなファイルを分割
    split_result = split_input_files(
        file_paths=file_paths,
        rows_per_file=rows_per_file,
        split_dir=work_dir,
    )

    # Step 2: 小さなファイルをチャンクにまとめる
    # rows_per_file の半分未満のファイルを「小さい」と判定
    threshold = rows_per_file // 2
    small_files: list[Path] = []
    ok_files: list[Path] = []

    for fp in split_result:
        try:
            import polars as pl

            row_count = len(pl.scan_ipc(fp).collect())
        except Exception:
            ok_files.append(fp)
            continue

        if row_count < threshold:
            small_files.append(fp)
        else:
            ok_files.append(fp)

    if len(small_files) > 1:
        chunked = chunk_input_files(
            file_paths=small_files,
            rows_per_chunk=rows_per_file,
            chunk_dir=work_dir,
        )
        ok_files.extend(chunked)
    else:
        ok_files.extend(small_files)

    return ok_files


def transform(
    *,
    datasource: DataSource,
    output_dir: Optional[Path],
    feature_store: Optional[FeatureStore] = None,
    max_workers: Optional[int] = None,
    intermediate_cache_dir: Optional[Path] = None,
    intermediate_batch_size: int = 50_000,
    win_rate_threshold: int = 2,
) -> str:
    """Transform HCPE data into neural network training features.

    Args:
        datasource: Source of HCPE data to process
        output_dir: Optional directory for output files
        feature_store: Optional storage backend for features
        max_workers: Number of parallel workers for CPU processing
        intermediate_cache_dir: Directory for intermediate data cache
        intermediate_batch_size: DuckDBへのフラッシュ前に蓄積するレコード数．
            Google Colab A100 High Memory (83GB RAM) ではデフォルト50,000を推奨．
        win_rate_threshold: Minimum position occurrence count for per-move
            win rate calculation. Positions with count below this threshold
            use uniform 1/N fallback. (default: 2)

    Returns:
        JSON string with processing results
    """
    if output_dir is not None:
        output_dir_init(output_dir)

    # 並列処理数 (デフォルトCPU数か3の小さい方)
    if max_workers is None:
        max_workers = min(3, os.cpu_count() or 1)
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
        intermediate_cache_dir=intermediate_cache_dir,
        intermediate_batch_size=intermediate_batch_size,
        win_rate_threshold=win_rate_threshold,
    ).transform(option)

    return json.dumps(pre_process_result)
