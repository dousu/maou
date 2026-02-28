"""Rust-based Arrow IPC I/O with Polars DataFrames．

このモジュールはRustで実装されたArrow IPC（Feather形式）ファイルI/Oを，
Polars DataFrameとして扱うためのPythonラッパーを提供する．

主な特徴:
- ゼロコピー変換（Polars ↔ Arrow RecordBatch）
- LZ4圧縮による高速I/Oとストレージ削減
- 型安全なインターフェース
"""

from pathlib import Path
from typing import Union, cast

import polars as pl

try:
    from maou._rust.maou_io import (
        load_feather_file,
        load_hcpe_feather,
        load_preprocessing_feather,
    )
    from maou._rust.maou_io import (
        merge_feather_files as _merge_feather_files,
    )
    from maou._rust.maou_io import (
        save_feather_file,
        save_hcpe_feather,
        save_preprocessing_feather,
        split_feather_file,
    )

    RUST_BACKEND_AVAILABLE = True
except ImportError as e:
    RUST_BACKEND_AVAILABLE = False
    _import_error = e


def _check_rust_backend() -> None:
    """Rustバックエンドが利用可能かチェックする．"""
    if not RUST_BACKEND_AVAILABLE:
        raise ImportError(
            f"Rust backend not available. Build with: uv run maturin develop\n"
            f"Original error: {_import_error}"
        )


def save_hcpe_df(
    df: pl.DataFrame, file_path: Union[Path, str]
) -> None:
    """HCPE DataFrameを.featherファイルに保存する（Rustバックエンド使用）．

    Polars DataFrameをArrow RecordBatchに変換し，Rustで実装された
    高速I/O関数でLZ4圧縮付きFeather形式ファイルとして保存する．

    Args:
        df: HCPEスキーマを持つPolars DataFrame
        file_path: 出力ファイルパス（.feather拡張子推奨）

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル書き込みエラーの場合
    """
    _check_rust_backend()

    # Create parent directories if they don't exist
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Polars → Arrow Table → RecordBatch（ゼロコピー）
    arrow_table = df.to_arrow()
    # Convert Table to RecordBatch (combine all batches)
    arrow_batch = (
        arrow_table.to_batches()[0]
        if len(arrow_table) > 0
        else arrow_table.to_batches(max_chunksize=None)[0]
    )

    # Rust関数を呼び出し
    save_hcpe_feather(arrow_batch, str(file_path))


def load_hcpe_df(file_path: Union[Path, str]) -> pl.DataFrame:
    """HCPE DataFrameを.featherファイルから読み込む（Rustバックエンド使用）．

    Rustで実装された高速I/O関数でFeather形式ファイルを読み込み，
    Arrow RecordBatchをPolars DataFrameに変換して返す．
    Stream形式とFile形式の両方に自動対応．

    Args:
        file_path: 入力ファイルパス（.feather拡張子）

    Returns:
        HCPEスキーマを持つPolars DataFrame

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル読み込みエラーの場合
    """
    _check_rust_backend()

    # Rust関数を呼び出し（Stream/File形式自動判定）
    arrow_batch = load_hcpe_feather(str(file_path))
    # Arrow → Polars（ゼロコピー）
    return cast(pl.DataFrame, pl.from_arrow(arrow_batch))


def save_preprocessing_df(
    df: pl.DataFrame, file_path: Union[Path, str]
) -> None:
    """前処理済みDataFrameを.featherファイルに保存する（Rustバックエンド使用）．

    Args:
        df: 前処理スキーマを持つPolars DataFrame
        file_path: 出力ファイルパス（.feather拡張子推奨）

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル書き込みエラーの場合
    """
    _check_rust_backend()

    # Create parent directories if they don't exist
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Polars → Arrow Table → RecordBatch
    arrow_table = df.to_arrow()
    arrow_batch = (
        arrow_table.to_batches()[0]
        if len(arrow_table) > 0
        else arrow_table.to_batches(max_chunksize=None)[0]
    )
    save_preprocessing_feather(arrow_batch, str(file_path))


def load_preprocessing_df(
    file_path: Union[Path, str],
) -> pl.DataFrame:
    """前処理済みDataFrameを.featherファイルから読み込む（Rustバックエンド使用）．

    Stream形式とFile形式の両方に自動対応．

    Args:
        file_path: 入力ファイルパス（.feather拡張子）

    Returns:
        前処理スキーマを持つPolars DataFrame

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル読み込みエラーの場合
    """
    _check_rust_backend()

    # Rust関数を呼び出し（Stream/File形式自動判定）
    arrow_batch = load_preprocessing_feather(str(file_path))
    return cast(pl.DataFrame, pl.from_arrow(arrow_batch))


def save_stage1_df(
    df: pl.DataFrame, file_path: Union[Path, str]
) -> None:
    """Stage 1 DataFrameを.featherファイルに保存する（Rustバックエンド使用）．

    Args:
        df: Stage 1スキーマを持つPolars DataFrame
        file_path: 出力ファイルパス（.feather拡張子推奨）

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル書き込みエラーの場合

    Example:
        >>> from maou.domain.data.schema import get_stage1_polars_schema
        >>> df = pl.DataFrame(data, schema=get_stage1_polars_schema())
        >>> save_stage1_df(df, "stage1_data.feather")
    """
    _check_rust_backend()

    # Create parent directories if they don't exist
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Polars → Arrow Table → RecordBatch（ゼロコピー）
    arrow_table = df.to_arrow()
    arrow_batch = (
        arrow_table.to_batches()[0]
        if len(arrow_table) > 0
        else arrow_table.to_batches(max_chunksize=None)[0]
    )

    save_feather_file(arrow_batch, str(file_path))


def load_stage1_df(file_path: Union[Path, str]) -> pl.DataFrame:
    """Stage 1 DataFrameを.featherファイルから読み込む（Rustバックエンド使用）．

    Stream形式とFile形式の両方に自動対応．

    Args:
        file_path: 入力ファイルパス（.feather拡張子）

    Returns:
        Stage 1スキーマを持つPolars DataFrame

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル読み込みエラーの場合

    Example:
        >>> df = load_stage1_df("stage1_data.feather")
        >>> print(f"Loaded {len(df)} Stage 1 records")
    """
    _check_rust_backend()

    arrow_batch = load_feather_file(str(file_path))
    return cast(pl.DataFrame, pl.from_arrow(arrow_batch))


def save_stage2_df(
    df: pl.DataFrame, file_path: Union[Path, str]
) -> None:
    """Stage 2 DataFrameを.featherファイルに保存する（Rustバックエンド使用）．

    Args:
        df: Stage 2スキーマを持つPolars DataFrame
        file_path: 出力ファイルパス（.feather拡張子推奨）

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル書き込みエラーの場合

    Example:
        >>> from maou.domain.data.schema import get_stage2_polars_schema
        >>> df = pl.DataFrame(data, schema=get_stage2_polars_schema())
        >>> save_stage2_df(df, "stage2_data.feather")
    """
    _check_rust_backend()

    # Create parent directories if they don't exist
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Polars → Arrow Table → RecordBatch（ゼロコピー）
    arrow_table = df.to_arrow()
    arrow_batch = (
        arrow_table.to_batches()[0]
        if len(arrow_table) > 0
        else arrow_table.to_batches(max_chunksize=None)[0]
    )

    save_feather_file(arrow_batch, str(file_path))


def load_stage2_df(file_path: Union[Path, str]) -> pl.DataFrame:
    """Stage 2 DataFrameを.featherファイルから読み込む（Rustバックエンド使用）．

    Stream形式とFile形式の両方に自動対応．

    Args:
        file_path: 入力ファイルパス（.feather拡張子）

    Returns:
        Stage 2スキーマを持つPolars DataFrame

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル読み込みエラーの場合

    Example:
        >>> df = load_stage2_df("stage2_data.feather")
        >>> print(f"Loaded {len(df)} Stage 2 records")
    """
    _check_rust_backend()

    arrow_batch = load_feather_file(str(file_path))
    return cast(pl.DataFrame, pl.from_arrow(arrow_batch))


def split_hcpe_feather(
    file_path: Union[Path, str],
    output_dir: Union[Path, str],
    rows_per_file: int = 500_000,
) -> list[Path]:
    """HCPEの.featherファイルを指定行数ごとに分割する（Rustバックエンド使用）．

    大きなHCPEファイルを複数の小さなファイルに分割し，
    並列処理時のワーカー間負荷分散を改善する．
    LZ4圧縮を維持したままRust側でArrow RecordBatchのゼロコピー
    スライスを使用して高速に分割する．

    Google Colab A100 High Memory (83GB RAM, 12 CPU cores) では
    rows_per_file=500,000を推奨．8ワーカーでの並列処理に適した
    ファイル粒度を実現する．

    Args:
        file_path: 入力HCPEファイルパス（.feather拡張子）
        output_dir: 分割ファイルの出力ディレクトリ
        rows_per_file: 1ファイルあたりの最大行数（デフォルト: 500,000）

    Returns:
        分割されたファイルパスのリスト．
        入力ファイルが rows_per_file 以下の場合は元のパスをそのまま返す．

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイルI/Oエラーの場合
        ValueError: rows_per_file が0以下の場合
    """
    _check_rust_backend()

    if rows_per_file <= 0:
        raise ValueError(
            f"rows_per_file must be positive, got {rows_per_file}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths = split_feather_file(
        str(file_path), str(output_dir), rows_per_file
    )

    return [Path(p) for p in result_paths]


def merge_hcpe_feather_files(
    file_paths: list[Union[Path, str]],
    output_dir: Union[Path, str],
    rows_per_chunk: int = 500_000,
    output_prefix: str = "merged",
) -> list[Path]:
    """複数の小さなHCPE .featherファイルをチャンクにまとめる（Rustバックエンド使用）．

    複数の小さなファイルを指定行数ごとにまとめて，
    並列処理に適したファイル粒度に統合する．
    LZ4圧縮を維持したままRust側でArrow RecordBatchの
    結合を行い高速に処理する．

    Google Colab A100 High Memory (83GB RAM, 12 CPU cores) では
    rows_per_chunk=500,000を推奨．8ワーカーでの並列処理に適した
    ファイル粒度を実現する．

    Args:
        file_paths: 入力HCPEファイルパスのリスト
        output_dir: 出力ディレクトリ
        rows_per_chunk: 1チャンクあたりの目標行数（デフォルト: 500,000）
        output_prefix: 出力ファイル名のプレフィックス

    Returns:
        チャンクされたファイルパスのリスト

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイルI/Oエラーの場合
        ValueError: rows_per_chunk が0以下の場合
    """
    _check_rust_backend()

    if rows_per_chunk <= 0:
        raise ValueError(
            f"rows_per_chunk must be positive, "
            f"got {rows_per_chunk}"
        )

    if not file_paths:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    str_paths = [str(p) for p in file_paths]
    result_paths = _merge_feather_files(
        str_paths,
        str(output_dir),
        rows_per_chunk,
        output_prefix,
    )

    return [Path(p) for p in result_paths]
