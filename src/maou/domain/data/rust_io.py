"""Rust-based Arrow IPC I/O with Polars DataFrames．

このモジュールはRustで実装されたArrow IPC（Feather形式）ファイルI/Oを，
Polars DataFrameとして扱うためのPythonラッパーを提供する．

主な特徴:
- ゼロコピー変換（Polars ↔ Arrow RecordBatch）
- LZ4圧縮による高速I/Oとストレージ削減
- 型安全なインターフェース
"""

import polars as pl
from pathlib import Path
from typing import Union

try:
    from maou._rust.maou_io import (
        save_hcpe_feather,
        load_hcpe_feather,
        save_preprocessing_feather,
        load_preprocessing_feather,
    )

    RUST_BACKEND_AVAILABLE = True
except ImportError as e:
    RUST_BACKEND_AVAILABLE = False
    _import_error = e


def _check_rust_backend() -> None:
    """Rustバックエンドが利用可能かチェックする．"""
    if not RUST_BACKEND_AVAILABLE:
        raise ImportError(
            f"Rust backend not available. Build with: poetry run maturin develop\n"
            f"Original error: {_import_error}"
        )


def save_hcpe_df(df: pl.DataFrame, file_path: Union[Path, str]) -> None:
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

    # Polars → Arrow Table → RecordBatch（ゼロコピー）
    arrow_table = df.to_arrow()
    # Convert Table to RecordBatch (combine all batches)
    arrow_batch = arrow_table.to_batches()[0] if len(arrow_table) > 0 else arrow_table.to_batches(max_chunksize=None)[0]

    # Rust関数を呼び出し
    save_hcpe_feather(arrow_batch, str(file_path))


def load_hcpe_df(file_path: Union[Path, str]) -> pl.DataFrame:
    """HCPE DataFrameを.featherファイルから読み込む（Rustバックエンド使用）．

    Rustで実装された高速I/O関数でFeather形式ファイルを読み込み，
    Arrow RecordBatchをPolars DataFrameに変換して返す．

    Args:
        file_path: 入力ファイルパス（.feather拡張子）

    Returns:
        HCPEスキーマを持つPolars DataFrame

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル読み込みエラーの場合
    """
    _check_rust_backend()

    # Rust関数を呼び出し
    arrow_batch = load_hcpe_feather(str(file_path))

    # Arrow → Polars（ゼロコピー）
    return pl.from_arrow(arrow_batch)


def save_preprocessing_df(df: pl.DataFrame, file_path: Union[Path, str]) -> None:
    """前処理済みDataFrameを.featherファイルに保存する（Rustバックエンド使用）．

    Args:
        df: 前処理スキーマを持つPolars DataFrame
        file_path: 出力ファイルパス（.feather拡張子推奨）

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル書き込みエラーの場合
    """
    _check_rust_backend()

    # Polars → Arrow Table → RecordBatch
    arrow_table = df.to_arrow()
    arrow_batch = arrow_table.to_batches()[0] if len(arrow_table) > 0 else arrow_table.to_batches(max_chunksize=None)[0]
    save_preprocessing_feather(arrow_batch, str(file_path))


def load_preprocessing_df(file_path: Union[Path, str]) -> pl.DataFrame:
    """前処理済みDataFrameを.featherファイルから読み込む（Rustバックエンド使用）．

    Args:
        file_path: 入力ファイルパス（.feather拡張子）

    Returns:
        前処理スキーマを持つPolars DataFrame

    Raises:
        ImportError: Rustバックエンドが利用不可の場合
        IOError: ファイル読み込みエラーの場合
    """
    _check_rust_backend()

    arrow_batch = load_preprocessing_feather(str(file_path))
    return pl.from_arrow(arrow_batch)
