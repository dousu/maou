"""ファイル単位ストリーミングデータソースモジュール．

全ファイルを一度にメモリにロードする ``FileManager`` とは異なり，
1ファイルずつ読み込み → ``ColumnarBatch`` に変換 → yield → GC解放の
ストリーミングパターンを使用する．

ピークメモリは「1ファイル分のPolars DataFrame + 対応するColumnarBatch」に抑えられる．
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal

import polars as pl

# Arrow IPC の File/Stream 判定は domain に置いてある．
# ここは後方互換のための再輸出で，`streaming_file_source.scan_row_count`
# を差し替えるテストと，モジュール名で参照する既存の import を保つ．
from maou.domain.data.arrow_format import (  # noqa: F401
    is_arrow_ipc_file_format,
    scan_row_count,
)
from maou.interface.data_io import (
    ColumnarBatch,
    convert_preprocessing_df_to_columnar,
    convert_stage1_df_to_columnar,
    convert_stage2_df_to_columnar,
    load_hcpe_df,
    load_preprocessing_df,
    load_stage1_df,
    load_stage2_df,
)

logger = logging.getLogger(__name__)

# Feather loader dispatch table
_FEATHER_LOADERS: dict[str, Callable[[Path], pl.DataFrame]] = {
    "hcpe": load_hcpe_df,
    "preprocessing": load_preprocessing_df,
    "stage1": load_stage1_df,
    "stage2": load_stage2_df,
}

# ColumnarBatch converter dispatch table
_COLUMNAR_CONVERTERS: dict[
    str, Callable[[pl.DataFrame], ColumnarBatch]
] = {
    "preprocessing": convert_preprocessing_df_to_columnar,
    "stage1": convert_stage1_df_to_columnar,
    "stage2": convert_stage2_df_to_columnar,
}


class StreamingFileSource:
    """ファイル単位のストリーミングデータソース．

    全ファイルを一度にメモリにロードする ``FileManager`` とは異なり，
    1ファイルずつ読み込み → 消費 → 解放のストリーミングパターンを使用する．

    Attributes:
        file_paths: featherファイルパスのリスト．
        total_rows: 全ファイルの合計行数(初期化時にスキャン)．
    """

    def __init__(
        self,
        file_paths: list[Path],
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ],
    ) -> None:
        """ストリーミングデータソースを初期化する．

        初期化時にはファイルパスとローダーの設定のみを行い，
        行数スキャンは ``total_rows`` の初回アクセス時まで遅延する．

        Args:
            file_paths: featherファイルパスのリスト
            array_type: データタイプ("hcpe", "preprocessing", "stage1", "stage2")

        Raises:
            ValueError: サポートされていないarray_typeの場合
        """
        if array_type not in _FEATHER_LOADERS:
            raise ValueError(
                f"Unsupported array_type: {array_type}. "
                f"Supported: {list(_FEATHER_LOADERS.keys())}"
            )
        if array_type not in _COLUMNAR_CONVERTERS:
            raise ValueError(
                "hcpe array_type is not supported for streaming columnar conversion. "
                "Use 'preprocessing', 'stage1', or 'stage2'."
            )

        self._file_paths = list(file_paths)
        self._array_type = array_type
        self._loader = _FEATHER_LOADERS[array_type]
        self._converter = _COLUMNAR_CONVERTERS[array_type]

        # 行数スキャンは遅延実行(total_rows初回アクセス時)
        self._total_rows: int | None = None
        self._row_counts: list[int] | None = None

        logger.info(
            "StreamingFileSource initialized (lazy): "
            "%d files, array_type=%s",
            len(self._file_paths),
            array_type,
        )

    @property
    def file_paths(self) -> list[Path]:
        """ファイルパスのリスト(シャッフル・worker分割用に公開)."""
        return list(self._file_paths)

    def _ensure_row_counts(self) -> None:
        """行数スキャンを実行する(未実行の場合のみ)．

        初回呼び出し時に全ファイルの行数をスキャンし，
        ``_total_rows`` と ``_row_counts`` を設定する．
        2回目以降の呼び出しでは何もしない．
        """
        if self._row_counts is not None:
            return

        # スキャン完了後にまとめて代入する．ループ中に代入すると，
        # 途中で scan_row_count が例外を投げたときに打ち切られた
        # カウントが memo (`_row_counts is not None`) として残り，
        # 以降 total_rows が過少申告され steps_per_epoch が狂う．
        row_counts: list[int] = []
        total_rows = 0
        for fp in self._file_paths:
            row_count = scan_row_count(fp)
            row_counts.append(row_count)
            total_rows += row_count

        self._row_counts = row_counts
        self._total_rows = total_rows

        logger.info(
            "StreamingFileSource scanned: "
            "%d files, %d total rows",
            len(self._file_paths),
            self._total_rows,
        )

    @property
    def total_rows(self) -> int:
        """全ファイルの合計行数(初回アクセス時にスキャン実行)."""
        self._ensure_row_counts()
        assert self._total_rows is not None
        return self._total_rows

    @property
    def row_counts(self) -> list[int]:
        """各ファイルの行数リスト(初回アクセス時にスキャン実行)."""
        self._ensure_row_counts()
        assert self._row_counts is not None
        return list(self._row_counts)

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        """ファイル単位で ``ColumnarBatch`` をyieldする．

        各ファイルについて:

        1. featherファイルをPolars DataFrameとして読み込み
        2. Polars DataFrame → ``ColumnarBatch`` に変換
        3. ``ColumnarBatch`` をyield
        4. (次のイテレーションでPolars DataFrameの参照が外れ，GC解放対象になる)

        Yields:
            1ファイル分のデータを含む ``ColumnarBatch``
        """
        yield from self.iter_files_columnar_subset(
            self._file_paths
        )

    def iter_files_columnar_subset(
        self,
        file_paths: list[Path],
    ) -> Generator[ColumnarBatch, None, None]:
        """指定されたファイルパスのみを読み込み ``ColumnarBatch`` をyieldする．

        workerファイル分割時に，各workerが担当ファイルのみを読み込むために使用する．
        DEBUG レベルでファイル読込・変換のタイミングを出力し，ボトルネック特定に使用する．

        Args:
            file_paths: 読み込むファイルパスのリスト

        Yields:
            1ファイル分のデータを含む ``ColumnarBatch``
        """
        n = len(file_paths)
        for i, fp in enumerate(file_paths):
            logger.debug(
                "Loading file %d/%d: %s",
                i + 1,
                n,
                fp.name,
            )
            t0 = time.perf_counter()
            df = self._loader(fp)
            t_load = time.perf_counter() - t0
            logger.debug(
                "File loaded: %d rows in %.2fs, converting...",
                len(df),
                t_load,
            )
            t1 = time.perf_counter()
            batch = self._converter(df)
            t_conv = time.perf_counter() - t1
            logger.debug(
                "Conversion complete: %.2fs (file %d/%d)",
                t_conv,
                i + 1,
                n,
            )
            del df  # DF参照を即座に切る(GC対象にする)
            yield batch
