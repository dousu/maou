"""preprocess向けストリーミングHCPEデータソースモジュール．

全ファイルを一度にメモリにロードする ``FileDataSource(FileManager)`` とは異なり，
1ファイルずつ読み込み → numpy変換 → yield → GC解放のストリーミングパターンを使用する．

ピークメモリは「1ファイル分のPolars DataFrame + 対応するnumpy array」に抑えられる．
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np

from maou.app.pre_process.hcpe_transform import DataSource
from maou.domain.data.rust_io import load_hcpe_df
from maou.domain.data.schema import convert_hcpe_df_to_numpy

logger = logging.getLogger(__name__)


class StreamingHcpeDataSource(DataSource):
    """preprocess向けストリーミングHCPEデータソース．

    ``FileDataSource(FileManager)`` が全ファイルを初期化時にメモリにロードするのに対し，
    このクラスは ``iter_batches()`` 呼び出し時に1ファイルずつ遅延ロードする．

    - 初期化コスト: ファイルメタデータスキャンのみ(行数取得)
    - ピークメモリ: 1ファイル分のデータ(DataFrame + numpy array)
    - 用途: preprocessコマンド専用(ランダムアクセス不要)

    Attributes:
        file_paths: featherファイルパスのリスト．
    """

    def __init__(self, file_paths: list[Path]) -> None:
        """ストリーミングHCPEデータソースを初期化する．

        初期化時にはファイルパスの設定のみを行い，
        行数スキャンは ``__len__()`` の初回アクセス時まで遅延する．

        Args:
            file_paths: HCPEのfeatherファイルパスのリスト
        """
        self._file_paths = list(file_paths)
        self._total_rows: int | None = None

        logger.info(
            "StreamingHcpeDataSource initialized (lazy): %d files",
            len(self._file_paths),
        )

    @property
    def file_paths(self) -> list[Path]:
        """ファイルパスのリスト."""
        return list(self._file_paths)

    def _ensure_row_counts(self) -> None:
        """行数スキャンを実行する(未実行の場合のみ)．

        初回呼び出し時に全ファイルの行数をスキャンし，
        ``_total_rows`` を設定する．
        2回目以降の呼び出しでは何もしない．
        """
        if self._total_rows is not None:
            return

        from maou.infra.file_system.streaming_file_source import (
            scan_row_count,
        )

        t0 = time.perf_counter()
        total = 0
        n = len(self._file_paths)
        for fp in self._file_paths:
            total += scan_row_count(fp)

        self._total_rows = total
        elapsed = time.perf_counter() - t0
        logger.info(
            "StreamingHcpeDataSource scanned: "
            "%d files, %d total rows in %.1fs",
            n,
            self._total_rows,
            elapsed,
        )

    def __len__(self) -> int:
        """全ファイルの合計行数(初回アクセス時にスキャン実行)."""
        self._ensure_row_counts()
        assert self._total_rows is not None  # noqa: S101
        return self._total_rows

    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """ファイル単位でnumpy structured arrayをyieldする．

        各ファイルについて:

        1. featherファイルをPolars DataFrameとして読み込み
        2. numpy structured arrayに変換
        3. DataFrameを即座に解放(GC対象にする)
        4. numpy arrayをyield

        Yields:
            tuple[str, np.ndarray]: (ファイル名, numpy structured array)
        """
        n = len(self._file_paths)
        for i, fp in enumerate(self._file_paths):
            t0 = time.perf_counter()
            df = load_hcpe_df(fp)
            t_load = time.perf_counter() - t0

            arr = convert_hcpe_df_to_numpy(df)
            t_total = time.perf_counter() - t0
            del df  # DataFrame参照を即座に切る(GC対象にする)

            logger.debug(
                "Streaming file %d/%d: %s "
                "(%d rows, load=%.2fs, total=%.2fs)",
                i + 1,
                n,
                fp.name,
                len(arr),
                t_load,
                t_total,
            )
            yield fp.name, arr

    def total_pages(self) -> int:
        """バッチ(ファイル)数を返す."""
        return len(self._file_paths)
