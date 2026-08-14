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
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl

from maou.interface.data_io import load_hcpe_df
from maou.interface.data_schema import convert_hcpe_df_to_numpy
from maou.interface.preprocess import DataSource

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
        self._row_counts: list[int] | None = None

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
        ``_row_counts`` と ``_total_rows`` を設定する．
        2回目以降の呼び出しでは何もしない．

        スキャンは ``StreamingFileSource`` と同じ
        :func:`~maou.domain.data.arrow_format.scan_row_counts` を使う．
        以前は両者が別々にループを書いており，「例外で打ち切られた
        カウントを memo しない」性質を各々が独立に守っていた
        (安全なのは構造が違う結果の偶然だった)．
        """
        if self._row_counts is not None:
            return

        from maou.domain.data.arrow_format import (
            scan_row_counts,
        )

        t0 = time.perf_counter()
        # 部分結果を返さないことは scan_row_counts が保証する．
        row_counts = scan_row_counts(self._file_paths)

        self._row_counts = row_counts
        self._total_rows = sum(row_counts)
        elapsed = time.perf_counter() - t0
        logger.info(
            "StreamingHcpeDataSource scanned: "
            "%d files, %d total rows in %.1fs",
            len(self._file_paths),
            self._total_rows,
            elapsed,
        )

    @property
    def row_counts(self) -> list[int]:
        """各ファイルの行数リスト(初回アクセス時にスキャン実行)．

        以前は per-file のカウントを合計した時点で捨てていたため，
        sharding に必要な行数リストを提供できなかった．
        """
        self._ensure_row_counts()
        assert self._row_counts is not None
        return list(self._row_counts)

    def __len__(self) -> int:
        """全ファイルの合計行数(初回アクセス時にスキャン実行)."""
        self._ensure_row_counts()
        assert self._total_rows is not None
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

    def iter_batches_df(
        self,
    ) -> Generator[tuple[str, pl.DataFrame], None, None]:
        """ファイル単位で Polars DataFrame を yield する．

        この実装は **HCPE スキーマ決め打ち**である．2026-08-14 まで
        ``preprocess.DataSource`` の**既定実装**として基底に置かれて
        いたもので，``preprocessing`` 型のソースが override を忘れると
        黙って誤ったスキーマを当てていた (backlog 行 N6-2)．基底の
        既定実装を廃して abstract にし，本体を「HCPE 専用であることが
        クラス名から明らかな」唯一の継承者であるここへ移した．

        Yields:
            tuple[str, pl.DataFrame]: (ファイル名, DataFrame)
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is required for DataFrame iteration. "
                "Install with: uv add polars"
            )

        from maou.domain.data.schema import (
            get_hcpe_polars_schema,
        )

        schema = get_hcpe_polars_schema()

        for name, array in self.iter_batches():
            # structured array を列ごとの list へ展開する
            data = {}
            assert array.dtype.names is not None
            assert array.dtype.fields is not None
            for field in array.dtype.names:
                field_data = array[field]
                field_dtype = array.dtype.fields[field][0]

                # バイナリ列 (uint8 の多次元) は bytes へ畳む
                if field == "hcp" or (
                    field_dtype.shape
                    and field_dtype.base == np.dtype("uint8")
                ):
                    data[field] = [
                        bytes(row)
                        if hasattr(row, "__iter__")
                        else bytes([row])
                        for row in field_data
                    ]
                else:
                    data[field] = field_data.tolist()

            yield name, pl.DataFrame(data, schema=schema)

    def total_pages(self) -> int:
        """バッチ(ファイル)数を返す."""
        return len(self._file_paths)
