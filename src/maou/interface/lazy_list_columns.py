"""List型カラムへのファイルベース遅延アクセス．

preprocessデータの moveLabel / moveWinRate カラムは
全ファイル一括読み込みするとメモリに収まらないため，
1ファイル分のみキャッシュして省メモリでアクセスする．
"""

from __future__ import annotations

import bisect
import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from maou.app.process_info import get_rss_mb

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class FileBackedListColumns:
    """List型カラム(moveLabel, moveWinRate)へのファイルベース遅延アクセス．

    全ファイルのList型カラムを一度にメモリに載せることを避け，
    1ファイル分のみキャッシュして省メモリでアクセスする．
    """

    def __init__(
        self,
        file_paths: list[Path],
        file_row_counts: list[int],
    ) -> None:
        """初期化する．

        Args:
            file_paths: 各ファイルのパス(読み込み順)
            file_row_counts: 各ファイルの行数
        """
        if len(file_paths) != len(file_row_counts):
            raise ValueError(
                f"file_paths({len(file_paths)}) と "
                f"file_row_counts({len(file_row_counts)}) の"
                "長さが一致しません．"
            )
        self._file_paths = file_paths
        # 累積行数の境界: [0, n0, n0+n1, ...]
        self._boundaries: list[int] = [0]
        for n in file_row_counts:
            self._boundaries.append(self._boundaries[-1] + n)
        self._cached_file_idx: int = -1
        self._cached_labels: pl.Series | None = None
        self._cached_win_rates: pl.Series | None = None
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    @property
    def cache_hits(self) -> int:
        """キャッシュヒット数．"""
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """キャッシュミス数．"""
        return self._cache_misses

    def get(
        self, global_row: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """指定行のmoveLabel, moveWinRateをNumPy配列で返す．

        Args:
            global_row: 全ファイルを連結した場合のグローバル行インデックス

        Returns:
            (moveLabel, moveWinRate) のNumPy float32配列タプル
        """
        file_idx = (
            bisect.bisect_right(self._boundaries, global_row)
            - 1
        )
        local_row = global_row - self._boundaries[file_idx]

        if file_idx != self._cached_file_idx:
            self._load_file(file_idx)
            self._cache_misses += 1
        else:
            self._cache_hits += 1

        assert self._cached_labels is not None
        assert self._cached_win_rates is not None

        labels = np.array(
            self._cached_labels[local_row],
            dtype=np.float32,
        )
        win_rates = np.array(
            self._cached_win_rates[local_row],
            dtype=np.float32,
        )
        return labels, win_rates

    def _load_file(self, file_idx: int) -> None:
        """指定ファイルのList型カラムをロードする(既存キャッシュは解放)."""
        import polars as pl

        # 既存キャッシュを解放
        self._cached_labels = None
        self._cached_win_rates = None
        gc.collect()

        path = self._file_paths[file_idx]
        logger.info(
            "List型カラム読み込み: [%d/%d] %s, RSS=%d MB",
            file_idx + 1,
            len(self._file_paths),
            path.name,
            get_rss_mb(),
        )

        df = pl.read_ipc(
            path,
            columns=["moveLabel", "moveWinRate"],
            memory_map=True,
        )
        self._cached_labels = df["moveLabel"]
        self._cached_win_rates = df["moveWinRate"]
        self._cached_file_idx = file_idx
        # NOTE: Series が元データへの参照を保持するため，
        # del df で実際のメモリ解放は起きないが，
        # DataFrame のメタデータ分は解放される
        del df

        logger.info(
            "List型カラム読み込み完了: %s 行, RSS=%d MB",
            f"{len(self._cached_labels):,}",
            get_rss_mb(),
        )

    def log_stats(self) -> None:
        """キャッシュ統計をログに出力する．"""
        total = self._cache_hits + self._cache_misses
        if total > 0:
            hit_rate = self._cache_hits / total * 100
            logger.info(
                "List型カラムキャッシュ統計: "
                "ヒット=%s, ミス=%s, ヒット率=%.1f%%",
                f"{self._cache_hits:,}",
                f"{self._cache_misses:,}",
                hit_rate,
            )
