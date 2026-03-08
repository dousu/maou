"""List型カラムへのファイルベース遅延アクセス．

preprocessデータの moveLabel / moveWinRate カラムは
全ファイル一括読み込みするとメモリに収まらないため，
LRUキャッシュで限られたファイル数のみ保持して省メモリでアクセスする．
"""

from __future__ import annotations

import bisect
import gc
import logging
from collections import OrderedDict
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
    LRUキャッシュで指定数のファイルのみ保持して省メモリでアクセスする．

    Note:
        ファイル順にアクセスする場合に最もキャッシュ効率が高い．
        ファイルをまたぐランダムアクセスではキャッシュミスが頻発し，
        ファイル I/O と gc.collect() のコストが増大する．
    """

    def __init__(
        self,
        file_paths: list[Path],
        file_row_counts: list[int],
        max_cache_files: int = 1,
    ) -> None:
        """初期化する．

        Args:
            file_paths: 各ファイルのパス(読み込み順)
            file_row_counts: 各ファイルの行数
            max_cache_files: キャッシュするファイル数の上限(LRU方式)．
                1ファイルあたり約11.5GBのメモリを使用する
                (100万行 × 1496要素 × 4bytes × 2列)．
        """
        if len(file_paths) != len(file_row_counts):
            raise ValueError(
                f"file_paths({len(file_paths)}) と "
                f"file_row_counts({len(file_row_counts)}) の"
                "長さが一致しません．"
            )
        if not file_paths:
            raise ValueError(
                "file_paths が空です．"
                "少なくとも1つのファイルが必要です．"
            )
        if max_cache_files < 1:
            raise ValueError(
                f"max_cache_files({max_cache_files}) は"
                "1以上でなければなりません．"
            )
        self._file_paths: list[Path] = []
        # 空ファイル(行数0)を除外して警告
        filtered_row_counts: list[int] = []
        for i, n in enumerate(file_row_counts):
            if n == 0:
                logger.warning(
                    "file_row_counts[%d] が 0 です(空ファイル: %s)．"
                    "このファイルはスキップされます．",
                    i,
                    file_paths[i],
                )
            else:
                self._file_paths.append(file_paths[i])
                filtered_row_counts.append(n)
        # 累積行数の境界: [0, n0, n0+n1, ...]
        self._boundaries: list[int] = [0]
        for n in filtered_row_counts:
            self._boundaries.append(self._boundaries[-1] + n)
        self._total_rows = self._boundaries[-1]
        if not self._file_paths:
            raise ValueError(
                "フィルタ後に有効なファイルが残りませんでした"
                "(全ファイルが空行数です)．"
            )
        self._max_cache_files = max_cache_files
        # LRUキャッシュ: file_idx → (labels, win_rates)
        # OrderedDict で最後にアクセスしたエントリを末尾に移動する
        self._cache: OrderedDict[
            int, tuple[pl.Series, pl.Series]
        ] = OrderedDict()
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

        Raises:
            IndexError: global_row が範囲外の場合
        """
        if global_row < 0 or global_row >= self._total_rows:
            raise IndexError(
                f"global_row={global_row} は範囲外です "
                f"(総行数={self._total_rows})"
            )
        file_idx = (
            bisect.bisect_right(self._boundaries, global_row)
            - 1
        )
        local_row = global_row - self._boundaries[file_idx]

        if file_idx in self._cache:
            self._cache.move_to_end(file_idx)
            self._cache_hits += 1
        else:
            self._load_file(file_idx)
            self._cache_misses += 1

        cached_labels, cached_win_rates = self._cache[file_idx]

        raw_labels = cached_labels[local_row]
        raw_win_rates = cached_win_rates[local_row]
        if raw_labels is None or raw_win_rates is None:
            raise ValueError(
                f"global_row={global_row} (file_idx={file_idx}, "
                f"local_row={local_row}) に null 値が含まれています"
            )
        labels = np.array(raw_labels, dtype=np.float32)
        win_rates = np.array(raw_win_rates, dtype=np.float32)
        return labels, win_rates

    def _load_file(self, file_idx: int) -> None:
        """指定ファイルのList型カラムをロードする(LRUで古いエントリを解放)．"""
        import polars as pl

        # キャッシュが上限に達している場合，最も古いエントリを破棄
        while len(self._cache) >= self._max_cache_files:
            evicted_idx, _ = self._cache.popitem(last=False)
            logger.debug(
                "LRUキャッシュ破棄: file_idx=%d", evicted_idx
            )
        gc.collect()

        path = self._file_paths[file_idx]
        logger.debug(
            "List型カラム読み込み: [%d/%d] %s, RSS=%d MB",
            file_idx + 1,
            len(self._file_paths),
            path.name,
            get_rss_mb(),
        )

        # memory_map=False: Python ヒープにロードすることで，
        # キャッシュ切り替え時の gc.collect() で確実にメモリを解放する．
        # mmap だと OS のページキャッシュに残り RSS が下がらない場合がある
        df = pl.read_ipc(
            path,
            columns=["moveLabel", "moveWinRate"],
            memory_map=False,
        )
        loaded_labels = df["moveLabel"]
        loaded_win_rates = df["moveWinRate"]
        del df

        expected_rows = (
            self._boundaries[file_idx + 1]
            - self._boundaries[file_idx]
        )
        actual_rows = len(loaded_labels)
        if actual_rows != expected_rows:
            raise ValueError(
                f"ファイル {path.name} の List カラム行数"
                f"({actual_rows}) が"
                f"スカラー読み込み時の行数"
                f"({expected_rows}) と一致しません"
            )

        self._cache[file_idx] = (
            loaded_labels,
            loaded_win_rates,
        )

        logger.debug(
            "List型カラム読み込み完了: %s 行, RSS=%d MB",
            f"{actual_rows:,}",
            get_rss_mb(),
        )

    def log_stats(self) -> None:
        """キャッシュ統計をログに出力する．"""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            logger.info("List型カラムキャッシュ: アクセスなし")
            return
        hit_rate = self._cache_hits / total * 100
        logger.info(
            "List型カラムキャッシュ統計: "
            "ヒット=%s, ミス=%s, ヒット率=%.1f%%, "
            "max_cache_files=%d",
            f"{self._cache_hits:,}",
            f"{self._cache_misses:,}",
            hit_rate,
            self._max_cache_files,
        )
