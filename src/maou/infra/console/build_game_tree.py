"""ゲームツリー構築CLIコマンド."""

from __future__ import annotations

import bisect
import gc
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np

from maou.app.process_info import get_rss_mb
from maou.infra.app_logging import app_logger
from maou.infra.console.common import handle_exception
from maou.infra.file_system.file_system import FileSystem

if TYPE_CHECKING:
    import polars as pl


class _FileBackedListColumns:
    """List型カラム(moveLabel, moveWinRate)へのファイルベース遅延アクセス．

    全ファイルのList型カラムを一度にメモリに載せることを避け，
    1ファイル分のみキャッシュして省メモリでアクセスする．
    """

    def __init__(
        self,
        file_paths: list[Path],
        file_row_counts: list[int],
    ) -> None:
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
        app_logger.info(
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

        app_logger.info(
            "List型カラム読み込み完了: %s 行, RSS=%d MB",
            f"{len(self._cached_labels):,}",
            get_rss_mb(),
        )

    def log_stats(self) -> None:
        """キャッシュ統計をログに出力する．"""
        total = self._cache_hits + self._cache_misses
        if total > 0:
            hit_rate = self._cache_hits / total * 100
            app_logger.info(
                "List型カラムキャッシュ統計: "
                "ヒット=%s, ミス=%s, ヒット率=%.1f%%",
                f"{self._cache_hits:,}",
                f"{self._cache_misses:,}",
                hit_rate,
            )


@click.command("build-game-tree")
@click.option(
    "--input-path",
    help="preprocessデータのディレクトリまたはファイルパス．",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    help="ツリーデータの出力先ディレクトリ．",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--max-depth",
    help="最大探索深さ．",
    type=int,
    default=30,
    show_default=True,
)
@click.option(
    "--min-probability",
    help="指し手の最小確率閾値(表示時より小さく設定)．",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.001,
    show_default=True,
)
@click.option(
    "--initial-hash",
    help="開始局面のZobrist hash(preprocessのID)．省略時は平手初期局面．",
    type=int,
    default=None,
)
@click.option(
    "--initial-sfen",
    help="開始局面のSFEN文字列．--initial-hash 指定時に必須．",
    type=str,
    default=None,
)
@handle_exception
def build_game_tree(
    input_path: Path,
    output_dir: Path,
    max_depth: int,
    min_probability: float,
    initial_hash: int | None,
    initial_sfen: str | None,
) -> None:
    """preprocessデータからゲームツリーを構築する．"""
    if initial_hash is not None and initial_sfen is None:
        raise click.ClickException(
            "--initial-hash を指定する場合は --initial-sfen も指定してください．"
        )
    import polars as pl

    from maou.app.game_tree.builder import GameTreeBuilder
    from maou.interface.game_tree_io import GameTreeIO

    # 入力ファイル収集
    input_files = sorted(
        FileSystem.collect_files(input_path, ext=".feather")
    )
    if not input_files:
        raise click.ClickException(
            f"入力パスに .feather ファイルが見つかりません: {input_path}"
        )

    app_logger.info(
        "入力ファイル数: %d, RSS=%d MB",
        len(input_files),
        get_rss_mb(),
    )

    # スカラーカラムのみ読み込み(List型カラムは遅延アクセスで省メモリ化)
    # 全ファイルのList型カラムを一括読み込みすると
    # 41M行 × 1496要素 × 4bytes × 2列 ≈ 477GB 必要なため
    scalar_dfs: list[pl.DataFrame] = []
    file_row_counts: list[int] = []
    scalar_columns = ["id", "resultValue", "bestMoveWinRate"]
    for i, f in enumerate(input_files):
        app_logger.info(
            "スカラーカラム読み込み: [%d/%d] %s, RSS=%d MB",
            i + 1,
            len(input_files),
            f.name,
            get_rss_mb(),
        )
        df = pl.read_ipc(f, columns=scalar_columns)
        file_row_counts.append(len(df))
        scalar_dfs.append(df)
    preprocess_df: pl.DataFrame = (
        pl.concat(scalar_dfs)
        if len(scalar_dfs) > 1
        else scalar_dfs[0]
    )
    del scalar_dfs

    app_logger.info(
        "スカラーカラム読み込み完了: 局面数=%s, RSS=%d MB",
        f"{len(preprocess_df):,}",
        get_rss_mb(),
    )

    # List型カラムの遅延アクセス用オブジェクト
    list_columns = _FileBackedListColumns(
        input_files, file_row_counts
    )

    # ツリー構築
    from tqdm import tqdm

    builder = GameTreeBuilder()
    pbar = tqdm(desc="ツリー構築中", unit="局面")

    def progress_callback(processed: int, total: int) -> None:
        # totalは発見済み局面数(BFS中に増加する)
        pbar.total = total
        pbar.n = processed
        pbar.refresh()

    try:
        nodes, edges = builder.build(
            preprocess_df,
            max_depth=max_depth,
            min_probability=min_probability,
            progress_callback=progress_callback,
            initial_hash=initial_hash,
            initial_sfen=initial_sfen,
            list_column_fn=list_columns.get,
        )
    finally:
        pbar.close()
        list_columns.log_stats()

    # 出力
    io = GameTreeIO()
    io.save(nodes, edges, output_dir)
    io.save_metadata(
        output_dir,
        {
            "initial_sfen": initial_sfen,
        },
    )

    click.echo(
        f"完了: ノード数={len(nodes):,}, エッジ数={len(edges):,}"
    )
    click.echo(f"出力先: {output_dir}")
