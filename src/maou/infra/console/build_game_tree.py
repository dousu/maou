"""ゲームツリー構築CLIコマンド."""

from pathlib import Path
from typing import Any

import click
import polars as pl

from maou.infra.app_logging import app_logger


def _handle_exception(func: Any) -> Any:
    """例外ハンドリングデコレータ(common.pyの依存を回避)."""
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            app_logger.exception(
                "Error occurred", stack_info=True
            )

    return wrapper


def _collect_feather_files(p: Path) -> list[Path]:
    """指定パスから .feather ファイルを収集する．"""
    if p.is_file():
        return [p]
    elif p.is_dir():
        return [
            f
            for f in p.glob("**/*")
            if f.is_file() and ".feather" in f.suffixes
        ]
    else:
        raise ValueError(
            f"Path `{p}` is neither a file nor a directory."
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
    type=float,
    default=0.001,
    show_default=True,
)
@_handle_exception
def build_game_tree(
    input_path: Path,
    output_dir: Path,
    max_depth: int,
    min_probability: float,
) -> None:
    """preprocessデータからゲームツリーを構築する．"""
    from maou.app.game_tree.builder import GameTreeBuilder
    from maou.interface.game_tree_io import GameTreeIO

    # 入力ファイル収集
    input_files = _collect_feather_files(input_path)
    if not input_files:
        raise click.ClickException(
            f"入力パスに .feather ファイルが見つかりません: {input_path}"
        )

    app_logger.info(f"入力ファイル数: {len(input_files)}")

    # データ読み込み
    dfs = [pl.read_ipc(f) for f in input_files]
    preprocess_df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]

    app_logger.info(f"局面数: {len(preprocess_df):,}")

    # ツリー構築
    builder = GameTreeBuilder()
    with click.progressbar(
        length=len(preprocess_df),
        label="ツリー構築中",
    ) as bar:
        last_processed = 0

        def progress_callback(
            processed: int, total: int
        ) -> None:
            nonlocal last_processed
            bar.update(processed - last_processed)
            last_processed = processed

        nodes, edges = builder.build(
            preprocess_df,
            max_depth=max_depth,
            min_probability=min_probability,
            progress_callback=progress_callback,
        )

    # 出力
    io = GameTreeIO()
    io.save(nodes, edges, output_dir)

    click.echo(
        f"完了: ノード数={len(nodes):,}, エッジ数={len(edges):,}"
    )
    click.echo(f"出力先: {output_dir}")
