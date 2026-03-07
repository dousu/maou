"""ゲームツリー構築CLIコマンド."""

from pathlib import Path

import click

from maou.infra.app_logging import app_logger
from maou.infra.console.common import handle_exception
from maou.infra.file_system.file_system import FileSystem


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
    from maou.interface.data_io import load_preprocessing_df
    from maou.interface.game_tree_io import GameTreeIO

    # 入力ファイル収集
    input_files = sorted(
        FileSystem.collect_files(input_path, ext=".feather")
    )
    if not input_files:
        raise click.ClickException(
            f"入力パスに .feather ファイルが見つかりません: {input_path}"
        )

    app_logger.info("入力ファイル数: %d", len(input_files))

    # データ読み込み(Rustバックエンド使用)
    dfs = [load_preprocessing_df(f) for f in input_files]
    preprocess_df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]

    app_logger.info("局面数: %s", f"{len(preprocess_df):,}")

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
        )
    finally:
        pbar.close()

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
