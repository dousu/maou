"""ゲームツリー構築CLIコマンド."""

from pathlib import Path

import click

from maou.infra.app_logging import app_logger
from maou.infra.console.common import handle_exception


def _collect_feather_files(p: Path) -> list[Path]:
    """指定パスから .feather ファイルを収集する．"""
    if p.is_file():
        if p.suffix != ".feather":
            raise ValueError(
                f"ファイルは .feather 形式でなければなりません: {p}"
            )
        return [p]
    elif p.is_dir():
        return sorted(
            f
            for f in p.glob("**/*")
            if f.is_file() and f.suffix == ".feather"
        )
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
    import polars as pl

    from maou.app.game_tree.builder import GameTreeBuilder
    from maou.domain.data.rust_io import load_preprocessing_df
    from maou.interface.game_tree_io import GameTreeIO

    # 入力ファイル収集
    input_files = _collect_feather_files(input_path)
    if not input_files:
        raise click.ClickException(
            f"入力パスに .feather ファイルが見つかりません: {input_path}"
        )

    app_logger.info(f"入力ファイル数: {len(input_files)}")

    # データ読み込み(Rustバックエンド使用)
    dfs = [load_preprocessing_df(f) for f in input_files]
    preprocess_df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]

    app_logger.info(f"局面数: {len(preprocess_df):,}")

    # ツリー構築
    builder = GameTreeBuilder()
    with click.progressbar(
        length=1,
        label="ツリー構築中",
    ) as bar:
        last_processed = 0

        def progress_callback(
            processed: int, total: int
        ) -> None:
            nonlocal last_processed
            # totalは発見済み局面数(BFS中に増加する)
            # NOTE: bar.length の動的更新はClick公式APIではないが，
            # BFSの総ノード数が事前に不明なため必要．Clickバージョン更新時に要確認．
            bar.length = total
            bar.update(processed - last_processed)
            last_processed = processed

        nodes, edges = builder.build(
            preprocess_df,
            max_depth=max_depth,
            min_probability=min_probability,
            progress_callback=progress_callback,
            initial_hash=initial_hash,
            initial_sfen=initial_sfen,
        )

    # 出力
    io = GameTreeIO()
    io.save(nodes, edges, output_dir)

    click.echo(
        f"完了: ノード数={len(nodes):,}, エッジ数={len(edges):,}"
    )
    click.echo(f"出力先: {output_dir}")
