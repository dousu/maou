"""棋譜を対局単位で train / val に分割する CLI コマンド (インフラ層)．

``maou utility split-kifu``．重い依存 (torch 等) を持たないため
学習環境が無くても実行できる．
"""

from pathlib import Path

import click

from maou.infra.console.common import (
    FileSystem,
    handle_exception,
)


@click.command("split-kifu")
@click.option(
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Kifu directory (or file) to split. Walks recursively.",
)
@click.option(
    "--train-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for the training split.",
)
@click.option(
    "--val-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for the validation split.",
)
@click.option(
    "--val-ratio",
    type=click.FloatRange(
        min=0.0, max=1.0, min_open=True, max_open=True
    ),
    default=0.1,
    show_default=True,
    help="Fraction of files assigned to the validation split.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Shuffle seed. The same input and seed always give the same split.",
)
@click.option(
    "--ext",
    type=str,
    default=None,
    help="Only collect files with this extension (e.g. '.csa'). "
    "Omit to take every file.",
)
@click.option(
    "--mode",
    type=click.Choice(["copy", "symlink", "hardlink"]),
    default="copy",
    show_default=True,
    help="How to place files: 'copy' duplicates them, 'symlink' and "
    "'hardlink' avoid extra disk usage ('hardlink' needs the same filesystem).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report the counts without creating any file.",
)
@handle_exception
def split_kifu(
    input_path: Path,
    train_dir: Path,
    val_dir: Path,
    val_ratio: float,
    seed: int,
    ext: str | None,
    mode: str,
    dry_run: bool,
) -> None:
    """Split kifu files into a training set and a validation set.

    The split is per file. floodgate stores one game per file, so this is a
    game-level split: no game ends up on both sides.

    This matters because `maou pre-process` aggregates positions by Zobrist
    hash across the whole corpus, so game identity cannot be recovered after
    aggregation. Split here, run `pre-process` on each side, then pass the
    validation output to `learn-model --stage3-validation-data-path`.
    """
    from maou.interface import kifu_split_interface

    result = kifu_split_interface.split_kifu(
        collect_files=FileSystem.collect_files,
        input_path=input_path,
        train_dir=train_dir,
        val_dir=val_dir,
        val_ratio=val_ratio,
        seed=seed,
        ext=ext,
        mode=mode,  # type: ignore[arg-type]
        dry_run=dry_run,
    )
    click.echo(result)
