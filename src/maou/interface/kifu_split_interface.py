"""棋譜の train/val 分割ユースケースのインタフェース層．"""

import json
import logging
from pathlib import Path

from maou.app.utility.kifu_split import (
    TransferMode,
    apply_split,
    plan_split,
)

logger: logging.Logger = logging.getLogger(__name__)


def split_kifu(
    *,
    collect_files: "object",
    input_path: Path,
    train_dir: Path,
    val_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    ext: str | None = None,
    mode: TransferMode = "copy",
    dry_run: bool = False,
) -> str:
    """棋譜ファイルを対局単位で学習用と検証用に分割する．

    Args:
        collect_files: ``(Path, ext=str | None) -> list[Path]`` の呼び出し可能．
            ``FileSystem.collect_files`` を渡す．
        input_path: 棋譜のディレクトリまたはファイル．
        train_dir: 学習側の出力ディレクトリ．
        val_dir: 検証側の出力ディレクトリ．
        val_ratio: 検証側に回す割合 (0 < ratio < 1)．
        seed: シャッフルの乱数シード．同じ入力と同じ seed なら同じ分割．
        ext: 収集する拡張子 (例 ``".csa"``)．None で全ファイル．
        mode: ``copy`` / ``symlink`` / ``hardlink``．
        dry_run: True で件数だけ数えてファイルを作らない．

    Returns:
        結果の JSON 文字列．

    Raises:
        ValueError: 入力にファイルが無い場合，または出力先が入力と同じ場合．
    """
    if train_dir.resolve() == val_dir.resolve():
        raise ValueError(
            "--train-dir and --val-dir must differ"
        )

    files = collect_files(input_path, ext=ext)  # type: ignore[operator]
    if not files:
        raise ValueError(
            f"no kifu files found under {input_path}"
            + (f" with ext={ext}" if ext else "")
        )

    plan = plan_split(files, val_ratio=val_ratio, seed=seed)
    logger.info(
        f"splitting {len(files)} kifu file(s) by game: "
        f"train={len(plan.train)} val={len(plan.val)} "
        f"(seed={seed}, mode={mode}"
        + (", dry-run" if dry_run else "")
        + ")"
    )

    input_root = (
        input_path if input_path.is_dir() else input_path.parent
    )
    counts = apply_split(
        plan,
        input_root=input_root,
        train_dir=train_dir,
        val_dir=val_dir,
        mode=mode,
        dry_run=dry_run,
    )

    return json.dumps(
        {
            "input_path": str(input_path),
            "train_dir": str(train_dir),
            "val_dir": str(val_dir),
            "total_files": len(files),
            "seed": seed,
            "mode": mode,
            "dry_run": dry_run,
            **counts,
        }
    )
