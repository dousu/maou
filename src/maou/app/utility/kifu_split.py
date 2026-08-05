"""棋譜ファイルを対局単位で学習用と検証用に分割する．

前処理 (``maou pre-process``) は局面を Zobrist hash で全コーパス横断に
集約するため，**集約後に対局の同一性は復元できない**．したがって
「同一対局の局面が train と val の両方に入る」ことを避けるには，
棋譜の段階で分割してからそれぞれ前処理する必要がある．

分割の単位はファイルである．floodgate の棋譜は 1 ファイル 1 対局なので
これが対局単位の分割になる．複数対局を含むファイルは分割されず，
まとめて片側に入る．
"""

import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger: logging.Logger = logging.getLogger(__name__)

TransferMode = Literal["copy", "symlink", "hardlink"]


@dataclass(frozen=True)
class KifuSplitPlan:
    """分割の計画．実際のファイル操作を行う前の決定内容．

    Attributes:
        train: 学習側に割り当てられたファイル．
        val: 検証側に割り当てられたファイル．
    """

    train: list[Path]
    val: list[Path]


def plan_split(
    files: list[Path],
    *,
    val_ratio: float,
    seed: int,
) -> KifuSplitPlan:
    """ファイル一覧を決定的にシャッフルして分割する．

    入力順に依存しないよう，シャッフル前にパスでソートする．
    同じ ``files`` の集合と同じ ``seed`` からは常に同じ分割が得られる．

    Args:
        files: 分割対象のファイル．
        val_ratio: 検証側に回す割合 (0 < ratio < 1)．
        seed: シャッフルの乱数シード．

    Returns:
        分割計画．

    Raises:
        ValueError: ``val_ratio`` が範囲外，または分割の結果どちらかが
            空になる場合．
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            f"val_ratio must satisfy 0 < ratio < 1, got {val_ratio}"
        )
    if not files:
        raise ValueError("no kifu files to split")

    ordered = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n_val = round(len(ordered) * val_ratio)
    # 端数で 0 件になるのを防ぐ．2 件未満は分割不能．
    n_val = max(1, min(n_val, len(ordered) - 1))
    if len(ordered) < 2:
        raise ValueError(
            f"need at least 2 files to split, got {len(ordered)}"
        )

    return KifuSplitPlan(
        train=ordered[n_val:], val=ordered[:n_val]
    )


def transfer(
    src: Path,
    dst: Path,
    *,
    mode: TransferMode,
) -> None:
    """``src`` を ``dst`` へ配置する．

    Args:
        src: 元ファイル．
        dst: 配置先．親ディレクトリは自動作成する．
        mode: ``"copy"`` は実体コピー，``"symlink"`` はシンボリックリンク，
            ``"hardlink"`` はハードリンク (同一ファイルシステムのみ)．

    Raises:
        ValueError: 未知の ``mode``．
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    else:  # pragma: no cover - Choice で弾かれる
        raise ValueError(f"unknown transfer mode: {mode}")


def apply_split(
    plan: KifuSplitPlan,
    *,
    input_root: Path,
    train_dir: Path,
    val_dir: Path,
    mode: TransferMode,
    dry_run: bool = False,
) -> dict[str, int]:
    """分割計画に従ってファイルを配置する．

    入力ディレクトリからの相対パスを出力先でも保つ
    (floodgate の ``YYYY/MM/DD`` 構造がそのまま残る)．

    Args:
        plan: ``plan_split`` の結果．
        input_root: 相対パスの基準となる入力ディレクトリ．
        train_dir: 学習側の出力先．
        val_dir: 検証側の出力先．
        mode: 配置方法．
        dry_run: True の場合，件数だけ数えてファイルを作らない．

    Returns:
        ``{"train_files": int, "val_files": int}``．
    """
    for files, out_dir in (
        (plan.train, train_dir),
        (plan.val, val_dir),
    ):
        if dry_run:
            continue
        for src in files:
            try:
                rel = src.relative_to(input_root)
            except ValueError:
                rel = Path(src.name)
            transfer(src, out_dir / rel, mode=mode)

    return {
        "train_files": len(plan.train),
        "val_files": len(plan.val),
    }
