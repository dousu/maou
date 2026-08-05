"""Tests for game-level kifu splitting."""

from __future__ import annotations

from pathlib import Path

import pytest

from maou.app.utility.kifu_split import (
    apply_split,
    plan_split,
    transfer,
)


def _paths(n: int) -> list[Path]:
    return [
        Path(f"kifu/2026/03/0{i % 3 + 1}/g{i:04d}.csa")
        for i in range(n)
    ]


class TestPlanSplit:
    def test_partition_is_complete_and_disjoint(self) -> None:
        files = _paths(100)
        plan = plan_split(files, val_ratio=0.1, seed=42)
        assert len(plan.train) + len(plan.val) == 100
        assert set(plan.train).isdisjoint(plan.val)
        assert set(plan.train) | set(plan.val) == set(files)

    def test_ratio_is_respected(self) -> None:
        plan = plan_split(_paths(100), val_ratio=0.25, seed=1)
        assert len(plan.val) == 25

    def test_same_seed_gives_same_split(self) -> None:
        a = plan_split(_paths(50), val_ratio=0.2, seed=7)
        b = plan_split(_paths(50), val_ratio=0.2, seed=7)
        assert a.val == b.val

    def test_different_seed_gives_different_split(self) -> None:
        a = plan_split(_paths(200), val_ratio=0.2, seed=1)
        b = plan_split(_paths(200), val_ratio=0.2, seed=2)
        assert a.val != b.val

    def test_input_order_does_not_matter(self) -> None:
        files = _paths(60)
        a = plan_split(files, val_ratio=0.1, seed=3)
        b = plan_split(
            list(reversed(files)), val_ratio=0.1, seed=3
        )
        assert sorted(a.val) == sorted(b.val)

    def test_small_ratio_still_yields_one_val_file(
        self,
    ) -> None:
        # 端数切り捨てで 0 件になると検証集合が空になってしまう
        plan = plan_split(_paths(5), val_ratio=0.01, seed=0)
        assert len(plan.val) == 1
        assert len(plan.train) == 4

    def test_large_ratio_leaves_one_train_file(self) -> None:
        plan = plan_split(_paths(5), val_ratio=0.99, seed=0)
        assert len(plan.train) == 1

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_ratio_rejected(self, ratio: float) -> None:
        with pytest.raises(ValueError, match="val_ratio"):
            plan_split(_paths(10), val_ratio=ratio, seed=0)

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="no kifu files"):
            plan_split([], val_ratio=0.1, seed=0)

    def test_single_file_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="at least 2 files"
        ):
            plan_split(_paths(1), val_ratio=0.5, seed=0)


class TestApplySplit:
    def _make_tree(self, root: Path, n: int) -> list[Path]:
        files = []
        for i in range(n):
            p = (
                root
                / "2026"
                / "03"
                / f"0{i % 2 + 1}"
                / f"g{i}.csa"
            )
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"game {i}")
            files.append(p)
        return files

    def test_preserves_relative_layout(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "in"
        files = self._make_tree(src, 10)
        plan = plan_split(files, val_ratio=0.2, seed=5)
        counts = apply_split(
            plan,
            input_root=src,
            train_dir=tmp_path / "train",
            val_dir=tmp_path / "val",
            mode="copy",
        )
        assert counts == {"train_files": 8, "val_files": 2}
        for p in plan.val:
            assert (
                tmp_path / "val" / p.relative_to(src)
            ).read_text() == p.read_text()

    def test_no_game_appears_on_both_sides(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "in"
        files = self._make_tree(src, 20)
        plan = plan_split(files, val_ratio=0.25, seed=11)
        apply_split(
            plan,
            input_root=src,
            train_dir=tmp_path / "train",
            val_dir=tmp_path / "val",
            mode="copy",
        )
        train = {
            p.name for p in (tmp_path / "train").rglob("*.csa")
        }
        val = {
            p.name for p in (tmp_path / "val").rglob("*.csa")
        }
        assert train.isdisjoint(val)
        assert len(train) + len(val) == 20

    def test_dry_run_creates_nothing(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "in"
        files = self._make_tree(src, 10)
        plan = plan_split(files, val_ratio=0.2, seed=5)
        counts = apply_split(
            plan,
            input_root=src,
            train_dir=tmp_path / "train",
            val_dir=tmp_path / "val",
            mode="copy",
            dry_run=True,
        )
        assert counts == {"train_files": 8, "val_files": 2}
        assert not (tmp_path / "train").exists()
        assert not (tmp_path / "val").exists()

    @pytest.mark.parametrize(
        "mode", ["copy", "symlink", "hardlink"]
    )
    def test_transfer_modes_produce_readable_files(
        self, tmp_path: Path, mode: str
    ) -> None:
        src = tmp_path / "a.csa"
        src.write_text("hello")
        dst = tmp_path / "out" / "a.csa"
        transfer(src, dst, mode=mode)  # type: ignore[arg-type]
        assert dst.read_text() == "hello"

    def test_transfer_overwrites_existing(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "a.csa"
        src.write_text("new")
        dst = tmp_path / "out" / "a.csa"
        dst.parent.mkdir(parents=True)
        dst.write_text("old")
        transfer(src, dst, mode="copy")
        assert dst.read_text() == "new"
