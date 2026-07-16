"""interface.analyzer のバリデーションのテスト．"""

from pathlib import Path

import pytest

from maou.app.analysis.game_analyzer import (
    EqualDivisionAllocator,
    FixedPlayoutsAllocator,
    FixedTimeAllocator,
)
from maou.interface import analyzer


class TestResolveInputFormat:
    def test_csa_extension(self) -> None:
        assert (
            analyzer.resolve_input_format(Path("a/b.csa"), None)
            == "csa"
        )

    def test_kif_extensions(self) -> None:
        assert (
            analyzer.resolve_input_format(Path("a.kif"), None)
            == "kif"
        )
        assert (
            analyzer.resolve_input_format(Path("a.kifu"), None)
            == "kif"
        )
        # 大文字拡張子も判定できる
        assert (
            analyzer.resolve_input_format(Path("a.CSA"), None)
            == "csa"
        )

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="判定できません"):
            analyzer.resolve_input_format(Path("a.txt"), None)

    def test_explicit_overrides_extension(self) -> None:
        assert (
            analyzer.resolve_input_format(Path("a.csa"), "kif")
            == "kif"
        )

    def test_invalid_explicit_raises(self) -> None:
        with pytest.raises(
            ValueError, match="未対応の棋譜形式"
        ):
            analyzer.resolve_input_format(Path("a.csa"), "sgf")


class TestBuildAllocator:
    def test_default_is_1000ms_per_position(self) -> None:
        allocator = analyzer.build_allocator(
            time_ms=None, total_time_ms=None, playouts=None
        )
        assert allocator == FixedTimeAllocator(time_ms=1000)

    def test_time_ms(self) -> None:
        allocator = analyzer.build_allocator(
            time_ms=250, total_time_ms=None, playouts=None
        )
        assert allocator == FixedTimeAllocator(time_ms=250)

    def test_total_time_ms(self) -> None:
        allocator = analyzer.build_allocator(
            time_ms=None, total_time_ms=60_000, playouts=None
        )
        assert allocator == EqualDivisionAllocator(
            total_time_ms=60_000
        )

    def test_playouts(self) -> None:
        allocator = analyzer.build_allocator(
            time_ms=None, total_time_ms=None, playouts=800
        )
        assert allocator == FixedPlayoutsAllocator(playouts=800)

    def test_multiple_budgets_rejected(self) -> None:
        with pytest.raises(ValueError, match="1 つまで"):
            analyzer.build_allocator(
                time_ms=100, total_time_ms=None, playouts=800
            )

    def test_non_positive_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="正の値"):
            analyzer.build_allocator(
                time_ms=0, total_time_ms=None, playouts=None
            )
