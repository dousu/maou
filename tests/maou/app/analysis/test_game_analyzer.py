"""GameAnalyzer / BudgetAllocator / 棋譜デコードのテスト．"""

from pathlib import Path
from typing import Any

import pytest

from maou._rust.maou_shogi import parse_kif_str
from maou.app.analysis.game_analyzer import (
    EqualDivisionAllocator,
    FixedPlayoutsAllocator,
    FixedTimeAllocator,
    GameAnalyzer,
    PositionBudget,
    decode_kifu_bytes,
)

RESOURCES = Path(__file__).parent / "resources"
CONVERTER_INPUT = (
    Path(__file__).parents[1]
    / "converter"
    / "resources"
    / "test_dir"
    / "input"
)


class TestBudgetAllocators:
    def test_fixed_time(self) -> None:
        allocator = FixedTimeAllocator(time_ms=500)
        budgets = allocator.allocate(3)
        assert (
            budgets
            == [PositionBudget(max_playouts=None, time_ms=500)]
            * 3
        )
        assert allocator.describe() == {
            "mode": "fixed_time",
            "time_ms": 500,
        }

    def test_equal_division(self) -> None:
        allocator = EqualDivisionAllocator(total_time_ms=1000)
        budgets = allocator.allocate(3)
        # 床関数: 端数は切り捨てて全体上限を超えない側に倒す
        assert (
            budgets
            == [PositionBudget(max_playouts=None, time_ms=333)]
            * 3
        )
        assert allocator.describe() == {
            "mode": "equal_division",
            "total_time_ms": 1000,
        }

    def test_equal_division_clamps_to_1ms(self) -> None:
        allocator = EqualDivisionAllocator(total_time_ms=2)
        budgets = allocator.allocate(4)
        assert all(b.time_ms == 1 for b in budgets)

    def test_equal_division_empty(self) -> None:
        allocator = EqualDivisionAllocator(total_time_ms=1000)
        assert allocator.allocate(0) == []

    def test_fixed_playouts(self) -> None:
        allocator = FixedPlayoutsAllocator(playouts=100)
        budgets = allocator.allocate(2)
        assert (
            budgets
            == [PositionBudget(max_playouts=100, time_ms=None)]
            * 2
        )
        assert allocator.describe() == {
            "mode": "fixed_playouts",
            "playouts": 100,
        }


class TestDecodeKifuBytes:
    def test_utf8(self) -> None:
        assert (
            decode_kifu_bytes("将棋".encode("utf-8")) == "将棋"
        )

    def test_cp932_fallback(self) -> None:
        text = "先手：将棋太郎"
        assert decode_kifu_bytes(text.encode("cp932")) == text

    def test_invalid_both_raises(self) -> None:
        # 0x81 0x20 は UTF-8 でも cp932 でも不正 (不正トレイル)
        with pytest.raises(UnicodeDecodeError):
            decode_kifu_bytes(b"\x81\x20")

    def test_sjis_kif_fixture_parses(self) -> None:
        data = (
            CONVERTER_INPUT / "test_data_sjis.kif"
        ).read_bytes()
        record = parse_kif_str(decode_kifu_bytes(data))
        assert len(record.moves) == 163


class TestGameAnalyzer:
    def _analyze(
        self, input_path: Path, input_format: str
    ) -> dict[str, Any]:
        option = GameAnalyzer.AnalyzeOption(
            input_path=input_path,
            input_format=input_format,
            allocator=FixedPlayoutsAllocator(playouts=8),
            root_dfpn=False,
            leaf_mate=False,
        )
        return GameAnalyzer().analyze(option)

    def test_analyze_mini_csa(self) -> None:
        result = self._analyze(RESOURCES / "mini.csa", "csa")

        assert set(result) == {
            "input",
            "engine",
            "budget",
            "positions",
            "summary",
        }
        assert result["input"]["names"] == [
            "black_engine",
            "white_engine",
        ]
        assert result["input"]["win"] == 2
        assert result["input"]["endgame"] == "%TORYO"
        assert result["input"]["n_moves"] == 4
        assert result["budget"] == {
            "mode": "fixed_playouts",
            "playouts": 8,
            "per_position": {
                "max_playouts": 8,
                "time_ms": None,
            },
        }
        assert result["engine"]["model_path"] is None

        positions = result["positions"]
        assert [p["ply"] for p in positions] == [1, 2, 3, 4]
        assert [p["side_to_move"] for p in positions] == [
            "b",
            "w",
            "b",
            "w",
        ]
        assert [p["played_move"] for p in positions] == [
            "7g7f",
            "3c3d",
            "2g2f",
            "8c8d",
        ]
        assert positions[0]["sfen"].startswith(
            "lnsgkgsnl/1r5b1/ppppppppp"
        )
        assert [p["record_time_s"] for p in positions] == [
            1,
            2,
            3,
            4,
        ]
        assert all(
            p["stop"] == "playout_limit" for p in positions
        )
        assert all(
            p["best_move"] is not None for p in positions
        )
        assert all(len(p["candidates"]) <= 5 for p in positions)
        # 候補手の先頭は best_move (SearchRunner と同じ整列)
        assert all(
            p["candidates"][0]["usi"] == p["best_move"]
            for p in positions
        )

        summary = result["summary"]
        assert set(summary["match_rate"]) == {"black", "white"}
        assert summary["total_playouts"] >= 4 * 8
        assert isinstance(summary["worst_moves"], list)
        assert summary["mates_found"] == []

    def test_multi_game_csa_rejected(self) -> None:
        with pytest.raises(ValueError, match="2 局"):
            self._analyze(
                CONVERTER_INPUT / "csa_multi_game.csa", "csa"
            )

    def test_no_moves_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="指し手がありません"
        ):
            self._analyze(
                CONVERTER_INPUT / "test_data_no_moves.csa",
                "csa",
            )

    def test_unknown_format_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="未対応の棋譜形式"
        ):
            self._analyze(RESOURCES / "mini.csa", "sgf")
