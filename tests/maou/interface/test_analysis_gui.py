"""analysis_gui (棋譜解析 GUI の interface アダプタ) のテスト．"""

import json
from pathlib import Path
from typing import Any

import pytest

from maou.interface.analysis_gui import (
    SessionView,
    board_svg,
    candidates_table,
    eval_figure,
    load_session,
    move_table,
    position_analysis,
    position_info,
    sente_eval_cp,
    sente_winrate,
    summary_markdown,
    usi_to_move_arrow,
)

RESOURCES = (
    Path(__file__).parents[1] / "app" / "analysis" / "resources"
)


def _make_report(view_doc: Any) -> dict[str, Any]:
    """mini.csa (4 手) に整合する合成レポートを作る．"""
    doc = view_doc
    base = {
        "playouts": 160,
        "elapsed_ms": 10,
        "stop": "time_limit",
        "record_time_s": None,
        "record_score": None,
        "record_comment": None,
    }
    positions = [
        {
            "ply": 1,
            "side_to_move": "b",
            "sfen": doc.snapshots[0].sfen,
            "played_move": "7g7f",
            "best_move": "2g2f",
            "match": False,
            "winrate": 0.52,
            "eval_cp": 30.0,
            "played_move_winrate": 0.50,
            "winrate_loss": 0.03,
            "pv": ["2g2f", "8c8d", "2f2e"],
            "candidates": [
                {
                    "usi": "2g2f",
                    "visits": 100,
                    "winrate": 0.53,
                    "prior": 0.3,
                    "proven": None,
                },
                {
                    "usi": "7g7f",
                    "visits": 60,
                    "winrate": 0.50,
                    "prior": 0.2,
                    "proven": None,
                },
            ],
            "mate_found": False,
            **base,
        },
        {
            "ply": 2,
            "side_to_move": "w",
            "sfen": doc.snapshots[1].sfen,
            "played_move": "3c3d",
            "best_move": "3c3d",
            "match": True,
            "winrate": 0.45,
            "eval_cp": -35.0,
            "played_move_winrate": 0.45,
            "winrate_loss": 0.0,
            "pv": ["3c3d"],
            "candidates": [
                {
                    "usi": "3c3d",
                    "visits": 120,
                    "winrate": 0.45,
                    "prior": 0.4,
                    "proven": None,
                }
            ],
            "mate_found": False,
            **base,
        },
        {
            "ply": 3,
            "side_to_move": "b",
            "sfen": doc.snapshots[2].sfen,
            "played_move": "2g2f",
            "best_move": "2g2f",
            "match": True,
            "winrate": 0.55,
            "eval_cp": 60.0,
            "played_move_winrate": 0.55,
            "winrate_loss": 0.0,
            "pv": ["2g2f"],
            "candidates": [
                {
                    "usi": "2g2f",
                    "visits": 150,
                    "winrate": 0.55,
                    "prior": 0.5,
                    "proven": 1.0,
                }
            ],
            "mate_found": True,
            **base,
        },
        {
            "ply": 4,
            "side_to_move": "w",
            "sfen": doc.snapshots[3].sfen,
            "played_move": "8c8d",
            "best_move": "8c8d",
            "match": True,
            "winrate": 0.42,
            "eval_cp": -55.0,
            "played_move_winrate": None,
            "winrate_loss": None,
            "pv": [],
            "candidates": [],
            "mate_found": False,
            **base,
        },
    ]
    return {
        "input": {"n_moves": 4},
        "engine": {"model_path": None},
        "budget": {"mode": "fixed_time", "time_ms": 10},
        "positions": positions,
        "summary": {
            "match_rate": {"black": 0.5, "white": 1.0},
            "mean_winrate_loss": {"black": 0.015, "white": 0.0},
            "worst_moves": [
                {
                    "ply": 1,
                    "side": "b",
                    "played": "7g7f",
                    "best": "2g2f",
                    "winrate_loss": 0.03,
                }
            ],
            "mates_found": [{"ply": 3, "side": "b"}],
            "total_elapsed_ms": 40,
            "total_playouts": 640,
        },
    }


@pytest.fixture
def kifu_bytes() -> bytes:
    return (RESOURCES / "mini.csa").read_bytes()


@pytest.fixture
def plain_view(kifu_bytes: bytes) -> SessionView:
    """レポートなしの SessionView．"""
    return load_session(kifu_bytes, "mini.csa")


@pytest.fixture
def analyzed_view(kifu_bytes: bytes) -> SessionView:
    """合成レポート付きの SessionView．"""
    plain = load_session(kifu_bytes, "mini.csa")
    report = _make_report(plain.document)
    return load_session(
        kifu_bytes, "mini.csa", json.dumps(report)
    )


class TestPerspective:
    """手番視点 → 先手視点の変換テスト．"""

    def test_sente_winrate(self) -> None:
        assert sente_winrate(0.6, "b") == pytest.approx(0.6)
        assert sente_winrate(0.6, "w") == pytest.approx(0.4)

    def test_sente_eval_cp(self) -> None:
        assert sente_eval_cp(120.0, "b") == pytest.approx(120.0)
        assert sente_eval_cp(120.0, "w") == pytest.approx(
            -120.0
        )


class TestUsiToMoveArrow:
    """USI → MoveArrow 変換のテスト．"""

    def test_normal_move(self) -> None:
        arrow = usi_to_move_arrow("7g7f")
        assert arrow.from_square == 60
        assert arrow.to_square == 59
        assert arrow.is_drop is False

    def test_drop_move(self) -> None:
        arrow = usi_to_move_arrow("P*5c")
        assert arrow.from_square is None
        assert arrow.to_square == 38
        assert arrow.is_drop is True
        assert arrow.drop_piece_type == 0

    def test_invalid_usi_raises(self) -> None:
        with pytest.raises(ValueError):
            usi_to_move_arrow("xx")
        with pytest.raises(ValueError):
            usi_to_move_arrow("0a1b")


class TestLoadSession:
    """load_session のテスト．"""

    def test_without_report(
        self, plain_view: SessionView
    ) -> None:
        assert plain_view.report is None
        assert plain_view.move_labels == [
            "▲7六歩",
            "△3四歩",
            "▲2六歩",
            "△8四歩",
        ]

    def test_with_report(
        self, analyzed_view: SessionView
    ) -> None:
        assert analyzed_view.report is not None
        assert position_analysis(analyzed_view, 0) is not None
        # 最終局面 (ply = n_moves) に解析はない
        assert position_analysis(analyzed_view, 4) is None

    def test_mismatched_report_raises(
        self, kifu_bytes: bytes
    ) -> None:
        with pytest.raises(ValueError):
            load_session(
                kifu_bytes,
                "mini.csa",
                json.dumps({"positions": []}),
            )


class TestBoardSvg:
    """盤面 SVG 生成のテスト．"""

    def test_initial_position_no_arrow(
        self, plain_view: SessionView
    ) -> None:
        svg = board_svg(plain_view, 0)
        assert "<svg" in svg
        assert "marker-end" not in svg

    def test_last_move_highlight(
        self, plain_view: SessionView
    ) -> None:
        svg = board_svg(plain_view, 1)
        # 最終手 (7g7f) の移動元 + 移動先がハイライトされる
        assert svg.count('fill="rgba(0,112,243,0.12)"') == 2

    def test_candidate_arrows(
        self, analyzed_view: SessionView
    ) -> None:
        svg = board_svg(
            analyzed_view, 0, show_candidates=True, top_n=2
        )
        assert svg.count("marker-end") == 2
        # 順位ラベル
        assert ">1</text>" in svg
        assert ">2</text>" in svg

    def test_pv_arrows(
        self, analyzed_view: SessionView
    ) -> None:
        svg = board_svg(
            analyzed_view,
            0,
            show_candidates=True,
            show_pv=True,
            top_n=2,
        )
        # 候補手 2 本 + PV 連鎖 2 本 (pv[1:3])
        assert svg.count("marker-end") == 4

    def test_candidates_disabled(
        self, analyzed_view: SessionView
    ) -> None:
        svg = board_svg(analyzed_view, 0, show_candidates=False)
        assert "marker-end" not in svg


class TestEvalFigure:
    """評価値グラフのテスト．"""

    def test_no_report_placeholder(
        self, plain_view: SessionView
    ) -> None:
        fig = eval_figure(plain_view, 0)
        assert not fig.data

    def test_winrate_sente_perspective(
        self, analyzed_view: SessionView
    ) -> None:
        fig = eval_figure(analyzed_view, 0, "winrate")
        line = fig.data[0]
        assert list(line.x) == [1, 2, 3, 4]
        # 2 手目 (後手番, 0.45) は先手視点 0.55
        assert line.y[1] == pytest.approx(0.55)

    def test_eval_cp_sente_perspective(
        self, analyzed_view: SessionView
    ) -> None:
        fig = eval_figure(analyzed_view, 0, "eval_cp")
        line = fig.data[0]
        assert line.y[1] == pytest.approx(35.0)

    def test_mate_marker_trace(
        self, analyzed_view: SessionView
    ) -> None:
        fig = eval_figure(analyzed_view, 0)
        assert len(fig.data) == 2
        assert list(fig.data[1].x) == [3]


class TestTables:
    """棋譜/候補手テーブルのテスト．"""

    def test_move_table_with_report(
        self, analyzed_view: SessionView
    ) -> None:
        rows = move_table(analyzed_view)
        assert len(rows) == 4
        # 1 手目: 不一致，先手視点勝率
        assert rows[0][0] == "1"
        assert rows[0][1] == "▲7六歩"
        assert rows[0][2] == ""
        assert rows[0][3] == "0.520"
        # 2 手目: 一致，後手番の先手視点変換
        assert rows[1][2] == "✓"
        assert rows[1][3] == "0.550"
        assert rows[1][4] == "35"
        # 3 手目: 詰み発見
        assert rows[2][6] == "★"
        # 4 手目: 損失 null は空欄
        assert rows[3][5] == ""

    def test_move_table_without_report(
        self, plain_view: SessionView
    ) -> None:
        rows = move_table(plain_view)
        assert len(rows) == 4
        assert rows[0][1] == "▲7六歩"
        assert all(cell == "" for cell in rows[0][2:])

    def test_candidates_table(
        self, analyzed_view: SessionView
    ) -> None:
        rows = candidates_table(analyzed_view, 0, top_n=5)
        assert len(rows) == 2
        assert rows[0][0] == "1"
        assert rows[0][1] == "▲2六歩"
        assert rows[0][2] == "100"
        # 最終局面は候補手なし
        assert candidates_table(analyzed_view, 4) == []

    def test_candidates_table_proven(
        self, analyzed_view: SessionView
    ) -> None:
        rows = candidates_table(analyzed_view, 2, top_n=5)
        assert rows[0][5] == "1"


class TestSummaryAndInfo:
    """サマリと局面情報のテスト．"""

    def test_summary_with_report(
        self, analyzed_view: SessionView
    ) -> None:
        text = summary_markdown(analyzed_view)
        assert "一致率" in text
        assert "50.0%" in text
        assert "詰み発見" in text

    def test_summary_without_report(
        self, plain_view: SessionView
    ) -> None:
        text = summary_markdown(plain_view)
        assert "解析レポート未読込" in text

    def test_position_info_initial(
        self, plain_view: SessionView
    ) -> None:
        sfen, position_str, note = position_info(plain_view, 0)
        assert sfen == plain_view.document.snapshots[0].sfen
        assert position_str == f"position sfen {sfen}"
        assert "初期局面" in note

    def test_position_info_after_moves(
        self, analyzed_view: SessionView
    ) -> None:
        _, position_str, note = position_info(analyzed_view, 2)
        assert position_str.endswith("moves 7g7f 3c3d")
        assert "2手目" in note
        assert "△3四歩" in note
        # 2 手目は一致
        assert "一致" in note

    def test_position_info_mismatch_note(
        self, analyzed_view: SessionView
    ) -> None:
        _, _, note = position_info(analyzed_view, 1)
        assert "エンジン最善: ▲2六歩" in note
        assert "0.030" in note
