"""analysis_gui (棋譜解析 GUI の interface アダプタ) のテスト．"""

import json
from pathlib import Path
from typing import Any

import pytest

from maou.app.analysis.analysis_session import (
    _snapshot,
    legal_move_infos,
)
from maou.domain.board.shogi import Board
from maou.interface.analysis_gui import (
    ClickState,
    SessionView,
    advance_move,
    board_svg,
    breadcrumb_markdown,
    build_variation_tree,
    candidate_usi,
    candidates_table,
    click_overlays,
    click_status_text,
    current_node,
    eval_figure,
    goto_node,
    handle_board_click,
    legal_move_choices,
    load_session,
    mainline_ply,
    move_table,
    node_board_svg,
    node_candidates_table,
    node_legal_moves,
    node_position_info,
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


@pytest.fixture
def plain_tree(plain_view: SessionView) -> Any:
    """レポートなしの分岐木を作る．"""
    return build_variation_tree(plain_view.document, None)


@pytest.fixture
def analyzed_tree(analyzed_view: SessionView) -> Any:
    """レポート取込済みの分岐木を作る．"""
    return build_variation_tree(
        analyzed_view.document, analyzed_view.report
    )


class TestBoardClickStateMachine:
    """盤面クリック入力の状態機械のテスト．"""

    def test_select_then_play(self, plain_tree: Any) -> None:
        """自駒選択 → 行き先クリックで USI が確定する．"""
        legal = node_legal_moves(plain_tree)
        state, usi = handle_board_click(
            legal, ClickState(), "sq:60", "b"
        )
        assert usi is None
        assert state.selected == "sq:60"
        state, usi = handle_board_click(
            legal, state, "sq:59", "b"
        )
        assert usi == "7g7f"
        assert state.selected is None

    def test_click_empty_square_noop(
        self, plain_tree: Any
    ) -> None:
        """起点にならないマスのクリックは何も起きない．"""
        legal = node_legal_moves(plain_tree)
        state, usi = handle_board_click(
            legal, ClickState(), "sq:40", "b"
        )
        assert usi is None
        assert state.selected is None

    def test_reclick_deselects(self, plain_tree: Any) -> None:
        """同じ起点の再クリックで選択解除される．"""
        legal = node_legal_moves(plain_tree)
        state, _ = handle_board_click(
            legal, ClickState(), "sq:60", "b"
        )
        state, usi = handle_board_click(
            legal, state, "sq:60", "b"
        )
        assert usi is None
        assert state.selected is None

    def test_switch_selection(self, plain_tree: Any) -> None:
        """別の自駒クリックで選択が切り替わる．"""
        legal = node_legal_moves(plain_tree)
        state, _ = handle_board_click(
            legal, ClickState(), "sq:60", "b"
        )
        # 2g = (2-1)*9 + (7-1) = 15
        state, usi = handle_board_click(
            legal, state, "sq:15", "b"
        )
        assert usi is None
        assert state.selected == "sq:15"

    def test_promotion_pending(self) -> None:
        """成/不成の両方が合法な行き先で選択待ちに入る．"""
        board = Board()
        board.set_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        snapshot = _snapshot(board, 0, None)
        legal = legal_move_infos(snapshot)
        # 5d = 39, 5c = 38
        state, _ = handle_board_click(
            legal, ClickState(), "sq:39", "b"
        )
        state, usi = handle_board_click(
            legal, state, "sq:38", "b"
        )
        assert usi is None
        assert state.pending_usis == ("5d5c+", "5d5c")

    def test_pending_cancelled_by_board_click(self) -> None:
        """選択待ち中の盤面クリックはキャンセルして再解釈する．"""
        board = Board()
        board.set_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        snapshot = _snapshot(board, 0, None)
        legal = legal_move_infos(snapshot)
        state, _ = handle_board_click(
            legal, ClickState(), "sq:39", "b"
        )
        state, _ = handle_board_click(
            legal, state, "sq:38", "b"
        )
        assert state.pending_usis
        state, usi = handle_board_click(
            legal, state, "sq:39", "b"
        )
        assert usi is None
        assert state.pending_usis == ()
        assert state.selected == "sq:39"

    def test_hand_drop(self) -> None:
        """持ち駒クリック → 空きマスで駒打ちが確定する．"""
        board = Board()
        board.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        snapshot = _snapshot(board, 0, None)
        legal = legal_move_infos(snapshot)
        state, _ = handle_board_click(
            legal, ClickState(), "hand:b:0", "b"
        )
        assert state.selected == "hand:b:0"
        # 5e = (5-1)*9 + (5-1) = 40
        state, usi = handle_board_click(
            legal, state, "sq:40", "b"
        )
        assert usi == "P*5e"

    def test_opponent_hand_ignored(self) -> None:
        """相手側の持ち駒クリックは選択されない．"""
        board = Board()
        board.set_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        snapshot = _snapshot(board, 0, None)
        legal = legal_move_infos(snapshot)
        state, usi = handle_board_click(
            legal, ClickState(), "hand:w:0", "b"
        )
        assert usi is None
        assert state.selected is None

    def test_click_overlays(self, plain_tree: Any) -> None:
        """選択中は選択マスと行き先が row-major で得られる．"""
        legal = node_legal_moves(plain_tree)
        state, _ = handle_board_click(
            legal, ClickState(), "sq:60", "b"
        )
        selected, destinations = click_overlays(
            legal, state, "b"
        )
        # 7g: col=6, row=6 → row-major 6*9+6 = 60 (対角なので同値)
        assert selected == [60]
        # 7f: col=6, row=5 → 5*9+6 = 51
        assert destinations == [51]
        # 未選択は空
        assert click_overlays(legal, ClickState(), "b") == (
            [],
            [],
        )

    def test_click_status_text(self, plain_tree: Any) -> None:
        """クリック状態の説明テキスト．"""
        snapshot = current_node(plain_tree).snapshot
        assert "クリック" in click_status_text(
            snapshot, ClickState()
        )
        text = click_status_text(
            snapshot, ClickState(selected="sq:60")
        )
        assert "7七歩" in text
        text = click_status_text(
            snapshot, ClickState(selected="hand:b:0")
        )
        assert "持ち駒の歩" in text
        text = click_status_text(
            snapshot,
            ClickState(pending_usis=("5d5c+", "5d5c")),
        )
        assert "成" in text


class TestLegalMoveChoices:
    """合法手 Dropdown 選択肢のテスト．"""

    def test_choices_labels(self, plain_tree: Any) -> None:
        snapshot = current_node(plain_tree).snapshot
        legal = node_legal_moves(plain_tree)
        choices = legal_move_choices(snapshot, legal)
        assert len(choices) == 30
        labels = {label for label, _usi in choices}
        values = {usi for _label, usi in choices}
        assert "7g7f" in values
        assert any("▲7六歩" in label for label in labels)


class TestVariationNavigation:
    """分岐木ナビゲーション表示のテスト．"""

    def test_breadcrumb_mainline(self, plain_tree: Any) -> None:
        assert "初期局面" in breadcrumb_markdown(plain_tree)
        goto_node(plain_tree, plain_tree.mainline_ids[2])
        assert "2手目" in breadcrumb_markdown(plain_tree)

    def test_breadcrumb_branch(self, plain_tree: Any) -> None:
        advance_move(plain_tree, "7g7f")
        advance_move(plain_tree, "8c8d")  # 分岐
        text = breadcrumb_markdown(plain_tree)
        assert "本譜 1手目" in text
        assert "▶" in text
        assert "△8四歩" in text

    def test_mainline_ply(self, plain_tree: Any) -> None:
        advance_move(plain_tree, "7g7f")
        assert mainline_ply(plain_tree) == 1
        advance_move(plain_tree, "8c8d")  # 分岐
        assert mainline_ply(plain_tree) == 1

    def test_node_board_svg_interactive(
        self, analyzed_tree: Any
    ) -> None:
        legal = node_legal_moves(analyzed_tree)
        svg = node_board_svg(
            analyzed_tree,
            click_state=ClickState(selected="sq:60"),
            legal=legal,
            interactive=True,
        )
        assert 'data-click="sq:60"' in svg
        assert "rgba(255,152,0,0.35)" in svg  # 選択マス
        assert "rgba(76,175,80,0.28)" in svg  # 行き先

    def test_node_candidates_table(
        self, analyzed_tree: Any
    ) -> None:
        rows = node_candidates_table(analyzed_tree, top_n=5)
        assert rows
        assert rows[0][0] == "1"

    def test_candidate_usi(self, analyzed_tree: Any) -> None:
        analysis = current_node(analyzed_tree).analysis
        assert candidate_usi(analysis, 0, 5) == "2g2f"
        assert candidate_usi(analysis, 99, 5) is None
        assert candidate_usi(None, 0, 5) is None

    def test_node_position_info_mainline(
        self, analyzed_view: SessionView, analyzed_tree: Any
    ) -> None:
        goto_node(analyzed_tree, analyzed_tree.mainline_ids[1])
        sfen, position_str, _note = node_position_info(
            analyzed_view, analyzed_tree
        )
        assert position_str.endswith("moves 7g7f")
        assert sfen == (
            analyzed_view.document.snapshots[1].sfen
        )

    def test_node_position_info_branch(
        self, analyzed_view: SessionView, analyzed_tree: Any
    ) -> None:
        advance_move(analyzed_tree, "2g2f")  # 本譜は 7g7f
        sfen, position_str, note = node_position_info(
            analyzed_view, analyzed_tree
        )
        assert position_str.endswith("moves 2g2f")
        assert "分岐" in note
        assert "未解析" in note
        assert sfen.split(" ")[1] == "w"
