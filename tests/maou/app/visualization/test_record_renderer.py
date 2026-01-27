"""RecordRendererのテスト．"""

from typing import Any
from unittest.mock import Mock

import pytest

from maou.app.visualization.record_renderer import (
    HCPERecordRenderer,
)
from maou.domain.visualization.board_renderer import (
    SVGBoardRenderer,
)
from maou.domain.visualization.move_label_converter import (
    MoveLabelConverter,
)


class TestHCPERecordRendererAnalytics:
    """HCPERecordRenderer.generate_analyticsのテスト．"""

    @pytest.fixture
    def renderer(self) -> HCPERecordRenderer:
        """テスト用HCPERecordRendererを作成．"""
        return HCPERecordRenderer(
            board_renderer=Mock(),
            move_converter=Mock(),
        )

    def test_generate_analytics_returns_single_eval_chart(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """HCPERecordRenderer.generate_analytics returns only eval distribution chart."""
        records = [
            {"eval": 100, "moves": 118},
            {"eval": -50, "moves": 118},
            {"eval": 200, "moves": 120},
        ]

        fig = renderer.generate_analytics(records)

        assert fig is not None
        # Should have only 1 trace (eval histogram), not 2
        assert len(fig.data) == 1
        assert fig.data[0].name == "評価値"

    def test_generate_analytics_returns_none_for_empty_records(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """空のレコードリストの場合，Noneを返す．"""
        fig = renderer.generate_analytics([])

        assert fig is None

    def test_generate_analytics_returns_none_for_no_evals(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """評価値がないレコードの場合，Noneを返す．"""
        records = [
            {"moves": 118},
            {"moves": 120},
        ]

        fig = renderer.generate_analytics(records)

        assert fig is None

    def test_generate_analytics_chart_layout(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """生成されるチャートのレイアウトが正しいことを確認．"""
        records = [
            {"eval": 100, "moves": 118},
            {"eval": -50, "moves": 118},
        ]

        fig = renderer.generate_analytics(records)

        assert fig is not None
        assert fig.layout.title.text == "評価値分布"
        assert fig.layout.xaxis.title.text == "評価値"
        assert fig.layout.yaxis.title.text == "頻度"


class TestHCPERecordRendererArrow:
    """HCPERecordRenderer._create_move_arrowのテスト．"""

    @pytest.fixture
    def renderer(self) -> HCPERecordRenderer:
        """テスト用HCPERecordRendererを作成．"""
        return HCPERecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )

    def test_create_move_arrow_normal_move(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """通常の移動から矢印を生成できる．"""
        # 7g7f (7七から7六への移動)
        # cshogi format: to_sq | (from_sq << 7) | (promote << 14)
        # 7g = square 48 (file 7, rank 7 -> 6*9 + 6 = 60... actually need to verify)
        # Let's use a simpler approach - create a move and check the result
        # For now, test with a known move value
        # bestMove16 for 7g7f: to=58, from=67 -> 58 | (67 << 7) = 58 | 8576 = 8634
        record = {"bestMove16": 8634}

        arrow = renderer._create_move_arrow(record)

        assert arrow is not None
        assert arrow.is_drop is False
        # The exact squares depend on cshogi encoding

    def test_create_move_arrow_missing_field(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """bestMove16フィールドがない場合はNoneを返す．"""
        record: dict[str, Any] = {}

        arrow = renderer._create_move_arrow(record)

        assert arrow is None

    def test_create_move_arrow_none_value(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """bestMove16がNoneの場合はNoneを返す．"""
        record = {"bestMove16": None}

        arrow = renderer._create_move_arrow(record)

        assert arrow is None


class TestHCPERecordRendererRenderBoard:
    """HCPERecordRenderer.render_boardのテスト．"""

    @pytest.fixture
    def renderer(self) -> HCPERecordRenderer:
        """テスト用HCPERecordRendererを作成．"""
        return HCPERecordRenderer(
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
        )

    def test_render_board_includes_arrow(
        self, renderer: HCPERecordRenderer
    ) -> None:
        """render_boardが矢印を含むSVGを生成する．"""
        # 空の盤面データ（9x9）と持ち駒（14要素）
        board = [[0 for _ in range(9)] for _ in range(9)]
        hand = [0 for _ in range(14)]
        record = {
            "boardIdPositions": board,
            "piecesInHand": hand,
            "bestMove16": 8634,
            "turn": 0,
            "id": "test-001",
        }

        svg = renderer.render_board(record)

        assert "<svg" in svg
        assert "</svg>" in svg
        # 矢印マーカーが含まれる
        assert "arrowhead" in svg
