"""RecordRendererのテスト．"""

from unittest.mock import Mock

import pytest

from maou.app.visualization.record_renderer import (
    HCPERecordRenderer,
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
