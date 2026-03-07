"""ゲームツリー可視化インターフェースのテスト."""

from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from maou.domain.game_tree.schema import (
    get_game_tree_edges_schema,
    get_game_tree_nodes_schema,
)
from maou.interface.game_tree_visualization import (
    GameTreeVisualizationInterface,
)


def _make_nodes(
    rows: list[dict],
) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=get_game_tree_nodes_schema()
    )


def _make_edges(
    rows: list[dict],
) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=get_game_tree_edges_schema()
    )


def _build_simple_tree() -> tuple[pl.DataFrame, pl.DataFrame]:
    """テスト用の単純なツリー(ルートのみ)."""
    nodes = _make_nodes(
        [
            {
                "position_hash": 100,
                "result_value": 0.52,
                "best_move_win_rate": 0.53,
                "num_branches": 0,
                "depth": 0,
                "is_depth_cutoff": False,
            },
        ]
    )
    edges = _make_edges([])
    return nodes, edges


class TestGetCytoscapeElements:
    """get_cytoscape_elements のテスト."""

    def test_single_node(self) -> None:
        """単一ノードのCytoscape elementsを生成する."""
        nodes, edges = _build_simple_tree()
        viz = GameTreeVisualizationInterface(nodes, edges)
        elements = viz.get_cytoscape_elements(100, 3, 0.01)
        assert len(elements["nodes"]) == 1
        assert len(elements["edges"]) == 0
        node_data = elements["nodes"][0]["data"]
        assert node_data["id"] == "100"
        assert node_data["label"] == "ROOT"
        assert node_data["result_value"] == pytest.approx(0.52)

    def test_elements_structure(self) -> None:
        """Cytoscape elementsの構造が正しい."""
        nodes, edges = _build_simple_tree()
        viz = GameTreeVisualizationInterface(nodes, edges)
        elements = viz.get_cytoscape_elements(100, 3, 0.01)
        assert "nodes" in elements
        assert "edges" in elements
        for node in elements["nodes"]:
            assert "data" in node
            assert "id" in node["data"]
            assert "result_value" in node["data"]


class TestGetNodeStats:
    """get_node_stats のテスト."""

    def test_stats_format(self) -> None:
        """統計情報が正しいフォーマットで返される."""
        nodes, edges = _build_simple_tree()
        viz = GameTreeVisualizationInterface(nodes, edges)
        stats = viz.get_node_stats(100)
        assert "局面ハッシュ" in stats
        assert "勝率" in stats
        assert "最善手勝率" in stats
        assert "深さ" in stats
        assert "分岐数" in stats
        assert stats["勝率"] == "52.0%"

    def test_missing_node(self) -> None:
        """存在しないノードは空辞書を返す."""
        nodes, edges = _build_simple_tree()
        viz = GameTreeVisualizationInterface(nodes, edges)
        assert viz.get_node_stats(999) == {}


class TestGetMoveTable:
    """get_move_table のテスト."""

    def test_empty_moves(self) -> None:
        """子がないノードは空リストを返す."""
        nodes, edges = _build_simple_tree()
        viz = GameTreeVisualizationInterface(nodes, edges)
        moves = viz.get_move_table(100)
        assert moves == []


class TestUsiToJapanese:
    """_usi_to_japanese のテスト."""

    def test_normal_move(self) -> None:
        """通常の指し手をjapanese表記に変換する."""
        result = (
            GameTreeVisualizationInterface._usi_to_japanese(
                "7g7f"
            )
        )
        assert result == "7六"

    def test_normal_move_with_piece(self) -> None:
        """通常の指し手に駒名を含めて変換する."""
        result = (
            GameTreeVisualizationInterface._usi_to_japanese(
                "7g7f", piece_name="歩"
            )
        )
        assert result == "7六歩"

    def test_drop_move(self) -> None:
        """駒打ちをjapanese表記に変換する."""
        result = (
            GameTreeVisualizationInterface._usi_to_japanese(
                "P*5e"
            )
        )
        assert result == "5五歩打"

    def test_promotion_move(self) -> None:
        """成りの指し手をjapanese表記に変換する."""
        result = (
            GameTreeVisualizationInterface._usi_to_japanese(
                "8h2b+"
            )
        )
        assert result == "2二成"

    def test_promotion_move_with_piece(self) -> None:
        """成りの指し手に駒名を含めて変換する."""
        result = (
            GameTreeVisualizationInterface._usi_to_japanese(
                "8h2b+", piece_name="角"
            )
        )
        assert result == "2二角成"


def _build_tree_with_edge() -> tuple[
    pl.DataFrame, pl.DataFrame
]:
    """テスト用のツリー(ルート + 子ノード1つ)．

    指し手は7g7f(move16=7739)．平手初期局面からの合法手．
    """
    nodes = _make_nodes(
        [
            {
                "position_hash": 100,
                "result_value": 0.52,
                "best_move_win_rate": 0.53,
                "num_branches": 1,
                "depth": 0,
                "is_depth_cutoff": False,
            },
            {
                "position_hash": 200,
                "result_value": 0.48,
                "best_move_win_rate": 0.49,
                "num_branches": 0,
                "depth": 1,
                "is_depth_cutoff": False,
            },
        ]
    )
    edges = _make_edges(
        [
            {
                "parent_hash": 100,
                "child_hash": 200,
                "move16": 7739,
                "move_label": 10,
                "probability": 0.6,
                "win_rate": 0.52,
                "is_leaf": False,
            },
        ]
    )
    return nodes, edges


class TestGetBoardSvg:
    """get_board_svg のテスト."""

    def test_root_node_returns_svg(self) -> None:
        """ルートノードの盤面SVGを生成する."""
        nodes, edges = _build_simple_tree()
        viz = GameTreeVisualizationInterface(nodes, edges)
        result = viz.get_board_svg(100)
        assert "<svg" in result
        assert "</svg>" in result

    def test_child_node_returns_svg_with_arrow(self) -> None:
        """子ノードの盤面SVGを矢印付きで生成する."""
        nodes, edges = _build_tree_with_edge()
        viz = GameTreeVisualizationInterface(nodes, edges)
        result = viz.get_board_svg(200)
        assert "<svg" in result
        assert "</svg>" in result

    def test_reconstruct_failure_returns_error(self) -> None:
        """_reconstruct_board_from_pathがNoneを返した場合はエラーHTMLを返す."""
        nodes, edges = _build_tree_with_edge()
        viz = GameTreeVisualizationInterface(nodes, edges)
        with patch.object(
            viz,
            "_reconstruct_board_from_path",
            return_value=None,
        ):
            result = viz.get_board_svg(200)
        assert "盤面を復元できません" in result


class TestGetAnalyticsData:
    """get_analytics_data のテスト."""

    def test_empty_analytics(self) -> None:
        """子がないノードは空のデータを返す."""
        nodes, edges = _build_simple_tree()
        viz = GameTreeVisualizationInterface(nodes, edges)
        data = viz.get_analytics_data(100)
        assert data["moves"] == []
        assert data["probabilities"] == []
        assert data["win_rates"] == []
