"""ゲームグラフ可視化インターフェースのテスト．"""

from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from maou.app.game_graph.layout import (
    GameGraphLayoutService,
    GraphLayout,
)
from maou.domain.board.shogi import Board
from maou.domain.game_graph.schema import (
    get_game_graph_edges_schema,
    get_game_graph_nodes_schema,
)
from maou.interface.game_graph_visualization import (
    GameGraphVisualizationInterface,
    MoveRow,
)


def _make_nodes(
    rows: list[dict],
) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=get_game_graph_nodes_schema()
    )


def _make_edges(
    rows: list[dict],
) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=get_game_graph_edges_schema()
    )


def _build_simple_tree() -> tuple[pl.DataFrame, pl.DataFrame]:
    """テスト用の単純なグラフ(ルートのみ)."""
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


def _make_layout(
    nodes: pl.DataFrame, edges: pl.DataFrame, root_hash: int
) -> GraphLayout:
    """テスト用のレイアウトを計算する．"""
    svc = GameGraphLayoutService()
    return svc.compute_layout(nodes, edges, root_hash)


class TestGetCanvasData:
    """get_canvas_data のテスト."""

    def test_single_node(self) -> None:
        """単一ノードの Canvas データを生成する."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        layout = _make_layout(nodes, edges, 100)
        data = viz.get_canvas_data(100, 3, 0.01, layout)
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 0
        node = data["nodes"][0]
        assert node["id"] == "100"
        assert node["label"] == "ROOT"
        # depth=0, 先手番 → sente_result_value == result_value
        assert node["sente_result_value"] == pytest.approx(0.52)
        # 座標が含まれる
        assert "x" in node
        assert "y" in node

    def test_sente_result_value_depth1_flipped(self) -> None:
        """depth=1(後手番)ではsente_result_valueが反転する."""
        nodes, edges = _build_tree_with_edge()
        viz = GameGraphVisualizationInterface(nodes, edges)
        layout = _make_layout(nodes, edges, 100)
        data = viz.get_canvas_data(100, 3, 0.01, layout)
        node_map = {n["id"]: n for n in data["nodes"]}
        # depth=0: 先手番 → そのまま
        assert node_map["100"][
            "sente_result_value"
        ] == pytest.approx(0.52)
        # depth=1: 後手番 → 1 - 0.48 = 0.52
        assert node_map["200"][
            "sente_result_value"
        ] == pytest.approx(0.52)

    def test_sente_result_value_gote_root(self) -> None:
        """後手番ルート(initial_sfen="w")ではdepth=0が反転する."""
        nodes, edges = _build_tree_with_edge()
        gote_sfen = (
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/"
            "PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        )
        viz = GameGraphVisualizationInterface(
            nodes, edges, initial_sfen=gote_sfen
        )
        layout = _make_layout(nodes, edges, 100)
        data = viz.get_canvas_data(100, 3, 0.01, layout)
        node_map = {n["id"]: n for n in data["nodes"]}
        # depth=0: 後手番ルート → 1 - 0.52 = 0.48
        assert node_map["100"][
            "sente_result_value"
        ] == pytest.approx(0.48)
        # depth=1: 先手番 → そのまま 0.48
        assert node_map["200"][
            "sente_result_value"
        ] == pytest.approx(0.48)

    def test_data_structure(self) -> None:
        """Canvas データの構造が正しい."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        layout = _make_layout(nodes, edges, 100)
        data = viz.get_canvas_data(100, 3, 0.01, layout)
        assert "nodes" in data
        assert "edges" in data
        assert "bounds" in data
        for node in data["nodes"]:
            assert "id" in node
            assert "x" in node
            assert "y" in node
            assert "sente_result_value" in node

    def test_edges_have_coordinates(self) -> None:
        """エッジに source/target 座標が含まれる."""
        nodes, edges = _build_tree_with_edge()
        viz = GameGraphVisualizationInterface(nodes, edges)
        layout = _make_layout(nodes, edges, 100)
        data = viz.get_canvas_data(100, 3, 0.01, layout)
        assert len(data["edges"]) == 1
        edge = data["edges"][0]
        assert "source_x" in edge
        assert "source_y" in edge
        assert "target_x" in edge
        assert "target_y" in edge
        assert "source_id" in edge
        assert "target_id" in edge


class TestGetNodeStats:
    """get_node_stats のテスト."""

    def test_stats_format(self) -> None:
        """統計情報が正しいフォーマットで返される."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        stats = viz.get_node_stats(100)
        assert "Zobrist Hash" in stats
        assert "勝率(手番視点)" in stats
        assert "最善手勝率" in stats
        assert "深さ" in stats
        assert "分岐数" in stats
        assert stats["勝率(手番視点)"] == "52.0%"

    def test_missing_node(self) -> None:
        """存在しないノードは空辞書を返す."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        assert viz.get_node_stats(999) == {}


class TestGetMoveTable:
    """get_move_table のテスト."""

    def test_empty_moves(self) -> None:
        """子がないノードは空リストを返す."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        moves = viz.get_move_table(100)
        assert moves == []

    def test_returns_move_rows(self) -> None:
        """子がある場合はMoveRowのリストを返す."""
        nodes, edges = _build_tree_with_edge()
        viz = GameGraphVisualizationInterface(nodes, edges)
        moves = viz.get_move_table(100)
        assert len(moves) == 1
        row = moves[0]
        assert isinstance(row, MoveRow)
        assert "7六" in row.japanese
        assert row.probability == "60.0%"
        assert row.win_rate == "52.0%"
        assert row.child_hash == "200"


class TestUsiToJapanese:
    """usi_to_japanese のテスト．"""

    def test_normal_move(self) -> None:
        """通常の指し手をjapanese表記に変換する."""
        result = (
            GameGraphVisualizationInterface.usi_to_japanese(
                "7g7f"
            )
        )
        assert result == "7六"

    def test_normal_move_with_piece(self) -> None:
        """通常の指し手に駒名を含めて変換する."""
        result = (
            GameGraphVisualizationInterface.usi_to_japanese(
                "7g7f", piece_name="歩"
            )
        )
        assert result == "7六歩"

    def test_drop_move(self) -> None:
        """駒打ちをjapanese表記に変換する."""
        result = (
            GameGraphVisualizationInterface.usi_to_japanese(
                "P*5e"
            )
        )
        assert result == "5五歩打"

    def test_promotion_move(self) -> None:
        """成りの指し手をjapanese表記に変換する."""
        result = (
            GameGraphVisualizationInterface.usi_to_japanese(
                "8h2b+"
            )
        )
        assert result == "2二成"

    def test_promotion_move_with_piece(self) -> None:
        """成りの指し手に駒名を含めて変換する."""
        result = (
            GameGraphVisualizationInterface.usi_to_japanese(
                "8h2b+", piece_name="角"
            )
        )
        assert result == "2二角成"


def _build_tree_with_edge() -> tuple[
    pl.DataFrame, pl.DataFrame
]:
    """テスト用のグラフ(ルート + 子ノード1つ)．

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
        viz = GameGraphVisualizationInterface(nodes, edges)
        result = viz.get_board_svg(100)
        assert "<svg" in result
        assert "</svg>" in result

    def test_child_node_returns_svg_with_arrow(self) -> None:
        """子ノードの盤面SVGを矢印付きで生成する."""
        nodes, edges = _build_tree_with_edge()
        viz = GameGraphVisualizationInterface(nodes, edges)
        result = viz.get_board_svg(200)
        assert "<svg" in result
        assert "</svg>" in result

    def test_reconstruct_failure_returns_error(self) -> None:
        """_reconstruct_board_from_pathがNoneを返した場合はエラーHTMLを返す."""
        nodes, edges = _build_tree_with_edge()
        viz = GameGraphVisualizationInterface(nodes, edges)
        with patch.object(
            viz,
            "_reconstruct_board_from_path",
            return_value=None,
        ):
            result = viz.get_board_svg(200)
        assert "盤面を復元できません" in result


class TestGetPieceName:
    """get_piece_name のテスト．"""

    def test_pawn(self) -> None:
        """歩の駒名を取得する(move16=7739: 7g7f)．"""
        board = Board()
        result = GameGraphVisualizationInterface.get_piece_name(
            board, 7739
        )
        assert result == "歩"

    def test_silver(self) -> None:
        """銀の駒名を取得する(move16=3362: 3i4h)．"""
        board = Board()
        result = GameGraphVisualizationInterface.get_piece_name(
            board, 3362
        )
        assert result == "銀"

    def test_gold(self) -> None:
        """金の駒名を取得する(move16=6845: 6i7h)．"""
        board = Board()
        result = GameGraphVisualizationInterface.get_piece_name(
            board, 6845
        )
        assert result == "金"

    def test_rook(self) -> None:
        """飛の駒名を取得する(move16=2055: 2h1h)．"""
        board = Board()
        result = GameGraphVisualizationInterface.get_piece_name(
            board, 2055
        )
        assert result == "飛"


class TestGetAnalyticsData:
    """get_analytics_data のテスト."""

    def test_empty_analytics(self) -> None:
        """子がないノードは空のデータを返す."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        data = viz.get_analytics_data(100)
        assert data["moves"] == []
        assert data["probabilities"] == []
        assert data["win_rates"] == []


class TestGetBreadcrumbData:
    """get_breadcrumb_data のテスト."""

    def test_root_breadcrumb(self) -> None:
        """ルートノードのパンくずリストは初期局面のみ."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        breadcrumb = viz.get_breadcrumb_data(100)
        assert len(breadcrumb) == 1
        assert breadcrumb[0]["label"] == "初期局面"
        assert breadcrumb[0]["hash"] == "100"

    def test_child_breadcrumb(self) -> None:
        """子ノードのパンくずリストは2要素."""
        nodes, edges = _build_tree_with_edge()
        viz = GameGraphVisualizationInterface(nodes, edges)
        breadcrumb = viz.get_breadcrumb_data(200)
        assert len(breadcrumb) == 2
        assert breadcrumb[0]["label"] == "初期局面"
        assert breadcrumb[1]["hash"] == "200"
        # 7g7f → 日本語表記(歩を含む)
        assert "7六" in breadcrumb[1]["label"]

    def test_missing_node(self) -> None:
        """存在しないノードは空リストを返す．

        get_path_to_rootは未知のhashで空リストを返すため，
        パンくずリストも空になる．
        """
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        breadcrumb = viz.get_breadcrumb_data(999)
        assert len(breadcrumb) == 0


class TestGetOpeningName:
    """get_opening_name のテスト."""

    def test_root_has_no_opening(self) -> None:
        """ルートノードに定跡名はない."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        result = viz.get_opening_name(100)
        assert result is None

    def test_single_move_no_match(self) -> None:
        """1手のみでは定跡に一致しない(パターンが短すぎる場合を除く)."""
        nodes, edges = _build_tree_with_edge()
        viz = GameGraphVisualizationInterface(nodes, edges)
        result = viz.get_opening_name(200)
        # 7g7f のみではデフォルト定跡に一致しない
        assert result is None


class TestExportSfenPath:
    """export_sfen_path のテスト."""

    def test_root_sfen(self) -> None:
        """ルートノードのSFEN出力は初期局面."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        result = viz.export_sfen_path(100)
        assert result == "position startpos"

    def test_child_sfen(self) -> None:
        """子ノードのSFEN出力にmovesが含まれる."""
        nodes, edges = _build_tree_with_edge()
        viz = GameGraphVisualizationInterface(nodes, edges)
        result = viz.export_sfen_path(200)
        assert result.startswith("position startpos moves")
        assert "7g7f" in result

    def test_missing_node_returns_startpos(self) -> None:
        """存在しないノードはmovesなしの初期局面を返す．

        get_path_to_rootは未知のhashで空リストを返すため，
        指し手のない初期局面のposition文字列が返される．
        """
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        result = viz.export_sfen_path(999)
        assert result == "position startpos"


class TestExportSubgraphCsv:
    """export_subgraph_csv のテスト."""

    def test_csv_with_edges(self) -> None:
        """エッジのあるサブグラフをCSV出力する."""
        nodes, edges = _build_tree_with_edge()
        viz = GameGraphVisualizationInterface(nodes, edges)
        csv_content = viz.export_subgraph_csv(
            100, max_depth=3, min_probability=0.01
        )
        assert "parent_hash" in csv_content
        assert "child_hash" in csv_content
        assert "move" in csv_content
        lines = csv_content.strip().split("\n")
        assert len(lines) >= 2  # ヘッダー + データ行

    def test_csv_empty_tree(self) -> None:
        """エッジのないグラフのCSV出力はヘッダーのみ."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        csv_content = viz.export_subgraph_csv(
            100, max_depth=3, min_probability=0.01
        )
        lines = csv_content.strip().split("\n")
        assert len(lines) == 1  # ヘッダーのみ


class TestGetInitialSfen:
    """get_initial_sfen のテスト."""

    def test_default_returns_startpos(self) -> None:
        """initial_sfen=None の場合 "startpos" を返す."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        assert viz.get_initial_sfen() == "startpos"

    def test_custom_sfen(self) -> None:
        """initial_sfen が指定されている場合そのまま返す."""
        nodes, edges = _build_simple_tree()
        custom_sfen = (
            "lnsgkgsnl/1r5b1/ppppppppp/"
            "9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        )
        viz = GameGraphVisualizationInterface(
            nodes, edges, initial_sfen=custom_sfen
        )
        assert viz.get_initial_sfen() == custom_sfen

    def test_empty_string_returns_startpos(self) -> None:
        """initial_sfen が空文字列の場合 "startpos" を返す."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(
            nodes, edges, initial_sfen=""
        )
        assert viz.get_initial_sfen() == "startpos"


class TestNodeStatsWithOpening:
    """get_node_stats の定跡名統合テスト."""

    def test_stats_without_opening(self) -> None:
        """定跡に一致しないノードの統計には定跡キーがない."""
        nodes, edges = _build_simple_tree()
        viz = GameGraphVisualizationInterface(nodes, edges)
        stats = viz.get_node_stats(100)
        assert "定跡" not in stats
