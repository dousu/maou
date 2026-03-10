"""GameGraphLayoutService のテスト．"""

from __future__ import annotations

import polars as pl
import pytest

from maou.app.game_graph.layout import (
    GameGraphLayoutService,
)
from maou.domain.game_graph.schema import (
    get_game_graph_edges_schema,
    get_game_graph_nodes_schema,
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


def _default_node(position_hash: int, depth: int) -> dict:
    """テスト用のデフォルトノード行を作成する．"""
    return {
        "position_hash": position_hash,
        "result_value": 0.5,
        "best_move_win_rate": 0.5,
        "num_branches": 0,
        "depth": depth,
        "is_depth_cutoff": False,
    }


def _default_edge(
    parent_hash: int,
    child_hash: int,
    probability: float = 0.5,
) -> dict:
    """テスト用のデフォルトエッジ行を作成する．"""
    return {
        "parent_hash": parent_hash,
        "child_hash": child_hash,
        "move16": 0,
        "move_label": 0,
        "probability": probability,
        "win_rate": 0.5,
        "is_leaf": False,
    }


class TestComputeLayout:
    """compute_layout のテスト．"""

    def test_empty(self) -> None:
        """空の DataFrame でエラーにならない．"""
        nodes = _make_nodes([])
        edges = _make_edges([])
        svc = GameGraphLayoutService()
        layout = svc.compute_layout(nodes, edges, 0)
        assert layout.node_positions == {}
        assert layout.bounds == (0, 0, 0, 0)

    def test_single_node(self) -> None:
        """単一ノード(ルートのみ)の座標が (0, 0) になる．"""
        nodes = _make_nodes([_default_node(100, 0)])
        edges = _make_edges([])
        svc = GameGraphLayoutService()
        layout = svc.compute_layout(nodes, edges, 100)
        assert layout.node_positions[100] == (0.0, 0.0)
        assert layout.bounds == (0.0, 0.0, 0.0, 0.0)

    def test_linear_chain(self) -> None:
        """線形チェーン: 各 depth の Y 座標が正しい．"""
        nodes = _make_nodes(
            [
                _default_node(1, 0),
                _default_node(2, 1),
                _default_node(3, 2),
            ]
        )
        edges = _make_edges(
            [
                _default_edge(1, 2, 1.0),
                _default_edge(2, 3, 1.0),
            ]
        )
        svc = GameGraphLayoutService()
        layout = svc.compute_layout(
            nodes, edges, 1, rank_spacing=100.0
        )
        assert layout.node_positions[1][1] == 0.0
        assert layout.node_positions[2][1] == 100.0
        assert layout.node_positions[3][1] == 200.0
        # 全て X=0 (一本道)
        assert layout.node_positions[1][0] == 0.0
        assert layout.node_positions[2][0] == 0.0
        assert layout.node_positions[3][0] == 0.0

    def test_binary_tree(self) -> None:
        """二分木: 子が親の X を中心に対称配置される．"""
        nodes = _make_nodes(
            [
                _default_node(1, 0),
                _default_node(2, 1),
                _default_node(3, 1),
            ]
        )
        edges = _make_edges(
            [
                _default_edge(1, 2, 0.6),
                _default_edge(1, 3, 0.4),
            ]
        )
        svc = GameGraphLayoutService()
        layout = svc.compute_layout(
            nodes, edges, 1, sibling_spacing=60.0
        )
        x2 = layout.node_positions[2][0]
        x3 = layout.node_positions[3][0]
        # 親 x=0 を中心に配置
        assert x2 == pytest.approx(-30.0)
        assert x3 == pytest.approx(30.0)
        # 確率降順: hash=2 (prob=0.6) が左
        assert x2 < x3

    def test_multiple_parents(self) -> None:
        """複数の親を持つノードでもエラーにならない．"""
        nodes = _make_nodes(
            [
                _default_node(1, 0),
                _default_node(2, 1),
                _default_node(3, 1),
                _default_node(4, 2),  # 2 と 3 の両方から辺
            ]
        )
        edges = _make_edges(
            [
                _default_edge(1, 2, 0.6),
                _default_edge(1, 3, 0.4),
                _default_edge(2, 4, 0.8),
                _default_edge(3, 4, 0.3),
            ]
        )
        svc = GameGraphLayoutService()
        layout = svc.compute_layout(nodes, edges, 1)
        # ノード 4 は primary parent=2 (確率 0.8 > 0.3) の下に配置
        assert 4 in layout.node_positions
        # primary parent (hash=2) の X に近い位置
        x2 = layout.node_positions[2][0]
        x4 = layout.node_positions[4][0]
        assert x4 == pytest.approx(x2)

    def test_overlap_resolution(self) -> None:
        """重なりが解消される．"""
        svc = GameGraphLayoutService()
        placed = [(1, 0.0), (2, 5.0), (3, 10.0)]
        result = svc._resolve_overlaps(placed, 40.0)
        # 間隔が 40 未満のペアは押し広げられる
        for i in range(1, len(result)):
            assert result[i][1] - result[i - 1][1] >= 40.0

    def test_large_tree_performance(self) -> None:
        """1000ノードのレイアウトが1秒以内に完了する．"""
        import time

        node_rows = [
            _default_node(i, min(i, 20)) for i in range(1000)
        ]
        edge_rows = []
        for i in range(1, 1000):
            parent = max(0, i - 1)
            edge_rows.append(
                _default_edge(
                    parent, i, 1.0 / max(1, i % 5 + 1)
                )
            )
        nodes = _make_nodes(node_rows)
        edges = _make_edges(edge_rows)

        svc = GameGraphLayoutService()
        start = time.monotonic()
        layout = svc.compute_layout(nodes, edges, 0)
        elapsed = time.monotonic() - start

        assert len(layout.node_positions) == 1000
        assert elapsed < 1.0

    def test_all_nodes_have_positions(self) -> None:
        """全ノードに座標が割り当てられる．"""
        nodes = _make_nodes(
            [
                _default_node(1, 0),
                _default_node(2, 1),
                _default_node(3, 2),
                _default_node(4, 1),
            ]
        )
        edges = _make_edges(
            [
                _default_edge(1, 2, 0.7),
                _default_edge(1, 4, 0.3),
                _default_edge(2, 3, 1.0),
            ]
        )
        svc = GameGraphLayoutService()
        layout = svc.compute_layout(nodes, edges, 1)
        for row in nodes.iter_rows(named=True):
            h = row["position_hash"]
            assert h in layout.node_positions, (
                f"Node {h} missing from positions"
            )

    def test_back_edge_cycle_no_infinite_loop(self) -> None:
        """閉路(バックエッジ)を持つグラフで無限ループにならない．"""
        # depth=0 → depth=1 → depth=0 の循環エッジを含むグラフ
        nodes = _make_nodes(
            [
                _default_node(1, 0),
                _default_node(2, 1),
            ]
        )
        edges = _make_edges(
            [
                _default_edge(1, 2, 0.8),  # フォワードエッジ
                _default_edge(2, 1, 0.2),  # バックエッジ(循環)
            ]
        )
        svc = GameGraphLayoutService()
        layout = svc.compute_layout(nodes, edges, 1)
        assert 1 in layout.node_positions
        assert 2 in layout.node_positions
        # Y座標が depth に基づいている
        assert layout.node_positions[1][1] == 0.0
        assert layout.node_positions[2][1] == pytest.approx(
            80.0
        )

    def test_bounds_correct(self) -> None:
        """bounds が全ノードの座標を包含する．"""
        nodes = _make_nodes(
            [
                _default_node(1, 0),
                _default_node(2, 1),
                _default_node(3, 1),
            ]
        )
        edges = _make_edges(
            [
                _default_edge(1, 2, 0.6),
                _default_edge(1, 3, 0.4),
            ]
        )
        svc = GameGraphLayoutService()
        layout = svc.compute_layout(nodes, edges, 1)
        min_x, min_y, max_x, max_y = layout.bounds
        for x, y in layout.node_positions.values():
            assert x >= min_x
            assert x <= max_x
            assert y >= min_y
            assert y <= max_y
