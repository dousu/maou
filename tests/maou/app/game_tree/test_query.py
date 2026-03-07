"""ゲームツリー検索・フィルタリングのテスト."""

from __future__ import annotations

import polars as pl
import pytest

from maou.app.game_tree.query import GameTreeQuery
from maou.domain.game_tree.schema import (
    get_game_tree_edges_schema,
    get_game_tree_nodes_schema,
)


def _make_nodes(
    rows: list[dict],
) -> pl.DataFrame:
    """テスト用ノードDataFrameを生成する．"""
    return pl.DataFrame(
        rows, schema=get_game_tree_nodes_schema()
    )


def _make_edges(
    rows: list[dict],
) -> pl.DataFrame:
    """テスト用エッジDataFrameを生成する．"""
    return pl.DataFrame(
        rows, schema=get_game_tree_edges_schema()
    )


def _build_simple_tree() -> (
    tuple[pl.DataFrame, pl.DataFrame]
):
    """シンプルなテスト用ツリーを構築する．

    ROOT(depth=0) -> A(depth=1, prob=0.6) -> C(depth=2, prob=0.3)
                  -> B(depth=1, prob=0.4)
    """
    nodes = _make_nodes(
        [
            {
                "position_hash": 100,
                "result_value": 0.52,
                "best_move_win_rate": 0.53,
                "num_branches": 2,
                "depth": 0,
                "is_depth_cutoff": False,
            },
            {
                "position_hash": 200,
                "result_value": 0.48,
                "best_move_win_rate": 0.50,
                "num_branches": 1,
                "depth": 1,
                "is_depth_cutoff": False,
            },
            {
                "position_hash": 300,
                "result_value": 0.55,
                "best_move_win_rate": 0.56,
                "num_branches": 0,
                "depth": 1,
                "is_depth_cutoff": False,
            },
            {
                "position_hash": 400,
                "result_value": 0.45,
                "best_move_win_rate": 0.46,
                "num_branches": 0,
                "depth": 2,
                "is_depth_cutoff": False,
            },
        ]
    )
    edges = _make_edges(
        [
            {
                "parent_hash": 100,
                "child_hash": 200,
                "move16": 1000,
                "move_label": 10,
                "probability": 0.6,
                "win_rate": 0.52,
                "is_leaf": False,
            },
            {
                "parent_hash": 100,
                "child_hash": 300,
                "move16": 1001,
                "move_label": 11,
                "probability": 0.4,
                "win_rate": 0.50,
                "is_leaf": False,
            },
            {
                "parent_hash": 200,
                "child_hash": 400,
                "move16": 1002,
                "move_label": 12,
                "probability": 0.3,
                "win_rate": 0.48,
                "is_leaf": False,
            },
        ]
    )
    return nodes, edges


class TestGetSubtree:
    """get_subtree のテスト."""

    def test_depth_1(self) -> None:
        """depth=1ではルートと直接の子のみ取得する."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        sub_nodes, sub_edges = query.get_subtree(
            100, max_depth=1
        )
        # ルート + 2子 = 3ノード
        assert len(sub_nodes) == 3
        # 2エッジ(ルートから子へ)
        assert len(sub_edges) == 2

    def test_depth_2(self) -> None:
        """depth=2では全ノードを取得する."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        sub_nodes, sub_edges = query.get_subtree(
            100, max_depth=2
        )
        assert len(sub_nodes) == 4
        assert len(sub_edges) == 3

    def test_min_probability_filter(self) -> None:
        """min_probabilityでエッジをフィルタする."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        sub_nodes, sub_edges = query.get_subtree(
            100, max_depth=2, min_probability=0.5
        )
        # probability >= 0.5 のエッジのみ: ROOT->A (0.6)
        assert len(sub_edges) == 1
        assert sub_edges["child_hash"][0] == 200

    def test_empty_result(self) -> None:
        """存在しないハッシュからは空の結果を返す."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        sub_nodes, sub_edges = query.get_subtree(
            999, max_depth=3
        )
        # 存在しないノードでも visited に含まれる
        assert len(sub_edges) == 0


class TestGetNodeDetail:
    """get_node_detail のテスト."""

    def test_existing_node(self) -> None:
        """存在するノードの詳細を取得する."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        detail = query.get_node_detail(100)
        assert detail["position_hash"] == 100
        assert detail["depth"] == 0
        assert detail["num_branches"] == 2

    def test_missing_node(self) -> None:
        """存在しないノードは空辞書を返す."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        assert query.get_node_detail(999) == {}


class TestGetChildren:
    """get_children のテスト."""

    def test_children_sorted_by_probability(self) -> None:
        """子エッジが確率降順でソートされる."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        children = query.get_children(100)
        assert len(children) == 2
        # 0.6 > 0.4 の順
        assert children["probability"][0] == pytest.approx(
            0.6
        )
        assert children["probability"][1] == pytest.approx(
            0.4
        )

    def test_no_children(self) -> None:
        """子がないノードは空DataFrameを返す."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        children = query.get_children(400)
        assert len(children) == 0


class TestGetPathToRoot:
    """get_path_to_root のテスト."""

    def test_root_path(self) -> None:
        """ルート自身のパスはルートのみ."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        path = query.get_path_to_root(100)
        assert path == [100]

    def test_depth_1_path(self) -> None:
        """depth=1ノードのパスはルート→ノード."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        path = query.get_path_to_root(200)
        assert path == [100, 200]

    def test_depth_2_path(self) -> None:
        """depth=2ノードのパスはルート→中間→ノード."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        path = query.get_path_to_root(400)
        assert path == [100, 200, 400]


class TestGetEdgeBetween:
    """get_edge_between のテスト."""

    def test_existing_edge(self) -> None:
        """存在するエッジの情報を取得する."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        edge = query.get_edge_between(100, 200)
        assert edge is not None
        assert edge["move16"] == 1000
        assert edge["probability"] == pytest.approx(0.6)

    def test_missing_edge(self) -> None:
        """存在しないエッジはNoneを返す."""
        nodes, edges = _build_simple_tree()
        query = GameTreeQuery(nodes, edges)
        assert query.get_edge_between(100, 400) is None
