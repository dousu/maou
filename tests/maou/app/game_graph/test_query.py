"""ゲームグラフ検索・フィルタリングのテスト."""

from __future__ import annotations

import polars as pl
import pytest

from maou.app.game_graph.query import GameGraphQuery
from maou.domain.game_graph.schema import (
    get_game_graph_edges_schema,
    get_game_graph_nodes_schema,
)


def _make_nodes(
    rows: list[dict],
) -> pl.DataFrame:
    """テスト用ノードDataFrameを生成する．"""
    return pl.DataFrame(
        rows, schema=get_game_graph_nodes_schema()
    )


def _make_edges(
    rows: list[dict],
) -> pl.DataFrame:
    """テスト用エッジDataFrameを生成する．"""
    return pl.DataFrame(
        rows, schema=get_game_graph_edges_schema()
    )


def _build_simple_tree() -> tuple[pl.DataFrame, pl.DataFrame]:
    """シンプルなテスト用グラフを構築する．

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


def _build_broken_chain_tree() -> tuple[
    pl.DataFrame, pl.DataFrame
]:
    """親チェーンが壊れた枝を含むグラフを構築する．

    ``_build_simple_tree()`` に 2 種類の壊れ方を足す．

    - 500 (depth=1): 入力エッジが一本もない孤児
    - 600 (depth=2): 入力エッジはあるが親 400 も depth=2 で，
      depth-1 に親がいない

    ビルダはこの形を出力しないので，実運用では到達しない．
    """
    nodes, edges = _build_simple_tree()
    extra_nodes = _make_nodes(
        [
            {
                "position_hash": 500,
                "result_value": 0.50,
                "best_move_win_rate": 0.51,
                "num_branches": 0,
                "depth": 1,
                "is_depth_cutoff": False,
            },
            {
                "position_hash": 600,
                "result_value": 0.49,
                "best_move_win_rate": 0.49,
                "num_branches": 0,
                "depth": 2,
                "is_depth_cutoff": False,
            },
        ]
    )
    extra_edges = _make_edges(
        [
            {
                "parent_hash": 400,
                "child_hash": 600,
                "move16": 1003,
                "move_label": 13,
                "probability": 0.5,
                "win_rate": 0.49,
                "is_leaf": False,
            },
        ]
    )
    return (
        pl.concat([nodes, extra_nodes]),
        pl.concat([edges, extra_edges]),
    )


class TestGetSubtree:
    """get_subgraph のテスト."""

    def test_depth_1(self) -> None:
        """depth=1ではルートと直接の子のみ取得する."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        sub_nodes, sub_edges = query.get_subgraph(
            100, max_depth=1
        )
        # ルート + 2子 = 3ノード
        assert len(sub_nodes) == 3
        # 2エッジ(ルートから子へ)
        assert len(sub_edges) == 2

    def test_depth_2(self) -> None:
        """depth=2では全ノードを取得する."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        sub_nodes, sub_edges = query.get_subgraph(
            100, max_depth=2
        )
        assert len(sub_nodes) == 4
        assert len(sub_edges) == 3

    def test_min_probability_filter(self) -> None:
        """min_probabilityでエッジをフィルタする."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        _sub_nodes, sub_edges = query.get_subgraph(
            100, max_depth=2, min_probability=0.5
        )
        # probability >= 0.5 のエッジのみ: ROOT->A (0.6)
        assert len(sub_edges) == 1
        assert sub_edges["child_hash"][0] == 200

    def test_empty_result(self) -> None:
        """存在しないハッシュからは空の結果を返す."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        _sub_nodes, sub_edges = query.get_subgraph(
            999, max_depth=3
        )
        # 存在しないノードでも visited に含まれる
        assert len(sub_edges) == 0


class TestGetNodeDetail:
    """get_node_detail のテスト."""

    def test_existing_node(self) -> None:
        """存在するノードの詳細を取得する."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        detail = query.get_node_detail(100)
        assert detail["position_hash"] == 100
        assert detail["depth"] == 0
        assert detail["num_branches"] == 2

    def test_missing_node(self) -> None:
        """存在しないノードは空辞書を返す."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        assert query.get_node_detail(999) == {}


class TestGetChildren:
    """get_children のテスト."""

    def test_children_sorted_by_probability(self) -> None:
        """子エッジが確率降順でソートされる."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        children = query.get_children(100)
        assert len(children) == 2
        # 0.6 > 0.4 の順
        assert children["probability"][0] == pytest.approx(0.6)
        assert children["probability"][1] == pytest.approx(0.4)

    def test_no_children(self) -> None:
        """子がないノードは空DataFrameを返す."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        children = query.get_children(400)
        assert len(children) == 0


class TestGetPathToRoot:
    """get_path_to_root のテスト."""

    def test_root_path(self) -> None:
        """ルート自身のパスはルートのみ."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        path = query.get_path_to_root(100)
        assert path == [100]

    def test_depth_1_path(self) -> None:
        """depth=1ノードのパスはルート→ノード."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        path = query.get_path_to_root(200)
        assert path == [100, 200]

    def test_depth_2_path(self) -> None:
        """depth=2ノードのパスはルート→中間→ノード."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        path = query.get_path_to_root(400)
        assert path == [100, 200, 400]

    def test_missing_node_returns_empty(self) -> None:
        """グラフにないノードは例外でなく空リストを返す."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        assert query.get_path_to_root(999) == []

    def test_orphan_node_raises(self) -> None:
        """depth>0 なのに入力エッジがないノードは例外."""
        nodes, edges = _build_broken_chain_tree()
        query = GameGraphQuery(nodes, edges)
        with pytest.raises(
            ValueError, match="no incoming edge"
        ):
            query.get_path_to_root(500)

    def test_parent_at_wrong_depth_raises(self) -> None:
        """親が depth-1 にいないノードは例外."""
        nodes, edges = _build_broken_chain_tree()
        query = GameGraphQuery(nodes, edges)
        with pytest.raises(
            ValueError, match="no parent at depth 1"
        ):
            query.get_path_to_root(600)

    def test_never_returns_partial_path(self) -> None:
        """壊れたチェーンで先頭がルートでないパスを返さない.

        修正前は break して部分パスを reversed で返していた
        ため，``path[0]`` がルート(depth 0)でないリストが
        呼び出し側に届いていた．
        """
        nodes, edges = _build_broken_chain_tree()
        query = GameGraphQuery(nodes, edges)
        for broken in (500, 600):
            with pytest.raises(ValueError):
                query.get_path_to_root(broken)

    def test_broken_chain_not_cached(self) -> None:
        """例外時に部分パスをキャッシュしない."""
        nodes, edges = _build_broken_chain_tree()
        query = GameGraphQuery(nodes, edges)
        for _ in range(2):
            with pytest.raises(ValueError):
                query.get_path_to_root(600)
        assert 600 not in query._path_cache

    def test_intact_chain_unaffected(self) -> None:
        """壊れた枝が同居していても健全な枝は従来通り."""
        nodes, edges = _build_broken_chain_tree()
        query = GameGraphQuery(nodes, edges)
        assert query.get_path_to_root(400) == [100, 200, 400]


class TestGetEdgeBetween:
    """get_edge_between のテスト."""

    def test_existing_edge(self) -> None:
        """存在するエッジの情報を取得する."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        edge = query.get_edge_between(100, 200)
        assert edge is not None
        assert edge["move16"] == 1000
        assert edge["probability"] == pytest.approx(0.6)

    def test_missing_edge(self) -> None:
        """存在しないエッジはNoneを返す."""
        nodes, edges = _build_simple_tree()
        query = GameGraphQuery(nodes, edges)
        assert query.get_edge_between(100, 400) is None
