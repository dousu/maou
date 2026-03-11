"""ゲームグラフスキーマのテスト."""

from __future__ import annotations

import polars as pl

from maou.domain.game_graph.schema import (
    create_empty_edges_df,
    create_empty_nodes_df,
    get_game_graph_edges_schema,
    get_game_graph_nodes_schema,
)


class TestNodesSchema:
    """nodes スキーマのテスト."""

    def test_schema_columns(self) -> None:
        """必要なカラムが全て含まれる."""
        schema = get_game_graph_nodes_schema()
        assert set(schema.keys()) == {
            "position_hash",
            "result_value",
            "best_move_win_rate",
            "num_branches",
            "depth",
            "is_depth_cutoff",
        }

    def test_schema_types(self) -> None:
        """型が正しい."""
        schema = get_game_graph_nodes_schema()
        assert schema["position_hash"] == pl.UInt64()
        assert schema["result_value"] == pl.Float32()
        assert schema["best_move_win_rate"] == pl.Float32()
        assert schema["num_branches"] == pl.UInt16()
        assert schema["depth"] == pl.UInt16()
        assert schema["is_depth_cutoff"] == pl.Boolean()

    def test_create_empty_df(self) -> None:
        """空の DataFrame を生成できる."""
        df = create_empty_nodes_df()
        assert len(df) == 0
        assert set(df.columns) == set(
            get_game_graph_nodes_schema().keys()
        )

    def test_create_empty_df_with_size(self) -> None:
        """指定サイズの DataFrame を生成できる."""
        df = create_empty_nodes_df(5)
        assert len(df) == 5


class TestEdgesSchema:
    """edges スキーマのテスト."""

    def test_schema_columns(self) -> None:
        """必要なカラムが全て含まれる."""
        schema = get_game_graph_edges_schema()
        assert set(schema.keys()) == {
            "parent_hash",
            "child_hash",
            "move16",
            "move_label",
            "probability",
            "win_rate",
            "is_leaf",
        }

    def test_schema_types(self) -> None:
        """型が正しい."""
        schema = get_game_graph_edges_schema()
        assert schema["parent_hash"] == pl.UInt64()
        assert schema["child_hash"] == pl.UInt64()
        assert schema["move16"] == pl.UInt16()
        assert schema["move_label"] == pl.UInt16()
        assert schema["probability"] == pl.Float32()
        assert schema["win_rate"] == pl.Float32()
        assert schema["is_leaf"] == pl.Boolean()

    def test_create_empty_df(self) -> None:
        """空の DataFrame を生成できる."""
        df = create_empty_edges_df()
        assert len(df) == 0
        assert set(df.columns) == set(
            get_game_graph_edges_schema().keys()
        )

    def test_create_empty_df_with_size(self) -> None:
        """指定サイズの DataFrame を生成できる."""
        df = create_empty_edges_df(3)
        assert len(df) == 3
