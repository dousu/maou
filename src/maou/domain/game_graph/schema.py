"""ゲームグラフデータのPolarsスキーマ定義．"""

from __future__ import annotations

import polars as pl


def get_game_graph_nodes_schema() -> dict[str, pl.DataType]:
    """nodes.feather のPolarsスキーマを返す．

    Returns:
        dict[str, pl.DataType]: ノードデータのスキーマ定義
    """
    return {
        "position_hash": pl.UInt64(),
        "result_value": pl.Float32(),
        "best_move_win_rate": pl.Float32(),
        "num_branches": pl.UInt16(),
        "depth": pl.UInt16(),
        "is_depth_cutoff": pl.Boolean(),
    }


def get_game_graph_edges_schema() -> dict[str, pl.DataType]:
    """edges.feather のPolarsスキーマを返す．

    Returns:
        dict[str, pl.DataType]: エッジデータのスキーマ定義
    """
    return {
        "parent_hash": pl.UInt64(),
        "child_hash": pl.UInt64(),
        "move16": pl.UInt16(),
        "move_label": pl.UInt16(),
        "probability": pl.Float32(),
        "win_rate": pl.Float32(),
        "is_leaf": pl.Boolean(),
    }
