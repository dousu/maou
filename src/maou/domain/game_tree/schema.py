"""ゲームツリーデータのPolarsスキーマ定義."""

from __future__ import annotations

import polars as pl


def get_game_tree_nodes_schema() -> dict[str, pl.DataType]:
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
    }


def get_game_tree_edges_schema() -> dict[str, pl.DataType]:
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
    }


def _create_empty_df(
    schema: dict[str, pl.DataType], size: int
) -> pl.DataFrame:
    """指定スキーマで空のDataFrameを生成する．"""
    if size == 0:
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(
        {
            col: pl.Series(
                values=[], dtype=dtype
            ).extend_constant(None, size)
            for col, dtype in schema.items()
        }
    )


def create_empty_nodes_df(size: int = 0) -> pl.DataFrame:
    """空の nodes DataFrame を生成する．

    Args:
        size: 作成する行数(デフォルト: 0)

    Returns:
        ノードスキーマを持つ空のDataFrame
    """
    return _create_empty_df(get_game_tree_nodes_schema(), size)


def create_empty_edges_df(size: int = 0) -> pl.DataFrame:
    """空の edges DataFrame を生成する．

    Args:
        size: 作成する行数(デフォルト: 0)

    Returns:
        エッジスキーマを持つ空のDataFrame
    """
    return _create_empty_df(get_game_tree_edges_schema(), size)
