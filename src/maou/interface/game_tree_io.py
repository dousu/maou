"""ゲームツリーデータのI/O."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl

from maou.domain.game_tree.model import (
    GameTreeEdge,
    GameTreeNode,
)
from maou.domain.game_tree.schema import (
    get_game_tree_edges_schema,
    get_game_tree_nodes_schema,
)

NODES_FILENAME = "nodes.feather"
EDGES_FILENAME = "edges.feather"


class GameTreeIO:
    """ゲームツリーデータのI/O."""

    def save(
        self,
        nodes: list[GameTreeNode],
        edges: list[GameTreeEdge],
        output_dir: Path,
    ) -> None:
        """nodes.feather, edges.feather を Arrow IPC (LZ4圧縮) で出力する．

        Args:
            nodes: ノードのリスト
            edges: エッジのリスト
            output_dir: 出力先ディレクトリ
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        nodes_df = pl.DataFrame(
            [dataclasses.asdict(n) for n in nodes],
            schema=get_game_tree_nodes_schema(),
        )

        edges_df = pl.DataFrame(
            [dataclasses.asdict(e) for e in edges],
            schema=get_game_tree_edges_schema(),
        )

        nodes_df.write_ipc(
            output_dir / NODES_FILENAME, compression="lz4"
        )
        edges_df.write_ipc(
            output_dir / EDGES_FILENAME, compression="lz4"
        )

    def load(
        self, tree_dir: Path
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """nodes.feather, edges.feather を読み込む．

        Args:
            tree_dir: ツリーデータのディレクトリ

        Returns:
            (nodes_df, edges_df) のタプル

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: スキーマが一致しない場合
        """
        nodes_path = tree_dir / NODES_FILENAME
        edges_path = tree_dir / EDGES_FILENAME

        try:
            nodes_df = pl.read_ipc(nodes_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{NODES_FILENAME} が見つかりません: {nodes_path}"
            ) from None
        try:
            edges_df = pl.read_ipc(edges_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{EDGES_FILENAME} が見つかりません: {edges_path}"
            ) from None

        # スキーマ検証
        expected_nodes_cols = set(
            get_game_tree_nodes_schema().keys()
        )
        actual_nodes_cols = set(nodes_df.columns)
        if expected_nodes_cols != actual_nodes_cols:
            raise ValueError(
                f"nodes.feather のカラムが不正: "
                f"期待={expected_nodes_cols}, 実際={actual_nodes_cols}"
            )

        expected_edges_cols = set(
            get_game_tree_edges_schema().keys()
        )
        actual_edges_cols = set(edges_df.columns)
        if expected_edges_cols != actual_edges_cols:
            raise ValueError(
                f"edges.feather のカラムが不正: "
                f"期待={expected_edges_cols}, 実際={actual_edges_cols}"
            )

        return nodes_df, edges_df
