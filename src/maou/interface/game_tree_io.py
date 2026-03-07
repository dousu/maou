"""ゲームツリーデータのI/O."""

from __future__ import annotations

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

        # nodes DataFrame
        nodes_df = pl.DataFrame(
            {
                "position_hash": pl.Series(
                    [n.position_hash for n in nodes],
                    dtype=pl.UInt64,
                ),
                "result_value": pl.Series(
                    [n.result_value for n in nodes],
                    dtype=pl.Float32,
                ),
                "best_move_win_rate": pl.Series(
                    [n.best_move_win_rate for n in nodes],
                    dtype=pl.Float32,
                ),
                "num_branches": pl.Series(
                    [n.num_branches for n in nodes],
                    dtype=pl.UInt16,
                ),
                "depth": pl.Series(
                    [n.depth for n in nodes],
                    dtype=pl.UInt16,
                ),
            }
        )

        # edges DataFrame
        edges_df = pl.DataFrame(
            {
                "parent_hash": pl.Series(
                    [e.parent_hash for e in edges],
                    dtype=pl.UInt64,
                ),
                "child_hash": pl.Series(
                    [e.child_hash for e in edges],
                    dtype=pl.UInt64,
                ),
                "move16": pl.Series(
                    [e.move16 for e in edges],
                    dtype=pl.UInt16,
                ),
                "move_label": pl.Series(
                    [e.move_label for e in edges],
                    dtype=pl.UInt16,
                ),
                "probability": pl.Series(
                    [e.probability for e in edges],
                    dtype=pl.Float32,
                ),
                "win_rate": pl.Series(
                    [e.win_rate for e in edges],
                    dtype=pl.Float32,
                ),
            }
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

        if not nodes_path.exists():
            raise FileNotFoundError(
                f"{NODES_FILENAME} が見つかりません: {nodes_path}"
            )
        if not edges_path.exists():
            raise FileNotFoundError(
                f"{EDGES_FILENAME} が見つかりません: {edges_path}"
            )

        nodes_df = pl.read_ipc(nodes_path)
        edges_df = pl.read_ipc(edges_path)

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
