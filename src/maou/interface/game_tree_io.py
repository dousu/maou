"""ゲームツリーデータのI/O(Rustバックエンド使用)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import cast

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


def _save_df_feather(df: pl.DataFrame, path: Path) -> None:
    """DataFrameをRust I/Oで保存する．空DataFrameはPolarsで保存する．

    Args:
        df: 保存するDataFrame
        path: 出力ファイルパス
    """
    if len(df) == 0:
        df.write_ipc(path, compression="lz4")
        return

    from maou._rust.maou_io import save_feather_file
    from maou.domain.data.rust_io import _df_to_single_batch

    save_feather_file(
        _df_to_single_batch(df),
        str(path),
    )


def _load_df_feather(path: Path) -> pl.DataFrame:
    """Rust I/OでfeatherファイルをDataFrameとして読み込む．

    空ファイル(0レコード)の場合はPolarsで読み込む．

    Args:
        path: 入力ファイルパス

    Returns:
        読み込んだDataFrame
    """
    from maou._rust.maou_io import load_feather_file

    try:
        arrow_batch = load_feather_file(str(path))
    except OSError:
        # 空ファイル(0 record batches)はRust I/Oで読めないためPolarsで読み込む
        return pl.read_ipc(path)
    return cast(pl.DataFrame, pl.from_arrow(arrow_batch))


class GameTreeIO:
    """ゲームツリーデータのI/O(Rustバックエンド使用)."""

    def save(
        self,
        nodes: list[GameTreeNode],
        edges: list[GameTreeEdge],
        output_dir: Path,
    ) -> None:
        """nodes.feather, edges.feather を Rust I/O で出力する．

        Args:
            nodes: ノードのリスト
            edges: エッジのリスト
            output_dir: 出力先ディレクトリ
        """
        from maou.domain.data.rust_io import _check_rust_backend

        _check_rust_backend()
        output_dir.mkdir(parents=True, exist_ok=True)

        nodes_df = pl.DataFrame(
            [dataclasses.asdict(n) for n in nodes],
            schema=get_game_tree_nodes_schema(),
        )

        edges_df = pl.DataFrame(
            [dataclasses.asdict(e) for e in edges],
            schema=get_game_tree_edges_schema(),
        )

        _save_df_feather(nodes_df, output_dir / NODES_FILENAME)
        _save_df_feather(edges_df, output_dir / EDGES_FILENAME)

    def load(
        self, tree_dir: Path
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """nodes.feather, edges.feather を Rust I/O で読み込む．

        Args:
            tree_dir: ツリーデータのディレクトリ

        Returns:
            (nodes_df, edges_df) のタプル

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: スキーマが一致しない場合
        """
        from maou.domain.data.rust_io import _check_rust_backend

        _check_rust_backend()

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

        nodes_df = _load_df_feather(nodes_path)
        edges_df = _load_df_feather(edges_path)

        # スキーマ検証(カラム名 + データ型)
        self._validate_schema(
            nodes_df,
            get_game_tree_nodes_schema(),
            NODES_FILENAME,
        )
        self._validate_schema(
            edges_df,
            get_game_tree_edges_schema(),
            EDGES_FILENAME,
        )

        return nodes_df, edges_df

    @staticmethod
    def _validate_schema(
        df: pl.DataFrame,
        expected_schema: dict[str, pl.DataType],
        filename: str,
    ) -> None:
        """DataFrameのカラム名とデータ型を検証する．

        Args:
            df: 検証対象のDataFrame
            expected_schema: 期待するスキーマ(カラム名 → データ型)
            filename: エラーメッセージ用のファイル名

        Raises:
            ValueError: カラム名またはデータ型が一致しない場合
        """
        expected_cols = set(expected_schema.keys())
        actual_cols = set(df.columns)
        if expected_cols != actual_cols:
            raise ValueError(
                f"{filename} のカラムが不正: "
                f"期待={expected_cols}, 実際={actual_cols}"
            )

        for col, expected_dtype in expected_schema.items():
            actual_dtype = df[col].dtype
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"{filename} のカラム '{col}' の型が不正: "
                    f"期待={expected_dtype}, 実際={actual_dtype}"
                )
