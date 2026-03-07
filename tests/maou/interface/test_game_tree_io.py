"""ゲームツリーI/Oのテスト."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from maou.domain.game_tree.model import (
    GameTreeEdge,
    GameTreeNode,
)
from maou.interface.game_tree_io import (
    EDGES_FILENAME,
    NODES_FILENAME,
    GameTreeIO,
)


class TestGameTreeIO:
    """GameTreeIO のテスト."""

    def _sample_nodes(self) -> list[GameTreeNode]:
        return [
            GameTreeNode(
                position_hash=100,
                result_value=0.52,
                best_move_win_rate=0.53,
                num_branches=2,
                depth=0,
            ),
            GameTreeNode(
                position_hash=200,
                result_value=0.48,
                best_move_win_rate=0.49,
                num_branches=1,
                depth=1,
            ),
        ]

    def _sample_edges(self) -> list[GameTreeEdge]:
        return [
            GameTreeEdge(
                parent_hash=100,
                child_hash=200,
                move16=7654,
                move_label=53,
                probability=0.45,
                win_rate=0.52,
            ),
        ]

    def test_save_and_load_roundtrip(self) -> None:
        """save → load のラウンドトリップテスト."""
        io = GameTreeIO()
        nodes = self._sample_nodes()
        edges = self._sample_edges()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "tree"
            io.save(nodes, edges, output_dir)

            # ファイルが生成されている
            assert (output_dir / NODES_FILENAME).exists()
            assert (output_dir / EDGES_FILENAME).exists()

            # 読み込み
            nodes_df, edges_df = io.load(output_dir)

            assert len(nodes_df) == 2
            assert len(edges_df) == 1

            # 値の検証
            assert nodes_df["position_hash"].to_list() == [
                100,
                200,
            ]
            assert edges_df["parent_hash"].to_list() == [100]
            assert edges_df["child_hash"].to_list() == [200]

    def test_save_empty_data(self) -> None:
        """空データの保存・読み込み."""
        io = GameTreeIO()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "empty"
            io.save([], [], output_dir)

            nodes_df, edges_df = io.load(output_dir)
            assert len(nodes_df) == 0
            assert len(edges_df) == 0

    def test_load_missing_file(self) -> None:
        """ファイルが存在しない場合のエラー."""
        io = GameTreeIO()

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                io.load(Path(tmp))

    def test_creates_output_dir(self) -> None:
        """出力先ディレクトリが存在しない場合は作成する."""
        io = GameTreeIO()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "a" / "b" / "c"
            io.save([], [], output_dir)
            assert output_dir.exists()
