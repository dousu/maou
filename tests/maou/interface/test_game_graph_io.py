"""ゲームグラフI/Oのテスト."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from maou.domain.game_graph.model import (
    GameGraphEdge,
    GameGraphNode,
)
from maou.interface.game_graph_io import (
    EDGES_FILENAME,
    NODES_FILENAME,
    GameGraphIO,
)


class TestGameGraphIO:
    """GameGraphIO のテスト."""

    def _sample_nodes(self) -> list[GameGraphNode]:
        return [
            GameGraphNode(
                position_hash=100,
                result_value=0.52,
                best_move_win_rate=0.53,
                num_branches=2,
                depth=0,
                is_depth_cutoff=False,
            ),
            GameGraphNode(
                position_hash=200,
                result_value=0.48,
                best_move_win_rate=0.49,
                num_branches=1,
                depth=1,
                is_depth_cutoff=True,
            ),
        ]

    def _sample_edges(self) -> list[GameGraphEdge]:
        return [
            GameGraphEdge(
                parent_hash=100,
                child_hash=200,
                move16=7654,
                move_label=53,
                probability=0.45,
                win_rate=0.52,
                is_leaf=False,
            ),
        ]

    def test_save_and_load_roundtrip(self) -> None:
        """save → load のラウンドトリップテスト."""
        io = GameGraphIO()
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
        io = GameGraphIO()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "empty"
            io.save([], [], output_dir)

            nodes_df, edges_df = io.load(output_dir)
            assert len(nodes_df) == 0
            assert len(edges_df) == 0

    def test_load_missing_file(self) -> None:
        """ファイルが存在しない場合のエラー."""
        io = GameGraphIO()

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                io.load(Path(tmp))

    def test_creates_output_dir(self) -> None:
        """出力先ディレクトリが存在しない場合は作成する."""
        io = GameGraphIO()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "a" / "b" / "c"
            io.save([], [], output_dir)
            assert output_dir.exists()

    def test_load_schema_mismatch_nodes(self) -> None:
        """nodes.feather のカラムが不正な場合 ValueError."""
        io = GameGraphIO()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "tree"
            output_dir.mkdir()

            # 不正なカラムを持つ nodes.feather
            wrong_df = pl.DataFrame({"wrong_col": [1, 2, 3]})
            wrong_df.write_ipc(
                output_dir / NODES_FILENAME,
                compression="lz4",
            )

            # 正しい edges.feather
            io.save([], [], output_dir)
            # edgesは正しいままにする(nodesだけ壊す)
            wrong_df.write_ipc(
                output_dir / NODES_FILENAME,
                compression="lz4",
            )

            with pytest.raises(ValueError, match="カラム"):
                io.load(output_dir)

    def test_load_schema_mismatch_edges(self) -> None:
        """edges.feather のカラムが不正な場合 ValueError."""
        io = GameGraphIO()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "tree"

            # まず正しいデータを保存
            io.save(
                self._sample_nodes(),
                self._sample_edges(),
                output_dir,
            )

            # edges.feather を不正なカラムで上書き
            wrong_df = pl.DataFrame({"bad_column": [1]})
            wrong_df.write_ipc(
                output_dir / EDGES_FILENAME,
                compression="lz4",
            )

            with pytest.raises(ValueError, match="カラム"):
                io.load(output_dir)

    def test_roundtrip_preserves_dtypes(self) -> None:
        """save → load でデータ型が保持される."""
        from maou.domain.game_graph.schema import (
            get_game_graph_edges_schema,
            get_game_graph_nodes_schema,
        )

        io = GameGraphIO()
        nodes = self._sample_nodes()
        edges = self._sample_edges()

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "tree"
            io.save(nodes, edges, output_dir)
            nodes_df, edges_df = io.load(output_dir)

            expected_nodes = get_game_graph_nodes_schema()
            for col, dtype in expected_nodes.items():
                assert nodes_df[col].dtype == dtype

            expected_edges = get_game_graph_edges_schema()
            for col, dtype in expected_edges.items():
                assert edges_df[col].dtype == dtype
