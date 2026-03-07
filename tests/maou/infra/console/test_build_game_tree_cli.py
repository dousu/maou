"""build-game-tree CLI コマンドのスモークテスト."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
from click.testing import CliRunner

from maou.domain.board import shogi
from maou.domain.move.label import MOVE_LABELS_NUM
from maou.infra.console.build_game_tree import build_game_tree


class TestBuildGameTreeCLI:
    """build-game-tree CLI のテスト."""

    def test_help(self) -> None:
        """--help が正常に表示される."""
        runner = CliRunner()
        result = runner.invoke(build_game_tree, ["--help"])
        assert result.exit_code == 0
        assert "--input-path" in result.output
        assert "--output-dir" in result.output

    def test_basic_execution(self) -> None:
        """最小限のデータで正常に実行できる."""
        runner = CliRunner()

        board = shogi.Board()
        move_labels = [0.0] * MOVE_LABELS_NUM
        move_win_rates = [0.0] * MOVE_LABELS_NUM

        df = pl.DataFrame(
            {
                "id": pl.Series(
                    [board.hash()], dtype=pl.UInt64
                ),
                "moveLabel": [move_labels],
                "moveWinRate": [move_win_rates],
                "resultValue": pl.Series(
                    [0.5], dtype=pl.Float32
                ),
                "bestMoveWinRate": pl.Series(
                    [0.5], dtype=pl.Float32
                ),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input"
            input_path.mkdir()
            output_dir = Path(tmp) / "output"

            # 入力ファイル作成
            df.write_ipc(
                input_path / "test.feather",
                compression="lz4",
            )

            result = runner.invoke(
                build_game_tree,
                [
                    "--input-path",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--max-depth",
                    "1",
                ],
            )

            assert result.exit_code == 0
            assert "完了" in result.output
            assert (output_dir / "nodes.feather").exists()
            assert (output_dir / "edges.feather").exists()

    def test_no_feather_files(self) -> None:
        """入力パスに .feather ファイルがない場合のエラー."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "empty_input"
            input_path.mkdir()
            output_dir = Path(tmp) / "output"

            result = runner.invoke(
                build_game_tree,
                [
                    "--input-path",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            # handle_exception でキャッチされるため exit_code は 0
            # 出力ファイルが生成されていないことで確認
            assert not (output_dir / "nodes.feather").exists()
