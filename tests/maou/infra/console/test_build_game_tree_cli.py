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

    def test_initial_hash_without_sfen(self) -> None:
        """--initial-hash のみ指定時にエラーになる."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input"
            input_path.mkdir()
            output_dir = Path(tmp) / "output"

            result = runner.invoke(
                build_game_tree,
                [
                    "--input-path",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--initial-hash",
                    "12345",
                ],
            )

            assert result.exit_code == 1
            assert "--initial-sfen" in result.output

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

            assert result.exit_code == 1
            assert "feather" in result.output
            assert not (output_dir / "nodes.feather").exists()

    def test_nested_directory_collection(self) -> None:
        """ネストされたディレクトリからも .feather ファイルを収集する."""
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
            # ネストされたディレクトリに配置
            nested_dir = Path(tmp) / "input" / "sub1" / "sub2"
            nested_dir.mkdir(parents=True)
            output_dir = Path(tmp) / "output"

            df.write_ipc(
                nested_dir / "data.feather",
                compression="lz4",
            )

            # 非.featherファイルも配置(無視されるべき)
            (Path(tmp) / "input" / "readme.txt").write_text(
                "ignore"
            )

            result = runner.invoke(
                build_game_tree,
                [
                    "--input-path",
                    str(Path(tmp) / "input"),
                    "--output-dir",
                    str(output_dir),
                    "--max-depth",
                    "0",
                ],
            )

            assert result.exit_code == 0
            assert "完了" in result.output
            assert (output_dir / "nodes.feather").exists()

    def test_single_file_input(self) -> None:
        """単一ファイルを直接入力できる."""
        runner = CliRunner()

        board = shogi.Board()
        df = pl.DataFrame(
            {
                "id": pl.Series(
                    [board.hash()], dtype=pl.UInt64
                ),
                "moveLabel": [[0.0] * MOVE_LABELS_NUM],
                "moveWinRate": [[0.0] * MOVE_LABELS_NUM],
                "resultValue": pl.Series(
                    [0.5], dtype=pl.Float32
                ),
                "bestMoveWinRate": pl.Series(
                    [0.5], dtype=pl.Float32
                ),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "single.feather"
            output_dir = Path(tmp) / "output"

            df.write_ipc(input_file, compression="lz4")

            result = runner.invoke(
                build_game_tree,
                [
                    "--input-path",
                    str(input_file),
                    "--output-dir",
                    str(output_dir),
                    "--max-depth",
                    "0",
                ],
            )

            assert result.exit_code == 0
            assert "完了" in result.output
