"""`maou search` CLI のテスト．"""

from click.testing import CliRunner

from maou.infra.console.search_board import search_board

MATE_IN_1 = "4k4/9/4P4/9/9/9/9/9/9 b G 1"


class TestSearchBoardCli:
    def test_expected_options_exist(self) -> None:
        param_names = [p.name for p in search_board.params]
        for expected in [
            "sfen",
            "moves",
            "model_path",
            "threads",
            "batch_size",
            "playouts",
            "time_ms",
            "num_moves",
            "root_dfpn",
            "cuda",
            "tensorrt",
            "trt_cache_dir",
        ]:
            assert expected in param_names
        sfen_param = next(
            p for p in search_board.params if p.name == "sfen"
        )
        assert sfen_param.required is True

    def test_search_outputs_eval_and_winrate(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            search_board,
            ["--sfen", MATE_IN_1, "--playouts", "2000"],
        )
        assert result.exit_code == 0
        assert "Bestmove: G*5b" in result.output
        assert "Eval:" in result.output
        assert "WinRate: 1.0000" in result.output
        assert "PV: G*5b" in result.output
        assert "stop=root_proven" in result.output

    def test_cuda_without_model_path_is_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            search_board, ["--sfen", MATE_IN_1, "--cuda"]
        )
        assert "Bestmove" not in result.output
