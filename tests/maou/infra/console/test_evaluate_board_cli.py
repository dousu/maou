"""`maou evaluate` CLI のテスト．

推論本体は Rust (`maou._rust.maou_search.evaluate`) にあるため，モデル未指定
時の mock 評価器で CLI 〜 interface の配線を検証する．
"""

from click.testing import CliRunner

from maou.infra.console.evaluate_board import evaluate_board

HIRATE = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/"
    "PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
)
# 後手番に合法手が無い局面 (頭金)
CHECKMATED = "4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1"


class TestEvaluateBoardCli:
    def test_expected_options_exist(self) -> None:
        param_names = [p.name for p in evaluate_board.params]
        for expected in [
            "sfen",
            "model_path",
            "num_moves",
            "cuda",
            "tensorrt",
            "trt_cache_dir",
        ]:
            assert expected in param_names
        sfen_param = next(
            p for p in evaluate_board.params if p.name == "sfen"
        )
        assert sfen_param.required is True

    def test_removed_options_are_gone(self) -> None:
        """Python TensorRT 経路と共に廃止したオプション．"""
        param_names = [p.name for p in evaluate_board.params]
        assert "model_type" not in param_names
        assert "engine_path" not in param_names
        assert "trt_workspace_size" not in param_names

    def test_outputs_policy_eval_and_winrate(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            evaluate_board, ["--sfen", HIRATE]
        )
        assert result.exit_code == 0
        assert "Policy:" in result.output
        assert "Eval:" in result.output
        assert "WinRate:" in result.output
        # 盤面表示
        assert "先手の持駒" in result.output

    def test_num_moves_limits_candidates(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            evaluate_board,
            ["--sfen", HIRATE, "--num-moves", "3"],
        )
        assert result.exit_code == 0
        policy_line = next(
            line
            for line in result.output.splitlines()
            if line.startswith("Policy:")
        )
        assert len(policy_line.split(", ")) == 3

    def test_candidates_are_legal_moves_only(self) -> None:
        """非合法ラベルが混ざらない (旧実装の 'failed to convert' 廃止)．"""
        runner = CliRunner()
        result = runner.invoke(
            evaluate_board,
            ["--sfen", HIRATE, "--num-moves", "30"],
        )
        assert result.exit_code == 0
        assert "failed to convert" not in result.output
        policy_line = next(
            line
            for line in result.output.splitlines()
            if line.startswith("Policy:")
        )
        # 平手初期局面の合法手は 30 手
        assert len(policy_line.split(", ")) == 30

    def test_checkmated_position_has_no_candidates(
        self,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            evaluate_board, ["--sfen", CHECKMATED]
        )
        assert result.exit_code == 0
        assert "Policy: (no legal moves)" in result.output

    def test_zero_num_moves_is_distinguished_from_checkmate(
        self,
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            evaluate_board,
            ["--sfen", HIRATE, "--num-moves", "0"],
        )
        assert result.exit_code == 0
        assert "no legal moves" not in result.output
        assert "30 legal moves" in result.output

    def test_negative_num_moves_is_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            evaluate_board,
            ["--sfen", HIRATE, "--num-moves", "-1"],
        )
        assert result.exit_code != 0
        assert "--num-moves" in result.output

    def test_cuda_requires_model_path(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            evaluate_board, ["--sfen", HIRATE, "--cuda"]
        )
        assert result.exit_code != 0
        assert (
            "--cuda / --tensorrt require --model-path."
            in result.output
        )

    def test_tensorrt_requires_model_path(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            evaluate_board, ["--sfen", HIRATE, "--tensorrt"]
        )
        assert result.exit_code != 0
        assert (
            "--cuda / --tensorrt require --model-path."
            in result.output
        )
