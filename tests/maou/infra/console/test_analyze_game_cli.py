"""`maou analyze-game` CLI のテスト．"""

import json
from pathlib import Path

from click.testing import CliRunner

from maou.infra.console.analyze_game import analyze_game

MINI_CSA = (
    Path(__file__).parents[2]
    / "app"
    / "analysis"
    / "resources"
    / "mini.csa"
)


class TestAnalyzeGameCli:
    def test_expected_options_exist(self) -> None:
        param_names = [p.name for p in analyze_game.params]
        for expected in [
            "input_path",
            "input_format",
            "model_path",
            "time_ms",
            "total_time_ms",
            "playouts",
            "num_candidates",
            "output",
            "threads",
            "batch_size",
            "root_dfpn",
            "leaf_mate",
            "cuda",
            "tensorrt",
            "trt_cache_dir",
        ]:
            assert expected in param_names
        input_path_param = next(
            p
            for p in analyze_game.params
            if p.name == "input_path"
        )
        assert input_path_param.required is True

    def test_writes_json_and_prints_summary(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(
            analyze_game,
            [
                "--input-path",
                str(MINI_CSA),
                "--playouts",
                "4",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Match rate:" in result.output
        assert "black_engine (black)" in result.output
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["input"]["n_moves"] == 4
        assert len(data["positions"]) == 4
        assert data["input"]["format"] == "csa"

    def test_json_to_stdout_without_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            analyze_game,
            ["--input-path", str(MINI_CSA), "--playouts", "4"],
        )
        assert result.exit_code == 0, result.output
        # tqdm の progress 行 (stderr 由来) が混ざり得るため，JSON の開始
        # 位置から末尾までをパースする
        json_start = result.output.index("{")
        data = json.loads(result.output[json_start:])
        assert len(data["positions"]) == 4

    def test_budget_options_are_exclusive(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            analyze_game,
            [
                "--input-path",
                str(MINI_CSA),
                "--playouts",
                "4",
                "--time-ms",
                "100",
            ],
        )
        assert result.exit_code != 0
        assert "at most one" in result.output

    def test_cuda_without_model_path_is_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            analyze_game,
            ["--input-path", str(MINI_CSA), "--cuda"],
        )
        assert result.exit_code != 0
        assert "require --model-path" in result.output
