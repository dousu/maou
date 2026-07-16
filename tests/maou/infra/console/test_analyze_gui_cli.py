"""analyze-gui CLI コマンドのテスト．"""

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

pytest.importorskip("gradio")

from maou.infra.console.analyze_gui import (  # noqa: E402
    analyze_gui,
)

RESOURCES = (
    Path(__file__).parents[2] / "app" / "analysis" / "resources"
)
MINI_CSA = RESOURCES / "mini.csa"


class TestAnalyzeGuiCli:
    """analyze-gui コマンドのオプション検証テスト．"""

    def test_report_requires_input_path(
        self, tmp_path: Path
    ) -> None:
        """--report のみの指定はエラー．"""
        report = tmp_path / "report.json"
        report.write_text("{}", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            analyze_gui, ["--report", str(report)]
        )
        assert result.exit_code != 0
        assert "--input-path" in result.output

    def test_invalid_num_candidates(self) -> None:
        """--num-candidates 0 はエラー．"""
        runner = CliRunner()
        result = runner.invoke(
            analyze_gui,
            [
                "--input-path",
                str(MINI_CSA),
                "--num-candidates",
                "0",
            ],
        )
        assert result.exit_code != 0
        assert "--num-candidates" in result.output

    def test_launch_invoked_with_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """オプションがサーバー起動関数へ渡る (起動はモック)．"""
        recorded: dict[str, Any] = {}

        def fake_launch(**kwargs: Any) -> None:
            recorded.update(kwargs)

        monkeypatch.setattr(
            "maou.infra.visualization.analysis_gui_server."
            "launch_analysis_gui_server",
            fake_launch,
        )
        runner = CliRunner()
        result = runner.invoke(
            analyze_gui,
            [
                "--input-path",
                str(MINI_CSA),
                "--num-candidates",
                "7",
                "--port",
                "7999",
                "--server-name",
                "0.0.0.0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert recorded["kifu_path"] == MINI_CSA
        assert recorded["report_path"] is None
        assert recorded["num_candidates"] == 7
        assert recorded["port"] == 7999
        assert recorded["server_name"] == "0.0.0.0"
