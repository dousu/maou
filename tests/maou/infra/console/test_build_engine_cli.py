"""build-engine CLIコマンドのテスト．"""

import click

from maou.infra.console.app import LAZY_COMMANDS
from maou.infra.console.build_engine import build_engine


class TestBuildEngineCLI:
    """build-engine コマンドのCLIオプションテスト．"""

    def test_build_engine_has_required_options(self) -> None:
        """--model-path と --output が必須オプションとして存在すること．"""
        assert isinstance(build_engine, click.Command)
        param_names = [p.name for p in build_engine.params]
        assert "model_path" in param_names
        assert "output" in param_names

        # 必須チェック
        for param in build_engine.params:
            if param.name == "model_path":
                assert param.required is True
            if param.name == "output":
                assert param.required is True

    def test_build_engine_has_workspace_size_option(
        self,
    ) -> None:
        """--trt-workspace-size オプションがデフォルト256で存在すること．"""
        param_names = [p.name for p in build_engine.params]
        assert "trt_workspace_size" in param_names

        for param in build_engine.params:
            if param.name == "trt_workspace_size":
                assert param.default == 256
                assert param.required is False

    def test_build_engine_command_registered(self) -> None:
        """app.py の LAZY_COMMANDS に build-engine が登録されていること．"""
        assert "build-engine" in LAZY_COMMANDS
        spec = LAZY_COMMANDS["build-engine"]
        assert (
            spec.module_path
            == "maou.infra.console.build_engine"
        )
        assert spec.attr_name == "build_engine"
