"""LazyGroup の依存チェック機能テスト．"""

from __future__ import annotations

from importlib.machinery import ModuleSpec
from typing import Callable
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from maou.infra.console.app import (
    LazyCommandSpec,
    LazyGroup,
    PackageRequirement,
)


@pytest.fixture
def runner() -> CliRunner:
    """Click テスト用ランナー．"""
    return CliRunner()


def _make_group(
    lazy_commands: dict[str, LazyCommandSpec],
) -> click.Group:
    """テスト用 LazyGroup を作成する．"""

    @click.group(cls=LazyGroup, lazy_commands=lazy_commands)
    def cli() -> None:
        pass

    return cli


def _mock_find_spec(
    missing_packages: set[str],
) -> Callable[[str], ModuleSpec | None]:
    """指定パッケージを不存在として扱う find_spec モックを返す．"""

    def _find_spec(name: str) -> ModuleSpec | None:
        if name in missing_packages:
            return None
        return MagicMock(spec=ModuleSpec)

    return _find_spec


class TestCheckPackages:
    """_check_packages メソッドのテスト．"""

    def test_single_missing_package(
        self, runner: CliRunner
    ) -> None:
        """1パッケージ不足時に具体的なエラーメッセージが表示される．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            required_packages=(
                PackageRequirement("torch", ("cpu", "cuda")),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec({"torch"}),
        ):
            result = runner.invoke(cli, ["test-cmd"])

        assert result.exit_code != 0
        assert "torch (not installed)" in result.output
        assert "uv sync --extra cpu" in result.output
        assert "uv sync --extra cuda" in result.output

    def test_multiple_missing_packages(
        self, runner: CliRunner
    ) -> None:
        """複数パッケージ不足時に全パッケージがリスト表示される．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            required_packages=(
                PackageRequirement("torch", ("cpu", "cuda")),
                PackageRequirement(
                    "torchinfo", ("cpu", "cuda")
                ),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec({"torch", "torchinfo"}),
        ):
            result = runner.invoke(cli, ["test-cmd"])

        assert result.exit_code != 0
        assert "torch (not installed)" in result.output
        assert "torchinfo (not installed)" in result.output

    def test_all_packages_present(
        self, runner: CliRunner
    ) -> None:
        """全パッケージが存在する場合はモジュール import に到達する．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            required_packages=(
                PackageRequirement("click", ("cpu",)),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec(set()),
        ):
            result = runner.invoke(cli, ["test-cmd"])

        # パッケージチェックは通過し "not installed" は表示されない
        # (fake.module は存在しないため別のエラーが出る)
        assert "not installed" not in result.output

    def test_extras_deduplication(
        self, runner: CliRunner
    ) -> None:
        """同一 extras が複数パッケージに指定されても重複なくリスト表示される．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            required_packages=(
                PackageRequirement("torch", ("cpu", "cuda")),
                PackageRequirement(
                    "torchinfo", ("cpu", "cuda")
                ),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec({"torch", "torchinfo"}),
        ):
            result = runner.invoke(cli, ["test-cmd"])

        # "uv sync --extra cpu" が1回だけ出現する
        assert result.output.count("uv sync --extra cpu") == 1
        assert result.output.count("uv sync --extra cuda") == 1

    def test_fallback_caching(self, runner: CliRunner) -> None:
        """同じコマンドを2回呼び出すとフォールバックが再利用される．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            required_packages=(
                PackageRequirement("torch", ("cpu",)),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec({"torch"}),
        ):
            result1 = runner.invoke(cli, ["test-cmd"])
            result2 = runner.invoke(cli, ["test-cmd"])

        assert result1.exit_code != 0
        assert result2.exit_code != 0
        assert "torch (not installed)" in result1.output
        assert "torch (not installed)" in result2.output

    def test_no_required_packages(
        self, runner: CliRunner
    ) -> None:
        """required_packages 未指定の場合は従来どおり import 試行する．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            missing_help="Custom missing help message.",
        )
        cli = _make_group({"test-cmd": spec})

        result = runner.invoke(cli, ["test-cmd"])

        # required_packages がないため _check_packages は呼ばれず，
        # import_module のフォールバックで missing_help が使われる
        assert "not installed" not in result.output

    def test_missing_help_not_used_when_required_packages_set(
        self, runner: CliRunner
    ) -> None:
        """required_packages 指定時は missing_help より新チェックが優先される．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            missing_help="Old fallback message should not appear.",
            required_packages=(
                PackageRequirement("torch", ("cpu",)),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec({"torch"}),
        ):
            result = runner.invoke(cli, ["test-cmd"])

        assert "Old fallback message" not in result.output
        assert "torch (not installed)" in result.output

    def test_error_message_format(
        self, runner: CliRunner
    ) -> None:
        """エラーメッセージが指定フォーマットに準拠する．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            required_packages=(
                PackageRequirement("torch", ("cpu", "cuda")),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec({"torch"}),
        ):
            result = runner.invoke(cli, ["test-cmd"])

        assert (
            "Command 'test-cmd' requires the following packages:"
            in result.output
        )
        assert "Install with one of:" in result.output


class TestHelpDisplay:
    """--help 表示時の挙動テスト．"""

    def test_help_shows_missing_packages(
        self, runner: CliRunner
    ) -> None:
        """--help 時にクラッシュせず不足パッケージ情報が表示される．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            required_packages=(
                PackageRequirement("torch", ("cpu", "cuda")),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec({"torch"}),
        ):
            result = runner.invoke(cli, ["test-cmd", "--help"])

        assert result.exit_code == 0
        assert "requires additional packages" in result.output
        assert "torch (not installed)" in result.output
        assert "uv sync --extra cpu" in result.output

    def test_help_with_all_packages_present(
        self, runner: CliRunner
    ) -> None:
        """全パッケージあり + --help 時は正常なコマンドヘルプが表示される．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            required_packages=(
                PackageRequirement("click", ("cpu",)),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec(set()),
        ):
            result = runner.invoke(cli, ["test-cmd", "--help"])

        # パッケージチェックは通過するため "requires additional packages" は表示されない
        assert (
            "requires additional packages" not in result.output
        )

    def test_group_help_shows_short_help(
        self, runner: CliRunner
    ) -> None:
        """親グループの --help で short_help にパッケージ名が表示される．"""
        spec = LazyCommandSpec(
            "fake.module",
            "fake_cmd",
            required_packages=(
                PackageRequirement("torch", ("cpu",)),
                PackageRequirement("torchinfo", ("cpu",)),
            ),
        )
        cli = _make_group({"test-cmd": spec})

        with patch(
            "maou.infra.console.app.find_spec",
            side_effect=_mock_find_spec({"torch", "torchinfo"}),
        ):
            # サブコマンドを一度解決させてフォールバックを登録
            runner.invoke(cli, ["test-cmd", "--help"])

            # 親グループの --help を表示
            result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "requires:" in result.output
