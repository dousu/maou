"""CLI-level regression tests for the benchmark-training command.

Verifies that stage-specific data paths bypass the main datasource
validation so that ``--stage1-data-path`` / ``--stage2-data-path`` can
be used without providing ``--stage3-data-path`` or a cloud provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import maou.infra.console.utility as utility


def _fake_benchmark_training(**kwargs: Any) -> str:
    """Return a minimal valid JSON result."""
    return '{"benchmark_results": {"Summary": "OK"}, "estimation": null}'


class TestStageDataPathValidation:
    """stage別data-pathのみ指定時にエラーにならないことを確認する."""

    def test_stage1_data_path_only_does_not_raise(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """--stage 1 --stage1-data-path のみ指定で正常終了する."""
        runner = CliRunner()

        monkeypatch.setattr(
            utility.utility_interface,
            "benchmark_training",
            _fake_benchmark_training,
        )

        stage1_dir = tmp_path / "stage1"
        stage1_dir.mkdir()
        (stage1_dir / "data.feather").touch()

        monkeypatch.setattr(
            utility,
            "FileSystem",
            type(
                "FakeFS",
                (),
                {
                    "collect_files": staticmethod(
                        lambda p, ext=None: [
                            stage1_dir / "data.feather"
                        ]
                    )
                },
            ),
        )

        result = runner.invoke(
            utility.benchmark_training,
            [
                "--stage",
                "1",
                "--stage1-data-path",
                str(stage1_dir),
                "--no-streaming",
            ],
        )

        assert result.exit_code == 0, result.output

    def test_stage2_data_path_only_does_not_raise(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """--stage 2 --stage2-data-path のみ指定で正常終了する."""
        runner = CliRunner()

        monkeypatch.setattr(
            utility.utility_interface,
            "benchmark_training",
            _fake_benchmark_training,
        )

        stage2_dir = tmp_path / "stage2"
        stage2_dir.mkdir()
        (stage2_dir / "data.feather").touch()

        monkeypatch.setattr(
            utility,
            "FileSystem",
            type(
                "FakeFS",
                (),
                {
                    "collect_files": staticmethod(
                        lambda p, ext=None: [
                            stage2_dir / "data.feather"
                        ]
                    )
                },
            ),
        )

        result = runner.invoke(
            utility.benchmark_training,
            [
                "--stage",
                "2",
                "--stage2-data-path",
                str(stage2_dir),
                "--no-streaming",
            ],
        )

        assert result.exit_code == 0, result.output

    def test_no_data_path_at_all_raises_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """データパスもクラウド設定も未指定時にエラーになることを確認する."""
        runner = CliRunner()

        captured_calls: list[dict[str, Any]] = []

        def tracking_benchmark_training(**kwargs: Any) -> str:
            captured_calls.append(kwargs)
            return (
                '{"benchmark_results": {"Summary": "OK"}, '
                '"estimation": null}'
            )

        monkeypatch.setattr(
            utility.utility_interface,
            "benchmark_training",
            tracking_benchmark_training,
        )

        runner.invoke(
            utility.benchmark_training,
            [
                "--stage",
                "1",
                "--no-streaming",
            ],
        )

        # benchmark_training should NOT be called because the
        # validation raises before reaching the interface call.
        assert len(captured_calls) == 0
