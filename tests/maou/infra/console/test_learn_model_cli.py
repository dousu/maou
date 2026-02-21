from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

import maou.infra.console.learn_model as learn_model
from maou.domain.data.schema import get_hcpe_dtype


def _create_sample_file(path: Path) -> None:
    array = np.zeros(2, dtype=get_hcpe_dtype())
    array.tofile(path)


@pytest.mark.skip(reason="Needs update for .feather files")
def test_learn_model_passes_cache_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()

    captured_kwargs: dict[str, Any] = {}

    def fake_learn(**kwargs: Any) -> str:
        captured_kwargs.update(kwargs)
        return "{}"

    monkeypatch.setattr(learn_model.learn, "learn", fake_learn)

    with runner.isolated_filesystem():
        input_path = Path("input.bin")
        _create_sample_file(input_path)

        result = runner.invoke(
            learn_model.learn_model,
            [
                "--input-path",
                str(input_path),
                "--input-cache-mode",
                "memory",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured_kwargs["input_cache_mode"] == "memory"


def test_learn_model_passes_stage_batch_sizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--stage1-batch-size and --stage2-batch-size are passed to learn_multi_stage."""
    runner = CliRunner()

    captured_kwargs: dict[str, Any] = {}

    def fake_learn_multi_stage(**kwargs: Any) -> str:
        captured_kwargs.update(kwargs)
        return "{}"

    monkeypatch.setattr(
        learn_model.learn,
        "learn_multi_stage",
        fake_learn_multi_stage,
    )

    # FileSystem.collect_files needs to return a list of paths
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    stage1_file = stage1_dir / "data.feather"
    stage1_file.touch()

    monkeypatch.setattr(
        learn_model,
        "FileSystem",
        type(
            "FakeFS",
            (),
            {
                "collect_files": staticmethod(
                    lambda p, ext=None: [stage1_file]
                )
            },
        ),
    )

    result = runner.invoke(
        learn_model.learn_model,
        [
            "--stage",
            "1",
            "--stage1-data-path",
            str(stage1_dir),
            "--stage1-batch-size",
            "32",
            "--stage2-batch-size",
            "64",
            "--batch-size",
            "256",
            "--no-streaming",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_kwargs["stage1_batch_size"] == 32
    assert captured_kwargs["stage2_batch_size"] == 64
    assert captured_kwargs["batch_size"] == 256


def test_learn_model_passes_stage_learning_rates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--stage1-learning-rate and --stage2-learning-rate are passed to learn_multi_stage."""
    runner = CliRunner()

    captured_kwargs: dict[str, Any] = {}

    def fake_learn_multi_stage(**kwargs: Any) -> str:
        captured_kwargs.update(kwargs)
        return "{}"

    monkeypatch.setattr(
        learn_model.learn,
        "learn_multi_stage",
        fake_learn_multi_stage,
    )

    # FileSystem.collect_files needs to return a list of paths
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    stage1_file = stage1_dir / "data.feather"
    stage1_file.touch()

    monkeypatch.setattr(
        learn_model,
        "FileSystem",
        type(
            "FakeFS",
            (),
            {
                "collect_files": staticmethod(
                    lambda p, ext=None: [stage1_file]
                )
            },
        ),
    )

    result = runner.invoke(
        learn_model.learn_model,
        [
            "--stage",
            "1",
            "--stage1-data-path",
            str(stage1_dir),
            "--stage1-learning-rate",
            "0.0001",
            "--stage2-learning-rate",
            "0.005",
            "--no-streaming",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_kwargs["stage1_learning_rate"] == 0.0001
    assert captured_kwargs["stage2_learning_rate"] == 0.005


@pytest.mark.skip(reason="Needs update for .feather files")
def test_learn_model_detect_anomaly_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()

    captured_kwargs: dict[str, Any] = {}

    def fake_learn(**kwargs: Any) -> str:
        captured_kwargs.update(kwargs)
        return "{}"

    monkeypatch.setattr(learn_model.learn, "learn", fake_learn)

    with runner.isolated_filesystem():
        input_path = Path("input.bin")
        _create_sample_file(input_path)

        result = runner.invoke(
            learn_model.learn_model,
            [
                "--input-path",
                str(input_path),
                "--detect-anomaly",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured_kwargs["detect_anomaly"] is True


def test_stage3_with_stage3_data_path_routes_to_multi_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--stage 3 --stage3-data-pathがlearn_multi_stage()を呼ぶことを確認する．"""
    runner = CliRunner()

    captured_kwargs: dict[str, Any] = {}

    def fake_learn_multi_stage(**kwargs: Any) -> str:
        captured_kwargs.update(kwargs)
        return "{}"

    monkeypatch.setattr(
        learn_model.learn,
        "learn_multi_stage",
        fake_learn_multi_stage,
    )

    stage3_dir = tmp_path / "stage3"
    stage3_dir.mkdir()
    stage3_file = stage3_dir / "data.feather"
    stage3_file.touch()

    monkeypatch.setattr(
        learn_model,
        "FileSystem",
        type(
            "FakeFS",
            (),
            {
                "collect_files": staticmethod(
                    lambda p, ext=None: [stage3_file]
                )
            },
        ),
    )

    result = runner.invoke(
        learn_model.learn_model,
        [
            "--stage",
            "3",
            "--stage3-data-path",
            str(stage3_dir),
            "--no-streaming",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "stage3_data_config" in captured_kwargs
    assert captured_kwargs["stage3_data_config"] is not None
