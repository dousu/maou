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
    assert captured_kwargs["datasource_type"] == "hcpe"


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
