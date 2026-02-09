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
