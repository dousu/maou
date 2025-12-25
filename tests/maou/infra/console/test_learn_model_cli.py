from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from maou.domain.data.schema import get_hcpe_dtype
from maou.infra.console import learn_model


def _create_sample_file(path: Path) -> None:
    array = np.zeros(2, dtype=get_hcpe_dtype())
    array.tofile(path)


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
                "--input-dir",
                str(input_path),
                "--input-cache-mode",
                "memory",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured_kwargs["input_cache_mode"] == "memory"
    assert captured_kwargs["datasource_type"] == "hcpe"


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
                "--input-dir",
                str(input_path),
                "--detect-anomaly",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured_kwargs["detect_anomaly"] is True
