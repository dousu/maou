from __future__ import annotations

from typing import Any

import pytest

from maou.interface import utility_interface


class _DummyDataSource:
    """Minimal stub for learning data source."""

    def train_test_split(self, *, test_ratio: float) -> tuple[Any, Any]:
        return self, self


def test_benchmark_training_detect_anomaly_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_config = None

    class FakeUseCase:
        def __init__(self) -> None:
            pass

        def execute(self, config: Any) -> str:
            nonlocal captured_config
            captured_config = config
            return "{}"

    monkeypatch.setattr(
        utility_interface,
        "TrainingBenchmarkUseCase",
        FakeUseCase,
    )

    datasource = _DummyDataSource()
    result = utility_interface.benchmark_training(
        datasource=datasource,
        datasource_type="hcpe",
        detect_anomaly=True,
    )

    assert result == "{}"
    assert captured_config is not None
    assert captured_config.detect_anomaly is True
