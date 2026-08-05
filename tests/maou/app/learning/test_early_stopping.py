"""Tests for early stopping and the separate Stage 3 validation data path."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from maou.app.learning.dl import Learning, should_stop_early
from maou.infra.console import learn_model


class TestShouldStopEarly:
    """``patience`` の意味 (連続未更新エポック数) を固定する．"""

    def test_disabled_when_patience_is_zero(self) -> None:
        for n in (0, 1, 10, 1000):
            assert should_stop_early(n, 0) is False

    def test_disabled_for_negative_patience(self) -> None:
        assert should_stop_early(5, -1) is False

    def test_stops_exactly_at_patience(self) -> None:
        # patience=3 は「3 エポック連続で更新なし」で停止する
        assert should_stop_early(2, 3) is False
        assert should_stop_early(3, 3) is True
        assert should_stop_early(4, 3) is True

    def test_patience_one_stops_after_single_bad_epoch(
        self,
    ) -> None:
        assert should_stop_early(0, 1) is False
        assert should_stop_early(1, 1) is True


class TestLearningOptionField:
    """``LearningOption`` が patience を保持することを確認する．"""

    def _option(self, **kw: object) -> Learning.LearningOption:
        base: dict[str, object] = {
            "datasource": None,
            "compilation": False,
            "test_ratio": 0.1,
            "epoch": 10,
            "batch_size": 32,
            "dataloader_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "policy_loss_ratio": 1.0,
            "value_loss_ratio": 1.0,
            "learning_ratio": 0.001,
            "momentum": 0.9,
            "optimizer_name": "adamw",
            "optimizer_beta1": 0.9,
            "optimizer_beta2": 0.999,
            "optimizer_eps": 1e-8,
            "log_dir": Path("."),
            "model_dir": Path("."),
        }
        base.update(kw)
        return Learning.LearningOption(**base)  # type: ignore[arg-type]

    def test_defaults_to_disabled(self) -> None:
        assert self._option().early_stopping_patience == 0

    def test_carries_value(self) -> None:
        opt = self._option(early_stopping_patience=5)
        assert opt.early_stopping_patience == 5


def _find_option(name: str) -> click.Option:
    for param in learn_model.learn_model.params:
        if (
            isinstance(param, click.Option)
            and name in param.opts
        ):
            return param
    pytest.fail(f"option {name} not found")


class TestCliOptions:
    def test_early_stopping_patience_defaults_to_zero(
        self,
    ) -> None:
        opt = _find_option("--early-stopping-patience")
        assert opt.default == 0

    def test_early_stopping_patience_rejects_negative(
        self,
    ) -> None:
        opt = _find_option("--early-stopping-patience")
        assert isinstance(opt.type, click.IntRange)
        assert opt.type.min == 0

    def test_validation_data_path_requires_existing_path(
        self,
    ) -> None:
        opt = _find_option("--stage3-validation-data-path")
        assert isinstance(opt.type, click.Path)
        assert opt.type.exists is True
        # click は未指定の default を Sentinel.UNSET で表す
        assert not opt.required
