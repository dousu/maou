"""Tests for early stopping and the separate Stage 3 validation data path."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from maou.app.learning.callbacks import ValidationMetrics
from maou.app.learning.dl import (
    Learning,
    monitored_value,
    should_stop_early,
)
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


def _metrics(
    *, policy_ce: float, value_brier: float
) -> ValidationMetrics:
    return ValidationMetrics(
        policy_cross_entropy=policy_ce,
        value_brier_score=value_brier,
        policy_top5_accuracy=0.4,
        policy_f1_score=0.26,
        value_high_confidence_rate=0.85,
    )


class TestMonitoredValue:
    """監視指標の選択が実測の分岐を再現することを固定する．

    2026-08-05 の学習 (対局単位分割 + patience 5) では検証の
    value Brier が epoch 11 で底を打つ一方，policy 交差エントロピーは
    epoch 21 まで下がり続け，合算値は epoch 16 で底を打った．
    合算値だけを見ていると value 側の最小点を通り過ぎる．
    """

    EP11 = _metrics(policy_ce=2.1562, value_brier=0.14949)
    EP16 = _metrics(policy_ce=2.0875, value_brier=0.15434)

    def test_total_is_the_combined_loss(self) -> None:
        assert (
            monitored_value("total", 2.6011, self.EP16)
            == 2.6011
        )

    def test_value_tracks_brier_score(self) -> None:
        assert (
            monitored_value("value", 2.6011, self.EP16)
            == 0.15434
        )

    def test_policy_tracks_cross_entropy(self) -> None:
        assert (
            monitored_value("policy", 2.6011, self.EP16)
            == 2.0875
        )

    def test_heads_disagree_about_the_better_epoch(
        self,
    ) -> None:
        # 合算値は epoch 16 を選ぶが value 側は epoch 11 の方が良い
        assert monitored_value(
            "total", 2.6011, self.EP16
        ) < monitored_value("total", 2.6478, self.EP11)
        assert monitored_value(
            "value", 2.6478, self.EP11
        ) < monitored_value("value", 2.6011, self.EP16)
        assert monitored_value(
            "policy", 2.6011, self.EP16
        ) < monitored_value("policy", 2.6478, self.EP11)


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

    def test_metric_defaults_to_value(self) -> None:
        """既定は ``value``．

        合算 ``total`` は policy に支配されるので value head の最小点より
        後ろで止まる．実測では合算の底と value の底で held-out ECE が
        0.0513 対 0.0285 と 1.8 倍違った．
        """
        assert self._option().early_stopping_metric == "value"

    def test_carries_metric(self) -> None:
        opt = self._option(early_stopping_metric="value")
        assert opt.early_stopping_metric == "value"


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

    def test_early_stopping_metric_defaults_to_value(
        self,
    ) -> None:
        opt = _find_option("--early-stopping-metric")
        assert opt.default == "value"

    def test_early_stopping_metric_choices(self) -> None:
        opt = _find_option("--early-stopping-metric")
        assert isinstance(opt.type, click.Choice)
        assert list(opt.type.choices) == [
            "total",
            "value",
            "policy",
        ]

    def test_validation_data_path_requires_existing_path(
        self,
    ) -> None:
        opt = _find_option("--stage3-validation-data-path")
        assert isinstance(opt.type, click.Path)
        assert opt.type.exists is True
        # click は未指定の default を Sentinel.UNSET で表す
        assert not opt.required
