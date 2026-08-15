"""GNS 計測間隔の既定値が本番経路に届くことを固定する回帰テスト．

`measurement_interval` の既定は 4 箇所にある:

1. ``GradientNoiseScaleEstimator.__init__`` (`app/learning/gradient_noise_scale.py`)
2. ``AdaptiveBatchConfig`` (`app/learning/adaptive_batch.py`)
3. ``learn-model --adaptive-batch-measurement-interval``
4. ``benchmark-training --adaptive-batch-measurement-interval``

**本番経路の挙動を決めているのは (3)(4) の CLI 既定だけ**である．
どちらの CLI も値を常に明示的に渡すため，(1)(2) だけを変えても
実行時の計測頻度は 1 ミリも変わらない．この罠を固定するため，
クラス既定と CLI 既定の両方を，別々に検証する．
"""

from __future__ import annotations

from typing import Any

import click

from maou.app.learning.adaptive_batch import (
    AdaptiveBatchConfig,
)
from maou.app.learning.gradient_noise_scale import (
    GradientNoiseScaleEstimator,
)
from maou.infra.console import learn_model, utility

_EXPECTED_DEFAULT = 5


def _option_default(
    command: click.Command, option_name: str
) -> Any:
    """click コマンドから指定オプションの既定値を取り出す．"""
    for param in command.params:
        if param.name == option_name:
            return param.default
    raise AssertionError(
        f"option {option_name!r} not found on {command.name!r}"
    )


def test_estimator_default_measurement_interval() -> None:
    """推定器のクラス既定が 5．"""
    estimator = GradientNoiseScaleEstimator(
        physical_batch_size=8
    )
    assert estimator._measurement_interval == _EXPECTED_DEFAULT


def test_adaptive_batch_config_default_measurement_interval() -> (
    None
):
    """AdaptiveBatchConfig のフィールド既定が 5．"""
    assert (
        AdaptiveBatchConfig().measurement_interval
        == _EXPECTED_DEFAULT
    )


def test_learn_model_cli_default_measurement_interval() -> None:
    """learn-model の CLI 既定が 5．

    CLI は値を常に明示的に渡すので，**ここがクラス既定を上書きする**．
    クラス既定だけを直してこのセルを 1 のままにすると，本番経路の
    挙動は変わらない — それがこのテストの守る罠である．
    """
    assert (
        _option_default(
            learn_model.learn_model,
            "adaptive_batch_measurement_interval",
        )
        == _EXPECTED_DEFAULT
    )


def test_benchmark_training_cli_default_measurement_interval() -> (
    None
):
    """benchmark-training の CLI 既定が learn-model と揃っている．"""
    assert (
        _option_default(
            utility.benchmark_training,
            "adaptive_batch_measurement_interval",
        )
        == _EXPECTED_DEFAULT
    )


def test_cli_default_reaches_adaptive_batch_config() -> None:
    """CLI 既定がそのまま AdaptiveBatchConfig に載る．

    既定値の変更が「宣言だけで消える」ことがないよう，
    CLI 既定 → 設定オブジェクトの経路を通しで固定する．
    """
    cli_default = _option_default(
        learn_model.learn_model,
        "adaptive_batch_measurement_interval",
    )
    config = learn_model._build_adaptive_batch_config(
        adaptive_batch=True,
        min_steps=2,
        max_steps=8,
        interval=50,
        smoothing=0.1,
        measurement_interval=cli_default,
    )
    assert config is not None
    assert config.measurement_interval == _EXPECTED_DEFAULT
