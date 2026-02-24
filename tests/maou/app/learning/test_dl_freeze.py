"""Tests for Learning freeze backward compatibility and optimizer integration."""

import logging

import pytest
import torch

from maou.app.learning.dl import Learning
from maou.app.learning.network import Network
from maou.app.learning.setup import LossOptimizerFactory
from maou.domain.model.resnet import ResidualBlock

# --- _resolve_trainable_layers tests ---


def _make_learning(
    *,
    trainable_layers: int | None = None,
    freeze_backbone: bool = False,
) -> Learning:
    """Create a Learning instance with freeze-related attributes set."""
    learning = Learning()
    learning.freeze_backbone = freeze_backbone
    learning.trainable_layers = trainable_layers
    return learning


def test_resolve_trainable_layers_no_freeze() -> None:
    """trainable_layers=None, freeze_backbone=False should return None."""
    learning = _make_learning(
        trainable_layers=None, freeze_backbone=False
    )

    result = learning._resolve_trainable_layers()

    assert result is None


def test_resolve_trainable_layers_freeze_backbone_true() -> (
    None
):
    """freeze_backbone=True alone should return 0 (full freeze)."""
    learning = _make_learning(
        trainable_layers=None, freeze_backbone=True
    )

    result = learning._resolve_trainable_layers()

    assert result == 0


def test_resolve_trainable_layers_with_trainable_layers() -> (
    None
):
    """trainable_layers=2 should return 2."""
    learning = _make_learning(
        trainable_layers=2, freeze_backbone=False
    )

    result = learning._resolve_trainable_layers()

    assert result == 2


def test_resolve_trainable_layers_both_specified_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both specified should return trainable_layers and log a warning."""
    learning = _make_learning(
        trainable_layers=2, freeze_backbone=True
    )

    # The "maou" logger has propagate=False, so caplog cannot capture
    # from the root. We temporarily enable propagation to allow caplog
    # to see the warning emitted by _resolve_trainable_layers().
    maou_logger = logging.getLogger("maou")
    original_propagate = maou_logger.propagate
    maou_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            result = learning._resolve_trainable_layers()
    finally:
        maou_logger.propagate = original_propagate

    assert result == 2
    assert any(
        "freeze_backbone" in record.message
        and "trainable_layers" in record.message
        for record in caplog.records
    )


# --- Optimizer integration tests ---


def _create_resnet_model() -> Network:
    """Create a small ResNet-based Network for optimizer tests."""
    return Network(
        architecture="resnet",
        embedding_dim=32,
        block=ResidualBlock,
        layers=(1, 1, 1, 1),
        strides=(1, 1, 1, 1),
        out_channels=(32, 64, 128, 256),
    )


def _get_optimizer_param_ids(
    optimizer: torch.optim.Optimizer,
) -> set[int]:
    """Collect all parameter ids from an optimizer's param_groups."""
    param_ids: set[int] = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            param_ids.add(id(p))
    return param_ids


def test_optimizer_excludes_frozen_params_full_freeze() -> None:
    """After freeze_except_last_n(0), optimizer should only have trainable params."""
    model = _create_resnet_model()
    model.freeze_except_last_n(0)

    optimizer = LossOptimizerFactory.create_optimizer(
        model, learning_ratio=0.01, momentum=0.9
    )

    opt_param_ids = _get_optimizer_param_ids(optimizer)

    # No frozen param should be in the optimizer
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert id(param) not in opt_param_ids, (
                f"Frozen param {name} should not be in optimizer"
            )

    # All trainable params should be in the optimizer
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert id(param) in opt_param_ids, (
                f"Trainable param {name} should be in optimizer"
            )


def test_optimizer_includes_unfrozen_groups_partial_freeze() -> (
    None
):
    """After freeze_except_last_n(2), optimizer should include last 2 groups + heads."""
    model = _create_resnet_model()
    model.freeze_except_last_n(2)

    optimizer = LossOptimizerFactory.create_optimizer(
        model, learning_ratio=0.01, momentum=0.9
    )

    opt_param_ids = _get_optimizer_param_ids(optimizer)

    # Verify frozen backbone params are excluded
    for param in model.backbone.layer1.parameters():
        assert id(param) not in opt_param_ids, (
            "Frozen layer1 param should not be in optimizer"
        )
    for param in model.backbone.layer2.parameters():
        assert id(param) not in opt_param_ids, (
            "Frozen layer2 param should not be in optimizer"
        )

    # Verify unfrozen backbone params are included
    for param in model.backbone.layer3.parameters():
        assert id(param) in opt_param_ids, (
            "Trainable layer3 param should be in optimizer"
        )
    for param in model.backbone.layer4.parameters():
        assert id(param) in opt_param_ids, (
            "Trainable layer4 param should be in optimizer"
        )

    # Verify heads are included
    for param in model.policy_head.parameters():
        assert id(param) in opt_param_ids, (
            "Policy head param should be in optimizer"
        )
    for param in model.value_head.parameters():
        assert id(param) in opt_param_ids, (
            "Value head param should be in optimizer"
        )
