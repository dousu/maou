"""Tests for the VisionTransformer evaluation network."""

from __future__ import annotations

import pytest
import torch

from torchinfo import summary

from maou.app.learning.network import VisionTransformer
from maou.domain.board.shogi import FEATURES_NUM


def test_vision_transformer_returns_scalar_scores() -> None:
    """The VisionTransformer should output a scalar per sample."""

    model = VisionTransformer()
    batch_size = 4
    inputs = torch.randn(batch_size, FEATURES_NUM, 9, 9)

    outputs = model(inputs)

    assert outputs.shape == (batch_size,)


def test_vision_transformer_rejects_invalid_shape() -> None:
    """Invalid spatial dimensions should raise a descriptive error."""

    model = VisionTransformer()
    bad_inputs = torch.randn(1, FEATURES_NUM, 8, 8)

    with pytest.raises(ValueError):
        model(bad_inputs)


def test_vision_transformer_summary_has_expected_layout() -> None:
    """torchinfo summary should reflect the expected transformer layout."""

    model = VisionTransformer()
    stats = summary(
        model,
        input_size=(1, FEATURES_NUM, 9, 9),
        verbose=0,
        depth=3,
    )

    assert stats.total_params == 19_011_073
    assert len(model.encoder) == model.config.num_layers
