"""Tests for the VisionTransformer evaluation network."""

from __future__ import annotations

import pytest
import torch

from torchinfo import summary

from maou.app.learning.network import HeadlessNetwork
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.model.vision_transformer import (
    VisionTransformer as DomainVisionTransformer,
    VisionTransformerConfig,
)


def test_headless_network_wraps_domain_vit() -> None:
    """HeadlessNetwork should compose the domain VisionTransformer."""

    model = HeadlessNetwork()

    assert isinstance(model.backbone, DomainVisionTransformer)
    assert model.backbone.head is None
    assert model.embedding_dim == model.backbone.embedding_dim


def test_headless_network_forward_returns_embeddings() -> None:
    """HeadlessNetwork.forward should yield pooled embeddings."""

    model = HeadlessNetwork()
    batch_size = 4
    inputs = torch.randn(batch_size, FEATURES_NUM, 9, 9)

    outputs = model(inputs)

    assert outputs.shape == (batch_size, model.embedding_dim)


def test_headless_network_rejects_invalid_shape() -> None:
    """Invalid spatial dimensions should raise a descriptive error."""

    model = HeadlessNetwork()
    bad_inputs = torch.randn(1, FEATURES_NUM, 8, 8)

    with pytest.raises(ValueError):
        model(bad_inputs)


def test_domain_vit_summary_has_expected_layout() -> None:
    """torchinfo summary should reflect the expected transformer layout."""

    config = VisionTransformerConfig()
    model = DomainVisionTransformer(config)
    stats = summary(
        model,
        input_size=(1, config.input_channels, config.board_size, config.board_size),
        verbose=0,
        depth=3,
    )

    assert stats.total_params == 19_011_073
    assert len(model.encoder) == config.num_layers
