"""Tests for the ResNet-based evaluation network."""

from __future__ import annotations

import pytest
import torch

from torchinfo import summary

from maou.app.learning.network import HeadlessNetwork
from maou.domain.board.shogi import FEATURES_NUM
from maou.domain.model.resnet import ResNet as DomainResNet
from maou.domain.model.mlp_mixer import ShogiMLPMixer
from maou.domain.model.vision_transformer import (
    VisionTransformer as DomainVisionTransformer,
    VisionTransformerConfig,
)


def test_headless_network_wraps_domain_resnet() -> None:
    """HeadlessNetwork should compose the domain ResNet."""

    model = HeadlessNetwork()

    assert isinstance(model.backbone, DomainResNet)
    assert model.embedding_dim == 2048


def test_headless_network_forward_returns_embeddings() -> None:
    """HeadlessNetwork.forward should yield pooled embeddings."""

    model = HeadlessNetwork()
    batch_size = 4
    inputs = torch.randn(batch_size, FEATURES_NUM, 9, 9)

    outputs = model(inputs)

    assert outputs.shape == (batch_size, model.embedding_dim)


def test_headless_network_supports_mlp_mixer_backbone() -> None:
    """HeadlessNetwork should construct an MLP-Mixer backbone."""

    model = HeadlessNetwork(architecture="mlp-mixer")
    batch_size = 2
    inputs = torch.randn(batch_size, FEATURES_NUM, 9, 9)

    outputs = model(inputs)

    assert isinstance(model.backbone, ShogiMLPMixer)
    assert outputs.shape == (batch_size, model.embedding_dim)


def test_headless_network_supports_vit_backbone() -> None:
    """HeadlessNetwork should construct a Vision Transformer backbone."""

    model = HeadlessNetwork(architecture="vit")
    batch_size = 2
    inputs = torch.randn(batch_size, FEATURES_NUM, 9, 9)

    outputs = model(inputs)

    assert isinstance(model.backbone, DomainVisionTransformer)
    assert outputs.shape == (batch_size, model.embedding_dim)


def test_headless_network_rejects_invalid_shape() -> None:
    """Invalid spatial dimensions should raise a descriptive error."""

    model = HeadlessNetwork()
    bad_inputs = torch.randn(1, FEATURES_NUM, 8, 8)

    with pytest.raises(ValueError):
        model(bad_inputs)


def test_headless_network_rejects_unknown_architecture() -> None:
    """Unsupported architecture names should raise a clear error."""

    with pytest.raises(ValueError):
        HeadlessNetwork(architecture="unknown")  # type: ignore[arg-type]


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
