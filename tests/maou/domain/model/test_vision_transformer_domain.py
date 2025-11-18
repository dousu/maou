"""Tests for the generic vision transformer model."""

import torch

from maou.domain.model.vision_transformer import (
    VisionTransformer,
    VisionTransformerConfig,
)


def test_vision_transformer_forward_shape() -> None:
    model = VisionTransformer()
    x = torch.randn(3, 104, 9, 9)
    out = model(x)
    assert out.shape == (3,)


def test_vision_transformer_respects_config() -> None:
    config = VisionTransformerConfig(
        input_channels=104,
        board_size=9,
        embed_dim=256,
        num_heads=4,
        mlp_ratio=2.0,
        num_layers=2,
        dropout=0.2,
        attention_dropout=0.0,
    )
    model = VisionTransformer(config)
    x = torch.randn(1, 104, 9, 9)
    out = model(x)
    assert out.shape == (1,)
    assert model.positional_embedding.shape == (
        1,
        config.board_size * config.board_size,
        config.embed_dim,
    )


def test_vision_transformer_backward_pass() -> None:
    model = VisionTransformer()
    x = torch.randn(2, 104, 9, 9, requires_grad=True)
    out = model(x).sum()
    out.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
