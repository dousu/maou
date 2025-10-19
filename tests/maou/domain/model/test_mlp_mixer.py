"""Tests for the lightweight MLP-Mixer architecture."""

import torch

from maou.domain.model.mlp_mixer import LightweightMLPMixer


def test_mlp_mixer_forward_shape() -> None:
    model = LightweightMLPMixer(num_classes=10)
    x = torch.randn(2, 104, 9, 9)
    out = model(x)
    assert out.shape == (2, 10)


def test_mlp_mixer_masked_tokens() -> None:
    model = LightweightMLPMixer(num_classes=5)
    x = torch.randn(2, 104, 9, 9)
    mask = torch.ones(2, 81)
    mask[:, 40:] = 0
    logits, tokens = model(
        x, token_mask=mask, return_tokens=True
    )
    assert logits.shape == (2, 5)
    assert tokens.shape == (2, 81, 104)
