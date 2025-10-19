"""Tests for the lightweight MLP-Mixer architecture."""

import warnings
from pathlib import Path

import torch
from torch.onnx import TracerWarning

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


def test_mlp_mixer_onnx_export_without_tracer_warning(tmp_path: Path) -> None:
    model = LightweightMLPMixer(num_classes=2)
    model.eval()
    x = torch.randn(1, 104, 9, 9)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=TracerWarning)
        torch.onnx.export(
            model,
            x,
            tmp_path / "lightweight_mixer.onnx",
            opset_version=12,
        )

    tracer_warnings = [w for w in caught if issubclass(w.category, TracerWarning)]
    assert not tracer_warnings, "Expected no TracerWarning during ONNX export"
