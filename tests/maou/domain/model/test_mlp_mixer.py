"""Tests for the shogi-specific MLP-Mixer architecture."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

if TYPE_CHECKING:
    # For type checking, use Any to avoid import errors
    TracerWarning: Any = Warning
else:
    try:  # pragma: no cover - import location varies across PyTorch versions
        from torch.onnx import TracerWarning
    except (
        ImportError,
        AttributeError,
    ):  # pragma: no cover - fallback path
        from torch.onnx.errors import (
            OnnxExporterWarning as TracerWarning,
        )

from maou.domain.model.mlp_mixer import (
    ShogiMLPMixer,
    print_model_summary,
)


def test_mlp_mixer_forward_shape() -> None:
    model = ShogiMLPMixer(num_classes=10)
    x = torch.randn(2, 104, 9, 9)
    out = model(x)
    assert out.shape == (2, 10)


def test_mlp_mixer_masked_tokens() -> None:
    model = ShogiMLPMixer(num_classes=5)
    x = torch.randn(2, 104, 9, 9)
    mask = torch.ones(2, 81)
    mask[:, 40:] = 0
    logits, tokens = model(
        x, token_mask=mask, return_tokens=True
    )
    assert logits.shape == (2, 5)
    assert tokens.shape == (2, 81, 256)


def test_mlp_mixer_parameter_count() -> None:
    model = ShogiMLPMixer(num_classes=10)
    param_count = sum(p.numel() for p in model.parameters())
    assert 8_500_000 <= param_count <= 9_500_000


def test_mlp_mixer_summary_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = ShogiMLPMixer(num_classes=2)
    print_model_summary(model)
    captured = capsys.readouterr().out
    assert "Parameters" in captured
    assert "Embedding dimension" in captured


def test_mlp_mixer_onnx_export_without_tracer_warning(
    tmp_path: Path,
) -> None:
    model = ShogiMLPMixer(num_classes=2)
    model.eval()
    x = torch.randn(1, 104, 9, 9)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=TracerWarning)
        torch.onnx.export(
            model,
            (x,),
            tmp_path / "shogi_mixer.onnx",
            opset_version=12,
        )

    tracer_warnings = [
        w
        for w in caught
        if issubclass(w.category, TracerWarning)
    ]
    assert not tracer_warnings, (
        "Expected no TracerWarning during ONNX export"
    )


class TestFlattenTokensValidation:
    """_flatten_tokens の入力検証とトレーシングガード動作を検証する．"""

    def test_flatten_tokens_raises_on_wrong_spatial_dims(
        self,
    ) -> None:
        """eager 実行時に不正な空間次元で ValueError が送出される．"""
        model = ShogiMLPMixer(num_classes=2)
        # 正しい空間次元は 9x9 = 81 tokens
        wrong_spatial = torch.randn(1, 104, 5, 5)
        with pytest.raises(
            ValueError, match="Spatial dimensions"
        ):
            model._flatten_tokens(wrong_spatial)

    def test_flatten_tokens_raises_on_wrong_channels(
        self,
    ) -> None:
        """eager 実行時に不正なチャンネル数で ValueError が送出される．"""
        model = ShogiMLPMixer(num_classes=2)
        # 正しいチャンネル数は 104
        wrong_channels = torch.randn(1, 50, 9, 9)
        with pytest.raises(ValueError, match="Input channels"):
            model._flatten_tokens(wrong_channels)

    def test_flatten_tokens_skips_validation_during_tracing(
        self,
    ) -> None:
        """トレーシング中は不正入力でもエラーが発生しない．"""
        from unittest.mock import patch

        model = ShogiMLPMixer(num_classes=2)
        wrong_spatial = torch.randn(1, 104, 5, 5)

        with patch(
            "maou.domain.model.mlp_mixer.is_tracing",
            return_value=True,
        ):
            result = model._flatten_tokens(wrong_spatial)

        # view + transpose で (1, 25, 104) になる（5*5=25 tokens）
        assert result.shape == (1, 25, 104)
