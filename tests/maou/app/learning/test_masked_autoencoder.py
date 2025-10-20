from typing import List

import pytest
import torch

from maou.app.learning.masked_autoencoder import (
    MaskedAutoencoderPretraining,
    ModelFactory,
    _MaskedAutoencoder,
)


class _DummyBackbone(torch.nn.Module):
    """Lightweight encoder stub returning a fixed embedding."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        return torch.ones((batch_size, self._embedding_dim), device=device)


def test_resolve_device_enables_tensorfloat32_for_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TensorFloat32 matmul precision should be enabled when using CUDA devices."""

    calls: List[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", calls.append)

    pretraining = MaskedAutoencoderPretraining()

    device = pretraining._resolve_device("cuda")

    assert device.type == "cuda"
    assert calls == ["high"]


def test_resolve_device_skips_tensorfloat32_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TensorFloat32 matmul precision should remain disabled on CPU devices."""

    calls: List[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", calls.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    pretraining = MaskedAutoencoderPretraining()

    device = pretraining._resolve_device(None)

    assert device.type == "cpu"
    assert calls == []


def test_masked_autoencoder_uses_encoder_embedding_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decoder input dimension must match the encoder embedding size."""

    embedding_dim = 32

    def _create_backbone(cls, device: torch.device) -> _DummyBackbone:
        return _DummyBackbone(embedding_dim=embedding_dim).to(device)

    monkeypatch.setattr(
        ModelFactory,
        "create_shogi_backbone",
        classmethod(_create_backbone),
    )

    feature_shape = (3, 4, 5)
    model = _MaskedAutoencoder(
        feature_shape=feature_shape,
        hidden_dim=16,
        device=torch.device("cpu"),
    )

    decoder_linear = model.decoder[0]
    assert isinstance(decoder_linear, torch.nn.Linear)
    assert decoder_linear.in_features == embedding_dim
    flattened_size = feature_shape[0] * feature_shape[1] * feature_shape[2]
    inputs = torch.randn(2, flattened_size)
    outputs = model(inputs)
    assert outputs.shape == (2, flattened_size)
