"""Tests for masked autoencoder pretraining utilities."""

from __future__ import annotations

from typing import List

import torch

from maou.app.learning.masked_autoencoder import (
    MaskedAutoencoderPretraining,
)


def test_resolve_device_enables_tensorfloat32_for_cuda(monkeypatch: "pytest.MonkeyPatch") -> None:
    """TensorFloat32 matmul precision should be enabled when using CUDA devices."""

    calls: List[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", calls.append)

    pretraining = MaskedAutoencoderPretraining()

    device = pretraining._resolve_device("cuda")

    assert device.type == "cuda"
    assert calls == ["high"]


def test_resolve_device_skips_tensorfloat32_on_cpu(monkeypatch: "pytest.MonkeyPatch") -> None:
    """TensorFloat32 matmul precision should remain disabled on CPU devices."""

    calls: List[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", calls.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    pretraining = MaskedAutoencoderPretraining()

    device = pretraining._resolve_device(None)

    assert device.type == "cpu"
    assert calls == []
