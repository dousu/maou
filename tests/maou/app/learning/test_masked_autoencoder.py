from pathlib import Path
from typing import List

import pytest
import torch
from torch.utils.data import DataLoader

from maou.app.learning import masked_autoencoder
from maou.app.learning.masked_autoencoder import (
    MaskedAutoencoderPretraining,
    ModelFactory,
    _MaskedAutoencoder,
)
from maou.domain.board.shogi import FEATURES_NUM


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


class _RecordingModel(torch.nn.Module):
    """Model stub that records the maximum batch size seen during forward."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(feature_dim, feature_dim)
        self.max_batch_size = 0
        self.forward_calls: List[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        self.max_batch_size = max(self.max_batch_size, batch_size)
        self.forward_calls.append(batch_size)
        return self.linear(x)


def test_run_epoch_uses_forward_chunking() -> None:
    """Batches should be split into micro-batches when a chunk size is set."""

    batch_tensor = torch.randn(10, 4)
    dataloader = DataLoader(batch_tensor, batch_size=10)
    model = _RecordingModel(feature_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    pretraining = MaskedAutoencoderPretraining()

    loss = pretraining._run_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        mask_ratio=0.0,
        epoch_index=0,
        total_epochs=1,
        progress_bar=False,
        forward_chunk_size=3,
    )

    assert loss >= 0.0
    assert model.max_batch_size == 3
    assert model.forward_calls == [3, 3, 3, 1]


def test_forward_chunk_size_resolution() -> None:
    """Requested chunk sizes should clamp to the dataloader batch size."""

    pretraining = MaskedAutoencoderPretraining()

    assert pretraining._resolve_forward_chunk_size(None, 32) == 32
    assert pretraining._resolve_forward_chunk_size(16, 32) == 16
    assert pretraining._resolve_forward_chunk_size(64, 32) == 32


def test_persisted_state_dict_matches_training_model(
    tmp_path: Path,
) -> None:
    """Persisted checkpoints should load into the full training network."""

    pretraining = MaskedAutoencoderPretraining()
    autoencoder = _MaskedAutoencoder(
        feature_shape=(FEATURES_NUM, 9, 9),
        hidden_dim=128,
        device=torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"

    pretraining._persist_encoder_state_dict(
        autoencoder,
        checkpoint_path,
    )

    state_dict = torch.load(checkpoint_path, weights_only=True)

    model = ModelFactory.create_shogi_model(torch.device("cpu"))
    model.load_state_dict(state_dict)

    assert any(
        key.startswith("policy_head") or key.startswith("value_head")
        for key in state_dict
    )


def test_log_model_summary_logs_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_log_model_summary should emit torchinfo statistics."""

    pretraining = MaskedAutoencoderPretraining()
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 10),
    )

    summary_calls: list[dict[str, object]] = []

    class _FakeStats:
        def __str__(self) -> str:
            return "fake summary"

    def _fake_summary(
        model: torch.nn.Module,
        *,
        input_size: tuple[int, int],
        device: str,
        verbose: int,
    ) -> _FakeStats:
        summary_calls.append(
            {"input_size": input_size, "device": device, "verbose": verbose}
        )
        return _FakeStats()

    monkeypatch.setattr(masked_autoencoder, "summary", _fake_summary)

    logged_messages: list[str] = []

    def _record(message: str, *args: object, **_: object) -> None:
        formatted = message % args if args else message
        logged_messages.append(formatted)

    monkeypatch.setattr(pretraining.logger, "info", _record)

    pretraining._log_model_summary(
        model=model,
        batch_size=5,
        num_features=10,
        device=torch.device("cpu"),
    )

    assert summary_calls
    summary_call = summary_calls[0]
    assert summary_call["input_size"] == (2, 10)
    assert summary_call["device"] == "cpu"
    assert summary_call["verbose"] == 0
    assert any("fake summary" in message for message in logged_messages)
