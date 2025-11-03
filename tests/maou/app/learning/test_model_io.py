from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from maou.app.learning.model_io import ModelIO
from maou.app.learning.network import Network


class DummyOnnxModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("onnx")
        self.shape_inference = types.SimpleNamespace(
            infer_shapes=lambda model: model,
        )

    @staticmethod
    def load(*, f: Path) -> str:
        return str(f)

    @staticmethod
    def save(model: object, f: Path) -> None:
        Path(f).write_bytes(b"onnx")


class DummyOnnxSimModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("onnxsim")

    @staticmethod
    def simplify(model: object) -> tuple[object, bool]:
        return model, True


class DummyFloat16Module(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("onnxruntime.transformers.float16")

    @staticmethod
    def convert_float_to_float16(
        *, model: object, keep_io_types: bool, op_block_list: list[str],
    ) -> object:
        return model


def _install_onnx_stubs(monkeypatch: pytest.MonkeyPatch) -> None:

    dummy_onnx = DummyOnnxModule()
    dummy_onnxsim = DummyOnnxSimModule()
    dummy_float16 = DummyFloat16Module()
    dummy_transformers = types.ModuleType("onnxruntime.transformers")
    dummy_transformers.float16 = dummy_float16
    dummy_onnxruntime = types.ModuleType("onnxruntime")
    dummy_onnxruntime.transformers = dummy_transformers
    monkeypatch.setitem(sys.modules, "onnx", dummy_onnx)
    monkeypatch.setitem(sys.modules, "onnxsim", dummy_onnxsim)
    monkeypatch.setitem(sys.modules, "onnxruntime", dummy_onnxruntime)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime.transformers",
        dummy_transformers,
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime.transformers.float16",
        dummy_float16,
    )


def test_save_model_preserves_trained_model_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    _install_onnx_stubs(monkeypatch)

    def fake_export(*args: object, **kwargs: object) -> None:
        onnx_path = Path(kwargs["f"])
        onnx_path.write_bytes(b"exported")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    device = torch.device("cpu")
    model = Network().to(device)
    model.train(True)

    original_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }

    ModelIO.save_model(
        trained_model=model,
        dir=tmp_path,
        id="test",
        epoch=1,
        device=device,
    )

    assert model.training is True
    for key, value in original_state.items():
        assert torch.equal(model.state_dict()[key], value)

    saved_state = torch.load(tmp_path / "model_test_1.pt")
    for key, value in original_state.items():
        assert key in saved_state
        assert torch.equal(saved_state[key], value)

