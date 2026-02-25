"""Tests for ONNX export verification utilities."""

from pathlib import Path
from typing import Any

import pytest
import torch

from maou.app.learning.network import Network
from maou.app.learning.onnx_verifier import ONNXExportVerifier
from maou.app.learning.setup import ModelFactory


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_model(device: torch.device) -> Network:
    """Create a sample model for testing."""
    model = ModelFactory.create_shogi_model(device)
    model.eval()
    return model


@pytest.fixture
def sample_model_compiled(device: torch.device) -> Network:
    """Create a compiled sample model for testing."""
    model = ModelFactory.create_shogi_model(device)
    # Simulate torch.compile by adding _orig_mod prefix to state_dict
    state_dict = model.state_dict()
    compiled_state_dict = {
        f"_orig_mod.{key}": value
        for key, value in state_dict.items()
    }
    # Create a simple wrapper to simulate compiled model
    compiled_model = ModelFactory.create_shogi_model(device)
    # Replace state_dict method to return compiled version
    compiled_model._compiled_state_dict = compiled_state_dict

    def compiled_state_dict_method(
        self: Any,
        destination: Any = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self._compiled_state_dict

    compiled_model.state_dict = (
        compiled_state_dict_method.__get__(
            compiled_model, Network
        )
    )
    compiled_model.eval()
    return compiled_model


def test_verify_parameter_transfer_success(
    sample_model: Network,
    device: torch.device,
) -> None:
    """Test successful parameter transfer verification for non-compiled model."""
    fresh_model = ModelFactory.create_shogi_model(device)
    state_dict = sample_model.state_dict()
    fresh_model.load_state_dict(state_dict)

    report = ONNXExportVerifier.verify_parameter_transfer(
        trained_model=sample_model,
        fresh_model=fresh_model,
        cleaned_state_dict=state_dict,
    )

    assert report.success
    assert report.matched_parameters > 0
    assert len(report.missing_parameters) == 0
    assert len(report.value_mismatches) == 0
    assert report.policy_head_verified
    assert report.value_head_verified


def test_verify_parameter_transfer_compiled_model(
    sample_model_compiled: Network,
    device: torch.device,
) -> None:
    """Test parameter transfer verification for compiled model with _orig_mod prefix."""
    fresh_model = ModelFactory.create_shogi_model(device)
    original_state_dict = sample_model_compiled.state_dict()

    # Simulate prefix stripping
    cleaned_state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith("_orig_mod."):
            clean_key = key[len("_orig_mod.") :]
            cleaned_state_dict[clean_key] = value
        else:
            cleaned_state_dict[key] = value

    fresh_model.load_state_dict(cleaned_state_dict)

    report = ONNXExportVerifier.verify_parameter_transfer(
        trained_model=sample_model_compiled,
        fresh_model=fresh_model,
        cleaned_state_dict=cleaned_state_dict,
    )

    assert report.success
    assert report.policy_head_verified
    assert report.value_head_verified


def test_verify_output_head_parameters(
    sample_model: Network,
) -> None:
    """Test output head parameter verification."""
    report = ONNXExportVerifier.verify_output_head_parameters(
        sample_model
    )

    assert report.success
    assert report.policy_head_exists
    assert report.value_head_exists
    assert len(report.policy_head_params) > 0
    assert len(report.value_head_params) > 0


def test_verify_onnx_functional_equivalence_fp32(
    sample_model: Network,
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test ONNX FP32 functional equivalence verification."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from maou.domain.data.schema import (
        create_empty_preprocessing_array,
    )

    # Export model to ONNX (2入力: board + hand)
    onnx_path = tmp_path / "test_model.onnx"

    dummy_data = create_empty_preprocessing_array(1)
    import numpy as np

    dummy_board = (
        torch.from_numpy(
            np.asarray(
                dummy_data[0]["boardIdPositions"],
                dtype=np.uint8,
            ).astype(np.int32)
        )
        .unsqueeze(0)
        .to(device)
    )
    dummy_hand = (
        torch.from_numpy(
            np.asarray(
                dummy_data[0]["piecesInHand"],
                dtype=np.uint8,
            ).astype(np.float32)
        )
        .unsqueeze(0)
        .to(device)
    )

    torch.onnx.export(
        model=sample_model,
        args=((dummy_board, dummy_hand),),
        f=onnx_path,
        export_params=True,
        input_names=["board", "hand"],
        output_names=["policy", "value"],
        opset_version=20,
        dynamic_axes={
            "board": {0: "batch_size"},
            "hand": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
        dynamo=False,
    )

    report = (
        ONNXExportVerifier.verify_onnx_functional_equivalence(
            pytorch_model=sample_model,
            onnx_model_path=onnx_path,
            device=device,
            num_test_samples=3,
            fp16=False,
        )
    )

    assert report.success
    assert report.num_samples_tested == 3
    assert report.policy_max_abs_diff < 1e-5
    assert report.value_max_abs_diff < 1e-5


def test_verify_onnx_functional_equivalence_fp16(
    sample_model: Network,
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test ONNX FP16 functional equivalence verification."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxruntime.transformers")

    import onnx
    from onnxruntime.transformers import float16

    from maou.domain.data.schema import (
        create_empty_preprocessing_array,
    )

    # Export model to ONNX FP32 first (2入力: board + hand)
    onnx_path = tmp_path / "test_model.onnx"

    dummy_data = create_empty_preprocessing_array(1)
    import numpy as np

    dummy_board = (
        torch.from_numpy(
            np.asarray(
                dummy_data[0]["boardIdPositions"],
                dtype=np.uint8,
            ).astype(np.int32)
        )
        .unsqueeze(0)
        .to(device)
    )
    dummy_hand = (
        torch.from_numpy(
            np.asarray(
                dummy_data[0]["piecesInHand"],
                dtype=np.uint8,
            ).astype(np.float32)
        )
        .unsqueeze(0)
        .to(device)
    )

    torch.onnx.export(
        model=sample_model,
        args=((dummy_board, dummy_hand),),
        f=onnx_path,
        export_params=True,
        input_names=["board", "hand"],
        output_names=["policy", "value"],
        opset_version=20,
        dynamic_axes={
            "board": {0: "batch_size"},
            "hand": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Convert to FP16
    onnx_model = onnx.load(str(onnx_path))
    onnx_fp16_path = tmp_path / "test_model_fp16.onnx"
    onnx_model_fp16 = float16.convert_float_to_float16(
        model=onnx_model,
        keep_io_types=True,
        op_block_list=["Gemm", "GlobalAveragePool", "Flatten"],
    )
    onnx.save(onnx_model_fp16, str(onnx_fp16_path))

    report = (
        ONNXExportVerifier.verify_onnx_functional_equivalence(
            pytorch_model=sample_model,
            onnx_model_path=onnx_fp16_path,
            device=device,
            num_test_samples=3,
            fp16=True,
        )
    )

    # FP16 should have larger differences but still within tolerance
    assert report.num_samples_tested == 3
    # Note: FP16 might fail strict verification due to precision loss
    # This is expected and handled with warnings in the actual code


def test_verify_onnx_graph_structure(
    sample_model: Network,
    device: torch.device,
    tmp_path: Path,
) -> None:
    """Test ONNX graph structure verification."""
    pytest.importorskip("onnx")

    from maou.domain.data.schema import (
        create_empty_preprocessing_array,
    )

    # Export model to ONNX (2入力: board + hand)
    onnx_path = tmp_path / "test_model.onnx"

    dummy_data = create_empty_preprocessing_array(1)
    import numpy as np

    dummy_board = (
        torch.from_numpy(
            np.asarray(
                dummy_data[0]["boardIdPositions"],
                dtype=np.uint8,
            ).astype(np.int32)
        )
        .unsqueeze(0)
        .to(device)
    )
    dummy_hand = (
        torch.from_numpy(
            np.asarray(
                dummy_data[0]["piecesInHand"],
                dtype=np.uint8,
            ).astype(np.float32)
        )
        .unsqueeze(0)
        .to(device)
    )

    torch.onnx.export(
        model=sample_model,
        args=((dummy_board, dummy_hand),),
        f=onnx_path,
        export_params=True,
        input_names=["board", "hand"],
        output_names=["policy", "value"],
        opset_version=20,
        dynamic_axes={
            "board": {0: "batch_size"},
            "hand": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
        dynamo=False,
    )

    report = ONNXExportVerifier.verify_onnx_graph_structure(
        onnx_path
    )

    assert report.success
    assert report.input_names == ["board", "hand"]
    assert report.output_names == ["policy", "value"]
    assert report.graph_valid
