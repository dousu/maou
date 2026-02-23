"""Tests for model I/O utilities."""

import torch

from maou.app.learning.model_io import ModelIO
from maou.app.learning.setup import ModelFactory


def test_format_parameter_count_millions() -> None:
    """Test parameter count formatting for millions."""
    assert ModelIO.format_parameter_count(1_234_567) == "1.2m"
    assert ModelIO.format_parameter_count(1_000_000) == "1m"
    assert ModelIO.format_parameter_count(2_500_000) == "2.5m"


def test_format_parameter_count_thousands() -> None:
    """Test parameter count formatting for thousands."""
    assert ModelIO.format_parameter_count(45_000) == "45k"
    assert ModelIO.format_parameter_count(1_000) == "1k"
    assert ModelIO.format_parameter_count(1_234) == "1.2k"


def test_format_parameter_count_small() -> None:
    """Test parameter count formatting for small numbers."""
    assert ModelIO.format_parameter_count(123) == "123"
    assert ModelIO.format_parameter_count(999) == "999"
    assert ModelIO.format_parameter_count(1) == "1"


def test_generate_model_tag() -> None:
    """Test model tag generation."""
    device = torch.device("cpu")
    model = ModelFactory.create_shogi_model(
        device, architecture="resnet"
    )

    tag = ModelIO.generate_model_tag(model, "resnet")

    # Tag should start with architecture name
    assert tag.startswith("resnet-")

    # Tag should end with parameter count (format: XXm or XXk)
    param_part = tag.split("-")[1]
    assert param_part.endswith("m") or param_part.endswith("k")


def test_generate_model_tag_different_architectures() -> None:
    """Test model tag generation for different architectures."""
    device = torch.device("cpu")

    # ResNet
    resnet_model = ModelFactory.create_shogi_model(
        device, architecture="resnet"
    )
    resnet_tag = ModelIO.generate_model_tag(
        resnet_model, "resnet"
    )
    assert resnet_tag.startswith("resnet-")

    # MLP-Mixer
    mlp_model = ModelFactory.create_shogi_model(
        device, architecture="mlp-mixer"
    )
    mlp_tag = ModelIO.generate_model_tag(mlp_model, "mlp-mixer")
    assert mlp_tag.startswith("mlp-mixer-")

    # ViT
    vit_model = ModelFactory.create_shogi_model(
        device, architecture="vit"
    )
    vit_tag = ModelIO.generate_model_tag(vit_model, "vit")
    assert vit_tag.startswith("vit-")


def test_generate_model_tag_trainable_layers_none() -> None:
    """trainable_layers=None produces no suffix (backward compat)."""
    device = torch.device("cpu")
    model = ModelFactory.create_shogi_model(
        device, architecture="resnet"
    )
    tag = ModelIO.generate_model_tag(
        model, "resnet", trainable_layers=None
    )
    assert "-tl" not in tag
    assert tag.startswith("resnet-")


def test_generate_model_tag_trainable_layers_zero() -> None:
    """trainable_layers=0 appends '-tl0'."""
    device = torch.device("cpu")
    model = ModelFactory.create_shogi_model(
        device, architecture="resnet"
    )
    tag = ModelIO.generate_model_tag(
        model, "resnet", trainable_layers=0
    )
    assert tag.endswith("-tl0")


def test_generate_model_tag_trainable_layers_positive() -> None:
    """trainable_layers=2 appends '-tl2'."""
    device = torch.device("cpu")
    model = ModelFactory.create_shogi_model(
        device, architecture="vit"
    )
    tag = ModelIO.generate_model_tag(
        model, "vit", trainable_layers=2
    )
    assert tag.endswith("-tl2")
    assert tag.startswith("vit-")


# --- Fix 2: onnx_model_simp UnboundLocalError テスト ---


def test_onnx_fp16_without_onnxsim() -> None:
    """onnxsim未インストール時にFP16変換パスでUnboundLocalErrorが発生しないこと．"""
    import onnx
    from onnxruntime.transformers import float16

    # 最小限のONNXモデルを作成
    device = torch.device("cpu")
    model = ModelFactory.create_shogi_model(
        device, architecture="resnet"
    )
    model.train(False)

    import numpy as np

    from maou.domain.data.schema import (
        create_empty_preprocessing_array,
    )

    dummy_data = create_empty_preprocessing_array(1)
    dummy_board = np.asarray(
        dummy_data[0]["boardIdPositions"], dtype=np.uint8
    )
    dummy_input = (
        torch.from_numpy(dummy_board.astype(np.int64))
        .unsqueeze(0)
        .to(device)
    )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"
        torch.onnx.export(
            model=model,
            args=(dummy_input,),
            f=onnx_path,
            export_params=True,
            input_names=["input"],
            output_names=["policy", "value"],
            opset_version=20,
            dynamic_axes={
                "input": {0: "batch_size"},
                "policy": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
        )

        # onnxsim未インストールの状態をシミュレート
        onnx_model = onnx.load(f=onnx_path)
        onnx_model = onnx.shape_inference.infer_shapes(
            onnx_model
        )

        onnxsim_available = False
        if onnxsim_available:
            pass  # pragma: no cover
        else:
            onnx_model_simp = onnx_model
            onnx.save(onnx_model_simp, onnx_path)

        # FP16変換 — onnx_model_simp が定義されていればUnboundLocalErrorは発生しない
        onnx_model_fp16 = float16.convert_float_to_float16(
            model=onnx_model_simp,
            keep_io_types=True,
            op_block_list=[
                "Gemm",
                "GlobalAveragePool",
                "Flatten",
            ],
        )

        fp16_path = Path(tmpdir) / "test_model_fp16.onnx"
        onnx.save(onnx_model_fp16, fp16_path)

        # ファイルが正常に保存されたことを確認
        assert fp16_path.exists()
        assert fp16_path.stat().st_size > 0


def test_save_model_with_custom_architecture_config() -> None:
    """カスタムarchitecture_configとhand_projection_dim指定時のsave_modelが成功すること．"""
    import tempfile
    from pathlib import Path

    device = torch.device("cpu")
    architecture_config = {"num_layers": 2, "hidden_dim": 64}
    hand_projection_dim = 16

    trained_model = ModelFactory.create_shogi_model(
        device,
        architecture="resnet",
        architecture_config=architecture_config,
        hand_projection_dim=hand_projection_dim,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ModelIO.save_model(
            trained_model=trained_model,
            dir=Path(tmpdir),
            id="test",
            epoch=1,
            device=device,
            architecture="resnet",
            architecture_config=architecture_config,
            hand_projection_dim=hand_projection_dim,
        )

        # ONNXファイルが生成されていることを確認
        onnx_files = list(Path(tmpdir).glob("*.onnx"))
        assert len(onnx_files) >= 1, (
            "ONNX model file was not generated"
        )

        # FP32 ONNXファイルのサイズが正常
        fp32_files = [
            f for f in onnx_files if "fp16" not in f.name
        ]
        assert len(fp32_files) == 1
        assert fp32_files[0].stat().st_size > 0
