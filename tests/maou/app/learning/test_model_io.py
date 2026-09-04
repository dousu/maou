"""Tests for model I/O utilities."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
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
    assert param_part.endswith(("m", "k"))


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


def test_onnx_fp16_with_onnxslim() -> None:
    """onnxslim.slim()を使ったONNXモデル最適化とFP16変換が正常に動作すること．"""
    import onnx
    import onnxslim
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
    dummy_board = (
        torch.from_numpy(
            np.asarray(
                dummy_data[0]["boardIdPositions"],
                dtype=np.uint8,
            ).astype(np.int64)
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

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"
        torch.onnx.export(
            model=model,
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

        # ONNX最適化: shape inference → onnxslim.slim()
        onnx_model = onnx.load(f=onnx_path)
        onnx_model = onnx.shape_inference.infer_shapes(
            onnx_model
        )
        onnx_model_simp = onnxslim.slim(onnx_model)
        onnx.save(onnx_model_simp, onnx_path)

        # FP16変換
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


# --- head ロード関数の _strip_orig_mod_prefix テスト ---


def _save_state_dict_with_prefix(
    state_dict: dict[str, torch.Tensor],
) -> Path:
    """_orig_mod. プレフィックス付きの state_dict を一時ファイルに保存する．"""
    prefixed = {
        f"_orig_mod.{k}": v for k, v in state_dict.items()
    }
    with tempfile.NamedTemporaryFile(
        suffix=".pt", delete=False
    ) as tmp:
        path = Path(tmp.name)
    torch.save(prefixed, path)
    return path


def test_load_policy_head_strips_orig_mod_prefix() -> None:
    """load_policy_head が _orig_mod. プレフィックスを除去すること．"""
    original = {"0.weight": torch.randn(2, 3)}
    path = _save_state_dict_with_prefix(original)
    result = ModelIO.load_policy_head(path, torch.device("cpu"))
    assert "0.weight" in result
    assert not any(k.startswith("_orig_mod.") for k in result)


def test_load_value_head_strips_orig_mod_prefix() -> None:
    """load_value_head が _orig_mod. プレフィックスを除去すること．"""
    original = {"0.weight": torch.randn(2, 3)}
    path = _save_state_dict_with_prefix(original)
    result = ModelIO.load_value_head(path, torch.device("cpu"))
    assert "0.weight" in result
    assert not any(k.startswith("_orig_mod.") for k in result)


def test_load_reachable_head_strips_orig_mod_prefix() -> None:
    """load_reachable_head が _orig_mod. プレフィックスを除去すること．"""
    original = {"0.weight": torch.randn(2, 3)}
    path = _save_state_dict_with_prefix(original)
    result = ModelIO.load_reachable_head(
        path, torch.device("cpu")
    )
    assert "0.weight" in result
    assert not any(k.startswith("_orig_mod.") for k in result)


def test_load_legal_moves_head_strips_orig_mod_prefix() -> None:
    """load_legal_moves_head が _orig_mod. プレフィックスを除去すること．"""
    original = {"0.weight": torch.randn(2, 3)}
    path = _save_state_dict_with_prefix(original)
    result = ModelIO.load_legal_moves_head(
        path, torch.device("cpu")
    )
    assert "0.weight" in result
    assert not any(k.startswith("_orig_mod.") for k in result)


class TestDedupeIdenticalNodes:
    """``convert_float_to_float16`` が吐く重複ノードの除去 (回帰)．

    fp16 変換器が ``name``/``op_type``/``input``/``output`` の
    すべてが同一な Cast を 2 回吐くことがあり，onnxruntime は
    ``two nodes with same node name`` として読み込みを拒否する．
    """

    @staticmethod
    def _model_with_duplicate() -> Any:
        from onnx import TensorProto, helper

        cast = helper.make_node(
            "Cast",
            inputs=["x"],
            outputs=["y"],
            name="/dup_cast_node",
            to=TensorProto.FLOAT,
        )
        graph = helper.make_graph(
            [
                cast,
                helper.make_node(
                    "Cast",
                    inputs=["x"],
                    outputs=["y"],
                    name="/dup_cast_node",
                    to=TensorProto.FLOAT,
                ),
            ],
            "g",
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT16, [1]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, [1]
                )
            ],
        )
        return helper.make_model(graph)

    def test_removes_exact_duplicate(self) -> None:
        from maou.app.learning.model_io import (
            _dedupe_identical_nodes,
        )

        model = self._model_with_duplicate()
        assert len(model.graph.node) == 2

        removed = _dedupe_identical_nodes(model)

        assert removed == 1
        assert len(model.graph.node) == 1
        # 残った 1 本は元と同じ入出力を保つ (計算が変わらない)
        assert list(model.graph.node[0].input) == ["x"]
        assert list(model.graph.node[0].output) == ["y"]

    def test_duplicate_makes_model_unloadable_and_dedupe_fixes_it(
        self, tmp_path: Path
    ) -> None:
        import onnx
        import onnxruntime as ort

        model = self._model_with_duplicate()
        broken = tmp_path / "broken.onnx"
        onnx.save(model, broken)
        with pytest.raises(
            Exception, match="two nodes with same node name"
        ):
            ort.InferenceSession(
                str(broken), providers=["CPUExecutionProvider"]
            )

        from maou.app.learning.model_io import (
            _dedupe_identical_nodes,
        )

        _dedupe_identical_nodes(model)
        fixed = tmp_path / "fixed.onnx"
        onnx.save(model, fixed)
        ort.InferenceSession(
            str(fixed), providers=["CPUExecutionProvider"]
        )

    def test_noop_when_no_duplicates(self) -> None:
        from maou.app.learning.model_io import (
            _dedupe_identical_nodes,
        )

        model = self._model_with_duplicate()
        del model.graph.node[1]

        assert _dedupe_identical_nodes(model) == 0
        assert len(model.graph.node) == 1


class TestAssertOnnxLoadable:
    """読み込めない ONNX を publish させないガード (回帰)．"""

    def test_raises_on_unloadable_model(
        self, tmp_path: Path
    ) -> None:
        from maou.app.learning.model_io import (
            _assert_onnx_loadable,
        )

        bad = tmp_path / "not_onnx.onnx"
        bad.write_bytes(b"definitely not a protobuf")

        with pytest.raises(RuntimeError, match="not loadable"):
            _assert_onnx_loadable(bad)

    def test_passes_on_loadable_model(
        self, tmp_path: Path
    ) -> None:
        import onnx
        from onnx import TensorProto, helper

        from maou.app.learning.model_io import (
            _assert_onnx_loadable,
        )

        node = helper.make_node(
            "Identity", inputs=["x"], outputs=["y"], name="/id"
        )
        graph = helper.make_graph(
            [node],
            "g",
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, [1]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, [1]
                )
            ],
        )
        good = tmp_path / "good.onnx"
        onnx.save(helper.make_model(graph), good)

        _assert_onnx_loadable(good)  # 例外が出なければ成功
