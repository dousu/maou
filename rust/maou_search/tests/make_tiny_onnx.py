"""OnnxEvaluator の検証用に I/O 契約互換の極小 ONNX モデルを生成する．

実モデル (model_io.py の torch.onnx.export) と同じ入出力契約を持つ:
- 入力 board: int32 (batch_size, 9, 9) / hand: float32 (batch_size, 14)
- 出力 policy: float32 (batch_size, 1496) / value: float32 (batch_size, 1)

重みは seed 固定の乱数 (推論スループットと配線の検証用で棋力はない)．
torch 不要 (onnx パッケージのみ)．使用例:

    uv run --with onnx python rust/maou_search/tests/make_tiny_onnx.py /tmp/tiny.onnx
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

MOVE_LABELS_NUM = 1496


def main(out_path: str) -> None:
    rng = np.random.default_rng(20260708)
    w_policy = rng.standard_normal((95, MOVE_LABELS_NUM)).astype(np.float32) * 0.1
    b_policy = np.zeros(MOVE_LABELS_NUM, dtype=np.float32)
    w_value = rng.standard_normal((95, 1)).astype(np.float32) * 0.1
    b_value = np.zeros(1, dtype=np.float32)

    nodes = [
        helper.make_node("Cast", ["board"], ["board_f"], to=TensorProto.FLOAT),
        helper.make_node("Flatten", ["board_f"], ["board_flat"], axis=1),
        helper.make_node("Concat", ["board_flat", "hand"], ["features"], axis=1),
        helper.make_node("Gemm", ["features", "w_policy", "b_policy"], ["policy"]),
        helper.make_node("Gemm", ["features", "w_value", "b_value"], ["value"]),
    ]
    graph = helper.make_graph(
        nodes,
        "maou_tiny_test_model",
        inputs=[
            helper.make_tensor_value_info(
                "board", TensorProto.INT32, ["batch_size", 9, 9]
            ),
            helper.make_tensor_value_info(
                "hand", TensorProto.FLOAT, ["batch_size", 14]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "policy", TensorProto.FLOAT, ["batch_size", MOVE_LABELS_NUM]
            ),
            helper.make_tensor_value_info(
                "value", TensorProto.FLOAT, ["batch_size", 1]
            ),
        ],
        initializer=[
            helper.make_tensor(
                "w_policy", TensorProto.FLOAT, w_policy.shape, w_policy.flatten()
            ),
            helper.make_tensor(
                "b_policy", TensorProto.FLOAT, b_policy.shape, b_policy
            ),
            helper.make_tensor(
                "w_value", TensorProto.FLOAT, w_value.shape, w_value.flatten()
            ),
            helper.make_tensor("b_value", TensorProto.FLOAT, b_value.shape, b_value),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10
    )
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny_test_model.onnx")
