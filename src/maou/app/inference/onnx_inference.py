import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger: logging.Logger = logging.getLogger(__name__)


class ONNXInference:
    @staticmethod
    def infer(
        path: Path,
        board_data: np.ndarray,
        hand_data: np.ndarray,
        num: int,
        cuda_available: bool,
    ) -> tuple[list[int], float]:
        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        options.intra_op_num_threads = 1
        options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )
        if cuda_available:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = [
                "CPUExecutionProvider",
            ]
        session = ort.InferenceSession(
            path, sess_options=options, providers=providers
        )
        batched_board = np.expand_dims(board_data, axis=0)
        batched_hand = np.expand_dims(hand_data, axis=0)
        outputs = session.run(
            ["policy", "value"],
            {"board": batched_board, "hand": batched_hand},
        )
        policy_labels: list[int] = list(
            np.argsort(outputs[0][0])[::-1][:num]  # type: ignore
        )
        value: float = outputs[1][0].item()  # type: ignore
        return policy_labels, value
