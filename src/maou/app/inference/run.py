import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from maou.app.inference.eval import Evaluation
from maou.app.inference.onnx_inference import ONNXInference
from maou.app.pre_process.feature import make_board_id_positions
from maou.app.pre_process.label import (
    IllegalMove,
    make_usi_move_from_label,
)
from maou.domain.board.shogi import Board

if TYPE_CHECKING:
    pass


class ModelType(Enum):
    ONNX = auto()
    TENSORRT = auto()


class InferenceRunner:
    """Inference Runner.
    ONNXやTensorRTで推論を実行した結果を上位n手計算する
    候補手はUSIで表現される
    評価値は勝率としても表現される
    GPUを使いたい場合はオプションで指定する
    局面情報はsfenまたはドメインのBoardで与える
    """

    logger: logging.Logger = logging.getLogger(__name__)

    @dataclass(kw_only=True, frozen=True)
    class InferenceOption:
        model_path: Path
        model_type: ModelType
        cuda: bool = False
        num_moves: int = 5
        board_view: bool = True
        sfen: Optional[str] = None
        board: Optional[Board] = None

    def infer(self, config: InferenceOption) -> Dict[str, str]:
        # 特徴量の作成
        input_data: np.ndarray
        board: Board
        if config.sfen is not None:
            board = Board()
            board.set_sfen(config.sfen)
        elif config.board is not None:
            board = config.board
        else:
            raise ValueError(
                "Either sfen or board must be provided."
            )
        input_data = make_board_id_positions(board).astype(
            np.int64
        )

        # 推論
        policy_labels: list[int] = []
        value: float = 0.0
        if config.model_type == ModelType.ONNX:
            policy_labels, value = ONNXInference.infer(
                config.model_path,
                input_data,
                config.num_moves,
                config.cuda,
            )
        elif config.model_type == ModelType.TENSORRT:
            try:
                from maou.app.inference.tensorrt_inference import (
                    TensorRTInference,
                )
            except ModuleNotFoundError as exc:
                missing_dep = exc.name or "tensorrt"
                raise RuntimeError(
                    "TensorRT inference requires optional dependency "
                    f"'{missing_dep}'. Install with "
                    "`poetry install -E tensorrt-infer` before using "
                    "`--model-type TENSORRT`."
                ) from exc
            policy_labels, value = TensorRTInference.infer(
                config.model_path,
                input_data,
                config.num_moves,
                config.cuda,
            )
        else:
            raise ValueError(
                f"Unsupported model type: {config.model_type.name}"
            )

        # 推論結果を評価
        result: Dict[str, str] = {}
        winrate = Evaluation.get_winrate_from_eval(
            board.get_turn(), value
        )
        eval = Evaluation.get_eval_from_winrate(winrate)
        usis: list[str] = []
        for policy_label in policy_labels:
            try:
                usi = make_usi_move_from_label(
                    board=board, label=policy_label
                )
            except IllegalMove as e:
                self.logger.error(
                    f"Failed to convert label {policy_label} to USI move: {e}"
                )
                usi = "failed to convert"
            usis.append(usi)
        result["Policy"] = ", ".join(usis)
        result["Eval"] = f"{eval:.2f}"
        result["WinRate"] = f"{winrate:.4f}"
        if config.board_view:
            result["Board"] = board.to_pretty_board()

        return result
