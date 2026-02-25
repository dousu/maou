import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from maou.app.inference.eval import Evaluation
from maou.app.inference.onnx_inference import ONNXInference
from maou.app.pre_process.feature import (
    make_board_id_positions,
    make_pieces_in_hand,
)
from maou.domain.board.shogi import Board
from maou.domain.move.label import (
    IllegalMove,
    make_usi_move_from_label,
)

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
        model_path: Optional[Path] = None
        model_type: ModelType = ModelType.ONNX
        cuda: bool = False
        num_moves: int = 5
        board_view: bool = True
        sfen: Optional[str] = None
        board: Optional[Board] = None
        trt_workspace_size_mb: int = 256
        engine_path: Optional[Path] = None

    def infer(self, config: InferenceOption) -> Dict[str, str]:
        # engine_path 指定時は自動的に TENSORRT 扱い
        effective_model_type = config.model_type
        if config.engine_path is not None:
            effective_model_type = ModelType.TENSORRT

        # model_path と engine_path のどちらかは必須
        if (
            config.model_path is None
            and config.engine_path is None
        ):
            raise ValueError(
                "Either model_path or engine_path must be provided."
            )

        # 特徴量の作成
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
        board_data = make_board_id_positions(board).astype(
            np.int32
        )
        hand_data = make_pieces_in_hand(board).astype(
            np.float32
        )

        # 推論
        policy_labels: list[int] = []
        value: float = 0.0
        if effective_model_type == ModelType.ONNX:
            policy_labels, value = ONNXInference.infer(
                config.model_path,
                board_data,
                hand_data,
                config.num_moves,
                config.cuda,
            )
        elif effective_model_type == ModelType.TENSORRT:
            try:
                from maou.app.inference.tensorrt_inference import (
                    TensorRTInference,
                )
            except ModuleNotFoundError as exc:
                missing_dep = exc.name or "tensorrt"
                raise RuntimeError(
                    "TensorRT inference requires optional dependency "
                    f"'{missing_dep}'. Install with "
                    "`uv sync --extra tensorrt-infer` before using "
                    "`--model-type TENSORRT`."
                ) from exc
            policy_labels, value = TensorRTInference.infer(
                config.model_path,
                board_data,
                hand_data,
                config.num_moves,
                config.cuda,
                workspace_size_mb=config.trt_workspace_size_mb,
                engine_path=config.engine_path,
            )
        else:
            raise ValueError(
                f"Unsupported model type: {effective_model_type.name}"
            )

        # 推論結果を評価
        result: Dict[str, str] = {}
        winrate = Evaluation.get_winrate_from_eval(value)
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
