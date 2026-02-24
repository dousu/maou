import logging
from pathlib import Path
from typing import Optional

from maou.app.inference.run import InferenceRunner, ModelType

logger: logging.Logger = logging.getLogger(__name__)


def infer(
    *,
    model_type: str,
    model_path: Optional[Path] = None,
    cuda: bool,
    num_moves: int,
    sfen: str,
    trt_workspace_size: int = 256,
    engine_path: Optional[Path] = None,
) -> str:
    """局面を推論して結果を文字列で返す．

    Args:
        model_type: モデル種別文字列（``"ONNX"`` or ``"TENSORRT"``）．
        model_path: ONNXモデルファイルパス．engine_path未指定時は必須．
        cuda: CUDA利用フラグ．
        num_moves: 上位候補手数．
        sfen: SFEN文字列．
        trt_workspace_size: TensorRTワークスペースサイズ(MB)．
        engine_path: ビルド済みTensorRTエンジンファイルパス．

    Returns:
        推論結果のフォーマット済み文字列．
    """
    try:
        model_type_enum = ModelType[model_type]
    except KeyError as e:
        raise ValueError(
            f"Invalid model type: {model_type}. Choose from {[mt.name for mt in ModelType]}"
        ) from e
    option = InferenceRunner.InferenceOption(
        model_path=model_path,
        model_type=model_type_enum,
        cuda=cuda,
        num_moves=num_moves,
        sfen=sfen,
        trt_workspace_size_mb=trt_workspace_size,
        engine_path=engine_path,
    )
    runner = InferenceRunner()
    result = runner.infer(option)
    return f"""

Policy: {result["Policy"]}
Eval: {result["Eval"]}
WinRate: {result["WinRate"]}
{result["Board"]}"""
