import logging
import time
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def build_engine(
    *,
    model_path: Path,
    output: Path,
    trt_workspace_size: int = 256,
) -> str:
    """ONNXモデルからTensorRTエンジンをビルドしてファイルに保存する．

    Args:
        model_path: ONNXモデルファイルパス．
        output: 出力エンジンファイルパス．
        trt_workspace_size: TensorRTワークスペースサイズ(MB)．

    Returns:
        ビルド時間を含む結果メッセージ．

    Raises:
        RuntimeError: TensorRT依存パッケージが未インストールの場合．
    """
    try:
        from maou.app.inference.tensorrt_inference import (
            TensorRTInference,
        )
    except ModuleNotFoundError as exc:
        missing_dep = exc.name or "tensorrt"
        raise RuntimeError(
            "build-engine requires optional dependency "
            f"'{missing_dep}'. Install with "
            "`uv sync --extra tensorrt-infer`."
        ) from exc

    start = time.perf_counter()
    serialized_engine = (
        TensorRTInference.build_engine_from_onnx(
            model_path,
            workspace_size_mb=trt_workspace_size,
        )
    )
    elapsed = time.perf_counter() - start
    TensorRTInference.save_engine(serialized_engine, output)
    return f"TensorRT engine built in {elapsed:.1f}s, saved to: {output}"
