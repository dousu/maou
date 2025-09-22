import logging
from pathlib import Path

from maou.app.inference.run import InferenceRunner, ModelType

logger: logging.Logger = logging.getLogger(__name__)


def infer(
    *,
    model_type: str,
    model_path: Path,
    cuda: bool,
    num_moves: int,
    sfen: str,
) -> str:
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
    )
    runner = InferenceRunner()
    result = runner.infer(option)
    return f"""

Policy: {result["Policy"]}
Eval: {result["Eval"]}
WinRate: {result["WinRate"]}
{result["Board"]}"""
