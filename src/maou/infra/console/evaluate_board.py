from pathlib import Path
from typing import Optional

import click

import maou.interface.infer as infer
from maou.infra.console.common import (
    handle_exception,
)


@click.command("evaluate")
@click.option(
    "--model-type",
    help="Input format: 'ONNX' or 'TENSORRT'.",
    type=str,
    default="ONNX",
    required=True,
)
@click.option(
    "--model-path",
    help="ONNX Model file path. "
    "Required when --engine-path is not specified.",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    required=False,
)
@click.option(
    "--cuda/--no-cuda",
    type=bool,
    is_flag=True,
    help="Enable CUDA inference.",
    default=False,
    required=False,
)
@click.option(
    "--num-moves",
    help="Number of candidate moves.",
    type=int,
    default=5,
    required=False,
)
@click.option(
    "--sfen",
    help="Set board position in SFEN notation.",
    type=str,
    required=True,
)
@click.option(
    "--trt-workspace-size",
    help="TensorRT workspace size in MB. "
    "Default is sufficient for this project's models. "
    "Increase for larger models or max speed. "
    "Decrease if GPU memory is limited.",
    type=int,
    default=256,
    show_default=True,
    required=False,
)
@click.option(
    "--engine-path",
    help="Pre-built TensorRT engine file path. "
    "When specified, the engine is loaded from this file "
    "and ONNX-to-TensorRT build is skipped. "
    "Build an engine first with `maou build-engine`.",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    required=False,
)
@handle_exception
def evaluate_board(
    model_type: str,
    model_path: Optional[Path],
    cuda: bool,
    num_moves: int,
    sfen: str,
    trt_workspace_size: int,
    engine_path: Optional[Path],
) -> None:
    """Evaluate a Shogi board position using a specified model.

    This command uses the provided model to evaluate the given Shogi board position
    in SFEN format. It returns the top moves, evaluation score, win rate, and a
    visual representation of the board.

    Args:
        model_type: Type of the model to use for inference (e.g., 'ONNX', 'TENSORRT').
        model_path: Path to the model file.
        num_moves: Number of top moves to return.
        sfen: SFEN string representing the board position.
        trt_workspace_size: TensorRT workspace size in MB.
        engine_path: Pre-built TensorRT engine file path.
    """
    if engine_path is None and model_path is None:
        raise click.UsageError(
            "--model-path or --engine-path is required."
        )
    click.echo(
        infer.infer(
            model_type=model_type,
            model_path=model_path,
            num_moves=num_moves,
            cuda=cuda,
            sfen=sfen,
            trt_workspace_size=trt_workspace_size,
            engine_path=engine_path,
        )
    )
