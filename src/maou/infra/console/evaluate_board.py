from pathlib import Path

import click

import maou.interface.infer as infer
from maou.infra.console.common import (
    exit_skipping_teardown,
    handle_exception,
)


@click.command("evaluate")
@click.option(
    "--sfen",
    help="Set board position in SFEN notation.",
    type=str,
    required=True,
)
@click.option(
    "--model-path",
    help="ONNX model file path. "
    "Without it a deterministic mock evaluator is used "
    "(for API checks only; the output is meaningless).",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    required=False,
)
@click.option(
    "--num-moves",
    help="Number of candidate moves.",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    required=False,
)
@click.option(
    "--cuda/--no-cuda",
    type=bool,
    is_flag=True,
    help="Enable CUDA Execution Provider "
    "(requires a wheel built with 'onnx-cuda').",
    default=False,
    required=False,
)
@click.option(
    "--tensorrt/--no-tensorrt",
    type=bool,
    is_flag=True,
    help="Enable TensorRT Execution Provider "
    "(requires a wheel built with 'onnx-tensorrt').",
    default=False,
    required=False,
)
@click.option(
    "--trt-cache-dir",
    help="TensorRT engine cache directory.",
    type=click.Path(path_type=Path),
    default=None,
    required=False,
)
@handle_exception
def evaluate_board(
    sfen: str,
    model_path: Path | None,
    num_moves: int,
    cuda: bool,
    tensorrt: bool,
    trt_cache_dir: Path | None,
) -> None:
    """Evaluate a Shogi board position with a single network pass.

    Runs no search: the network is evaluated once and the top candidate moves
    (legal moves only, ordered by policy prior), the evaluation score, the win
    rate, and a visual representation of the board are printed.

    Args:
        sfen: SFEN string representing the board position.
        model_path: Path to the ONNX model file.
        num_moves: Number of top moves to return.
        cuda: Enable CUDA Execution Provider.
        tensorrt: Enable TensorRT Execution Provider.
        trt_cache_dir: TensorRT engine cache directory.
    """
    if (cuda or tensorrt) and model_path is None:
        raise click.UsageError(
            "--cuda / --tensorrt require --model-path."
        )
    click.echo(
        infer.infer(
            model_path=model_path,
            num_moves=num_moves,
            sfen=sfen,
            cuda=cuda,
            tensorrt=tensorrt,
            trt_cache_dir=trt_cache_dir,
        )
    )
    # TensorRT EP の teardown はヒープを壊して abort する — 出力後に
    # destructor を経由せず終える (common.exit_skipping_teardown の docstring)
    exit_skipping_teardown(tensorrt=tensorrt)
