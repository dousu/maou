from pathlib import Path

import click

import maou.interface.usi as usi_interface
from maou.infra.console.common import (
    handle_exception,
)


@click.command("usi")
@click.option(
    "--model-path",
    help="ONNX model file path. When omitted, a deterministic mock "
    "evaluator is used (development only — announced via `info string` "
    "on isready). Can also be set from the GUI via "
    "`setoption name ModelPath`.",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    required=False,
)
@click.option(
    "--threads",
    help="Number of search threads.",
    type=int,
    default=1,
    show_default=True,
    required=False,
)
@click.option(
    "--batch-size",
    help="Evaluation batch size (use around 256 on GPU).",
    type=int,
    default=8,
    show_default=True,
    required=False,
)
@click.option(
    "--node-capacity",
    help="Node pool capacity (default 2^20 nodes).",
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--network-delay-ms",
    help="Communication overhead margin in milliseconds. The per-move "
    "search budget is reduced by this amount (the GUI/server measures "
    "elapsed time including transport).",
    type=int,
    default=1000,
    show_default=True,
    required=False,
)
@click.option(
    "--min-think-ms",
    help="Minimum thinking time in milliseconds.",
    type=int,
    default=100,
    show_default=True,
    required=False,
)
@click.option(
    "--draw-value-black",
    help="Draw value for Black in permille (default 500). The repetition / "
    "max-moves draw terminal is valued at this from the side-to-move view "
    "(Denryu-sen Black 0.4 win = 400). Also `setoption name DrawValueBlack`.",
    type=int,
    default=500,
    show_default=True,
    required=False,
)
@click.option(
    "--draw-value-white",
    help="Draw value for White in permille (default 500. Denryu-sen White "
    "0.6 win = 600). Also `setoption name DrawValueWhite`.",
    type=int,
    default=500,
    show_default=True,
    required=False,
)
@click.option(
    "--resign-value",
    help="Resign when the root win rate stays below this permille for "
    "--resign-consecutive moves (default 0 = never resign). Also "
    "`setoption name ResignValue`.",
    type=int,
    default=0,
    show_default=True,
    required=False,
)
@click.option(
    "--resign-consecutive",
    help="Consecutive below-threshold moves required to resign "
    "(with --resign-value > 0).",
    type=int,
    default=3,
    show_default=True,
    required=False,
)
@click.option(
    "--max-moves-to-draw",
    help="Move count for a drawn game (default 0 = disabled; Denryu-sen "
    "512). At/near the limit the engine always checks nyugyoku declaration "
    "and narrows its budget. Also `setoption name MaxMovesToDraw`.",
    type=int,
    default=0,
    show_default=True,
    required=False,
)
@click.option(
    "--usi-ponder/--no-usi-ponder",
    type=bool,
    is_flag=True,
    help="Enable pondering (thinking on the opponent's turn). When on, the "
    "engine declares the USI_Ponder option and appends the predicted reply "
    "to bestmove so the GUI sends `go ponder` (default on). Also "
    "`setoption name USI_Ponder`.",
    default=True,
    required=False,
)
@click.option(
    "--root-dfpn/--no-root-dfpn",
    type=bool,
    is_flag=True,
    help="Run dfpn mate search on the root position in parallel with "
    "MCTS (default on).",
    default=True,
    required=False,
)
@click.option(
    "--root-dfpn-nodes",
    help="Node budget for the root dfpn mate search.",
    type=int,
    default=2000000,
    show_default=True,
    required=False,
)
@click.option(
    "--root-dfpn-depth",
    help="Search depth limit for the root dfpn mate search (max 2047).",
    type=int,
    default=2047,
    show_default=True,
    required=False,
)
@click.option(
    "--leaf-mate/--no-leaf-mate",
    type=bool,
    is_flag=True,
    help="Enable short mate search at MCTS leaves (default on).",
    default=True,
    required=False,
)
@click.option(
    "--leaf-mate-nodes",
    help="Node budget per leaf-mate df-pn call.",
    type=int,
    default=50,
    show_default=True,
    required=False,
)
@click.option(
    "--leaf-mate-threads",
    help="Number of dedicated leaf-mate threads.",
    type=int,
    default=1,
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
def usi(
    model_path: Path | None,
    threads: int,
    batch_size: int,
    node_capacity: int | None,
    network_delay_ms: int,
    min_think_ms: int,
    draw_value_black: int,
    draw_value_white: int,
    resign_value: int,
    resign_consecutive: int,
    max_moves_to_draw: int,
    usi_ponder: bool,
    root_dfpn: bool,
    root_dfpn_nodes: int,
    root_dfpn_depth: int,
    leaf_mate: bool,
    leaf_mate_nodes: int,
    leaf_mate_threads: int,
    cuda: bool,
    tensorrt: bool,
    trt_cache_dir: Path | None,
) -> None:
    """Run the USI engine on stdin/stdout (for Shogi GUIs).

    Register this command as a USI engine in Shogidokoro / ShogiGUI /
    ShogiHome. The protocol loop, game-playing agent, and time management
    all run in Rust; stdout is reserved for the USI protocol (logs go to
    stderr). CLI flags provide initial values which USI `setoption`
    commands override. Heavy initialization (model load, TensorRT engine
    build) happens on `isready`.

    For GUIs that cannot pass command-line arguments, use the `maou-usi`
    entry point (no arguments) and configure via `setoption`.

    Args:
        model_path: Path to the ONNX model file (mock evaluator if omitted).
        threads: Number of search threads.
        batch_size: Evaluation batch size.
        node_capacity: Node pool capacity.
        network_delay_ms: Communication overhead margin in milliseconds.
        min_think_ms: Minimum thinking time in milliseconds.
        draw_value_black: Draw value for Black in permille (default 500).
        draw_value_white: Draw value for White in permille (default 500).
        resign_value: Resign win-rate threshold in permille (0 = never).
        resign_consecutive: Consecutive below-threshold moves to resign.
        max_moves_to_draw: Move count for a drawn game (0 = disabled).
        usi_ponder: Enable pondering on the opponent's turn (default on).
        root_dfpn: Run dfpn mate search on the root position in parallel.
        root_dfpn_nodes: Node budget for the root dfpn mate search.
        root_dfpn_depth: Search depth limit for the root dfpn mate search.
        leaf_mate: Enable short mate search at MCTS leaves (async).
        leaf_mate_nodes: Node budget per leaf-mate df-pn call.
        leaf_mate_threads: Number of dedicated leaf-mate threads.
        cuda: Enable CUDA Execution Provider.
        tensorrt: Enable TensorRT Execution Provider.
        trt_cache_dir: TensorRT engine cache directory.
    """
    usi_interface.usi(
        model_path=model_path,
        threads=threads,
        batch_size=batch_size,
        node_capacity=node_capacity,
        network_delay_ms=network_delay_ms,
        min_think_ms=min_think_ms,
        draw_value_black=draw_value_black,
        draw_value_white=draw_value_white,
        resign_value=resign_value,
        resign_consecutive=resign_consecutive,
        max_moves_to_draw=max_moves_to_draw,
        usi_ponder=usi_ponder,
        root_dfpn=root_dfpn,
        root_dfpn_nodes=root_dfpn_nodes,
        root_dfpn_depth=root_dfpn_depth,
        leaf_mate=leaf_mate,
        leaf_mate_nodes=leaf_mate_nodes,
        leaf_mate_threads=leaf_mate_threads,
        cuda=cuda,
        tensorrt=tensorrt,
        trt_engine_cache_dir=trt_cache_dir,
    )


def main() -> None:
    """`maou-usi` console script のエントリポイント (GUI 登録用)．

    引数なしで USI エンジンを起動する (起動引数を渡せない GUI 向け．
    設定は USI `setoption` で受ける)．
    """
    usi.main(args=[], prog_name="maou-usi")
