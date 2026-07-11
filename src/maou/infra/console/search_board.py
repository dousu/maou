from pathlib import Path

import click

import maou.interface.search as search_interface
from maou.infra.console.common import (
    handle_exception,
)


@click.command("search")
@click.option(
    "--sfen",
    help="Set board position in SFEN notation.",
    type=str,
    required=True,
)
@click.option(
    "--moves",
    help="Moves in USI format applied from the SFEN position "
    "(space-separated, like the USI `position ... moves ...` command). "
    "Intermediate positions are used as game history for "
    "repetition (sennichite) detection.",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--model-path",
    help="ONNX model file path. When omitted, a deterministic mock "
    "evaluator is used (API verification only — move quality is "
    "meaningless). Requires a wheel built with the 'onnx' cargo feature.",
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
    "--playouts",
    help="Maximum number of playouts.",
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--time-ms",
    help="Time limit in milliseconds. Defaults to 1000 when neither "
    "--playouts nor --time-ms is specified.",
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--num-moves",
    help="Number of candidate moves to display.",
    type=int,
    default=5,
    show_default=True,
    required=False,
)
@click.option(
    "--root-dfpn/--no-root-dfpn",
    type=bool,
    is_flag=True,
    help="Run dfpn mate search on the root position in parallel with "
    "MCTS (default on; NN-independent, ~free on quiet positions). When a "
    "mate is proven the search stops immediately and the mating sequence "
    "is returned as PV.",
    default=True,
    required=False,
)
@click.option(
    "--root-dfpn-nodes",
    help="Node budget for the root dfpn mate search. Larger reaches "
    "deeper mates (NN-independent) at the cost of a larger transposition "
    "table per search (~256MB at 2M).",
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
    help="Enable short mate search at MCTS leaves (default on). Search "
    "threads only enqueue mate requests (they never block); dedicated "
    "mate threads run the df-pn on spare CPU and mark proven leaves, so "
    "search NPS is unaffected (dlshogi-style leaf mate search).",
    default=True,
    required=False,
)
@click.option(
    "--leaf-mate-nodes",
    help="Node budget per leaf-mate df-pn call (smaller = cheaper and "
    "restricts to shorter mates).",
    type=int,
    default=50,
    show_default=True,
    required=False,
)
@click.option(
    "--leaf-mate-threads",
    help="Number of dedicated leaf-mate threads (use spare CPU cores).",
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
def search_board(
    sfen: str,
    moves: str | None,
    model_path: Path | None,
    threads: int,
    batch_size: int,
    playouts: int | None,
    time_ms: int | None,
    num_moves: int,
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
    """Search a Shogi position with the MCTS engine (Rust maou_search).

    This command runs the PUCT-based Monte Carlo tree search on the given
    SFEN position and prints the best move, evaluation score, win rate,
    principal variation, candidate moves, and search statistics. The
    evaluation score uses the same Ponanza-style conversion
    (eval = 600 x logit) as `maou evaluate`, applied to the searched
    win rate.

    Args:
        sfen: SFEN string representing the base position.
        moves: USI moves applied from the SFEN position (space-separated).
        model_path: Path to the ONNX model file.
        threads: Number of search threads.
        batch_size: Evaluation batch size.
        playouts: Maximum number of playouts.
        time_ms: Time limit in milliseconds.
        num_moves: Number of candidate moves to display.
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
    if (cuda or tensorrt) and model_path is None:
        raise click.UsageError(
            "--cuda / --tensorrt require --model-path."
        )
    click.echo(
        search_interface.search(
            sfen=sfen,
            moves=moves,
            model_path=model_path,
            threads=threads,
            batch_size=batch_size,
            playouts=playouts,
            time_ms=time_ms,
            num_moves=num_moves,
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
    )
