from pathlib import Path

import click

from maou.infra.console.common import handle_exception
from maou.interface import analyzer as analyzer_interface


@click.command("analyze-game")
@click.option(
    "--input-path",
    help="Path to the game record file (CSA / KIF).",
    type=click.Path(
        exists=True, dir_okay=False, path_type=Path
    ),
    required=True,
)
@click.option(
    "--input-format",
    help=(
        "Game record format. Auto-detected from the file extension "
        "(.csa / .kif / .kifu) when omitted."
    ),
    type=click.Choice(["csa", "kif"]),
    default=None,
    required=False,
)
@click.option(
    "--model-path",
    help=(
        "Path to the ONNX model file. Uses a deterministic mock evaluator "
        "when omitted (for development only)."
    ),
    type=click.Path(exists=True, path_type=Path),
    default=None,
    required=False,
)
@click.option(
    "--time-ms",
    help=(
        "Time budget per position in milliseconds. Defaults to 1000 when "
        "no budget option is given. Mutually exclusive with "
        "--total-time-ms / --playouts."
    ),
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--total-time-ms",
    help=(
        "Total time budget for the whole game in milliseconds, divided "
        "equally across positions. Mutually exclusive with "
        "--time-ms / --playouts."
    ),
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--playouts",
    help=(
        "Playout budget per position. Mutually exclusive with "
        "--time-ms / --total-time-ms."
    ),
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--num-candidates",
    help="Number of candidate moves recorded per position in the JSON.",
    type=int,
    default=5,
    required=False,
)
@click.option(
    "--output",
    help=(
        "Write the JSON report to this file and print a human-readable "
        "summary to stdout. When omitted, the JSON is printed to stdout."
    ),
    type=click.Path(path_type=Path),
    default=None,
    required=False,
)
@click.option(
    "--threads",
    help="Number of search threads.",
    type=int,
    default=1,
    required=False,
)
@click.option(
    "--batch-size",
    help="Evaluation batch size.",
    type=int,
    default=8,
    required=False,
)
@click.option(
    "--root-dfpn/--no-root-dfpn",
    help="Run dfpn mate search on each root position in parallel.",
    default=True,
    required=False,
)
@click.option(
    "--root-dfpn-nodes",
    help="Node budget for the root dfpn mate search.",
    type=int,
    default=2_000_000,
    required=False,
)
@click.option(
    "--root-dfpn-depth",
    help="Search depth limit for the root dfpn mate search (max 2047).",
    type=int,
    default=2047,
    required=False,
)
@click.option(
    "--leaf-mate/--no-leaf-mate",
    help="Enable short mate search at MCTS leaves (async).",
    default=True,
    required=False,
)
@click.option(
    "--leaf-mate-nodes",
    help="Node budget per leaf-mate df-pn call.",
    type=int,
    default=50,
    required=False,
)
@click.option(
    "--leaf-mate-threads",
    help="Number of dedicated leaf-mate threads.",
    type=int,
    default=1,
    required=False,
)
@click.option(
    "--cuda/--no-cuda",
    help="Enable CUDA Execution Provider.",
    default=False,
    required=False,
)
@click.option(
    "--tensorrt/--no-tensorrt",
    help="Enable TensorRT Execution Provider.",
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
def analyze_game(
    input_path: Path,
    input_format: str | None,
    model_path: Path | None,
    time_ms: int | None,
    total_time_ms: int | None,
    playouts: int | None,
    num_candidates: int,
    output: Path | None,
    threads: int,
    batch_size: int,
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
    """Analyze a whole game record with per-position MCTS search.

    This command parses a single game from a CSA/KIF file and runs the
    MCTS engine (the same engine as `maou search`) on the position before
    each move. The evaluator is loaded once and reused across all
    positions. The per-position budget is decided by a budget allocator
    (fixed time per position, total time divided equally, or fixed
    playouts) — the search itself only consumes the budget it is given.
    The result is a machine-readable JSON report (per-position best move,
    win rate, PV, comparison with the played move) plus a human-readable
    summary (best-move match rate, biggest win-rate losses, mates found).

    Args:
        input_path: Path to the game record file (CSA / KIF).
        input_format: Game record format; auto-detected when omitted.
        model_path: Path to the ONNX model file.
        time_ms: Time budget per position in milliseconds.
        total_time_ms: Total time budget divided equally across positions.
        playouts: Playout budget per position.
        num_candidates: Candidate moves recorded per position.
        output: JSON report destination; JSON goes to stdout when omitted.
        threads: Number of search threads.
        batch_size: Evaluation batch size.
        root_dfpn: Run dfpn mate search on each root position in parallel.
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
    if (
        sum(
            v is not None
            for v in (time_ms, total_time_ms, playouts)
        )
        > 1
    ):
        raise click.UsageError(
            "Specify at most one of --time-ms / --total-time-ms / "
            "--playouts."
        )
    json_str, summary = analyzer_interface.analyze_game(
        input_path=input_path,
        input_format=input_format,
        model_path=model_path,
        time_ms=time_ms,
        total_time_ms=total_time_ms,
        playouts=playouts,
        num_candidates=num_candidates,
        threads=threads,
        batch_size=batch_size,
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
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_str + "\n", encoding="utf-8")
        click.echo(summary)
        click.echo(f"JSON report: {output}")
    else:
        click.echo(json_str)
