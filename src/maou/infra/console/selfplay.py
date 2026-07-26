import json
import logging
from collections import Counter
from pathlib import Path

import click

import maou.interface.selfplay as selfplay_interface
from maou.infra.console.common import (
    handle_exception,
)

logger: logging.Logger = logging.getLogger(__name__)


@click.command("selfplay")
@click.option(
    "--model-path",
    help="ONNX model file path. When omitted, a deterministic mock "
    "evaluator is used (development only).",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    required=False,
)
@click.option(
    "--games",
    help="Number of games to play.",
    type=int,
    default=1,
    show_default=True,
    required=False,
)
@click.option(
    "--parallel",
    help="Number of games played concurrently (worker threads). The "
    "shared evaluator serializes GPU/ONNX calls internally.",
    type=int,
    default=1,
    show_default=True,
    required=False,
)
@click.option(
    "--playouts",
    help="Per-move playout budget (mutually exclusive with "
    "--movetime-ms; defaults to 800 when neither is given).",
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--movetime-ms",
    help="Per-move thinking time in milliseconds (mutually exclusive "
    "with --playouts).",
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--max-moves",
    help="Move count for a drawn game (Denryu-sen rule 512). A valid "
    "nyugyoku declaration at the limit still wins.",
    type=int,
    default=512,
    show_default=True,
    required=False,
)
@click.option(
    "--sfen",
    help="Starting position SFEN (default: even initial position).",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--opening-random-plies",
    help="Play this many uniformly random legal moves at the start of "
    "each game (game diversification; deterministic per --seed).",
    type=int,
    default=0,
    show_default=True,
    required=False,
)
@click.option(
    "--seed",
    help="Random seed for --opening-random-plies (mixed with the game "
    "index so every game gets a distinct sequence).",
    type=int,
    default=0,
    show_default=True,
    required=False,
)
@click.option(
    "--output",
    help="Write per-game records as JSON Lines to this file.",
    type=click.Path(path_type=Path),
    default=None,
    required=False,
)
@click.option(
    "--threads",
    help="Number of search threads per move.",
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
    help="Node pool capacity per agent (each game holds two agents; "
    "pools are pre-allocated, so keep this modest for many parallel "
    "games).",
    type=int,
    default=65536,
    show_default=True,
    required=False,
)
@click.option(
    "--draw-value-black",
    help="Draw value for Black in permille (Denryu-sen Black 0.4 win "
    "= 400).",
    type=int,
    default=500,
    show_default=True,
    required=False,
)
@click.option(
    "--draw-value-white",
    help="Draw value for White in permille (Denryu-sen White 0.6 win "
    "= 600).",
    type=int,
    default=500,
    show_default=True,
    required=False,
)
@click.option(
    "--resign-value",
    help="Resign when the root win rate stays below this permille for "
    "--resign-consecutive moves (default 0 = never resign).",
    type=int,
    default=0,
    show_default=True,
    required=False,
)
@click.option(
    "--resign-consecutive",
    help="Consecutive below-threshold moves required to resign.",
    type=int,
    default=3,
    show_default=True,
    required=False,
)
@click.option(
    "--opening-script",
    help="Forced opening move sequence in USI notation, space-separated. "
    "Applied to both agents (e.g. the HWT king-shuffle handicap).",
    type=str,
    default=None,
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
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress per-game progress lines on stderr.",
    default=False,
    required=False,
)
@handle_exception
def selfplay(
    model_path: Path | None,
    games: int,
    parallel: int,
    playouts: int | None,
    movetime_ms: int | None,
    max_moves: int,
    sfen: str | None,
    opening_random_plies: int,
    seed: int,
    output: Path | None,
    threads: int,
    batch_size: int,
    node_capacity: int | None,
    draw_value_black: int,
    draw_value_white: int,
    resign_value: int,
    resign_consecutive: int,
    opening_script: str | None,
    root_dfpn: bool,
    root_dfpn_nodes: int,
    root_dfpn_depth: int,
    leaf_mate: bool,
    leaf_mate_nodes: int,
    leaf_mate_threads: int,
    cuda: bool,
    tensorrt: bool,
    trt_cache_dir: Path | None,
    quiet: bool,
) -> None:
    """Run in-process self-play games and report the results.

    One game drives two agents (independent search trees for Black and
    White) directly, without stdio or subprocesses. The evaluator (ONNX
    session, TensorRT cache) is shared by every game in the process, so
    the model is loaded and warmed up exactly once. Game termination
    (nyugyoku declaration, sennichite incl. perpetual check, max moves,
    resignation) uses the same implementation as the USI engine.

    Args:
        model_path: Path to the ONNX model file (mock evaluator if
            omitted).
        games: Number of games to play.
        parallel: Number of games played concurrently.
        playouts: Per-move playout budget.
        movetime_ms: Per-move thinking time in milliseconds.
        max_moves: Move count for a drawn game.
        sfen: Starting position SFEN.
        opening_random_plies: Random opening plies per game.
        seed: Random seed for the opening randomization.
        output: JSON Lines output path for per-game records.
        threads: Number of search threads per move.
        batch_size: Evaluation batch size.
        node_capacity: Node pool capacity per agent.
        draw_value_black: Draw value for Black in permille.
        draw_value_white: Draw value for White in permille.
        resign_value: Resign win-rate threshold in permille (0 = never).
        resign_consecutive: Consecutive below-threshold moves to resign.
        opening_script: Forced opening moves in USI notation.
        root_dfpn: Run dfpn mate search on the root position in parallel.
        root_dfpn_nodes: Node budget for the root dfpn mate search.
        root_dfpn_depth: Search depth limit for the root dfpn mate search.
        leaf_mate: Enable short mate search at MCTS leaves (async).
        leaf_mate_nodes: Node budget per leaf-mate df-pn call.
        leaf_mate_threads: Number of dedicated leaf-mate threads.
        cuda: Enable CUDA Execution Provider.
        tensorrt: Enable TensorRT Execution Provider.
        trt_cache_dir: TensorRT engine cache directory.
        quiet: Suppress per-game progress lines.
    """
    records = selfplay_interface.selfplay(
        model_path=model_path,
        games=games,
        parallel=parallel,
        playouts=playouts,
        movetime_ms=movetime_ms,
        max_moves=max_moves,
        sfen=sfen,
        opening_random_plies=opening_random_plies,
        seed=seed,
        verbose=not quiet,
        threads=threads,
        batch_size=batch_size,
        node_capacity=node_capacity,
        draw_value_black=draw_value_black,
        draw_value_white=draw_value_white,
        resign_value=resign_value,
        resign_consecutive=resign_consecutive,
        opening_script=opening_script,
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
        with output.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )
        logger.info(
            "Wrote %d game record(s) to %s",
            len(records),
            output,
        )

    winners = Counter(str(r["winner"]) for r in records)
    reasons = Counter(str(r["reason"]) for r in records)
    total_plies = sum(int(r["plies"]) for r in records)
    total_playouts = sum(int(r["playouts"]) for r in records)
    total_ms = sum(int(r["elapsed_ms"]) for r in records)
    click.echo(f"games: {len(records)}")
    click.echo(
        "results: "
        f"black {winners.get('black', 0)} / "
        f"white {winners.get('white', 0)} / "
        f"draw {winners.get('None', 0)}"
    )
    click.echo(
        "reasons: "
        + ", ".join(
            f"{k} {v}" for k, v in sorted(reasons.items())
        )
    )
    click.echo(
        f"plies: {total_plies} total, playouts: {total_playouts} total, "
        f"game time: {total_ms / 1000.0:.1f}s summed"
    )
    reused = sum(int(r["reused_moves"]) for r in records)
    carried = sum(int(r["carried_visits"]) for r in records)
    pct = (
        100.0 * carried / total_playouts
        if total_playouts
        else 0.0
    )
    click.echo(
        f"subtree reuse: {reused} move(s) warm-started, "
        f"{carried} visits carried over ({pct:.1f}% of playouts)"
    )
