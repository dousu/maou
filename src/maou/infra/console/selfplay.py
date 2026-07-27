import json
import logging
from pathlib import Path
from typing import Any

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
    "--clock-ms",
    help="Real-clock mode: initial time per side in milliseconds "
    "(0 = off). Mutually exclusive with --playouts/--movetime-ms and "
    "requires --parallel 1, because time spent is measured on the wall "
    "clock. This is the mode that exercises the time strategy.",
    type=int,
    default=0,
    show_default=True,
    required=False,
)
@click.option(
    "--byoyomi-ms",
    help="Byoyomi in milliseconds (real-clock mode only).",
    type=int,
    default=0,
    show_default=True,
    required=False,
)
@click.option(
    "--inc-ms",
    help="Fischer increment per move in milliseconds (real-clock mode "
    "only).",
    type=int,
    default=0,
    show_default=True,
    required=False,
)
@click.option(
    "--min-think-ms",
    help="Minimum thinking time per move in milliseconds (engine "
    "default 100). Only meaningful in real-clock mode.",
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--ab-mode",
    help="Play an A/B match instead of plain self-play: player A runs "
    "the lever on, player B off, everything else identical. "
    "'subtree' = subtree reuse, 'maxmoves' = in-search draw terminal, "
    "'budget' = same config with a smaller budget for B (harness sanity "
    "check), 'horizon' = time-strategy horizon (needs --clock-ms).",
    type=click.Choice(
        ["subtree", "maxmoves", "budget", "horizon"]
    ),
    default=None,
    required=False,
)
@click.option(
    "--playouts-b",
    help="Per-move playout budget for player B (--ab-mode budget; "
    "defaults to one eighth of --playouts).",
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--horizon",
    help="Assumed remaining moves used to split the clock, player A "
    "(--ab-mode horizon; defaults to the engine value).",
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--horizon-b",
    help="Assumed remaining moves for player B (--ab-mode horizon; "
    "default 25).",
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--alternate-colors/--no-alternate-colors",
    type=bool,
    is_flag=True,
    help="Swap colors every game and let the pair (2n, 2n+1) share one "
    "opening sequence, so the pair comparison cancels opening variance "
    "(default: on when --ab-mode is given).",
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
    clock_ms: int,
    byoyomi_ms: int,
    inc_ms: int,
    min_think_ms: int | None,
    ab_mode: str | None,
    playouts_b: int | None,
    horizon: int | None,
    horizon_b: int | None,
    alternate_colors: bool | None,
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

    With ``--ab-mode`` the run becomes an A/B match: player A plays the
    lever on, player B off, and everything else stays identical. The
    summary then reports A's score with a Wilson 95% interval, the Elo
    conversion, the paired (color-swapped) statistics and how much the
    mechanism actually engaged — a win rate alone cannot separate "no
    effect" from "the mechanism never fired".

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
        clock_ms: Real-clock mode initial time per side (0 = off).
        byoyomi_ms: Byoyomi in milliseconds (real-clock mode).
        inc_ms: Fischer increment per move (real-clock mode).
        min_think_ms: Minimum thinking time per move.
        ab_mode: Lever compared in an A/B match (None = plain self-play).
        playouts_b: Player B playout budget (``--ab-mode budget``).
        horizon: Player A assumed remaining moves (``--ab-mode horizon``).
        horizon_b: Player B assumed remaining moves.
        alternate_colors: Swap colors per game and pair the openings.
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
    result = selfplay_interface.selfplay(
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
        min_think_ms=min_think_ms,
        ab_mode=ab_mode,
        playouts_b=playouts_b,
        horizon_moves=horizon,
        horizon_moves_b=horizon_b,
        alternate_colors=alternate_colors,
        clock_ms=clock_ms,
        byoyomi_ms=byoyomi_ms,
        inc_ms=inc_ms,
    )

    if output is not None:
        with output.open("w", encoding="utf-8") as f:
            for record in result.records:
                f.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )
        logger.info(
            "Wrote %d game record(s) to %s",
            len(result.records),
            output,
        )

    _echo_summary(
        result.summary,
        wall_ms=result.wall_ms,
        parallel=parallel,
        ab_mode=ab_mode,
    )


def _elo(value: float | None) -> str:
    """Elo 値の表示文字列 (0% / 100% は発散するため None = "n/a")."""
    return f"{value:+.0f}" if value is not None else "n/a"


def _echo_summary(
    summary: dict[str, Any],
    *,
    wall_ms: int,
    parallel: int,
    ab_mode: str | None,
) -> None:
    """Print the run summary produced by the Rust aggregator.

    Args:
        summary: Aggregated result (``maou_usi::ab::summarize``).
        wall_ms: Wall-clock duration of the whole run in milliseconds.
        parallel: Number of games played concurrently.
        ab_mode: Lever compared in the A/B match, if any.
    """
    total_playouts = int(summary["total_playouts"])
    click.echo(f"games: {summary['games']}")
    click.echo(
        "results: "
        f"black {summary['black_wins']} / "
        f"white {summary['white_wins']} / "
        f"draw {summary['draws']}"
    )
    click.echo(
        "reasons: "
        + ", ".join(
            f"{k} {v}"
            for k, v in sorted(summary["reasons"].items())
        )
    )
    click.echo(
        f"plies: {summary['total_plies']} total, "
        f"playouts: {total_playouts} total, "
        f"game time: {int(summary['total_game_ms']) / 1000.0:.1f}s summed"
    )
    # 空回り: 終端 (証明済み・千日手・最大手数・深さ上限) に当たって葉評価に
    # 至らなかった backprop．予算は playout との合計で消費されるため，比が
    # 大きいほど公称予算に対する実探索量が小さい
    spin = int(summary.get("total_terminal_backprops", 0))
    budget_used = total_playouts + spin
    spin_pct = (
        100.0 * spin / budget_used if budget_used else 0.0
    )
    click.echo(
        f"terminal spin: {spin} backprop(s) without leaf eval "
        f"({spin_pct:.1f}% of the consumed budget)"
    )
    # 並列スループット: 分母は wall clock (対局ごとの合計ではない)．
    # --parallel を振ってこの値が伸びるかがバッチ aggregator 採否の材料．
    # 分子は**実 playout のみ** (空回りを含めると NN 評価の物理上限を超えた
    # 水増し値になる)
    wall_s = wall_ms / 1000.0
    click.echo(
        f"throughput: {total_playouts / wall_s if wall_s else 0.0:.1f} "
        f"playouts/s over {wall_s:.1f}s wall clock (parallel={parallel})"
    )
    carried = int(summary["carried_visits"])
    pct = (
        100.0 * carried / total_playouts
        if total_playouts
        else 0.0
    )
    click.echo(
        f"subtree reuse: {summary['reused_moves']} move(s) warm-started, "
        f"{carried} visits carried over ({pct:.1f}% of playouts)"
    )

    ab = summary.get("ab")
    if ab is None:
        return
    click.echo(f"A/B mode: {ab_mode} (A = lever on, B = off)")
    click.echo(
        f"A result: {ab['wins']}W {ab['draws']}D {ab['losses']}L"
    )
    click.echo(
        f"A score: {ab['score']:.1f}/{summary['games']} = "
        f"{100.0 * ab['score_rate']:.1f}% "
        f"(Wilson 95% CI [{100.0 * ab['ci_lo']:.1f}%, "
        f"{100.0 * ab['ci_hi']:.1f}%])"
    )
    click.echo(
        f"A Elo: {_elo(ab['elo'])} "
        f"[{_elo(ab['elo_lo'])}, {_elo(ab['elo_hi'])}]"
    )
    if ab["pairs"]:
        click.echo(
            f"paired: {ab['pairs']} pairs, "
            f"mean {ab['paired_mean']:+.3f} "
            f"(SE {ab['paired_se']:.3f}), "
            f"t = {ab['paired_t']:+.2f}, "
            f"A ahead in {ab['pairs_a_ahead']}/{ab['pairs']} "
            f"(tied {ab['pairs_tied']})"
        )
    if ab["time_left_a_ms"] is not None:
        click.echo(
            "time left at end (avg): "
            f"A {ab['time_left_a_ms'] / 1000.0:.1f}s / "
            f"B {ab['time_left_b_ms'] / 1000.0:.1f}s"
        )
