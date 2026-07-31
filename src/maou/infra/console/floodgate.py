from pathlib import Path

import click

import maou.interface.floodgate as floodgate_interface
from maou.app.usi.floodgate import (
    DEFAULT_GAME_NAME,
    DEFAULT_HOST,
    DEFAULT_PORT,
    generate_trip,
)
from maou.infra.console.common import (
    exit_skipping_teardown,
    handle_exception,
)


@click.command("floodgate")
@click.option(
    "--login-name",
    help="Login name on the server. floodgate aggregates ratings per name "
    "and needs no registration; pick something unique (e.g. maou_test, "
    "maou_v1).",
    type=str,
    required=True,
)
@click.option(
    "--password",
    help="Secret string (trip) that distinguishes users sharing a login "
    "name. When omitted, a random one is generated and printed — pass that "
    "same value next time to keep playing as the same identity (ratings "
    "accumulate per login name + trip).",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--game-name",
    help="Game room name embedded in the CSA password field. The default is "
    "the floodgate rating room (300s + 10s increment). Pass an empty string "
    "for plain CSA servers that do not use this convention.",
    type=str,
    default=DEFAULT_GAME_NAME,
    show_default=True,
    required=False,
)
@click.option(
    "--host",
    help="CSA server hostname.",
    type=str,
    default=DEFAULT_HOST,
    show_default=True,
    required=False,
)
@click.option(
    "--port",
    help="CSA server port.",
    type=int,
    default=DEFAULT_PORT,
    show_default=True,
    required=False,
)
@click.option(
    "--games",
    help="Number of consecutive games to play (0 = unlimited). The server "
    "logs the client out after each game, so the client reconnects per game.",
    type=int,
    default=1,
    show_default=True,
    required=False,
)
@click.option(
    "--model-path",
    help="ONNX model file path. When omitted, a deterministic mock "
    "evaluator is used (connectivity checks only — it does not play well).",
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
    help="Evaluation batch size (around 64 on GPU, 8 on CPU).",
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
    help="Communication margin in milliseconds. The server measures the "
    "time consumed, so the round trip and the driver's own lag must be kept "
    "outside the search budget.",
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
    "--time-curve/--no-time-curve",
    type=bool,
    is_flag=True,
    help="Weight the per-move budget towards the conversion phase. The "
    "multiplier scales only the discretionary share (remaining time / "
    "horizon), so on `floodgate-300-10F` the 10-second Fischer increment "
    "stays intact while the 300-second bank moves to where it matters. On by "
    "default (self-play A/B: 65% over 20 games, +108 Elo).",
    default=True,
    show_default=True,
    required=False,
)
@click.option(
    "--time-curve-peak-ply",
    help="Ply where the time curve peaks (--time-curve).",
    type=int,
    default=100,
    show_default=True,
    required=False,
)
@click.option(
    "--time-curve-half-width-ply",
    help="Plies from the peak down to the floor weight (--time-curve).",
    type=int,
    default=55,
    show_default=True,
    required=False,
)
@click.option(
    "--time-curve-peak-permille",
    help="Multiplier at the peak, per mille (--time-curve).",
    type=int,
    default=2500,
    show_default=True,
    required=False,
)
@click.option(
    "--time-curve-opening-floor-permille",
    help="Multiplier before the peak, per mille (--time-curve). The opening "
    "is book-like, so search time buys little there.",
    type=int,
    default=300,
    show_default=True,
    required=False,
)
@click.option(
    "--time-curve-endgame-floor-permille",
    help="Multiplier after the peak, per mille (--time-curve). Above 1000 on "
    "purpose: the conversion phase is where a small time shortage costs whole "
    "games, so it gets more than the flat allocation, not less.",
    type=int,
    default=1200,
    show_default=True,
    required=False,
)
@click.option(
    "--keep-alive-sec",
    help="Keep-alive interval while waiting (seconds). The CSA protocol "
    "forbids sending one more often than every 30 seconds, so smaller "
    "values are raised to that floor.",
    type=int,
    default=60,
    show_default=True,
    required=False,
)
@click.option(
    "--connect-timeout-sec",
    help="Timeout for connecting and for the login response (seconds).",
    type=int,
    default=30,
    show_default=True,
    required=False,
)
@click.option(
    "--game-wait-sec",
    help="How long to wait for a game to be arranged (seconds, 0 = forever). "
    "floodgate pairs clients at :00 and :30 of every hour.",
    type=int,
    default=2400,
    show_default=True,
    required=False,
)
@click.option(
    "--reconnect-wait-sec",
    help="Delay between games before reconnecting (seconds).",
    type=int,
    default=5,
    show_default=True,
    required=False,
)
@click.option(
    "--draw-value-black",
    help="Draw value for Black in permille.",
    type=int,
    default=500,
    show_default=True,
    required=False,
)
@click.option(
    "--draw-value-white",
    help="Draw value for White in permille.",
    type=int,
    default=500,
    show_default=True,
    required=False,
)
@click.option(
    "--resign-value",
    help="Resign win-rate threshold in permille (0 = never resign).",
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
    help="Forced opening moves in USI notation (space separated).",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--root-dfpn/--no-root-dfpn",
    type=bool,
    is_flag=True,
    help="Run dfpn mate search on the root position in parallel.",
    default=True,
    show_default=True,
    required=False,
)
@click.option(
    "--root-dfpn-nodes",
    help="Node budget for the root dfpn mate search.",
    type=int,
    default=2_000_000,
    show_default=True,
    required=False,
)
@click.option(
    "--root-dfpn-depth",
    help="Search depth limit for the root dfpn mate search.",
    type=int,
    default=2047,
    show_default=True,
    required=False,
)
@click.option(
    "--defensive-mate/--no-defensive-mate",
    type=bool,
    is_flag=True,
    help="Enable defensive mate search: prove whether the side to move is "
    "being mated (root + leaves that are in check), and drop root moves that "
    "let the opponent force mate (default on). Without it the engine only asks "
    "whether it can mate, so a forced mate against it stays invisible until "
    "the last ply.",
    default=True,
    required=False,
)
@click.option(
    "--defensive-mate-threads",
    type=int,
    help="Parallelism for the defensive mate filter that screens root moves "
    "(default 1). Each root move is judged independently, so raising this "
    "uses spare CPU to reach a larger budget within the same wall clock. "
    "Memory scales with it (about 88MB per thread).",
    default=1,
    required=False,
)
@click.option(
    "--leaf-mate/--no-leaf-mate",
    type=bool,
    is_flag=True,
    help="Enable short mate search at MCTS leaves.",
    default=True,
    show_default=True,
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
    "--quiet/--no-quiet",
    type=bool,
    is_flag=True,
    help="Suppress the per-line protocol log on stderr.",
    default=False,
    required=False,
)
@handle_exception
def floodgate(
    login_name: str,
    password: str | None,
    game_name: str,
    host: str,
    port: int,
    games: int,
    model_path: Path | None,
    threads: int,
    batch_size: int,
    node_capacity: int | None,
    network_delay_ms: int,
    min_think_ms: int,
    time_curve: bool,
    time_curve_peak_ply: int,
    time_curve_half_width_ply: int,
    time_curve_peak_permille: int,
    time_curve_opening_floor_permille: int,
    time_curve_endgame_floor_permille: int,
    keep_alive_sec: int,
    connect_timeout_sec: int,
    game_wait_sec: int,
    reconnect_wait_sec: int,
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
    defensive_mate: bool,
    defensive_mate_threads: int,
    cuda: bool,
    tensorrt: bool,
    trt_cache_dir: Path | None,
    quiet: bool,
) -> None:
    """Play games on a CSA server (floodgate by default).

    Connects to the server, waits for a game to be arranged, plays it, and
    repeats for `--games` games. floodgate returns the client to a
    logged-out state after every game, so each game uses a fresh
    connection; the evaluator is loaded once and shared across them.

    Identity is the pair (login name, trip). No registration is needed:
    pick a unique `--login-name` and keep reusing the same `--password` so
    that ratings accumulate under one identity. When `--password` is
    omitted, a random trip is generated and printed — record it.

    Args:
        login_name: Login name on the server.
        password: Trip distinguishing users with the same login name.
        game_name: Game room name embedded in the CSA password field.
        host: CSA server hostname.
        port: CSA server port.
        games: Number of consecutive games (0 = unlimited).
        model_path: Path to the ONNX model file (mock evaluator if omitted).
        threads: Number of search threads.
        batch_size: Evaluation batch size.
        node_capacity: Node pool capacity.
        network_delay_ms: Communication overhead margin in milliseconds.
        min_think_ms: Minimum thinking time in milliseconds.
        time_curve: Weight the per-move budget towards the midgame.
        time_curve_peak_ply: Ply where the time curve peaks.
        time_curve_half_width_ply: Plies from the peak down to the floor.
        time_curve_peak_permille: Peak multiplier, per mille.
        time_curve_opening_floor_permille: Pre-peak multiplier, per mille.
        time_curve_endgame_floor_permille: Post-peak multiplier, per mille.
        keep_alive_sec: Keep-alive interval while waiting, in seconds.
        connect_timeout_sec: Connect and login timeout in seconds.
        game_wait_sec: Timeout waiting for a game (0 = forever).
        reconnect_wait_sec: Delay between games in seconds.
        draw_value_black: Draw value for Black in permille.
        draw_value_white: Draw value for White in permille.
        resign_value: Resign win-rate threshold in permille (0 = never).
        resign_consecutive: Consecutive below-threshold moves to resign.
        opening_script: Forced opening moves in USI notation.
        root_dfpn: Run dfpn mate search on the root position in parallel.
        root_dfpn_nodes: Node budget for the root dfpn mate search.
        root_dfpn_depth: Search depth limit for the root dfpn mate search.
        leaf_mate: Enable short mate search at MCTS leaves.
        leaf_mate_nodes: Node budget per leaf-mate df-pn call.
        leaf_mate_threads: Number of dedicated leaf-mate threads.
        defensive_mate: Detect being mated (defensive dfpn).
        defensive_mate_threads: Parallelism of the root move filter.
        cuda: Enable CUDA Execution Provider.
        tensorrt: Enable TensorRT Execution Provider.
        trt_cache_dir: TensorRT engine cache directory.
        quiet: Suppress the per-line protocol log.
    """
    trip = password if password else generate_trip()
    if not password:
        # 再開には同じ trip が要る (ログイン名 + trip で同一ユーザと見なされる)．
        # 通信ログ側 (stderr) ではなく stdout に出す — 生成した trip は失うと
        # 同じ識別子に戻れない唯一の出力で，ログを捨てる運用で消えてはならない
        click.echo(
            f"Generated trip for '{login_name}': {trip}\n"
            f"Reuse it to keep the same identity:\n"
            f"  maou floodgate --login-name {login_name} --password {trip}"
        )
    result = floodgate_interface.floodgate(
        login_name=login_name,
        trip=trip,
        game_name=game_name,
        host=host,
        port=port,
        games=games,
        keep_alive_sec=keep_alive_sec,
        connect_timeout_sec=connect_timeout_sec,
        game_wait_sec=game_wait_sec,
        reconnect_wait_sec=reconnect_wait_sec,
        verbose=not quiet,
        model_path=model_path,
        threads=threads,
        batch_size=batch_size,
        node_capacity=node_capacity,
        network_delay_ms=network_delay_ms,
        min_think_ms=min_think_ms,
        time_curve=time_curve,
        time_curve_peak_ply=time_curve_peak_ply,
        time_curve_half_width_ply=time_curve_half_width_ply,
        time_curve_peak_permille=time_curve_peak_permille,
        time_curve_opening_floor_permille=time_curve_opening_floor_permille,
        time_curve_endgame_floor_permille=time_curve_endgame_floor_permille,
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
        defensive_mate=defensive_mate,
        defensive_mate_threads=defensive_mate_threads,
        cuda=cuda,
        tensorrt=tensorrt,
        trt_engine_cache_dir=trt_cache_dir,
    )
    for index, game in enumerate(
        result.get("games", []), start=1
    ):
        click.echo(
            f"game {index}: {game['result']} ({game['reason']}) "
            f"as {game['my_color']} vs {game['opponent']} — "
            f"{game['moves']} moves, {game['elapsed_ms'] / 1000:.1f}s"
        )
    # TensorRT EP の teardown が abort するため destructor を経由せず終える
    exit_skipping_teardown(tensorrt=tensorrt)
