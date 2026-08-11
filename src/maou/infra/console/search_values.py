"""floodgate 局面を探索して value 教師を作る CLI コマンド (インフラ層)．

``maou utility search-values``．探索は Rust (`maou_search`) 側で行うため
torch を必要としない．
"""

from pathlib import Path

import click

from maou.infra.console.common import handle_exception


@click.command("search-values")
@click.option(
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="HCPE directory (or .feather file) to read positions from.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write shards into (part_NNNNNNNN.feather, each "
    "holding id, searchWinRate, playouts, stop, elapsedMs). Every flush "
    "adds one shard instead of rewriting the whole output, so the cost "
    "per flush stays flat as the run grows. Pass the directory to "
    "`maou pre-process --search-value-path`.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="ONNX model. Without it a deterministic mock evaluator is used "
    "(for API checks only; the values are meaningless).",
)
@click.option(
    "--min-ply",
    type=click.IntRange(min=0),
    default=60,
    show_default=True,
    help="Only search positions at or after this ply. Memorisation "
    "concentrates in the late game, and early positions are shared across "
    "many games so their targets are already averaged.",
)
@click.option(
    "--max-positions",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Cap on how many positions to search (0 = no cap). Use it to fit a "
    "GPU budget; the sample is drawn with --seed and is label-independent.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Sampling seed used when --max-positions applies.",
)
@click.option(
    "--playouts",
    type=click.IntRange(min=1),
    default=800,
    show_default=True,
    help="Playout budget per position.",
)
@click.option(
    "--time-ms",
    type=click.IntRange(min=1),
    default=None,
    help="Wall-clock budget per position in milliseconds (optional).",
)
@click.option(
    "--threads",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Search threads.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Evaluation batch size. Use 64 or more on GPU.",
)
@click.option(
    "--root-dfpn/--no-root-dfpn",
    default=True,
    show_default=True,
    help="Run the root parallel dfpn mate search.",
)
@click.option(
    "--root-dfpn-nodes",
    type=click.IntRange(min=1),
    default=None,
    help="Node budget for the root dfpn search (Rust default: 2,000,000). "
    "--min-ply 60 deliberately selects tactically dense positions, so mate "
    "proving can dominate the wall time; lower this to find out.",
)
@click.option(
    "--root-dfpn-depth",
    type=click.IntRange(min=1),
    default=None,
    help="Depth limit for the root dfpn search (Rust default).",
)
@click.option(
    "--leaf-mate/--no-leaf-mate",
    default=True,
    show_default=True,
    help="Run the short mate search at MCTS leaves.",
)
@click.option(
    "--leaf-mate-nodes",
    type=click.IntRange(min=1),
    default=None,
    help="Node budget per leaf-mate call (Rust default: 50).",
)
@click.option(
    "--leaf-mate-threads",
    type=click.IntRange(min=1),
    default=None,
    help="Dedicated leaf-mate threads (Rust default: 1).",
)
@click.option(
    "--defensive-mate/--no-defensive-mate",
    default=None,
    help="Run the defensive mate search (root losing-move filter). It is "
    "CPU-side work per position. Omit to keep the Rust default.",
)
@click.option(
    "--defensive-mate-threads",
    type=click.IntRange(min=1),
    default=None,
    help="Parallelism of the root losing-move filter (Rust default).",
)
@click.option(
    "--pad-buckets/--no-pad-buckets",
    default=None,
    help="Round TensorRT evaluation batches up to power-of-two buckets "
    "instead of padding every batch to --batch-size. This workload runs one "
    "cold search per position, so early playouts have few leaves and fixed "
    "padding makes a 1-leaf batch cost a full --batch-size batch. Omit to "
    "keep the Rust default (fixed padding).",
)
@click.option(
    "--cuda/--no-cuda",
    default=False,
    show_default=True,
    help="Enable CUDA Execution Provider (requires a wheel built with "
    "'onnx-cuda').",
)
@click.option(
    "--tensorrt/--no-tensorrt",
    default=False,
    show_default=True,
    help="Enable TensorRT Execution Provider (requires a wheel built with "
    "'onnx-tensorrt').",
)
@click.option(
    "--trt-cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="TensorRT engine cache directory.",
)
@click.option(
    "--flush-interval",
    type=click.IntRange(min=1),
    default=500,
    show_default=True,
    help="Write the results out every this many positions. An interrupted "
    "run keeps what it had, and --resume continues from there. Each flush "
    "adds a small pending_*.feather; it never rewrites what is already "
    "there.",
)
@click.option(
    "--shard-rows",
    type=click.IntRange(min=1),
    default=5_000_000,
    show_default=True,
    help="Target rows per finished shard (part_*.feather). Pending files "
    "are folded into one shard once this many rows have accumulated, so "
    "the run leaves a few large files rather than thousands of tiny ones. "
    "A row measures ~19.4 B, so the default is roughly 97 MB per shard. "
    "Raising it means fewer files but more rows left as pending when a "
    "run is interrupted.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip positions already present in --output-path (every feather "
    "under it is read) and append new shards. An interrupted run can "
    "simply be re-executed.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Delete the --output-path directory and start over. Without "
    "this or --resume, an existing output is an error: it usually "
    "represents days of GPU time.",
)
@handle_exception
def search_values(
    input_path: Path,
    output_path: Path,
    model_path: Path | None,
    min_ply: int,
    max_positions: int,
    seed: int,
    playouts: int,
    time_ms: int | None,
    threads: int,
    batch_size: int,
    root_dfpn: bool,
    root_dfpn_nodes: int | None,
    root_dfpn_depth: int | None,
    leaf_mate: bool,
    leaf_mate_nodes: int | None,
    leaf_mate_threads: int | None,
    defensive_mate: bool | None,
    defensive_mate_threads: int | None,
    pad_buckets: bool | None,
    cuda: bool,
    tensorrt: bool,
    trt_cache_dir: Path | None,
    resume: bool,
    overwrite: bool,
    flush_interval: int,
    shard_rows: int,
) -> None:
    """Search floodgate positions to build per-position value targets.

    The value target built from `gameResult` is constant within a game: all
    ~110 positions of one game share the same 0/1. Recalling which game a
    position came from is therefore a valid way to fit the training set, and
    it transfers nothing to unseen games. Search values differ position by
    position, so that shortcut stops paying.

    Selection uses ply and duplication only, never the label, so the training
    distribution is not biased towards the model's own mistakes.

    Positions are keyed by Zobrist hash, the same key `maou pre-process` uses,
    so the output can be joined back with `--search-value-path`. Positions
    that are absent keep their game-result value, which means a partial run is
    useful on its own.
    """
    from maou.interface import search_value_interface

    result = search_value_interface.collect_search_values(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        min_ply=min_ply,
        max_positions=max_positions,
        seed=seed,
        max_playouts=playouts,
        time_ms=time_ms,
        threads=threads,
        batch_size=batch_size,
        root_dfpn=root_dfpn,
        root_dfpn_nodes=root_dfpn_nodes,
        root_dfpn_depth=root_dfpn_depth,
        leaf_mate=leaf_mate,
        leaf_mate_nodes=leaf_mate_nodes,
        leaf_mate_threads=leaf_mate_threads,
        defensive_mate=defensive_mate,
        defensive_mate_threads=defensive_mate_threads,
        pad_buckets=pad_buckets,
        cuda=cuda,
        tensorrt=tensorrt,
        trt_engine_cache_dir=trt_cache_dir,
        resume=resume,
        overwrite=overwrite,
        flush_interval=flush_interval,
        shard_rows=shard_rows,
    )
    click.echo(result)
