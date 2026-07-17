"""棋譜解析 GUI の CLI コマンド (インフラ層)．

`maou analyze-gui` コマンドの実装を提供する．Gradio 依存は遅延 import
(``uv sync --extra visualize`` で導入)．エンジン系オプションは
`maou analyze-game` と同一に維持する (docs/design/game-analysis/gui.md §3)．
"""

from pathlib import Path

import click

from maou.infra.app_logging import app_logger
from maou.infra.console.common import (
    handle_exception,
    is_google_colab,
)


@click.command("analyze-gui")
@click.option(
    "--input-path",
    help=(
        "Path to the game record file (CSA / KIF) loaded at startup. "
        "Files can also be uploaded from the UI."
    ),
    type=click.Path(
        exists=True, dir_okay=False, path_type=Path
    ),
    default=None,
    required=False,
)
@click.option(
    "--report",
    help=(
        "Path to an analyze-game JSON report (the --output file) "
        "matching the game record. Requires --input-path."
    ),
    type=click.Path(
        exists=True, dir_okay=False, path_type=Path
    ),
    default=None,
    required=False,
)
@click.option(
    "--model-path",
    help=(
        "Path to the ONNX model file for in-GUI analysis. Uses a "
        "deterministic mock evaluator when omitted (for development "
        "only; clearly labeled in the UI)."
    ),
    type=click.Path(exists=True, path_type=Path),
    default=None,
    required=False,
)
@click.option(
    "--time-ms",
    help=(
        "Default time budget per in-GUI analysis in milliseconds "
        "(default 1000). Mutually exclusive with --playouts."
    ),
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--playouts",
    help=(
        "Default playout budget per in-GUI analysis. Mutually "
        "exclusive with --time-ms."
    ),
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--num-candidates",
    help="Maximum number of candidate moves shown in the UI.",
    type=int,
    default=5,
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
@click.option(
    "--port",
    help=(
        "Gradio server port. Auto-selected by Gradio when omitted."
    ),
    type=int,
    default=None,
    required=False,
)
@click.option(
    "--share",
    help="Create a public Gradio link.",
    is_flag=True,
    default=False,
)
@click.option(
    "--server-name",
    help="Server bind address (default: 127.0.0.1).",
    type=str,
    default="127.0.0.1",
)
@handle_exception
def analyze_gui(
    input_path: Path | None,
    report: Path | None,
    model_path: Path | None,
    time_ms: int | None,
    playouts: int | None,
    num_candidates: int,
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
    port: int | None,
    share: bool,
    server_name: str,
) -> None:
    """Launch the game-analysis GUI (Gradio) server.

    This command starts a browser UI for reviewing a game record
    (CSA / KIF) together with an analyze-game JSON report: board view
    with candidate-move arrows, win-rate / eval graph from the sente
    perspective, move list, and per-position candidate table. It also
    supports interactive analysis: playing moves on the board (click
    input with variation branches), analyzing the current position
    with the resident engine, and analyzing the whole mainline with a
    downloadable analyze-game compatible JSON report.

    Examples:
        # 棋譜と解析レポートを読み込んで起動
        maou analyze-gui --input-path game.csa --report report.json

        # 実モデルで GUI 内解析 (1 局面 1000ms)
        maou analyze-gui --input-path game.csa --model-path model.onnx

        # 空で起動して UI からアップロード
        maou analyze-gui

    Args:
        input_path: Game record file loaded at startup.
        report: analyze-game JSON report matching the game record.
        model_path: ONNX model file for in-GUI analysis.
        time_ms: Default time budget per in-GUI analysis.
        playouts: Default playout budget per in-GUI analysis.
        num_candidates: Maximum number of candidate moves shown.
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
        port: Gradio server port.
        share: Create a public Gradio link.
        server_name: Server bind address.
    """
    if report is not None and input_path is None:
        raise click.UsageError(
            "--report requires --input-path."
        )
    if num_candidates < 1:
        raise click.UsageError(
            "--num-candidates must be a positive integer."
        )
    if (cuda or tensorrt) and model_path is None:
        raise click.UsageError(
            "--cuda / --tensorrt require --model-path."
        )
    if time_ms is not None and playouts is not None:
        raise click.UsageError(
            "Specify at most one of --time-ms / --playouts."
        )

    try:
        from maou.app.analysis.interactive_analyzer import (
            EngineSettings,
        )
        from maou.infra.visualization.analysis_gui_server import (
            launch_analysis_gui_server,
        )
    except ImportError as e:
        error_msg = (
            "Gradio visualization dependencies not installed. "
            "Install with: uv sync --extra visualize"
        )
        app_logger.error(error_msg)
        raise click.ClickException(error_msg) from e

    # Google Colab環境では自動的にshareを有効化
    if not share and is_google_colab():
        share = True
        app_logger.info(
            "✅ Auto-enabled --share for Google Colab environment "
            "(localhost not accessible in Colab)"
        )

    engine_settings = EngineSettings(
        model_path=model_path,
        threads=threads,
        batch_size=batch_size,
        num_candidates=num_candidates,
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

    try:
        launch_analysis_gui_server(
            kifu_path=input_path,
            report_path=report,
            num_candidates=num_candidates,
            engine_settings=engine_settings,
            default_time_ms=time_ms,
            default_playouts=playouts,
            port=port,
            share=share,
            server_name=server_name,
        )
    except click.ClickException:
        raise
    except Exception as e:
        app_logger.exception(
            "Failed to launch analysis GUI server"
        )
        raise click.ClickException(
            f"Server launch failed: {e}"
        ) from e
