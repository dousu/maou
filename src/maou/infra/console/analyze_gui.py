"""棋譜解析 GUI の CLI コマンド (インフラ層)．

`maou analyze-gui` コマンドの実装を提供する．Gradio 依存は遅延 import
(``uv sync --extra visualize`` で導入)．
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
    "--num-candidates",
    help="Maximum number of candidate moves shown in the UI.",
    type=int,
    default=5,
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
    num_candidates: int,
    port: int | None,
    share: bool,
    server_name: str,
) -> None:
    """Launch the game-analysis GUI (Gradio) server.

    This command starts a browser UI for reviewing a game record
    (CSA / KIF) together with an analyze-game JSON report: board view
    with candidate-move arrows, win-rate / eval graph from the sente
    perspective, move list, and per-position candidate table.

    Examples:
        # 棋譜と解析レポートを読み込んで起動
        maou analyze-gui --input-path game.csa --report report.json

        # 空で起動して UI からアップロード
        maou analyze-gui

    Args:
        input_path: Game record file loaded at startup.
        report: analyze-game JSON report matching the game record.
        num_candidates: Maximum number of candidate moves shown.
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

    try:
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

    try:
        launch_analysis_gui_server(
            kifu_path=input_path,
            report_path=report,
            num_candidates=num_candidates,
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
