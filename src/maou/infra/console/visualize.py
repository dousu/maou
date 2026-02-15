"""可視化CLIコマンド実装（インフラ層）．

`maou visualize`コマンドの実装を提供する．
"""

import logging
from pathlib import Path
from typing import List, Optional

import click

from maou.infra.app_logging import app_logger
from maou.infra.console.common import (
    FileSystem,
    handle_exception,
)

logger = logging.getLogger(__name__)


def _is_google_colab() -> bool:
    """Google Colab環境で実行されているか検出する．

    Returns:
        bool: Google Colab環境の場合True
    """
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


@click.command("visualize")
@click.option(
    "--input-path",
    help="入力ファイルまたはディレクトリパス（複数指定可，--use-mock-dataの場合は不要）．",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    multiple=True,
)
@click.option(
    "--array-type",
    help="データ型: hcpe, preprocessing, stage1, stage2．",
    type=click.Choice(
        ["hcpe", "preprocessing", "stage1", "stage2"]
    ),
    default="hcpe",
    required=True,
)
@click.option(
    "--port",
    help="Gradioサーバーポート．未指定時は7860-7959の空きポートを自動選択．",
    type=int,
    default=None,
)
@click.option(
    "--share",
    help="Gradio公開リンクを作成する．",
    is_flag=True,
    default=False,
)
@click.option(
    "--server-name",
    help="サーバーバインドアドレス（デフォルト: 127.0.0.1）．",
    type=str,
    default="127.0.0.1",
)
@click.option(
    "--model-path",
    help="オプショナルなONNXモデルパス（評価表示用）．",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--debug-mode",
    help="詳細ログを有効化する．",
    is_flag=True,
    default=False,
)
@click.option(
    "--use-mock-data",
    help="モックデータを使用する（UIテスト用）．",
    is_flag=True,
    default=False,
)
@handle_exception
def visualize(
    input_path: tuple[Path, ...],
    array_type: str,
    port: Optional[int],
    share: bool,
    server_name: str,
    model_path: Optional[Path],
    debug_mode: bool,
    use_mock_data: bool,
) -> None:
    """Gradioデータ可視化サーバーを起動する．

    Examples:
        # モックデータでUIをテスト
        maou visualize --use-mock-data --array-type hcpe

        # ディレクトリからHCPEデータを可視化
        maou visualize --input-path ./data/hcpe --array-type hcpe

        # 特定のファイルを可視化
        maou visualize --input-path data1.feather --input-path data2.feather --array-type preprocessing

        # 公開リンクを作成
        maou visualize --input-path ./data --array-type hcpe --share
    """
    # デバッグモード設定
    if debug_mode:
        app_logger.setLevel(logging.DEBUG)

    # Gradio importを遅延（依存関係オプショナル）
    try:
        from maou.infra.visualization.gradio_server import (
            launch_server,
        )
    except ImportError as e:
        error_msg = (
            "Gradio visualization dependencies not installed. "
            "Install with: poetry install -E visualize"
        )
        app_logger.error(error_msg)
        raise click.ClickException(error_msg) from e

    # ファイルパス解決
    if use_mock_data:
        # モックデータモードの場合はダミーファイルパスを使用
        file_paths = [Path("mock_data.feather")]
        app_logger.info(
            "Using mock data mode - no actual files will be read"
        )
    elif input_path:
        file_paths = _resolve_file_paths(input_path)

        if not file_paths:
            raise click.ClickException(
                "No .feather files found in the specified input paths."
            )
    else:
        # Allow empty startup - user will load data from UI
        file_paths = []
        app_logger.info(
            "Starting visualization without data source - "
            "use UI to load data dynamically"
        )

    app_logger.info(
        f"Launching visualization server with {len(file_paths)} files"
    )
    app_logger.info(f"Array type: {array_type}")
    app_logger.info(f"Server address: {server_name}:{port}")

    # Google Colab環境では自動的にshareを有効化
    if not share and _is_google_colab():
        share = True
        app_logger.info(
            "✅ Auto-enabled --share for Google Colab environment "
            "(localhost not accessible in Colab)"
        )

    # Gradioサーバー起動（ブロッキング呼び出し）
    try:
        launch_server(
            file_paths=file_paths,
            array_type=array_type,
            port=port,
            share=share,
            server_name=server_name,
            model_path=model_path,
            debug=debug_mode,
            use_mock_data=use_mock_data,
        )
    except Exception as e:
        app_logger.exception("Failed to launch Gradio server")
        raise click.ClickException(
            f"Server launch failed: {e}"
        ) from e


def _resolve_file_paths(
    input_paths: tuple[Path, ...],
) -> List[Path]:
    """入力パス(ファイル・ディレクトリ混在可)からファイルパスを解決．

    Args:
        input_paths: 入力パスのタプル(ファイルまたはディレクトリ)

    Returns:
        解決されたファイルパスのリスト

    Raises:
        click.ClickException: 入力が不正な場合
    """
    result: List[Path] = []
    for p in input_paths:
        result.extend(
            FileSystem.collect_files(p, ext=".feather")
        )

    if not result:
        app_logger.warning(
            "No .feather files found in the specified input paths"
        )

    return sorted(set(result))
