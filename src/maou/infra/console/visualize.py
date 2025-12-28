"""可視化CLIコマンド実装（インフラ層）．

`maou visualize`コマンドの実装を提供する．
"""

import logging
from pathlib import Path
from typing import List, Optional

import click

from maou.infra.app_logging import app_logger
from maou.infra.console.common import handle_exception

logger = logging.getLogger(__name__)


@click.command("visualize")
@click.option(
    "--input-dir",
    help="データファイルを含むディレクトリパス．",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--input-files",
    help="カンマ区切りのデータファイルパスリスト．",
    type=str,
    required=False,
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
    help="Gradioサーバーポート（デフォルト: 7860）．",
    type=int,
    default=7860,
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
    input_dir: Optional[Path],
    input_files: Optional[str],
    array_type: str,
    port: int,
    share: bool,
    server_name: str,
    model_path: Optional[Path],
    debug_mode: bool,
    use_mock_data: bool,
) -> None:
    """Gradioデータ可視化サーバーを起動する．

    Examples:
        # ディレクトリからHCPEデータを可視化
        maou visualize --input-dir ./data/hcpe --array-type hcpe

        # 特定のファイルでpreprocessingデータを可視化
        maou visualize --input-files data1.feather,data2.feather --array-type preprocessing

        # 公開リンクを作成
        maou visualize --input-dir ./data --array-type hcpe --share
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
    file_paths = _resolve_file_paths(input_dir, input_files)

    if not file_paths:
        raise click.ClickException(
            "No input files found. Specify --input-dir or --input-files."
        )

    app_logger.info(
        f"Launching visualization server with {len(file_paths)} files"
    )
    app_logger.info(f"Array type: {array_type}")
    app_logger.info(f"Server address: {server_name}:{port}")

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
    input_dir: Optional[Path],
    input_files: Optional[str],
) -> List[Path]:
    """ディレクトリまたはカンマ区切りリストからファイルパスを解決．

    Args:
        input_dir: 入力ディレクトリパス
        input_files: カンマ区切りファイルパス文字列

    Returns:
        解決されたファイルパスのリスト

    Raises:
        click.ClickException: 入力が不正な場合
    """
    if input_files:
        # カンマ区切りリストから解析
        paths = [
            Path(f.strip()) for f in input_files.split(",")
        ]

        # 存在確認
        for path in paths:
            if not path.exists():
                raise click.ClickException(
                    f"File not found: {path}"
                )

        return paths

    elif input_dir:
        # ディレクトリから.featherファイルを検索
        feather_files = sorted(input_dir.glob("*.feather"))

        if not feather_files:
            app_logger.warning(
                f"No .feather files found in {input_dir}"
            )

        return feather_files

    else:
        raise click.ClickException(
            "Must provide either --input-dir or --input-files"
        )
