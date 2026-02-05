"""スクリーンショットCLIコマンド実装（インフラ層）．

`maou screenshot`コマンドの実装を提供する．
Playwright を使用してGradio UIのスクリーンショットを取得する．
"""

import base64
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from maou.infra.app_logging import app_logger
from maou.infra.console.common import handle_exception

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:7860"
DEFAULT_OUTPUT = "/tmp/gradio-screenshot.png"
DEFAULT_WAIT_FOR = ".gradio-container"
DEFAULT_TIMEOUT = 30000
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_SETTLE_TIME = 3000


def _capture_screenshot(
    url: str,
    output: Optional[Path],
    base64_output: bool,
    selector: Optional[str],
    full_page: bool,
    wait_for: str,
    timeout: int,
    width: int,
    height: int,
    settle_time: int,
) -> None:
    """Playwrightを使用してスクリーンショットを取得する．

    Args:
        url: 対象URL
        output: 出力ファイルパス（base64の場合はNone）
        base64_output: stdoutにbase64出力するか
        selector: キャプチャする要素のCSSセレクタ
        full_page: フルページキャプチャするか
        wait_for: 待機するセレクタ
        timeout: タイムアウト（ミリ秒）
        width: ビューポート幅
        height: ビューポート高さ
        settle_time: 動的コンテンツ安定待機時間（ミリ秒）
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        error_msg = (
            "Playwright is not installed. "
            "Install with: poetry install -E visualize && "
            "poetry run playwright install chromium"
        )
        raise click.ClickException(error_msg) from e

    with sync_playwright() as p:
        # Chromiumをヘッドレスモードで起動
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )

        try:
            context = browser.new_context(
                viewport={"width": width, "height": height}
            )
            page = context.new_page()

            # URLへナビゲート
            # Gradioは SSE接続を維持するため networkidle ではなく
            # domcontentloaded を使用
            app_logger.info(f"Navigating to {url}...")
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout,
            )

            # Gradioコンテナの表示を待機
            if wait_for:
                app_logger.info(f"Waiting for {wait_for}...")
                page.wait_for_selector(
                    wait_for,
                    state="visible",
                    timeout=timeout,
                )

            # 動的コンテンツの安定を待機
            page.wait_for_timeout(settle_time)

            # スクリーンショット取得
            if selector:
                app_logger.info(
                    f"Capturing element: {selector}"
                )
                element = page.query_selector(selector)
                if not element:
                    raise click.ClickException(
                        f"Element not found: {selector}"
                    )
                screenshot_bytes = element.screenshot(
                    type="png"
                )
            else:
                screenshot_bytes = page.screenshot(
                    type="png",
                    full_page=full_page,
                )

            # 出力処理
            if base64_output:
                b64_data = base64.b64encode(
                    screenshot_bytes
                ).decode("ascii")
                sys.stdout.write(b64_data)
                app_logger.info(
                    "Base64 output written to stdout"
                )
            else:
                output_path = output or Path(DEFAULT_OUTPUT)
                output_path.write_bytes(screenshot_bytes)
                app_logger.info(
                    f"Screenshot saved to {output_path}"
                )

        finally:
            browser.close()


@click.command("screenshot")
@click.option(
    "--url",
    help=f"Target URL (default: {DEFAULT_URL}).",
    type=str,
    default=DEFAULT_URL,
)
@click.option(
    "--output",
    "-o",
    help="Output file path.",
    type=click.Path(path_type=Path),
    required=False,
)
@click.option(
    "--base64",
    "base64_output",
    help="Output base64 to stdout instead of file.",
    is_flag=True,
    default=False,
)
@click.option(
    "--selector",
    "-s",
    help="CSS selector for element capture.",
    type=str,
    required=False,
)
@click.option(
    "--full-page/--no-full-page",
    help="Capture full scrollable page (default: enabled).",
    default=True,
)
@click.option(
    "--wait-for",
    help=f"Wait for selector before capture (default: {DEFAULT_WAIT_FOR}).",
    type=str,
    default=DEFAULT_WAIT_FOR,
)
@click.option(
    "--timeout",
    help=f"Navigation timeout in ms (default: {DEFAULT_TIMEOUT}).",
    type=int,
    default=DEFAULT_TIMEOUT,
)
@click.option(
    "--width",
    help=f"Viewport width (default: {DEFAULT_WIDTH}).",
    type=int,
    default=DEFAULT_WIDTH,
)
@click.option(
    "--height",
    help=f"Viewport height (default: {DEFAULT_HEIGHT}).",
    type=int,
    default=DEFAULT_HEIGHT,
)
@click.option(
    "--settle-time",
    help=f"Wait time for dynamic content to stabilize in ms (default: {DEFAULT_SETTLE_TIME}).",
    type=int,
    default=DEFAULT_SETTLE_TIME,
)
@handle_exception
def screenshot(
    url: str,
    output: Optional[Path],
    base64_output: bool,
    selector: Optional[str],
    full_page: bool,
    wait_for: str,
    timeout: int,
    width: int,
    height: int,
    settle_time: int,
) -> None:
    """Capture Gradio UI screenshot using Playwright.

    Examples:
        # Basic screenshot
        maou screenshot --url http://localhost:7860 --output /tmp/test.png

        # Base64 output for Claude Vision API
        maou screenshot --url http://localhost:7860 --base64

        # Capture specific element
        maou screenshot --url http://localhost:7860 --selector "#mode-badge"

        # Full page screenshot
        maou screenshot --url http://localhost:7860 --full-page

    Gradio UI Selectors:
        .gradio-container    Main container (default wait target)
        #mode-badge          Data mode display (MOCK/REAL)
        #id-search-input     Record ID search input
        #prev-page           Previous page button
        #next-page           Next page button
    """
    # 出力先の決定
    if not base64_output and not output:
        output = Path(DEFAULT_OUTPUT)

    _capture_screenshot(
        url=url,
        output=output,
        base64_output=base64_output,
        selector=selector,
        full_page=full_page,
        wait_for=wait_for,
        timeout=timeout,
        width=width,
        height=height,
        settle_time=settle_time,
    )
