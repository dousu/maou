"""Playwright スクリーンショットテスト: エッジラベルの視認性確認．

ゲームグラフのエッジに日本語の手ラベル(例: "7六歩 60.0%")が
背景ピル付きで明瞭に表示されることを視覚的に確認する．

Requires:
    - gradio>=6.7.0
    - playwright
    - Chromium browser (ms-playwright cache or system)
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import polars as pl
import pytest

from maou.domain.game_graph.schema import (
    get_game_graph_edges_schema,
    get_game_graph_nodes_schema,
)

# --- Playwright import (optional dep) ---
try:
    from playwright.sync_api import Page, Route, sync_playwright
except ImportError:
    pytest.skip(
        "playwright is not installed",
        allow_module_level=True,
    )

# --- Gradio import (optional dep) ---
try:
    import gradio as gr  # noqa: F401
except ImportError:
    pytest.skip(
        "gradio is not installed",
        allow_module_level=True,
    )


# =============================================
# Chromium 探索
# =============================================


def _find_chromium() -> str | None:
    """利用可能な Chromium バイナリパスを返す．"""
    env_path = os.environ.get("CHROMIUM_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    cache_dir = Path.home() / ".cache" / "ms-playwright"
    if cache_dir.exists():
        for d in sorted(cache_dir.iterdir(), reverse=True):
            if d.name.startswith("chromium-"):
                chrome = d / "chrome-linux" / "chrome"
                if chrome.exists():
                    return str(chrome)

    for path in (
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/usr/bin/google-chrome",
    ):
        if Path(path).exists():
            return path

    return None


_CHROMIUM_PATH = _find_chromium()

if _CHROMIUM_PATH is None:
    pytest.skip(
        "Chromium browser not found",
        allow_module_level=True,
    )


# =============================================
# テストデータ生成
# =============================================

_HIRATE_SFEN = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/"
    "PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
)

_ROOT_HASH = 100
_NODE_A_HASH = 200
_NODE_B_HASH = 300
_NODE_C_HASH = 400


def _build_test_graph(graph_dir: Path) -> None:
    """テスト用グラフデータを生成する．

    構造:
        ROOT(100, depth=0)
          ├── A(200, depth=1, prob=0.6)  ▲7六歩 (move16=7739)
          │   └── C(400, depth=2, prob=0.3)  △3四歩 (move16=2581)
          └── B(300, depth=1, prob=0.4)  ▲2六歩 (move16=1934)
    """
    nodes = pl.DataFrame(
        [
            {
                "position_hash": _ROOT_HASH,
                "result_value": 0.52,
                "best_move_win_rate": 0.53,
                "num_branches": 2,
                "depth": 0,
                "is_depth_cutoff": False,
            },
            {
                "position_hash": _NODE_A_HASH,
                "result_value": 0.48,
                "best_move_win_rate": 0.50,
                "num_branches": 1,
                "depth": 1,
                "is_depth_cutoff": False,
            },
            {
                "position_hash": _NODE_B_HASH,
                "result_value": 0.55,
                "best_move_win_rate": 0.56,
                "num_branches": 0,
                "depth": 1,
                "is_depth_cutoff": False,
            },
            {
                "position_hash": _NODE_C_HASH,
                "result_value": 0.45,
                "best_move_win_rate": 0.46,
                "num_branches": 0,
                "depth": 2,
                "is_depth_cutoff": False,
            },
        ],
        schema=get_game_graph_nodes_schema(),
    )
    edges = pl.DataFrame(
        [
            {
                "parent_hash": _ROOT_HASH,
                "child_hash": _NODE_A_HASH,
                "move16": 7739,
                "move_label": 10,
                "probability": 0.6,
                "win_rate": 0.52,
                "is_leaf": False,
            },
            {
                "parent_hash": _ROOT_HASH,
                "child_hash": _NODE_B_HASH,
                "move16": 1934,
                "move_label": 11,
                "probability": 0.4,
                "win_rate": 0.50,
                "is_leaf": False,
            },
            {
                "parent_hash": _NODE_A_HASH,
                "child_hash": _NODE_C_HASH,
                "move16": 2581,
                "move_label": 12,
                "probability": 0.3,
                "win_rate": 0.48,
                "is_leaf": False,
            },
        ],
        schema=get_game_graph_edges_schema(),
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    nodes.write_ipc(
        graph_dir / "nodes.feather", compression="lz4"
    )
    edges.write_ipc(
        graph_dir / "edges.feather", compression="lz4"
    )
    metadata = {"initial_sfen": _HIRATE_SFEN}
    (graph_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def _find_free_port() -> int:
    """空きポートを取得する．"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 30) -> None:
    """HTTP サーバーが応答するまで待機する．"""
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310
                return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    msg = f"Server at {url} did not start within {timeout}s"
    raise TimeoutError(msg)


# =============================================
# Fixtures
# =============================================


@pytest.fixture(scope="module")
def graph_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """テスト用グラフサーバーをバックグラウンドで起動する．"""
    graph_dir = tmp_path_factory.mktemp("graph_data")
    _build_test_graph(graph_dir)

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    from maou.infra.visualization.game_graph_server import (
        launch_game_graph_server,
    )

    def _run_server() -> None:
        launch_game_graph_server(
            graph_path=graph_dir,
            port=port,
            server_name="127.0.0.1",
        )

    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()

    _wait_for_server(url, timeout=30)
    return url


@pytest.fixture(scope="module")
def browser_page(
    graph_server: str,
) -> Page:
    """Playwright ブラウザページを提供する．"""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=_CHROMIUM_PATH,
        )
        page = browser.new_page(
            viewport={"width": 1280, "height": 960}
        )

        def _route_handler(route: Route) -> None:
            url: str = route.request.url
            if "127.0.0.1" in url or "localhost" in url:
                route.continue_()
            else:
                route.abort()

        page.route("**/*", _route_handler)

        page.goto(
            graph_server,
            wait_until="domcontentloaded",
            timeout=15000,
        )

        page.wait_for_function(
            "() => document.querySelectorAll('[id]').length > 5",
            timeout=30000,
        )

        page.wait_for_function(
            """() => {
                return !!(
                    window.__maou_select
                    && window.__maou_expand
                    && window.__maou_viewport
                );
            }""",
            timeout=15000,
        )

        yield page
        browser.close()


# =============================================
# テスト
# =============================================

_SCREENSHOT_DIR = Path(__file__).parent / "screenshots"


class TestEdgeLabelVisibility:
    """エッジラベルの視認性をスクリーンショットで確認するテスト．"""

    def test_edge_labels_rendered_with_background(
        self, browser_page: Page
    ) -> None:
        """エッジラベルが背景ピル付きで描画される．

        Canvas 上のエッジラベルが背景付きで描画されていることを，
        ノード展開後にスクリーンショットで確認する．
        """
        # ルートノードを展開してグラフを表示
        result = browser_page.evaluate(
            """async (nodeHash) => {
                const bridge = window.__maou_expand;
                if (!bridge || !bridge.server) return 'bridge_not_ready';
                const ok = await bridge.server.handle_expand(
                    String(nodeHash), 3, 0.01
                );
                if (!ok) return 'handle_expand_returned_false';
                bridge.trigger('change');
                return 'ok';
            }""",
            arg=_ROOT_HASH,
        )
        assert result == "ok", f"handle_expand failed: {result}"

        # グラフ再描画を待機
        browser_page.wait_for_function(
            """() => {
                const container = document.querySelector(
                    '#graph-view [data-canvas]'
                );
                return !!container;
            }""",
            timeout=10000,
        )

        # Canvas の描画完了をピクセル確認で待機
        browser_page.wait_for_function(
            """() => {
                const canvas = document.querySelector(
                    '#graph-view canvas'
                );
                if (!canvas) return false;
                const ctx = canvas.getContext('2d');
                const data = ctx.getImageData(
                    0, 0, canvas.width, canvas.height
                ).data;
                return data.some(v => v !== 0);
            }""",
            timeout=10000,
        )

        # スクリーンショットを保存
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

        screenshot_path = (
            _SCREENSHOT_DIR / "edge_labels_default_view.png"
        )
        browser_page.screenshot(path=str(screenshot_path))
        assert screenshot_path.exists(), "Screenshot was not saved"

    def test_edge_labels_visible_at_zoomed_view(
        self, browser_page: Page
    ) -> None:
        """ズーム時にエッジラベルが視認できる．

        renderer の zoom/offset を直接操作して拡大表示し，
        スクリーンショットでラベルの視認性を確認する．
        """
        # グラフが展開済みでなければ展開する(テストの独立実行対応)
        needs_expand = browser_page.evaluate(
            """() => {
                const r = window.__maou_get_renderer
                    ? window.__maou_get_renderer() : null;
                return !r || r.edges.size === 0;
            }"""
        )
        if needs_expand:
            result = browser_page.evaluate(
                """async (nodeHash) => {
                    const bridge = window.__maou_expand;
                    if (!bridge || !bridge.server)
                        return 'bridge_not_ready';
                    const ok = await bridge.server.handle_expand(
                        String(nodeHash), 3, 0.01
                    );
                    if (!ok) return 'handle_expand_returned_false';
                    bridge.trigger('change');
                    return 'ok';
                }""",
                arg=_ROOT_HASH,
            )
            assert result == "ok", (
                f"handle_expand failed: {result}"
            )
            browser_page.wait_for_function(
                """() => {
                    const canvas = document.querySelector(
                        '#graph-view canvas'
                    );
                    if (!canvas) return false;
                    const ctx = canvas.getContext('2d');
                    const data = ctx.getImageData(
                        0, 0, canvas.width, canvas.height
                    ).data;
                    return data.some(v => v !== 0);
                }""",
                timeout=10000,
            )

        # renderer を直接操作してズームイン + 中心に配置
        zoom_info = browser_page.evaluate(
            """() => {
                const r = window.__maou_get_renderer
                    ? window.__maou_get_renderer() : null;
                if (!r) return null;
                // zoom を 2.0 に設定してラベルが確実に表示される状態に
                r.zoom = 2.0;
                // ノード群の中心にビューを合わせる
                let sumX = 0, sumY = 0, count = 0;
                r.nodes.forEach(function (n) {
                    sumX += n.x; sumY += n.y; count++;
                });
                if (count > 0) {
                    const cx = sumX / count;
                    const cy = sumY / count;
                    const canvas = r.canvas;
                    const dw = canvas.clientWidth;
                    const dh = canvas.clientHeight;
                    r.offsetX = dw / 2 - cx * r.zoom;
                    r.offsetY = dh / 2 - cy * r.zoom;
                }
                r.requestRender();
                return {
                    zoom: r.zoom,
                    nodes: r.nodes.size,
                    edges: r.edges.size,
                };
            }"""
        )
        assert zoom_info is not None, (
            "__maou_get_renderer not available"
        )
        assert zoom_info["edges"] > 0, "No edges in renderer"

        # requestAnimationFrame の完了を待機
        browser_page.wait_for_timeout(500)

        screenshot_path = (
            _SCREENSHOT_DIR / "edge_labels_zoomed_in.png"
        )

        # graph-view 要素だけをキャプチャ
        graph_el = browser_page.query_selector("#graph-view")
        if graph_el:
            graph_el.screenshot(path=str(screenshot_path))
        else:
            browser_page.screenshot(path=str(screenshot_path))

        assert screenshot_path.exists()
