"""Playwright E2E テスト: ゲームグラフの JS→Python server_functions ブリッジ．

server_functions ブリッジが正しく動作し，
ノード選択(シングルクリック)・ノード展開(ダブルクリック)が
JS → Python → Gradio UI 更新のパイプラインを完走することを検証する．

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
from collections.abc import Generator
from pathlib import Path
from typing import Any

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
    # 環境変数で指定可能
    env_path = os.environ.get("CHROMIUM_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # ms-playwright cache の任意バージョンを検索
    cache_dir = Path.home() / ".cache" / "ms-playwright"
    if cache_dir.exists():
        for d in sorted(cache_dir.iterdir(), reverse=True):
            if d.name.startswith("chromium-"):
                chrome = d / "chrome-linux" / "chrome"
                if chrome.exists():
                    return str(chrome)

    # システムパス
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

# 平手初期局面の SFEN
_HIRATE_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

# テスト用ノードハッシュ
_ROOT_HASH = 100
_NODE_A_HASH = 200
_NODE_B_HASH = 300
_NODE_C_HASH = 400


def _make_nodes(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """テスト用ノード DataFrame を生成する．"""
    return pl.DataFrame(
        rows, schema=get_game_graph_nodes_schema()
    )


def _make_edges(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """テスト用エッジ DataFrame を生成する．"""
    return pl.DataFrame(
        rows, schema=get_game_graph_edges_schema()
    )


def _build_test_graph(graph_dir: Path) -> None:
    """テスト用グラフデータを graph_dir に保存する．

    構造:
        ROOT(100, depth=0)
          ├── A(200, depth=1, prob=0.6)  ▲7六歩 (move16=7739)
          │   └── C(400, depth=2, prob=0.3)  △3四歩 (move16=2581)
          └── B(300, depth=1, prob=0.4)  ▲2六歩 (move16=1934)

    move16 は cshogi の実際の指し手エンコーディング値を使用する．
    """
    nodes = _make_nodes(
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
        ]
    )
    # cshogi move16 エンコーディング:
    # ▲7六歩 (7g7f): move16=7739
    # ▲2六歩 (2g2f): move16=1934
    # △3四歩 (3c3d): move16=2581
    edges = _make_edges(
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
        ]
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
    """テスト用グラフサーバーをバックグラウンドで起動する．

    Returns:
        サーバーURL (例: "http://127.0.0.1:12345")
    """
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
) -> Generator[Page, None, None]:
    """Playwright ブラウザページを提供する．

    外部 CDN へのリクエストをブロックして
    ネットワーク制限環境でも動作するようにする．
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=_CHROMIUM_PATH,
        )
        page = browser.new_page()

        # 外部 CDN をブロック(ネットワーク制限環境対応)
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

        # Gradio がコンポーネントをレンダリングするまでポーリング
        page.wait_for_function(
            "() => document.querySelectorAll('[id]').length > 5",
            timeout=30000,
        )

        # server_functions ブリッジの初期化を待機
        # js_on_load が実行されると window.__maou_* が設定される
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


class TestServerFunctionsBridge:
    """server_functions ブリッジの E2E テスト．

    JS→Python 通信パイプライン(ノード選択・展開・ビューポートクエリ)が
    正しく動作することを検証する．
    """

    def test_bridge_objects_initialized(
        self, browser_page: Page
    ) -> None:
        """js_on_load で window.__maou_* ブリッジが初期化される．"""
        result = browser_page.evaluate(
            """() => {
                const s = window.__maou_select;
                const e = window.__maou_expand;
                const v = window.__maou_viewport;
                return {
                    select: !!(s && s.server && s.trigger),
                    expand: !!(e && e.server && e.trigger),
                    viewport: !!(v && v.server && v.trigger),
                };
            }"""
        )
        assert result["select"], (
            "window.__maou_select bridge not initialized"
        )
        assert result["expand"], (
            "window.__maou_expand bridge not initialized"
        )
        assert result["viewport"], (
            "window.__maou_viewport bridge not initialized"
        )

    def test_node_select_updates_detail_panel(
        self, browser_page: Page
    ) -> None:
        """ノード選択(server_functions)で詳細パネルが更新される．

        JS から handle_select を呼び出し，trigger("change") で
        on_select_result が発火してUIが更新されることを確認する．
        """
        # handle_select(_NODE_A_HASH) を呼び出し，
        # trigger("change") で Gradio の出力パイプラインを発火
        result = browser_page.evaluate(
            """async (nodeHash) => {
                const bridge = window.__maou_select;
                if (!bridge || !bridge.server) return 'bridge_not_ready';
                const ok = await bridge.server.handle_select(
                    String(nodeHash)
                );
                if (!ok) return 'handle_select_returned_false';
                bridge.trigger('change');
                return 'ok';
            }""",
            arg=_NODE_A_HASH,
        )
        assert result == "ok", f"handle_select failed: {result}"

        # Gradio が UI を更新するのを待機
        # ノード詳細パネル内にハッシュ値または評価値が表示される
        browser_page.wait_for_function(
            """(nodeHash) => {
                const panel = document.querySelector('#detail-panel')
                    || document.querySelector('[id*="detail"]');
                if (panel) {
                    const text = panel.innerText;
                    return text.includes(String(nodeHash))
                        || text.includes('0.48');
                }
                const body = document.body.innerText;
                return body.includes(String(nodeHash))
                    && body.includes('0.48');
            }""",
            arg=_NODE_A_HASH,
            timeout=10000,
        )

    def test_node_expand_updates_graph(
        self, browser_page: Page
    ) -> None:
        """ノード展開(server_functions)でグラフが再描画される．

        JS から handle_expand を呼び出し，trigger("change") で
        on_expand_result が発火してグラフHTMLが更新されることを確認する．
        """
        # 展開前の graph-view の data-canvas 属性を取得
        before = browser_page.evaluate(
            """() => {
                const container = document.querySelector(
                    '#graph-view [data-canvas]'
                );
                return container
                    ? container.getAttribute('data-canvas').substring(0, 50)
                    : '';
            }"""
        )

        # handle_expand(_NODE_A_HASH, depth=3, prob=0.01) を呼び出し
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
            arg=_NODE_A_HASH,
        )
        assert result == "ok", f"handle_expand failed: {result}"

        # Gradio がグラフを再描画するのを待機
        browser_page.wait_for_function(
            """(prev) => {
                const container = document.querySelector(
                    '#graph-view [data-canvas]'
                );
                if (!container) return false;
                const current = container.getAttribute(
                    'data-canvas'
                ).substring(0, 50);
                return current !== prev;
            }""",
            arg=before,
            timeout=10000,
        )

    def test_viewport_query_returns_data(
        self, browser_page: Page
    ) -> None:
        """ビューポートクエリが JSON データを返す．

        handle_viewport は trigger("change") を使わず，
        Promise で直接結果を返す．
        """
        result = browser_page.evaluate(
            """async () => {
                const bridge = window.__maou_viewport;
                if (!bridge || !bridge.server)
                    return 'bridge_not_ready';
                const data = await bridge.server.handle_viewport(
                    -100000, 100000, -100, 100000
                );
                if (!data) return 'no_data';
                const parsed = typeof data === 'string'
                    ? JSON.parse(data) : data;
                if (parsed.nodes && parsed.nodes.length > 0)
                    return 'ok';
                return 'empty_nodes';
            }"""
        )
        assert result == "ok", (
            f"viewport query failed: {result}"
        )
