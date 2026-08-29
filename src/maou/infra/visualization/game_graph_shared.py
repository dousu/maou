"""ゲームグラフ可視化の共有ユーティリティ(インフラ層)．

game_graph_server.py (スタンドアロンモード) と gradio_server.py (埋め込みモード)
の両方から使用されるHTML生成関数・Plotly生成関数・JS定数を提供する．
"""

import html
import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ========================================
# ワークベンチのアクションレーン
# ========================================
#
# 画面はワークベンチ 1 枚なので，UI 操作はすべて data-action 文字列として
# JS から届く．重い読み込みを nav と分けないと，インデックス構築中に
# 行送りが詰まる (analyze-gui の _LANES と同じ考え方)．

WORKBENCH_LANES: tuple[str, ...] = ("nav", "load")
"""ワークベンチのアクションレーン名．"""


def workbench_js_on_load(lane: str) -> str:
    """レーン名に対応する js_on_load スニペットを返す．

    Args:
        lane: レーン名

    Returns:
        JS スニペット
    """
    return (
        "window.__maou_viz = window.__maou_viz || {};"
        f"window.__maou_viz.{lane} = "
        "{server: server, trigger: trigger};"
    )


def make_workbench_bridge(
    buffer: dict[str, str],
) -> Callable[[str], bool]:
    """アクション受け渡し用の server_function を作る．

    WARNING: バッファはクロージャとして全ブラウザセッションで共有される
    (analysis_gui_server.py と同じ制約．ローカル可視化ツールとして
    単一利用者を前提とする)．

    Args:
        buffer: アクション文字列を控える辞書

    Returns:
        JS から呼ばれる server_function
    """

    def handle_action(value: str) -> bool:
        """JS から呼ばれる server_function．"""
        if not value:
            return False
        buffer["value"] = str(value)
        return True

    return handle_action


def as_int(text: Any, fallback: int) -> int:
    """文字列を整数に直す (失敗したら fallback)．

    Args:
        text: 入力値
        fallback: 失敗時の値

    Returns:
        整数
    """
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return fallback


def as_float(text: Any, fallback: float) -> float:
    """文字列を実数に直す (失敗したら fallback)．

    Args:
        text: 入力値
        fallback: 失敗時の値

    Returns:
        実数
    """
    try:
        return float(text)
    except (TypeError, ValueError):
        return fallback


_STATIC_DIR = Path(__file__).parent / "static"

# ワークベンチの見出し・ラベルに使う Archivo と本文用の Noto Sans JP．
# demo.launch(head=...) で head に注入する (CSS からは font-family
# 参照のみで，@font-face は Google Fonts 側が返す)．
FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Archivo:wght@400;600;800&family=Noto+Sans+JP:wght@400;500;700"
    '&display=swap">'
)


# ========================================
# Gradio component elem_id constants
# ========================================


ELEM_ID_SELECT_BRIDGE = "select-bridge"
"""ノード選択ブリッジ用 gr.HTML の elem_id．"""

ELEM_ID_EXPAND_BRIDGE = "expand-bridge"
"""ノード展開ブリッジ用 gr.HTML の elem_id．"""


ELEM_ID_VIEWPORT_BRIDGE = "viewport-bridge"
"""ビューポートクエリブリッジ用 gr.HTML の elem_id．"""

# ========================================
# JS constants (Gradio 6 server_functions bridge)
# ========================================

# Gradio 6 では JS から Textbox の値を変更しても .input() / .change()
# が発火しない(Issue #3471, #7954)．
# 代わりに gr.HTML の server_functions + js_on_load を使用し，
# JS → Python の直接呼び出しを実現する．
# server_functions で処理を実行した後 trigger("change") で
# .change() コールバックを発火し，Gradio の出力パイプラインで
# UI コンポーネントを更新する．

JS_ON_LOAD_SELECT = (
    "window.__maou_select = {server: server, trigger: trigger};"
)
"""select_bridge の js_on_load．server と trigger をグローバルに公開する．"""

JS_ON_LOAD_EXPAND = (
    "window.__maou_expand = {server: server, trigger: trigger};"
)
"""expand_bridge の js_on_load．server と trigger をグローバルに公開する．"""

JS_ON_LOAD_VIEWPORT = "window.__maou_viewport = {server: server, trigger: trigger};"
"""viewport_bridge の js_on_load．server と trigger をグローバルに公開する．"""


# ========================================
# Static file loader
# ========================================


@lru_cache(maxsize=8)
def load_static_file(filename: str) -> str:
    """staticディレクトリからファイルを読み込む(結果はキャッシュされる)．

    Args:
        filename: ファイル名

    Returns:
        ファイル内容の文字列
    """
    path = _STATIC_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning("Static file not found: %s", path)
    return ""


# ========================================
# HTML builders
# ========================================


def build_graph_html(canvas_data_json: str) -> str:
    """グラフ表示用HTMLを生成する．

    Canvas 2D レンダラーが data-canvas 属性の変更を検知して
    自動的にレンダリングを実行する．

    Args:
        canvas_data_json: Canvas 描画データの JSON 文字列

    Returns:
        HTML文字列
    """
    escaped_json = html.escape(canvas_data_json, quote=True)

    return f"""
<div class="game-graph-container">
    <div id="gt-canvas-container" data-canvas="{escaped_json}"></div>
    <div class="game-graph-legend">
        <span class="legend-item">
            <span class="legend-swatch" style="background:#2196F3;"></span>先手有利
        </span>
        <span class="legend-item">
            <span class="legend-swatch" style="background:#9E9E9E;"></span>互角
        </span>
        <span class="legend-item">
            <span class="legend-swatch" style="background:#F44336;"></span>後手有利
        </span>
        <span class="legend-item" style="margin-left:12px;">
            &#9679; サイズ = 確率 / 色 = 最善手勝率
        </span>
    </div>
    <div class="graph-export-overlay">
        <button class="export-btn" onclick="window.exportGraphPNG()">
            PNG出力
        </button>
    </div>
</div>
"""


# ========================================
# ワークベンチ用 HTML レンダラー
# ========================================
#
# gr.JSON / gr.Dataframe は Gradio 固有の枠・行番号・セル箱を伴い，
# ワークベンチのデザイン (罫線だけの表・キーと値の 2 段組) に寄せられない．
# 表示専用のパネルはサーバー側で HTML を組み，gr.HTML に流し込む．
# 行クリックが要るテーブルは data-row 属性を持たせ，
# static/visualize_workbench.js のブリッジが行番号をサーバーへ返す．


def build_workbench_head() -> str:
    """ワークベンチの行クリックブリッジ JS を head に注入する HTML を返す．

    gr.HTML は innerHTML で差し替わり ``<script>`` が実行されないため，
    head 側に置く (game_graph_server._build_head_scripts と同じ方針)．

    Returns:
        head に入れる HTML 文字列
    """
    return f"<script>{load_static_file('visualize_workbench.js')}</script>"
