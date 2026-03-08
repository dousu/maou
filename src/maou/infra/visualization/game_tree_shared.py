"""ゲームツリー可視化の共有ユーティリティ(インフラ層)．

game_tree_server.py (スタンドアロンモード) と gradio_server.py (埋め込みモード)
の両方から使用されるHTML生成関数・Plotly生成関数・JS定数を提供する．
"""

import html
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# ========================================
# Gradio component elem_id constants
# ========================================

ELEM_ID_CURRENT_ROOT = "current-root"
"""現在のルートハッシュ用 hidden Textbox の elem_id．"""

ELEM_ID_SELECT_BRIDGE = "select-bridge"
"""ノード選択ブリッジ用 gr.HTML の elem_id．"""

ELEM_ID_EXPAND_BRIDGE = "expand-bridge"
"""ノード展開ブリッジ用 gr.HTML の elem_id．"""

ELEM_ID_DEPTH_SLIDER = "gt-depth-slider"
"""表示深さスライダーの elem_id(JS からの値読み取り用)．"""

ELEM_ID_MIN_PROB_SLIDER = "gt-min-prob-slider"
"""最小確率スライダーの elem_id(JS からの値読み取り用)．"""

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


def build_tree_html(elements_json: str) -> str:
    """ツリー表示用HTMLを生成する．

    Cytoscape.jsのレンダリングはhead要素のMutationObserverが
    data-elements属性の変更を検知して自動的に実行する．

    Args:
        elements_json: Cytoscape elements の JSON 文字列

    Returns:
        HTML文字列
    """
    css_code = load_static_file("game_tree.css")
    escaped_json = html.escape(elements_json, quote=True)

    return f"""
<style>{css_code}</style>
<div class="game-tree-container">
    <div id="cy" data-elements="{escaped_json}"></div>
    <div class="game-tree-legend">
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
            &#9679; サイズ = 確率 / 色 = 勝率
        </span>
    </div>
    <div class="tree-export-overlay">
        <button class="export-btn" onclick="window.exportTreePNG()">
            PNG出力
        </button>
    </div>
</div>
"""


def build_breadcrumb_html(
    breadcrumb_data: list[dict[str, str]],
) -> str:
    """パンくずリストのHTMLを生成する．

    Args:
        breadcrumb_data: [{"hash": "...", "label": "..."}, ...]

    Returns:
        パンくずリストのHTML文字列
    """
    if not breadcrumb_data:
        return '<div class="breadcrumb-nav"></div>'

    items: list[str] = []
    last_idx = len(breadcrumb_data) - 1
    for i, item in enumerate(breadcrumb_data):
        if i > 0:
            items.append(
                '<span class="breadcrumb-sep">&gt;</span>'
            )
        escaped_label = html.escape(item["label"])
        escaped_hash = html.escape(item["hash"])
        if i == last_idx:
            # 現在のノード(クリック不可)
            items.append(
                f'<span class="breadcrumb-item active">'
                f"{escaped_label}</span>"
            )
        else:
            items.append(
                f'<span class="breadcrumb-item" '
                f'data-hash="{escaped_hash}">'
                f"{escaped_label}</span>"
            )

    return f'<div class="breadcrumb-nav">{"".join(items)}</div>'


# ========================================
# Plotly chart builders
# ========================================


def create_analytics_plot(
    analytics_data: dict[str, Any],
) -> go.Figure | None:
    """分岐分析のPlotlyチャートを生成する．

    Args:
        analytics_data: 分析データ(moves, probabilities, win_rates)

    Returns:
        Plotly Figure．データがない場合None．
    """
    moves = analytics_data.get("moves", [])
    probs = analytics_data.get("probabilities", [])
    win_rates = analytics_data.get("win_rates", [])

    if not moves:
        return None

    colors = []
    for wr in win_rates:
        if wr > 0.55:
            colors.append("#2196F3")
        elif wr < 0.45:
            colors.append("#F44336")
        else:
            colors.append("#9E9E9E")

    fig = go.Figure(
        data=[
            go.Bar(
                x=moves,
                y=[p * 100 for p in probs],
                marker_color=colors,
                text=[f"{wr * 100:.1f}%" for wr in win_rates],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>"
                + "確率: %{y:.1f}%<br>"
                + "勝率: %{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="上位指し手の確率分布",
        xaxis_title="指し手",
        yaxis_title="確率 (%)",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=40, b=60),
        font=dict(family="Noto Sans JP, sans-serif"),
    )
    return fig


def create_empty_plot() -> go.Figure:
    """空のPlotlyチャートを生成する．

    Returns:
        空のPlotly Figure
    """
    fig = go.Figure()
    fig.update_layout(
        title="分岐分析",
        template="plotly_white",
        height=300,
        annotations=[
            dict(
                text="ノードを選択してください",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#718096"),
            )
        ],
    )
    return fig
