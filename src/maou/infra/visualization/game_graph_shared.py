"""ゲームグラフ可視化の共有ユーティリティ(インフラ層)．

game_graph_server.py (スタンドアロンモード) と gradio_server.py (埋め込みモード)
の両方から使用されるHTML生成関数・Plotly生成関数・JS定数を提供する．
"""

import html
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _preimport_pandas() -> None:
    """pandas をメインスレッドで先に読み切っておく．

    plotly の配列判定 (_plotly_utils.basevalidators.is_homogeneous_array) は
    ``sys.modules`` にある pandas をそのまま参照する (optional_imports は
    import せず取得するだけ)．gr.Dataframe を廃してからは起動時に pandas を
    読むものが無く，最初の import がワーカースレッドで走るため，
    読み込み途中のモジュールを掴んで
    ``partially initialized module 'pandas' has no attribute 'Series'``
    で描画が落ちる．

    pandas は gradio / plotly の推移的依存だが，この経路では実質必須なので
    visualize extra に明示してある．
    """
    try:
        import pandas  # noqa: F401
    except (
        ImportError
    ):  # pragma: no cover - 無くても描画自体は動く
        logger.debug(
            "pandas is unavailable; plotly will treat arrays as lists"
        )


_preimport_pandas()

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

# ゲームグラフ凡例のHTML．色と輪郭の意味は Canvas レンダラー
# (static/game_graph_canvas.js の winRateToColor / ノード描画) に対応する．
GAME_GRAPH_LEGEND_HTML = """
<div class="vz-legend">
  <div class="vz-legend-item">
    <span class="vz-legend-swatch" style="background:rgb(25,118,210)"></span>
    先手有利 — 最善手勝率 &gt; 55%
  </div>
  <div class="vz-legend-item">
    <span class="vz-legend-swatch" style="background:#9E9E9E"></span>
    互角 — 45% 〜 55%
  </div>
  <div class="vz-legend-item">
    <span class="vz-legend-swatch" style="background:rgb(211,47,47)"></span>
    後手有利 — 最善手勝率 &lt; 45%
  </div>
  <div class="vz-legend-item">
    <span class="vz-legend-swatch"
          style="background:transparent;border:3px solid #0070f3"></span>
    選択中のノード
  </div>
  <div class="vz-legend-item">
    <span class="vz-legend-swatch"
          style="background:transparent;border:2px dashed #ff9800"></span>
    深さ打ち切り (未展開の子を持つ)
  </div>
  <div class="vz-legend-note">
    ノード径は親エッジ確率の平方根に比例し，エッジ線幅は確率に比例する．
  </div>
</div>
"""


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

ELEM_ID_ROW_BRIDGE = "vz-row-bridge"
"""結果一覧の行クリックを受けるブリッジ gr.HTML の elem_id．"""

ELEM_ID_MOVE_BRIDGE = "vz-move-bridge"
"""指し手一覧の行クリックを受けるブリッジ gr.HTML の elem_id．"""

JS_ON_LOAD_ROW = (
    "window.__maou_row = {server: server, trigger: trigger};"
)
"""row_bridge の js_on_load．static/visualize_workbench.js から使う．"""

JS_ON_LOAD_MOVE = "window.__maou_movesel = {server: server, trigger: trigger};"
"""move_bridge の js_on_load．static/visualize_workbench.js から使う．"""


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


def _fmt_value(value: Any) -> str:
    """表示用に値を整形する (エスケープ込み)．

    Args:
        value: 任意の値

    Returns:
        HTML エスケープ済みの文字列
    """
    if value is None:
        return "—"
    if isinstance(value, float):
        text = f"{value:.6g}"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return html.escape(text)


def build_workbench_head() -> str:
    """ワークベンチの行クリックブリッジ JS を head に注入する HTML を返す．

    gr.HTML は innerHTML で差し替わり ``<script>`` が実行されないため，
    head 側に置く (game_graph_server._build_head_scripts と同じ方針)．

    Returns:
        head に入れる HTML 文字列
    """
    return f"<script>{load_static_file('visualize_workbench.js')}</script>"


def build_kv_html(data: dict[str, Any] | None) -> str:
    """キーと値の一覧を vz-kv 行として組み立てる．

    Args:
        data: 表示する辞書．None または空なら空状態を返す．

    Returns:
        HTML 文字列
    """
    if not data:
        return '<div class="vz-empty">データがありません</div>'
    rows = [
        '<div class="vz-kv">'
        f"<span>{html.escape(str(key))}</span>"
        f'<span class="vz-m">{_fmt_value(value)}</span>'
        "</div>"
        for key, value in data.items()
    ]
    return '<div class="vz-kv-list">' + "".join(rows) + "</div>"


def build_stats_grid_html(data: dict[str, Any] | None) -> str:
    """統計値を 2 列のグリッドとして組み立てる．

    Args:
        data: 統計値の辞書．入れ子の辞書は 1 段だけ見出し付きで展開する．

    Returns:
        HTML 文字列
    """
    if not data:
        return '<div class="vz-empty">「更新」で統計を取得します</div>'

    parts: list[str] = []
    flat: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested[str(key)] = value
        else:
            flat[str(key)] = value

    if flat:
        parts.append(
            '<div class="vz-kv-grid">'
            + "".join(
                '<div class="vz-kv">'
                f"<span>{html.escape(k)}</span>"
                f'<span class="vz-m">{_fmt_value(v)}</span>'
                "</div>"
                for k, v in flat.items()
            )
            + "</div>"
        )
    for title, sub in nested.items():
        parts.append(
            f'<div class="vz-sub-lbl">{html.escape(title)}</div>'
            '<div class="vz-kv-grid">'
            + "".join(
                '<div class="vz-kv">'
                f"<span>{html.escape(str(k))}</span>"
                f'<span class="vz-m">{_fmt_value(v)}</span>'
                "</div>"
                for k, v in sub.items()
            )
            + "</div>"
        )
    return "".join(parts)


def _grid_template(n_cols: int) -> str:
    """列数から grid-template-columns を決める．

    先頭を通し番号，2 列目を可変幅の主キー，残りを数値列とみなす．

    Args:
        n_cols: 列数

    Returns:
        grid-template-columns の値
    """
    if n_cols <= 1:
        return "1fr"
    if n_cols == 2:
        return "1fr 72px"
    # 幅 340px のレールに 4 列まで収める．数値列は詰めて主キーに幅を回す．
    return "42px minmax(0, 1fr)" + " 58px" * (n_cols - 2)


def build_row_table_html(
    headers: list[str],
    rows: list[list[Any]],
    *,
    selected: int | None = None,
    clickable: bool = False,
    empty_message: str = "該当するレコードがありません",
) -> str:
    """罫線だけの表 (vz-h ヘッダー + vz-row 行) を組み立てる．

    Args:
        headers: 列見出し
        rows: 行データ (セルの並び)
        selected: 強調表示する行番号 (0 始まり)．None なら強調しない．
        clickable: True なら各行に data-row を付けてクリック可能にする．
        empty_message: 行が無いときに出す文言

    Returns:
        HTML 文字列
    """
    n_cols = (
        len(headers)
        if headers
        else (len(rows[0]) if rows else 0)
    )
    if n_cols == 0:
        return f'<div class="vz-empty">{html.escape(empty_message)}</div>'
    tmpl = _grid_template(n_cols)

    head = (
        f'<div class="vz-h" style="grid-template-columns:{tmpl}">'
        + "".join(
            f"<span{_align(i, n_cols)}>{html.escape(str(h))}</span>"
            for i, h in enumerate(headers)
        )
        + "</div>"
    )
    if not rows:
        return (
            '<div class="vz-table">'
            + head
            + f'<div class="vz-empty">{html.escape(empty_message)}</div>'
            + "</div>"
        )

    body: list[str] = []
    for i, row in enumerate(rows):
        cls = "vz-row" + (" on" if selected == i else "")
        attrs = f' data-row="{i}"' if clickable else ""
        cells = "".join(
            f'<span class="vz-m"{_align(j, n_cols)}>{_fmt_value(c)}</span>'
            for j, c in enumerate(row[:n_cols])
        )
        body.append(
            f'<div class="{cls}"{attrs} '
            f'style="grid-template-columns:{tmpl}">{cells}</div>'
        )
    return (
        '<div class="vz-table">'
        + head
        + '<div class="vz-table-body">'
        + "".join(body)
        + "</div></div>"
    )


def _align(col: int, n_cols: int) -> str:
    """数値列 (3 列目以降と最終列) を右寄せにする style 属性を返す．

    Args:
        col: 列番号 (0 始まり)
        n_cols: 列数

    Returns:
        style 属性の文字列 (不要なら空文字)
    """
    if n_cols >= 3 and col >= 2:
        return ' style="text-align:right"'
    if n_cols == 2 and col == 1:
        return ' style="text-align:right"'
    return ""


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
        margin={"l": 40, "r": 20, "t": 40, "b": 60},
        font={"family": "Noto Sans JP, sans-serif"},
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
            {
                "text": "ノードを選択してください",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": "#718096"},
            }
        ],
    )
    return fig
