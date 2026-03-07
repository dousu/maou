"""ゲームツリー可視化Gradioサーバー（インフラ層）．

構築済みゲームツリーをインタラクティブに可視化するGradio Webインターフェース．
maou visualize --array-type game-tree から起動される．
"""

import json
import logging
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.graph_objects as go

from maou.interface.game_tree_io import GameTreeIO
from maou.interface.game_tree_visualization import (
    GameTreeVisualizationInterface,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def _load_static_file(filename: str) -> str:
    """staticディレクトリからファイルを読み込む．

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


def _load_custom_css() -> str:
    """カスタムCSSファイルを読み込む(既存テーマ + ゲームツリー用)．

    Returns:
        結合されたCSS文字列
    """
    css_files = ["theme.css", "components.css", "game_tree.css"]
    css_parts = []
    for css_file in css_files:
        css_path = _STATIC_DIR / css_file
        if css_path.exists():
            css_parts.append(
                css_path.read_text(encoding="utf-8")
            )
    return "\n\n".join(css_parts)


def _build_head_scripts() -> str:
    """CDNスクリプトとゲームツリーJSをhead要素に注入するHTMLを生成する．

    demo.launch(head=...)パラメータで使用する．
    gr.HTMLコンポーネントはinnerHTMLで設定されるため<script>タグが
    実行されない問題を回避する．

    Returns:
        head要素に注入するHTML文字列
    """
    js_code = _load_static_file("game_tree.js")

    return f"""
<script>
(function() {{
    var CDN_SCRIPTS = [
        'https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js',
        'https://unpkg.com/dagre@0.8.5/dist/dagre.min.js',
        'https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js'
    ];
    var scriptsLoaded = false;

    function loadScriptsSequentially(urls, callback) {{
        if (urls.length === 0) {{ callback(); return; }}
        var url = urls[0];
        var existing = document.querySelector('script[src="' + url + '"]');
        if (existing) {{
            loadScriptsSequentially(urls.slice(1), callback);
            return;
        }}
        var s = document.createElement('script');
        s.src = url;
        s.onload = function() {{
            loadScriptsSequentially(urls.slice(1), callback);
        }};
        s.onerror = function() {{
            console.error('Failed to load: ' + url);
        }};
        document.head.appendChild(s);
    }}

    function initGameTreeJS() {{
        {js_code}
    }}

    function tryRender() {{
        var cy = document.getElementById('cy');
        if (!cy) return;
        var dataAttr = cy.getAttribute('data-elements');
        if (!dataAttr) return;
        if (cy._lastRendered === dataAttr) return;
        cy._lastRendered = dataAttr;

        function doRender() {{
            try {{
                var elements = JSON.parse(dataAttr);
                if (typeof window.renderGameTree === 'function') {{
                    window.renderGameTree(elements, 'cy');
                }}
            }} catch (e) {{
                console.error('Failed to render game tree:', e);
            }}
        }}

        if (scriptsLoaded) {{
            doRender();
        }} else {{
            loadScriptsSequentially(CDN_SCRIPTS, function() {{
                scriptsLoaded = true;
                initGameTreeJS();
                doRender();
            }});
        }}
    }}

    var observer = new MutationObserver(function() {{
        tryRender();
    }});

    function startObserving() {{
        observer.observe(document.body, {{
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['data-elements'],
        }});
        tryRender();
    }}

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', startObserving);
    }} else {{
        startObserving();
    }}
}})();
</script>
"""


def _build_tree_html(elements_json: str) -> str:
    """ツリー表示用HTMLを生成する．

    Cytoscape.jsのレンダリングはhead要素のMutationObserverが
    data-elements属性の変更を検知して自動的に実行する．

    Args:
        elements_json: Cytoscape elements の JSON 文字列

    Returns:
        HTML文字列
    """
    css_code = _load_static_file("game_tree.css")

    return f"""
<style>{css_code}</style>
<div class="game-tree-container">
    <div id="cy" data-elements='{elements_json}'></div>
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
</div>
"""


def _create_analytics_plot(
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


def _create_empty_plot() -> go.Figure:
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


def _update_tree_view(
    viz: GameTreeVisualizationInterface,
    root_hash: int,
    display_depth: int,
    min_prob: float,
) -> tuple[
    str, str, dict[str, str], list[list[str]], go.Figure
]:
    """ツリービューと詳細パネルを更新する．

    Args:
        viz: 可視化インターフェース
        root_hash: 表示するサブツリーのルートhash
        display_depth: 表示深さ
        min_prob: エッジの最小確率閾値

    Returns:
        (tree_html, board_svg, stats, moves, plot) のタプル
    """
    elements = viz.get_cytoscape_elements(
        root_hash, int(display_depth), min_prob
    )
    elements_json = json.dumps(elements, ensure_ascii=False)
    tree_html = _build_tree_html(elements_json)

    board_svg = viz.get_board_svg(root_hash)
    stats = viz.get_node_stats(root_hash)
    moves = viz.get_move_table(root_hash)
    analytics = viz.get_analytics_data(root_hash)
    plot = _create_analytics_plot(analytics)
    if plot is None:
        plot = _create_empty_plot()

    return tree_html, board_svg, stats, moves, plot


def launch_game_tree_server(
    tree_path: Path,
    port: int | None = None,
    share: bool = False,
    server_name: str = "127.0.0.1",
) -> None:
    """ゲームツリー可視化サーバーを起動する．

    gradio_server.launch_server() から array_type="game-tree" の場合に
    ディスパッチされる．

    Args:
        tree_path: ツリーデータディレクトリ(nodes.feather + edges.feather)
        port: サーバーポート．Noneの場合Gradioの自動選択に委任
        share: Gradio公開リンクを生成するか
        server_name: サーバーバインドアドレス
    """
    # データ読み込み
    io = GameTreeIO()
    nodes_df, edges_df = io.load(tree_path)
    logger.info(
        "Loaded tree: %d nodes, %d edges",
        len(nodes_df),
        len(edges_df),
    )

    # ルートノード特定(depth=0)
    root_nodes = nodes_df.filter(nodes_df["depth"] == 0)
    if len(root_nodes) == 0:
        raise ValueError(
            "ルートノード(depth=0)が見つかりません"
        )
    root_hash = int(root_nodes["position_hash"][0])

    viz = GameTreeVisualizationInterface(
        nodes_df, edges_df, root_hash
    )

    custom_css = _load_custom_css()
    head_scripts = _build_head_scripts()

    # --- コールバック定義 ---

    def on_load(
        display_depth: int,
        min_prob: float,
    ) -> tuple[
        str, str, dict[str, str], list[list[str]], go.Figure
    ]:
        """初期表示コールバック．"""
        return _update_tree_view(
            viz, root_hash, display_depth, min_prob
        )

    def on_refresh(
        display_depth: int,
        min_prob: float,
        current_root: str,
    ) -> tuple[
        str, str, dict[str, str], list[list[str]], go.Figure
    ]:
        """更新ボタンのコールバック．"""
        rh = int(current_root) if current_root else root_hash
        return _update_tree_view(
            viz, rh, display_depth, min_prob
        )

    def on_node_selected(
        node_id: str,
    ) -> tuple[str, dict[str, str], list[list[str]], go.Figure]:
        """ノードクリック時のコールバック．"""
        if not node_id:
            return ("", {}, [], _create_empty_plot())
        try:
            pos_hash = int(node_id)
        except ValueError:
            logger.warning("Invalid node_id: %s", node_id)
            return ("", {}, [], _create_empty_plot())
        board_svg = viz.get_board_svg(pos_hash)
        stats = viz.get_node_stats(pos_hash)
        moves = viz.get_move_table(pos_hash)
        analytics = viz.get_analytics_data(pos_hash)
        plot = _create_analytics_plot(analytics)
        if plot is None:
            plot = _create_empty_plot()
        return board_svg, stats, moves, plot

    def on_node_expanded(
        node_id: str,
        display_depth: int,
        min_prob: float,
    ) -> tuple[
        str,
        str,
        str,
        dict[str, str],
        list[list[str]],
        go.Figure,
    ]:
        """ノードダブルクリック(サブツリー展開)のコールバック．"""
        if not node_id:
            return (
                "",
                "",
                str(root_hash),
                {},
                [],
                _create_empty_plot(),
            )
        try:
            pos_hash = int(node_id)
        except ValueError:
            logger.warning("Invalid node_id: %s", node_id)
            return (
                "",
                "",
                str(root_hash),
                {},
                [],
                _create_empty_plot(),
            )
        tree_html, board_svg, stats, moves, plot = (
            _update_tree_view(
                viz, pos_hash, display_depth, min_prob
            )
        )
        return (
            tree_html,
            board_svg,
            str(pos_hash),
            stats,
            moves,
            plot,
        )

    def on_back_to_root(
        display_depth: int,
        min_prob: float,
    ) -> tuple[
        str,
        str,
        str,
        dict[str, str],
        list[list[str]],
        go.Figure,
    ]:
        """ルートに戻るボタンのコールバック．"""
        tree_html, board_svg, stats, moves, plot = (
            _update_tree_view(
                viz, root_hash, display_depth, min_prob
            )
        )
        return (
            tree_html,
            board_svg,
            str(root_hash),
            stats,
            moves,
            plot,
        )

    # --- UI構築 ---

    with gr.Blocks(
        title="Maou Game Tree Viewer",
    ) as demo:
        gr.Markdown("# Maou Game Tree Viewer")
        gr.Markdown(
            f"Nodes: **{len(nodes_df):,}** / "
            f"Edges: **{len(edges_df):,}** / "
            f"Root: `0x{root_hash:016X}`"
        )

        # コントロールバー
        with gr.Row():
            depth_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="表示深さ",
                scale=1,
            )
            min_prob_slider = gr.Slider(
                minimum=0.001,
                maximum=0.3,
                value=0.01,
                step=0.001,
                label="最小確率",
                scale=1,
            )
            refresh_btn = gr.Button(
                "更新", variant="primary", scale=0
            )
            back_btn = gr.Button(
                "ルートに戻る", variant="secondary", scale=0
            )

        # メインコンテンツ
        with gr.Row():
            with gr.Column(scale=3):
                tree_html = gr.HTML(
                    label="ツリー表示",
                    elem_id="tree-view",
                )

            with gr.Column(scale=2):
                board_html = gr.HTML(label="盤面")
                stats_json = gr.JSON(label="局面統計")
                move_table = gr.Dataframe(
                    headers=["指し手", "確率", "勝率"],
                    label="指し手一覧",
                    interactive=False,
                )
                analytics_plot = gr.Plot(label="分岐分析")

        # Hidden state
        selected_node = gr.Textbox(
            visible=False, elem_id="selected-node-id"
        )
        expand_node = gr.Textbox(
            visible=False, elem_id="expand-node-id"
        )
        current_root_state = gr.Textbox(
            value=str(root_hash),
            visible=False,
            elem_id="current-root",
        )

        # --- イベントハンドリング ---

        demo.load(
            fn=on_load,
            inputs=[depth_slider, min_prob_slider],
            outputs=[
                tree_html,
                board_html,
                stats_json,
                move_table,
                analytics_plot,
            ],
        )

        refresh_btn.click(
            fn=on_refresh,
            inputs=[
                depth_slider,
                min_prob_slider,
                current_root_state,
            ],
            outputs=[
                tree_html,
                board_html,
                stats_json,
                move_table,
                analytics_plot,
            ],
        )

        selected_node.change(
            fn=on_node_selected,
            inputs=[selected_node],
            outputs=[
                board_html,
                stats_json,
                move_table,
                analytics_plot,
            ],
        )

        expand_node.change(
            fn=on_node_expanded,
            inputs=[
                expand_node,
                depth_slider,
                min_prob_slider,
            ],
            outputs=[
                tree_html,
                board_html,
                current_root_state,
                stats_json,
                move_table,
                analytics_plot,
            ],
        )

        back_btn.click(
            fn=on_back_to_root,
            inputs=[depth_slider, min_prob_slider],
            outputs=[
                tree_html,
                board_html,
                current_root_state,
                stats_json,
                move_table,
                analytics_plot,
            ],
        )

    # サーバー起動
    launch_kwargs: dict[str, Any] = {
        "share": share,
        "server_name": server_name,
        "show_error": True,
        "css": custom_css,
        "head": head_scripts,
    }
    if port is not None:
        launch_kwargs["server_port"] = port

    demo.launch(**launch_kwargs)
