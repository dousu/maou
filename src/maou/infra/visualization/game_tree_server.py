"""ゲームツリー可視化Gradioサーバー(インフラ層)．

構築済みゲームツリーをインタラクティブに可視化するGradio Webインターフェース．
maou visualize --array-type game-tree から起動される．
"""

import atexit
import html
import json
import logging
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.graph_objects as go

from maou.interface.game_tree_io import GameTreeIO
from maou.interface.game_tree_visualization import (
    GameTreeVisualizationInterface,
)

logger = logging.getLogger(__name__)

# on_node_expanded / on_move_selected / on_back_to_root 共通の返却型
# (tree_html, board_svg, current_root, stats, moves, child_hashes,
#  plot, breadcrumb_html, sfen_text)
_ExpandResult = tuple[
    str,
    str,
    str,
    dict[str, str],
    list[list[str]],
    list[str],
    go.Figure,
    str,
    str,
]

_STATIC_DIR = Path(__file__).parent / "static"


@lru_cache(maxsize=None)
def _load_static_file(filename: str) -> str:
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


def _load_custom_css() -> str:
    """カスタムCSSファイルを読み込む(既存テーマ + ゲームツリー用)．

    Returns:
        結合されたCSS文字列
    """
    css_files = ["theme.css", "components.css"]
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


def _build_breadcrumb_html(
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


def _get_detail_outputs(
    viz: GameTreeVisualizationInterface,
    pos_hash: int,
) -> tuple[
    str,
    dict[str, str],
    list[list[str]],
    list[str],
    go.Figure,
    str,
    str,
]:
    """ノード詳細パネルの出力を生成する．

    Args:
        viz: 可視化インターフェース
        pos_hash: 対象ノードのposition_hash

    Returns:
        (board_svg, stats, display_moves, child_hashes,
         plot, breadcrumb_html, sfen_text)
    """
    board_svg = viz.get_board_svg(pos_hash)
    stats = viz.get_node_stats(pos_hash)
    moves_with_hash = viz.get_move_table(pos_hash)
    analytics = viz.get_analytics_data(pos_hash)
    plot = _create_analytics_plot(analytics)
    if plot is None:
        plot = _create_empty_plot()

    breadcrumb = viz.get_breadcrumb_data(pos_hash)
    breadcrumb_html = _build_breadcrumb_html(breadcrumb)

    sfen_text = viz.export_sfen_path(pos_hash)

    # 表示用データ(3列)とchild_hashリストを分離
    display_moves = [
        [r.japanese, r.probability, r.win_rate]
        for r in moves_with_hash
    ]
    child_hashes = [r.child_hash for r in moves_with_hash]

    return (
        board_svg,
        stats,
        display_moves,
        child_hashes,
        plot,
        breadcrumb_html,
        sfen_text,
    )


def _update_tree_view(
    viz: GameTreeVisualizationInterface,
    root_hash: int,
    display_depth: int,
    min_prob: float,
) -> tuple[
    str,
    str,
    dict[str, str],
    list[list[str]],
    list[str],
    go.Figure,
    str,
    str,
]:
    """ツリービューと詳細パネルを更新する．

    Args:
        viz: 可視化インターフェース
        root_hash: 表示するサブツリーのルートhash
        display_depth: 表示深さ
        min_prob: エッジの最小確率閾値

    Returns:
        (tree_html, board_svg, stats, display_moves, child_hashes,
         plot, breadcrumb_html, sfen_text)
    """
    elements = viz.get_cytoscape_elements(
        root_hash, int(display_depth), min_prob
    )
    elements_json = json.dumps(elements, ensure_ascii=False)
    tree_html = _build_tree_html(elements_json)

    (
        board_svg,
        stats,
        display_moves,
        child_hashes,
        plot,
        breadcrumb_html,
        sfen_text,
    ) = _get_detail_outputs(viz, root_hash)

    return (
        tree_html,
        board_svg,
        stats,
        display_moves,
        child_hashes,
        plot,
        breadcrumb_html,
        sfen_text,
    )


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
    tree_io = GameTreeIO()
    nodes_df, edges_df = tree_io.load(tree_path)
    metadata = tree_io.load_metadata(tree_path)
    logger.info(
        "Loaded tree: %d nodes, %d edges",
        len(nodes_df),
        len(edges_df),
    )

    viz = GameTreeVisualizationInterface(
        nodes_df,
        edges_df,
        initial_sfen=metadata.get("initial_sfen"),
    )
    root_hash = viz.get_root_hash()

    custom_css = _load_custom_css()
    head_scripts = _build_head_scripts()

    # --- コールバック定義 ---

    def on_load(
        display_depth: int,
        min_prob: float,
    ) -> tuple[
        str,
        str,
        dict[str, str],
        list[list[str]],
        list[str],
        go.Figure,
        str,
        str,
    ]:
        """初期表示コールバック．"""
        return _update_tree_view(
            viz, viz.get_root_hash(), display_depth, min_prob
        )

    def on_refresh(
        display_depth: int,
        min_prob: float,
        current_root: str,
    ) -> tuple[
        str,
        str,
        dict[str, str],
        list[list[str]],
        list[str],
        go.Figure,
        str,
        str,
    ]:
        """更新ボタンのコールバック．"""
        try:
            rh = (
                int(current_root)
                if current_root
                else viz.get_root_hash()
            )
        except ValueError:
            logger.warning(
                "Invalid current_root: %s", current_root
            )
            rh = viz.get_root_hash()
        return _update_tree_view(
            viz, rh, display_depth, min_prob
        )

    def on_node_selected(
        node_id: str,
    ) -> tuple[
        str,
        dict[str, str],
        list[list[str]],
        list[str],
        go.Figure,
        str,
        str,
    ]:
        """ノードクリック時のコールバック．"""
        if not node_id:
            return (
                "",
                {},
                [],
                [],
                _create_empty_plot(),
                "",
                "",
            )
        try:
            pos_hash = int(node_id)
        except ValueError:
            logger.warning("Invalid node_id: %s", node_id)
            return (
                "",
                {},
                [],
                [],
                _create_empty_plot(),
                "",
                "",
            )
        return _get_detail_outputs(viz, pos_hash)

    def on_move_selected(
        current_child_hashes: list[str],
        display_depth: int,
        min_prob: float,
        evt: gr.SelectData,
    ) -> _ExpandResult:
        """指し手一覧の行選択時のコールバック．

        Args:
            current_child_hashes: 現在表示中の局面の子ノードhashリスト(gr.State)
            display_depth: 表示深さ
            min_prob: エッジの最小確率閾値
            evt: Gradio の SelectData イベント
        """
        _empty: _ExpandResult = (
            "",
            "",
            "",
            {},
            [],
            [],
            _create_empty_plot(),
            "",
            "",
        )
        if not current_child_hashes or evt.index is None:
            return _empty
        row_idx = (
            evt.index[0]
            if isinstance(evt.index, (list, tuple))
            else evt.index
        )
        if row_idx < 0 or row_idx >= len(current_child_hashes):
            return _empty
        try:
            pos_hash = int(current_child_hashes[row_idx])
        except (ValueError, IndexError):
            return _empty
        (
            tree_html,
            board_svg,
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html,
            sfen_text,
        ) = _update_tree_view(
            viz, pos_hash, display_depth, min_prob
        )
        return (
            tree_html,
            board_svg,
            str(pos_hash),
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html,
            sfen_text,
        )

    def on_node_expanded(
        node_id: str,
        display_depth: int,
        min_prob: float,
    ) -> _ExpandResult:
        """ノードダブルクリック(サブツリー展開)のコールバック．"""
        if not node_id:
            return (
                "",
                "",
                str(viz.get_root_hash()),
                {},
                [],
                [],
                _create_empty_plot(),
                "",
                "",
            )
        try:
            pos_hash = int(node_id)
        except ValueError:
            logger.warning("Invalid node_id: %s", node_id)
            return (
                "",
                "",
                str(viz.get_root_hash()),
                {},
                [],
                [],
                _create_empty_plot(),
                "",
                "",
            )
        (
            tree_html,
            board_svg,
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html,
            sfen_text,
        ) = _update_tree_view(
            viz, pos_hash, display_depth, min_prob
        )
        return (
            tree_html,
            board_svg,
            str(pos_hash),
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html,
            sfen_text,
        )

    def on_back_to_root(
        display_depth: int,
        min_prob: float,
    ) -> _ExpandResult:
        """ルートに戻るボタンのコールバック．"""
        rh = viz.get_root_hash()
        (
            tree_html,
            board_svg,
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html,
            sfen_text,
        ) = _update_tree_view(viz, rh, display_depth, min_prob)
        return (
            tree_html,
            board_svg,
            str(rh),
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html,
            sfen_text,
        )

    # CSV一時ファイル用ディレクトリ(プロセス終了時に自動削除)
    _csv_tmp_dir = tempfile.TemporaryDirectory(
        prefix="maou_game_tree_csv_"
    )
    atexit.register(_csv_tmp_dir.cleanup)

    def on_export_csv(
        current_root: str,
        display_depth: int,
        min_prob: float,
    ) -> str | None:
        """CSV出力ボタンのコールバック．"""
        try:
            rh = (
                int(current_root)
                if current_root
                else viz.get_root_hash()
            )
        except ValueError:
            rh = viz.get_root_hash()

        csv_content = viz.export_subtree_csv(
            rh,
            int(display_depth),
            min_prob,
        )
        if not csv_content.strip():
            return None

        tmp_path = Path(_csv_tmp_dir.name) / (
            f"game_tree_{uuid.uuid4().hex}.csv"
        )
        tmp_path.write_text(csv_content, encoding="utf-8")
        return str(tmp_path)

    # --- UI構築 ---

    with gr.Blocks(
        title="Maou Game Tree Viewer",
    ) as demo:
        gr.Markdown("# Maou Game Tree Viewer")
        initial_sfen = viz.get_initial_sfen()
        gr.Markdown(
            f"Nodes: **{len(nodes_df):,}** / "
            f"Edges: **{len(edges_df):,}** / "
            f"Root: `{initial_sfen}`"
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

        # パンくずリスト
        breadcrumb_html = gr.HTML(
            value=_build_breadcrumb_html(
                [{"hash": str(root_hash), "label": "初期局面"}]
            ),
            label="パンくずリスト",
            show_label=False,
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

                # エクスポートセクション
                with gr.Accordion("エクスポート", open=False):
                    sfen_text = gr.Textbox(
                        label="USI position文字列",
                        interactive=False,
                        lines=2,
                    )
                    export_csv_btn = gr.Button(
                        "CSV出力",
                        variant="secondary",
                        size="sm",
                    )
                    csv_file = gr.File(
                        label="CSVダウンロード",
                        visible=True,
                    )

        # Hidden state
        # NOTE: visible=False だと Gradio 6 の Svelte 条件レンダリングにより
        # DOM要素が生成されず，JSからアクセスできない．
        # CSS で非表示にすることでDOMに残しつつ画面には表示しない．
        selected_node = gr.Textbox(
            elem_id="selected-node-id",
            elem_classes=["js-hidden"],
        )
        expand_node = gr.Textbox(
            elem_id="expand-node-id",
            elem_classes=["js-hidden"],
        )
        current_root_state = gr.Textbox(
            value=str(root_hash),
            elem_id="current-root",
            elem_classes=["js-hidden"],
        )
        # Hidden buttons (JSからクリックしてGradioコールバックを発火)
        select_trigger = gr.Button(
            elem_id="node-select-trigger",
            elem_classes=["js-hidden"],
        )
        expand_trigger = gr.Button(
            elem_id="node-expand-trigger",
            elem_classes=["js-hidden"],
        )
        # 指し手一覧の行選択用child_hashリスト
        child_hashes_state = gr.State([])

        # --- イベントハンドリング ---

        # 初期表示
        _load_outputs = [
            tree_html,
            board_html,
            stats_json,
            move_table,
            child_hashes_state,
            analytics_plot,
            breadcrumb_html,
            sfen_text,
        ]

        demo.load(
            fn=on_load,
            inputs=[depth_slider, min_prob_slider],
            outputs=_load_outputs,
        )

        # 更新ボタン
        refresh_btn.click(
            fn=on_refresh,
            inputs=[
                depth_slider,
                min_prob_slider,
                current_root_state,
            ],
            outputs=_load_outputs,
        )

        # ノード選択(シングルクリック) - hidden buttonクリックで発火
        _select_outputs = [
            board_html,
            stats_json,
            move_table,
            child_hashes_state,
            analytics_plot,
            breadcrumb_html,
            sfen_text,
        ]

        # ノード展開 / 指し手選択共通の出力(ツリー + 詳細パネル)
        _expand_outputs = [
            tree_html,
            board_html,
            current_root_state,
            stats_json,
            move_table,
            child_hashes_state,
            analytics_plot,
            breadcrumb_html,
            sfen_text,
        ]

        select_trigger.click(
            fn=on_node_selected,
            inputs=[selected_node],
            outputs=_select_outputs,
        )

        # 指し手一覧の行選択
        move_table.select(
            fn=on_move_selected,
            inputs=[
                child_hashes_state,
                depth_slider,
                min_prob_slider,
            ],
            outputs=_expand_outputs,
        )

        # ノード展開(ダブルクリック) - hidden buttonクリックで発火
        expand_trigger.click(
            fn=on_node_expanded,
            inputs=[
                expand_node,
                depth_slider,
                min_prob_slider,
            ],
            outputs=_expand_outputs,
        )

        # ルートに戻る
        back_btn.click(
            fn=on_back_to_root,
            inputs=[depth_slider, min_prob_slider],
            outputs=_expand_outputs,
        )

        # CSV出力
        export_csv_btn.click(
            fn=on_export_csv,
            inputs=[
                current_root_state,
                depth_slider,
                min_prob_slider,
            ],
            outputs=[csv_file],
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
