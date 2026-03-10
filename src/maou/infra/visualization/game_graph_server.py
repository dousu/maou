"""ゲームグラフ可視化Gradioサーバー(インフラ層)．

構築済みゲームグラフをインタラクティブに可視化するGradio Webインターフェース．
maou visualize --array-type game-graph から起動される．
"""

import atexit
import json
import logging
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.graph_objects as go

from maou.app.game_graph.layout import (
    GameGraphLayoutService,
    TreeLayout,
)
from maou.infra.visualization.game_graph_shared import (
    ELEM_ID_CURRENT_ROOT,
    ELEM_ID_DEPTH_SLIDER,
    ELEM_ID_EXPAND_BRIDGE,
    ELEM_ID_MIN_PROB_SLIDER,
    ELEM_ID_SELECT_BRIDGE,
    ELEM_ID_VIEWPORT_BRIDGE,
    JS_ON_LOAD_EXPAND,
    JS_ON_LOAD_SELECT,
    JS_ON_LOAD_VIEWPORT,
    build_breadcrumb_html,
    build_tree_html,
    create_analytics_plot,
    create_empty_plot,
    load_static_file,
)
from maou.interface.game_graph_io import GameGraphIO
from maou.interface.game_graph_visualization import (
    GameGraphVisualizationInterface,
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


def _load_custom_css() -> str:
    """カスタムCSSファイルを読み込む(既存テーマ + ゲームグラフ用)．

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
    """Canvas 2D ゲームグラフJSをhead要素に注入するHTMLを生成する．

    demo.launch(head=...)パラメータで使用する．
    gr.HTMLコンポーネントはinnerHTMLで設定されるため<script>タグが
    実行されない問題を回避する．

    CDN依存なし(Cytoscape.js/dagre を除去済み)．

    Returns:
        head要素に注入するHTML文字列
    """
    js_code = load_static_file("game_graph_canvas.js")

    return f"""
<script>
(function() {{
    var jsLoaded = false;

    function initGameGraphJS() {{
        {js_code}
    }}

    function tryRender() {{
        var container = document.getElementById('gt-canvas-container');
        if (!container) return;
        var dataAttr = container.getAttribute('data-canvas');
        if (!dataAttr) return;
        if (container._lastRendered === dataAttr) return;
        container._lastRendered = dataAttr;

        if (!jsLoaded) {{
            jsLoaded = true;
            initGameGraphJS();
        }}

        try {{
            var data = JSON.parse(dataAttr);
            if (typeof window.renderGameGraph === 'function') {{
                window.renderGameGraph(data, 'gt-canvas-container');
            }}
        }} catch (e) {{
            console.error('[maou] Failed to render game tree:', e);
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
            attributeFilter: ['data-canvas'],
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


def _get_detail_outputs(
    viz: GameGraphVisualizationInterface,
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
    plot = create_analytics_plot(analytics)
    if plot is None:
        plot = create_empty_plot()

    breadcrumb = viz.get_breadcrumb_data(pos_hash)
    breadcrumb_html = build_breadcrumb_html(breadcrumb)

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
    viz: GameGraphVisualizationInterface,
    root_hash: int,
    display_depth: int,
    min_prob: float,
    layout: TreeLayout | None = None,
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
        layout: 事前計算されたレイアウト

    Returns:
        (tree_html, board_svg, stats, display_moves, child_hashes,
         plot, breadcrumb_html, sfen_text)
    """
    if layout is None:
        layout = TreeLayout(
            node_positions={}, bounds=(0, 0, 0, 0)
        )
    canvas_data = viz.get_canvas_data(
        root_hash, int(display_depth), min_prob, layout
    )
    canvas_json = json.dumps(canvas_data, ensure_ascii=False)
    tree_html = build_tree_html(canvas_json)

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


def launch_game_graph_server(
    tree_path: Path,
    port: int | None = None,
    share: bool = False,
    server_name: str = "127.0.0.1",
) -> None:
    """ゲームグラフ可視化サーバーを起動する．

    gradio_server.launch_server() から array_type="game-graph" の場合に
    ディスパッチされる．

    Args:
        tree_path: ツリーデータディレクトリ(nodes.feather + edges.feather)
        port: サーバーポート．Noneの場合Gradioの自動選択に委任
        share: Gradio公開リンクを生成するか
        server_name: サーバーバインドアドレス
    """
    # データ読み込み
    tree_io = GameGraphIO()
    nodes_df, edges_df = tree_io.load(tree_path)
    metadata = tree_io.load_metadata(tree_path)
    logger.info(
        "Loaded tree: %d nodes, %d edges",
        len(nodes_df),
        len(edges_df),
    )

    viz = GameGraphVisualizationInterface(
        nodes_df,
        edges_df,
        initial_sfen=metadata.get("initial_sfen"),
    )
    root_hash = viz.get_root_hash()

    # レイアウト事前計算
    layout_svc = GameGraphLayoutService()
    tree_layout = layout_svc.compute_layout(
        nodes_df, edges_df, root_hash
    )
    logger.info(
        "Computed layout: %d positions, bounds=%s",
        len(tree_layout.node_positions),
        tree_layout.bounds,
    )

    # ビューポートクエリ用の空間インデックス
    _spatial_buckets: dict[tuple[int, int], list[int]] = (
        defaultdict(list)
    )
    _bucket_size = 500.0
    for h, (x, y) in tree_layout.node_positions.items():
        bx = int(x // _bucket_size)
        by = int(y // _bucket_size)
        _spatial_buckets[(bx, by)].append(h)

    custom_css = _load_custom_css()
    head_scripts = _build_head_scripts()

    # --- コールバック定義 ---

    # server_functions → .change() コールバック間のデータ受け渡し用
    # NOTE: Gradio はセッション内のリクエストを逐次処理するため，
    # 同一セッション内での競合は発生しない．
    # WARNING: _pending はクロージャとして全セッションに共有されるため，
    # 複数ユーザーが同時接続するマルチセッション環境では競合が発生する．
    # 本モジュール(スタンドアロンモード)は単一ユーザー前提で設計されており，
    # マルチセッション環境では gr.State を使ったセッション分離が必要．
    _pending: dict[str, Any] = {}

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
            viz,
            viz.get_root_hash(),
            display_depth,
            min_prob,
            tree_layout,
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
            viz, rh, display_depth, min_prob, tree_layout
        )

    # --- server_functions: JS から直接呼び出される Python 関数 ---
    # Gradio 6 では JS DOM 操作で Textbox の値を変更しても
    # .input() / .change() が発火しない(Issue #3471, #7954)．
    # gr.HTML の server_functions でデータを処理し，
    # trigger("change") で .change() コールバックを発火する．

    def handle_select(node_id_str: str) -> bool:
        """ノード選択の server_function．

        JS から呼ばれ，結果を _pending に格納する．
        """
        if not node_id_str:
            return False
        try:
            pos_hash = int(node_id_str)
        except (ValueError, TypeError):
            logger.warning("Invalid node_id: %s", node_id_str)
            return False
        _pending["select"] = {
            "data": _get_detail_outputs(viz, pos_hash),
            "hash": pos_hash,
        }
        return True

    def handle_expand(
        node_id_str: str | list[Any],
        display_depth: int | float = 3,
        min_prob: float = 0.01,
    ) -> bool:
        """ノード展開の server_function．

        JS から呼ばれ，結果を _pending に格納する．
        depth / prob は JS 側がスライダー DOM から読み取って渡す．

        Note:
            Gradio 6 の server_functions は複数の JS 引数を
            リストとして第1引数に渡す場合があるため，
            node_id_str がリストの場合は展開して処理する．
        """
        # server_functions が複数引数をリストで渡す場合の展開
        if isinstance(node_id_str, list):
            args = node_id_str
            node_id_str = str(args[0]) if args else ""
            if len(args) > 1:
                display_depth = args[1]
            if len(args) > 2:
                min_prob = args[2]
        if not node_id_str:
            return False
        try:
            pos_hash = int(node_id_str)
        except (ValueError, TypeError):
            logger.warning("Invalid node_id: %s", node_id_str)
            return False
        _pending["expand"] = {
            "data": _update_tree_view(
                viz,
                pos_hash,
                int(display_depth),
                min_prob,
                tree_layout,
            ),
            "hash": pos_hash,
        }
        return True

    def handle_viewport(
        args: list[Any] | float,
        max_x: float = 0,
        min_y: float = 0,
        max_y: float = 0,
    ) -> str:
        """ビューポート範囲内のノード・エッジを返す server_function．

        パン/ズーム後にフロントエンドから呼ばれ，
        可視領域のノード・エッジデータを返す．

        Note:
            Gradio 6 の server_functions は複数の JS 引数を
            リストとして第1引数に渡す場合がある．
        """
        # server_functions がリストで渡す場合の展開
        if isinstance(args, list):
            min_x_v = float(args[0]) if len(args) > 0 else 0
            max_x_v = float(args[1]) if len(args) > 1 else 0
            min_y_v = float(args[2]) if len(args) > 2 else 0
            max_y_v = float(args[3]) if len(args) > 3 else 0
        else:
            min_x_v = float(args)
            max_x_v = float(max_x)
            min_y_v = float(min_y)
            max_y_v = float(max_y)

        # 空間インデックスで該当バケットのノードを収集
        min_bx = int(min_x_v // _bucket_size) - 1
        max_bx = int(max_x_v // _bucket_size) + 1
        min_by = int(min_y_v // _bucket_size) - 1
        max_by = int(max_y_v // _bucket_size) + 1

        visible_hashes: set[int] = set()
        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                bucket = _spatial_buckets.get((bx, by), [])
                for h in bucket:
                    pos = tree_layout.node_positions.get(h)
                    if pos is None:
                        continue
                    x, y = pos
                    if (
                        min_x_v <= x <= max_x_v
                        and min_y_v <= y <= max_y_v
                    ):
                        visible_hashes.add(h)

        # 可視ノードのデータを構築
        canvas_data = viz.get_viewport_data(
            visible_hashes, tree_layout
        )
        return json.dumps(canvas_data, ensure_ascii=False)

    def on_select_result() -> tuple[
        str,
        dict[str, str],
        list[list[str]],
        list[str],
        go.Figure,
        str,
        str,
        str,
    ]:
        """select_bridge.change のコールバック．

        handle_select が格納した結果を返す．
        最後の要素は選択ノードのハッシュ文字列．
        """
        result = _pending.pop("select", None)
        if result:
            return (*result["data"], str(result["hash"]))
        return ("", {}, [], [], create_empty_plot(), "", "", "")

    def on_expand_result() -> _ExpandResult:
        """expand_bridge.change のコールバック．

        handle_expand が格納した結果を返す．
        """
        result = _pending.pop("expand", None)
        if result:
            (
                tree_html_v,
                board_svg,
                stats,
                display_moves,
                child_hashes,
                plot,
                breadcrumb_html_v,
                sfen_text_v,
            ) = result["data"]
            return (
                tree_html_v,
                board_svg,
                str(result["hash"]),
                stats,
                display_moves,
                child_hashes,
                plot,
                breadcrumb_html_v,
                sfen_text_v,
            )
        return (
            "",
            "",
            str(viz.get_root_hash()),
            {},
            [],
            [],
            create_empty_plot(),
            "",
            "",
        )

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
            create_empty_plot(),
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
            tree_html_v,
            board_svg,
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html_v,
            sfen_text_v,
        ) = _update_tree_view(
            viz, pos_hash, display_depth, min_prob, tree_layout
        )
        return (
            tree_html_v,
            board_svg,
            str(pos_hash),
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html_v,
            sfen_text_v,
        )

    def on_back_to_root(
        display_depth: int,
        min_prob: float,
    ) -> _ExpandResult:
        """ルートに戻るボタンのコールバック．"""
        rh = viz.get_root_hash()
        (
            tree_html_v,
            board_svg,
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html_v,
            sfen_text_v,
        ) = _update_tree_view(
            viz, rh, display_depth, min_prob, tree_layout
        )
        return (
            tree_html_v,
            board_svg,
            str(rh),
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html_v,
            sfen_text_v,
        )

    def on_set_as_root(
        selected_node: str,
        current_root: str,
        display_depth: int,
        min_prob: float,
    ) -> _ExpandResult:
        """選択ノードをルートに設定するボタンのコールバック．

        選択ノードが現在のルートと同じ場合は再描画をスキップする．
        """
        _noop: _ExpandResult = (
            gr.skip(),  # type: ignore[assignment]
            gr.skip(),  # type: ignore[assignment]
            gr.skip(),  # type: ignore[assignment]
            gr.skip(),  # type: ignore[assignment]
            gr.skip(),  # type: ignore[assignment]
            gr.skip(),  # type: ignore[assignment]
            gr.skip(),  # type: ignore[assignment]
            gr.skip(),  # type: ignore[assignment]
            gr.skip(),  # type: ignore[assignment]
        )
        if not selected_node:
            return _noop
        if selected_node == current_root:
            return _noop
        try:
            pos_hash = int(selected_node)
        except (ValueError, TypeError):
            return _noop
        (
            tree_html_v,
            board_svg,
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html_v,
            sfen_text_v,
        ) = _update_tree_view(
            viz, pos_hash, display_depth, min_prob, tree_layout
        )
        return (
            tree_html_v,
            board_svg,
            str(pos_hash),
            stats,
            display_moves,
            child_hashes,
            plot,
            breadcrumb_html_v,
            sfen_text_v,
        )

    # CSV一時ファイル用ディレクトリ(プロセス終了時に自動削除)
    _csv_tmp_dir = tempfile.TemporaryDirectory(
        prefix="maou_game_graph_csv_"
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
            f"game_graph_{uuid.uuid4().hex}.csv"
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
                maximum=20,
                value=3,
                step=1,
                label="表示深さ",
                scale=1,
                elem_id=ELEM_ID_DEPTH_SLIDER,
            )
            min_prob_slider = gr.Slider(
                minimum=0.001,
                maximum=0.3,
                value=0.01,
                step=0.001,
                label="最小確率",
                scale=1,
                elem_id=ELEM_ID_MIN_PROB_SLIDER,
            )
            refresh_btn = gr.Button(
                "更新", variant="primary", scale=0
            )
            back_btn = gr.Button(
                "ルートに戻る", variant="secondary", scale=0
            )
            set_root_btn = gr.Button(
                "ルートに設定", variant="secondary", scale=0
            )

        # パンくずリスト
        breadcrumb_html = gr.HTML(
            value=build_breadcrumb_html(
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

        # --- Bridge components (server_functions) ---
        # Gradio 6 では JS DOM 操作で Textbox の値を変更しても
        # .input() / .change() が発火しない．
        # gr.HTML の server_functions を使い，JS → Python を直接呼び出す．
        # 処理完了後に trigger("change") で .change() を発火し，
        # Gradio の出力パイプラインで UI コンポーネントを更新する．
        select_bridge = gr.HTML(
            value="",
            elem_id=ELEM_ID_SELECT_BRIDGE,
            elem_classes=["maou-hidden"],
            server_functions=[handle_select],
            js_on_load=JS_ON_LOAD_SELECT,
        )
        expand_bridge = gr.HTML(
            value="",
            elem_id=ELEM_ID_EXPAND_BRIDGE,
            elem_classes=["maou-hidden"],
            server_functions=[handle_expand],
            js_on_load=JS_ON_LOAD_EXPAND,
        )
        # ビューポートクエリ用ブリッジ(Phase 4: 遅延読み込み)
        gr.HTML(
            value="",
            elem_id=ELEM_ID_VIEWPORT_BRIDGE,
            elem_classes=["maou-hidden"],
            server_functions=[handle_viewport],
            js_on_load=JS_ON_LOAD_VIEWPORT,
        )

        # Hidden state
        current_root_state = gr.Textbox(
            label="",
            value=str(root_hash),
            elem_id=ELEM_ID_CURRENT_ROOT,
            elem_classes=["maou-hidden"],
        )
        # 指し手一覧の行選択用child_hashリスト
        child_hashes_state = gr.State([])
        # ツリー上で選択中のノードhash("ルートに設定"ボタン用)
        selected_node_state = gr.State("")

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

        # ノード選択(シングルクリック) - server_functions → trigger("change")
        _select_outputs = [
            board_html,
            stats_json,
            move_table,
            child_hashes_state,
            analytics_plot,
            breadcrumb_html,
            sfen_text,
            selected_node_state,
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

        # server_functions で処理後，trigger("change") で発火する
        select_bridge.change(
            fn=on_select_result,
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

        # ノード展開(ダブルクリック / パンくずクリック)
        expand_bridge.change(
            fn=on_expand_result,
            outputs=_expand_outputs,
        )

        # ルートに戻る
        back_btn.click(
            fn=on_back_to_root,
            inputs=[depth_slider, min_prob_slider],
            outputs=_expand_outputs,
        )

        # 選択ノードをルートに設定
        set_root_btn.click(
            fn=on_set_as_root,
            inputs=[
                selected_node_state,
                current_root_state,
                depth_slider,
                min_prob_slider,
            ],
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
