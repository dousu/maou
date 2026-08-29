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
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.graph_objects as go

from maou.infra.visualization.game_graph_shared import (
    ELEM_ID_EXPAND_BRIDGE,
    ELEM_ID_SELECT_BRIDGE,
    ELEM_ID_VIEWPORT_BRIDGE,
    FONT_LINKS,
    JS_ON_LOAD_EXPAND,
    JS_ON_LOAD_SELECT,
    JS_ON_LOAD_VIEWPORT,
    WORKBENCH_LANES,
    as_float,
    as_int,
    build_graph_html,
    build_workbench_head,
    load_static_file,
    make_workbench_bridge,
    workbench_js_on_load,
)
from maou.interface.game_graph_io import GameGraphIO
from maou.interface.game_graph_visualization import (
    GameGraphVisualizationInterface,
)
from maou.interface.visualize_workbench import (
    GraphData,
    StatusView,
    WorkbenchState,
    render_workbench,
)

logger = logging.getLogger(__name__)

# on_node_expanded / on_move_selected / on_back_to_root 共通の返却型
# (graph_html, board_svg, current_root, stats, moves, child_hashes,
#  plot, breadcrumb_html, sfen_text)
_ExpandResult = tuple[
    str,
    str,
    str,
    str,
    str,
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
    # 画面はワークベンチ 1 枚なので，Gradio コンポーネント向けの
    # theme.css / components.css は読み込まない (旧 UI の遺物であり，
    # 残すとワークベンチのトークンと競合する)．
    css_files = ["visualize_workbench.css"]
    css_parts = []
    for css_file in css_files:
        css_path = _STATIC_DIR / css_file
        if css_path.exists():
            css_parts.append(
                css_path.read_text(encoding="utf-8")
            )
    return "\n\n".join(css_parts)


def _build_head_scripts() -> str:
    """Canvas 2D ゲームグラフのCSS・JSをhead要素に注入するHTMLを生成する．

    demo.launch(head=...)パラメータで使用する．
    gr.HTMLコンポーネントはinnerHTMLで設定されるため<script>タグが
    実行されない問題を回避する．CSSも同様にhead注入することで
    gr.HTML更新ごとの重複注入を防止する．

    CDN依存なし(Cytoscape.js/dagre を除去済み)．

    Returns:
        head要素に注入するHTML文字列
    """
    css_code = load_static_file("game_graph.css")
    js_code = load_static_file("game_graph_canvas.js")

    return f"""
<style>{css_code}</style>
<script>
(function() {{
    // IIFE スコープに閉じたフラグ．ページ単位で一度だけ
    // initGameGraphJS() を呼び出すことを保証する．
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
            console.error('[maou] Failed to render game graph:', e);
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


def launch_game_graph_server(
    graph_path: Path,
    port: int | None = None,
    share: bool = False,
    server_name: str = "127.0.0.1",
) -> None:
    """ゲームグラフ可視化サーバーを起動する．

    gradio_server.launch_server() から array_type="game-graph" の場合に
    ディスパッチされる．

    Args:
        graph_path: グラフデータディレクトリ(nodes.feather + edges.feather)
        port: サーバーポート．Noneの場合Gradioの自動選択に委任
        share: Gradio公開リンクを生成するか
        server_name: サーバーバインドアドレス
    """
    # データ読み込み
    graph_io = GameGraphIO()
    nodes_df, edges_df = graph_io.load(graph_path)
    metadata = graph_io.load_metadata(graph_path)
    logger.info(
        "Loaded graph: %d nodes, %d edges",
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
    graph_layout = viz.compute_layout()
    logger.info(
        "Computed layout: %d positions, bounds=%s",
        len(graph_layout.node_positions),
        graph_layout.bounds,
    )

    # ビューポートクエリ用の空間インデックス
    _spatial_buckets: dict[tuple[int, int], list[int]] = (
        defaultdict(list)
    )
    _bucket_size = 500.0
    for h, (x, y) in graph_layout.node_positions.items():
        bx = int(x // _bucket_size)
        by = int(y // _bucket_size)
        _spatial_buckets[(bx, by)].append(h)

    custom_css = _load_custom_css()
    head_scripts = (
        FONT_LINKS
        + _build_head_scripts()
        + build_workbench_head()
    )

    # server_functions → .change() コールバック間のデータ受け渡し用．
    # WARNING: クロージャとして全ブラウザセッションで共有されるため，
    # 複数ユーザーの同時接続では競合する．本モジュール (スタンドアロン
    # モード) は単一利用者を前提とする．
    _pending: dict[str, Any] = {}

    # --- server_functions: JS から直接呼び出される Python 関数 ---
    # Gradio 6 では JS DOM 操作で Textbox の値を変更しても
    # .input() / .change() が発火しない(Issue #3471, #7954)．
    # gr.HTML の server_functions でデータを処理し，
    # trigger("change") で .change() コールバックを発火する．

    def handle_move_select(row_str: str) -> bool:
        """指し手一覧の行クリックの server_function．

        JS から呼ばれ，行番号を _pending に控える．
        """
        try:
            _pending["move_row"] = int(row_str)
        except (TypeError, ValueError):
            logger.warning("Invalid move row: %s", row_str)
            return False
        return True

    def handle_select(node_id_str: str) -> bool:
        """ノード選択の server_function．

        Canvas のシングルクリックで呼ばれ，選んだノードの hash を
        _pending に控える (描画はワークベンチ側でまとめて行う)．

        Args:
            node_id_str: ノードの position_hash 文字列

        Returns:
            控えられたら True
        """
        if not node_id_str:
            return False
        try:
            _pending["node"] = str(int(node_id_str))
        except (ValueError, TypeError):
            logger.warning("Invalid node_id: %s", node_id_str)
            return False
        return True

    def handle_expand(
        node_id_str: str | list[Any],
        display_depth: float = 3,
        min_prob: float = 0.01,
    ) -> bool:
        """ノード展開の server_function．

        Canvas のダブルクリックで呼ばれ，新しいルートにするノードの
        hash を _pending に控える．深さ・確率は JS がスライダー DOM
        から読んで渡すが，値の保持はワークベンチ状態側が持つので
        ここでは使わない．

        Note:
            Gradio 6 の server_functions は複数の JS 引数をリストとして
            第1引数に渡す場合があるため，リストなら展開して扱う．

        Args:
            node_id_str: ノードの position_hash 文字列 (または引数リスト)
            display_depth: 表示深さ (JS から渡るが未使用)
            min_prob: 最小確率 (JS から渡るが未使用)

        Returns:
            控えられたら True
        """
        if isinstance(node_id_str, list):
            args = node_id_str
            node_id_str = str(args[0]) if args else ""
        if not node_id_str:
            return False
        try:
            _pending["node"] = str(int(node_id_str))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid expand node_id: %s", node_id_str
            )
            return False
        return True

    def handle_viewport(
        min_x_or_args: list[Any] | float,
        max_x: float = 0,
        min_y: float = 0,
        max_y: float = 0,
    ) -> str:
        """ビューポート範囲内のノード・エッジを返す server_function．

        パン/ズーム後にフロントエンドから呼ばれ，
        可視領域のノード・エッジデータを返す．

        Args:
            min_x_or_args: ビューポートの min_x 値．Gradio 6 の
                server_functions が複数の JS 引数をリストとして
                第1引数に渡す場合は [min_x, max_x, min_y, max_y]．
            max_x: ビューポートの max_x (個別引数渡し時)
            min_y: ビューポートの min_y (個別引数渡し時)
            max_y: ビューポートの max_y (個別引数渡し時)
        """
        # server_functions がリストで渡す場合の展開
        if isinstance(min_x_or_args, list):
            args = min_x_or_args
            min_x_v = float(args[0]) if len(args) > 0 else 0
            max_x_v = float(args[1]) if len(args) > 1 else 0
            min_y_v = float(args[2]) if len(args) > 2 else 0
            max_y_v = float(args[3]) if len(args) > 3 else 0
        else:
            min_x_v = float(min_x_or_args)
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
                    pos = graph_layout.node_positions.get(h)
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
            visible_hashes, graph_layout
        )
        return json.dumps(canvas_data, ensure_ascii=False)

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

        csv_content = viz.export_subgraph_csv(
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
    #
    # 画面は gradio_server と同じワークベンチ 1 枚
    # (maou.interface.visualize_workbench)．操作は data-action として
    # static/visualize_workbench.js から，Canvas 上のクリックは従来どおり
    # select / expand / viewport ブリッジから届く．

    render_seq = {"n": 0}

    def _collect(state: WorkbenchState) -> GraphData:
        """状態からグラフ画面の表示データを組み立てる．

        Args:
            state: 操作状態

        Returns:
            GraphData
        """
        try:
            rh = (
                int(state.node)
                if state.node
                else viz.get_root_hash()
            )
        except (TypeError, ValueError):
            rh = viz.get_root_hash()

        canvas = viz.get_canvas_data(
            rh,
            int(state.depth),
            float(state.min_prob),
            graph_layout,
        )
        nodes_total, edges_total = viz.get_counts()
        return GraphData(
            graph_html=build_graph_html(
                json.dumps(canvas, ensure_ascii=False)
            ),
            breadcrumb=[
                (
                    str(c.get("label", "")),
                    str(c.get("hash", "")),
                )
                for c in viz.get_breadcrumb_data(rh)
            ],
            board_svg=viz.get_board_svg(rh),
            node_stats=viz.get_node_stats(rh),
            moves=[
                [r.japanese, r.probability, r.win_rate]
                for r in viz.get_move_table(rh)
            ],
            usi_line=viz.export_sfen_path(rh),
            node_count=len(canvas.get("nodes", [])),
            edge_count=edges_total,
            total_nodes=nodes_total,
        )

    def _status(state: WorkbenchState) -> StatusView:
        """トップバーの状態を組み立てる．

        Args:
            state: 操作状態

        Returns:
            StatusView
        """
        nodes_total, edges_total = viz.get_counts()
        return StatusView(
            badge="GRAPH",
            tone="ok",
            count_main=f"{nodes_total:,}",
            count_unit=f"nodes / {edges_total:,} edges",
            path_label=str(graph_path),
        )

    def _render(state: WorkbenchState) -> str:
        """ワークベンチ全体の HTML を返す．

        Args:
            state: 操作状態

        Returns:
            HTML 文字列
        """
        render_seq["n"] += 1
        return render_workbench(
            state,
            _status(state),
            graph=_collect(state),
            render_stamp=str(render_seq["n"]),
            # スタンドアロンはグラフ専用 (レコードのインデックスを持たない)
            types_enabled=False,
        )

    def _child_hash_at(state: WorkbenchState, row: int) -> str:
        """指し手一覧の行番号から子ノード hash を引く．

        Args:
            state: 操作状態
            row: 行番号 (0 始まり)

        Returns:
            子ノードの hash 文字列．引けなければ空文字．
        """
        try:
            rh = (
                int(state.node)
                if state.node
                else viz.get_root_hash()
            )
        except (TypeError, ValueError):
            return ""
        moves = viz.get_move_table(rh)
        if 0 <= row < len(moves):
            return str(moves[row].child_hash)
        return ""

    def on_workbench_action(
        action: str, state: WorkbenchState
    ) -> tuple[WorkbenchState, str, Any]:
        """data-action を状態に適用して再描画する．

        Args:
            action: JS から届いたアクション文字列
            state: 現在の操作状態

        Returns:
            (新しい状態, HTML, CSV ダウンロードの更新)
        """
        verb, _, rest = str(action).partition(":")
        csv_update: Any = gr.skip()

        if verb == "depth":
            state = replace(
                state, depth=as_int(rest, state.depth)
            )
        elif verb == "minprob":
            state = replace(
                state,
                min_prob=as_float(rest, state.min_prob),
            )
        elif verb == "node":
            state = replace(
                state,
                node="" if rest == "root" else rest,
            )
        elif verb == "move":
            child = _child_hash_at(state, as_int(rest, -1))
            if child:
                state = replace(state, node=child)
        elif verb == "setroot":
            # 選択中のノードが既にルート扱いなので再描画のみ
            pass
        elif verb == "csv":
            csv_update = on_export_csv(
                state.node,
                state.depth,
                state.min_prob,
            )
        elif verb in ("redraw", "refresh"):
            pass
        else:
            logger.debug("Unknown action: %s", action)

        return (state, _render(state), csv_update)

    def on_bridge_node(
        state: WorkbenchState,
    ) -> tuple[WorkbenchState, str]:
        """Canvas のクリック / ダブルクリックで選択ノードを移す．

        Args:
            state: 現在の操作状態

        Returns:
            (新しい状態, HTML)
        """
        pending = _pending.pop("node", None)
        if pending:
            state = replace(state, node=str(pending))
        return (state, _render(state))

    with gr.Blocks(
        title="Maou Game Graph Viewer",
    ) as demo:
        state = gr.State(
            WorkbenchState(
                array_type="game-graph",
                node=str(root_hash),
            )
        )
        workbench = gr.HTML(
            _render(
                WorkbenchState(
                    array_type="game-graph",
                    node=str(root_hash),
                )
            ),
            elem_id="viz-workbench-slot",
        )
        csv_file = gr.File(
            label="CSVダウンロード",
            interactive=False,
            elem_id="vz-csv-file",
        )

        # --- ワークベンチのアクションレーン ---
        buffers: dict[str, dict[str, str]] = {
            lane: {} for lane in WORKBENCH_LANES
        }
        lane_bridges = {
            lane: gr.HTML(
                value="",
                elem_id=f"vz-bridge-{lane}",
                elem_classes=["maou-hidden"],
                server_functions=[
                    make_workbench_bridge(buffers[lane])
                ],
                js_on_load=workbench_js_on_load(lane),
            )
            for lane in WORKBENCH_LANES
        }

        def _lane_handler(
            lane_name: str,
        ) -> Callable[[WorkbenchState], tuple[Any, ...]]:
            """レーン名を束縛したハンドラを返す．

            Args:
                lane_name: レーン名

            Returns:
                .change() に渡すハンドラ
            """

            def run(
                current: WorkbenchState,
            ) -> tuple[Any, ...]:
                """控えたアクションを適用して再描画する．"""
                return on_workbench_action(
                    buffers[lane_name].pop("value", ""),
                    current,
                )

            return run

        for lane in WORKBENCH_LANES:
            lane_bridges[lane].change(
                _lane_handler(lane),
                inputs=[state],
                outputs=[state, workbench, csv_file],
            )

        # --- Canvas レンダラー側のブリッジ ---
        # Gradio 6 では JS DOM 操作で Textbox の値を変更しても
        # .input() / .change() が発火しない．gr.HTML の server_functions で
        # JS → Python を直接呼び出し，trigger("change") で描画を回す．
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
        # ビューポートクエリ用ブリッジ(遅延読み込み)
        gr.HTML(
            value="",
            elem_id=ELEM_ID_VIEWPORT_BRIDGE,
            elem_classes=["maou-hidden"],
            server_functions=[handle_viewport],
            js_on_load=JS_ON_LOAD_VIEWPORT,
        )

        select_bridge.change(
            on_bridge_node,
            inputs=[state],
            outputs=[state, workbench],
        )
        expand_bridge.change(
            on_bridge_node,
            inputs=[state],
            outputs=[state, workbench],
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
