"""棋譜解析 GUI (analyze-gui) の Gradio サーバー (インフラ層)．

gr.Blocks の構築とイベント配線のみを担い，表示整形は interface 層
(:mod:`maou.interface.analysis_gui`)，セッション状態・分岐木・エンジン
呼び出しは app 層に委譲する (interface 経由)．セッション状態
(:class:`SessionView` / 分岐木 / クリック状態) は plain data で
``gr.State`` に保持する (ブラウザセッションごとに独立)．

エンジン (:class:`InteractiveAnalyzer`) はサーバープロセスで 1 個を
共有し，探索系イベントは ``concurrency_id="engine"`` +
``concurrency_limit=1`` で直列化する
(docs/design/game-analysis/gui.md §11)．
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.graph_objects as go

from maou.interface import analysis_gui
from maou.interface.analysis_gui import (
    ClickState,
    EngineSettings,
    InteractiveAnalyzer,
    SessionView,
    VariationTree,
)

logger: logging.Logger = logging.getLogger(__name__)

_EMPTY_BOARD_HTML = (
    "<p>棋譜が読み込まれていません．下の「棋譜/レポートの読み込み」"
    "からファイルを読み込んでください．</p>"
)

# Gradio 6 では JS から Textbox の値を変更しても .input()/.change() が
# 発火しない (game_graph_shared.py 参照)．gr.HTML の server_functions +
# js_on_load で JS → Python を直接呼び出し，trigger("change") で
# .change() コールバックを発火する．
_BOARD_CLICK_JS_ON_LOAD = (
    "window.__maou_board_click = "
    "{server: server, trigger: trigger};"
)

# 盤面 SVG のクリック標的 ([data-click] rect) の委譲リスナー．
# gr.HTML は innerHTML 差し替えのため，永続する外側コンテナ
# (#board-display) に 1 回だけリスナーを付ける．
_HEAD_SCRIPTS = """
<script>
(function() {
    function onBoardClick(e) {
        var target = e.target.closest('[data-click]');
        if (!target) return;
        var bridge = window.__maou_board_click;
        if (!bridge || !bridge.server) {
            console.warn('[maou] board click bridge not ready');
            return;
        }
        bridge.server.handle_board_click(target.getAttribute('data-click'))
            .then(function(ok) { if (ok) bridge.trigger('change'); })
            .catch(function(err) {
                console.error('[maou] board click failed:', err);
            });
    }
    function attach() {
        var container = document.getElementById('board-display');
        if (!container || container._maouClickAttached) return;
        container._maouClickAttached = true;
        container.addEventListener('click', onBoardClick);
    }
    var observer = new MutationObserver(attach);
    function start() {
        observer.observe(document.body, {childList: true, subtree: true});
        attach();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();
</script>
"""

_CUSTOM_CSS = ".maou-hidden {display: none !important;}"


def _empty_figure() -> go.Figure:
    """データ未読込時のプレースホルダ Figure を返す．"""
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=320,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    return fig


def _clamp_ply(view: SessionView | None, ply: Any) -> int:
    """スライダー値を有効なスナップショット番号に丸める．"""
    if view is None:
        return 0
    return max(0, min(int(ply), view.document.n_moves))


def _file_path(file_obj: Any) -> Path | None:
    """gr.File の値 (パス文字列 or file-like) を Path にする．"""
    if file_obj is None:
        return None
    name = getattr(file_obj, "name", file_obj)
    return Path(str(name))


class AnalysisGuiServer:
    """棋譜解析 GUI の Gradio サーバー．

    Attributes:
        num_candidates: 候補手表示数の上限 (スライダー最大値)．
        initial_view: CLI 引数から構築した初期セッション (任意)．
        initial_tree: 初期セッションの分岐木 (任意)．
    """

    def __init__(
        self,
        *,
        kifu_path: Path | None = None,
        report_path: Path | None = None,
        num_candidates: int = 5,
        engine_settings: EngineSettings | None = None,
        default_time_ms: int | None = None,
        default_playouts: int | None = None,
    ) -> None:
        """CLI 引数から初期状態を構築する．

        Args:
            kifu_path: 起動時に読み込む棋譜ファイル (CSA / KIF)．
            report_path: analyze-game の JSON レポート
                (kifu_path と併せて指定)．
            num_candidates: 候補手表示数の上限．
            engine_settings: GUI 内解析のエンジン設定．None なら
                mock 評価器のデフォルト設定 (開発検証専用)．
            default_time_ms: GUI 内解析のデフォルト時間予算 (ミリ秒)．
            default_playouts: GUI 内解析のデフォルト playout 数予算
                (指定時は時間予算より優先して初期選択)．

        Raises:
            ValueError: report_path のみ指定された場合，または棋譜/
                レポートの読み込みに失敗した場合．
        """
        if report_path is not None and kifu_path is None:
            raise ValueError(
                "--report は --input-path と併せて指定してください"
            )
        self.num_candidates = max(1, num_candidates)
        self._analyzer = InteractiveAnalyzer(
            engine_settings
            or EngineSettings(
                num_candidates=self.num_candidates
            )
        )
        self._default_playouts = default_playouts
        self._default_time_ms = (
            default_time_ms
            if default_time_ms is not None
            else analysis_gui.DEFAULT_TIME_MS
        )
        self._cancel_event = threading.Event()
        self.initial_view: SessionView | None = None
        self.initial_tree: VariationTree | None = None
        if kifu_path is not None:
            report_json = (
                report_path.read_text(encoding="utf-8")
                if report_path is not None
                else None
            )
            self.initial_view = analysis_gui.load_session(
                kifu_path.read_bytes(),
                kifu_path.name,
                report_json,
            )
            self.initial_tree = (
                analysis_gui.build_variation_tree(
                    self.initial_view.document,
                    self.initial_view.report,
                )
            )

    # ------------------------------------------------------------------
    # 表示ヘルパ
    # ------------------------------------------------------------------

    def _engine_note(self) -> str:
        """エンジン設定の説明行 (mock 明示) を返す．"""
        settings = self._analyzer.settings
        if self._analyzer.is_mock:
            return (
                "⚠️ **mock 評価器** (開発検証専用．実モデルは "
                "`--model-path` で指定)"
            )
        return f"エンジン: `{settings.model_path}`"

    def _budget(
        self, budget_mode: str, budget_value: Any
    ) -> tuple[int | None, int | None]:
        """予算 UI の値を (time_ms, max_playouts) にする．"""
        try:
            value = int(budget_value)
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            return self._default_time_ms, None
        if budget_mode == "playouts":
            return None, value
        return value, None

    def _render_node(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """現在ノードの表示系出力 13 要素 (node_outputs) を作る．

        戻り値の順序: (tree, click, board, plot, candidates, sfen,
        position, note, breadcrumb, dropdown 更新, click 状態表示,
        成るボタン可視更新, 成らずボタン可視更新)．
        成/不成の可視は gr.Row でなくボタン単位で更新する
        (Gradio 6 では Row への visible 更新が効かない)．
        """
        if view is None or tree is None:
            return (
                tree,
                ClickState(),
                _EMPTY_BOARD_HTML,
                _empty_figure(),
                [],
                "",
                "",
                "",
                "",
                gr.update(choices=[], value=None),
                "",
                gr.update(visible=False),
                gr.update(visible=False),
            )
        click = click or ClickState()
        snapshot = analysis_gui.current_node(tree).snapshot
        legal = analysis_gui.node_legal_moves(tree)
        board = analysis_gui.node_board_svg(
            tree,
            show_candidates=bool(show_arrows),
            show_pv=bool(show_pv),
            top_n=int(top_n),
            click_state=click,
            legal=legal,
            interactive=True,
        )
        fig = analysis_gui.eval_figure(
            view, analysis_gui.mainline_ply(tree), y_mode
        )
        candidates = analysis_gui.node_candidates_table(
            tree, int(top_n)
        )
        sfen, position_str, note = (
            analysis_gui.node_position_info(view, tree)
        )
        breadcrumb = analysis_gui.breadcrumb_markdown(tree)
        dropdown = gr.update(
            choices=analysis_gui.legal_move_choices(
                snapshot, legal
            ),
            value=None,
        )
        click_status = analysis_gui.click_status_text(
            snapshot, click
        )
        promo_visible = bool(click.pending_usis)
        return (
            tree,
            click,
            board,
            fig,
            candidates,
            sfen,
            position_str,
            note,
            breadcrumb,
            dropdown,
            click_status,
            gr.update(visible=promo_visible),
            gr.update(visible=promo_visible),
        )

    def _nav_outputs(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """node_outputs + スライダー更新 (14 要素) を作る．

        現在ノードが本譜上ならスライダー値を追随させ，分岐中は
        変更しない (スライダーは本譜ナビゲーション専用)．
        """
        rendered = self._render_node(
            view,
            tree,
            click,
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )
        slider: Any = gr.update()
        if tree is not None:
            node = analysis_gui.current_node(tree)
            if node.is_mainline:
                slider = gr.update(value=node.snapshot.ply)
        return (*rendered, slider)

    # ------------------------------------------------------------------
    # ナビゲーション系イベントハンドラ
    # ------------------------------------------------------------------

    def _on_slider(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
        ply: Any,
    ) -> tuple[Any, ...]:
        """スライダー操作: 本譜の該当局面へ移動する (選択は解除)．"""
        if view is not None and tree is not None:
            ply_int = _clamp_ply(view, ply)
            analysis_gui.goto_node(
                tree, tree.mainline_ids[ply_int]
            )
        return self._render_node(
            view,
            tree,
            ClickState(),
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _on_display(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """表示オプション変更: 現在ノードを再描画する．"""
        return self._render_node(
            view,
            tree,
            click,
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _goto_and_render(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        node_id: int | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """指定ノードへ移動して nav_outputs を返す (選択は解除)．"""
        if tree is not None and node_id is not None:
            analysis_gui.goto_node(tree, node_id)
        return self._nav_outputs(
            view,
            tree,
            ClickState(),
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _on_first(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """最初へ: root (初期局面) に移動する．"""
        node_id = (
            tree.mainline_ids[0] if tree is not None else None
        )
        return self._goto_and_render(
            view,
            tree,
            node_id,
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _on_last(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """最後へ: 本譜の最終局面に移動する．"""
        node_id = (
            tree.mainline_ids[-1] if tree is not None else None
        )
        return self._goto_and_render(
            view,
            tree,
            node_id,
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _on_prev(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """前へ: 親ノードに移動する (分岐中もそのまま遡れる)．"""
        node_id: int | None = None
        if tree is not None:
            node = analysis_gui.current_node(tree)
            node_id = (
                node.parent_id
                if node.parent_id is not None
                else node.node_id
            )
        return self._goto_and_render(
            view,
            tree,
            node_id,
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _on_next(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """次へ: 子ノードに移動する (本譜の子を優先)．"""
        node_id: int | None = None
        if tree is not None:
            node = analysis_gui.current_node(tree)
            if node.children:
                node_id = next(
                    (
                        cid
                        for cid in node.children
                        if tree.nodes[cid].is_mainline
                    ),
                    node.children[0],
                )
            else:
                node_id = node.node_id
        return self._goto_and_render(
            view,
            tree,
            node_id,
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _on_back_mainline(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """本譜へ戻る: 分岐点の本譜側ノードに移動する．"""
        node_id: int | None = None
        if tree is not None:
            node_id = analysis_gui.mainline_ancestor(
                tree
            ).node_id
        return self._goto_and_render(
            view,
            tree,
            node_id,
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _on_move_select(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
        evt: gr.SelectData,
    ) -> tuple[Any, ...]:
        """棋譜テーブルの行クリック: その手の後の本譜局面へ移動する．"""
        node_id: int | None = None
        if view is not None and tree is not None:
            ply = _clamp_ply(view, evt.index[0] + 1)
            node_id = tree.mainline_ids[ply]
        return self._goto_and_render(
            view,
            tree,
            node_id,
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    # ------------------------------------------------------------------
    # 指し手入力系イベントハンドラ (分岐)
    # ------------------------------------------------------------------

    def _play_usi(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        usi: str | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """指し手 1 手を現在ノードから進める (分岐の共通経路)．"""
        if view is None or tree is None:
            raise gr.Error("棋譜を読み込んでください")
        if usi:
            try:
                analysis_gui.advance_move(tree, usi)
            except ValueError as e:
                raise gr.Error(f"指せません: {e}") from e
        return self._nav_outputs(
            view,
            tree,
            ClickState(),
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _on_play_dropdown(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
        usi: str | None,
    ) -> tuple[Any, ...]:
        """合法手 Dropdown + 「指す」ボタン (フォールバック入力)．"""
        return self._play_usi(
            view, tree, usi, show_arrows, show_pv, top_n, y_mode
        )

    def _on_candidate_select(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
        evt: gr.SelectData,
    ) -> tuple[Any, ...]:
        """候補手テーブルの行クリック: その手で分岐して進める．"""
        usi: str | None = None
        if tree is not None:
            node = analysis_gui.current_node(tree)
            usi = analysis_gui.candidate_usi(
                node.analysis, int(evt.index[0]), int(top_n)
            )
        return self._play_usi(
            view, tree, usi, show_arrows, show_pv, top_n, y_mode
        )

    def _on_pv_play(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """PV 再生: 現在ノードの解析 PV を分岐として一括で進める．"""
        if view is None or tree is None:
            raise gr.Error("棋譜を読み込んでください")
        node = analysis_gui.current_node(tree)
        pv = list((node.analysis or {}).get("pv") or [])
        if not pv:
            raise gr.Error(
                "この局面に解析 PV がありません "
                "(先に「この局面を解析」を実行してください)"
            )
        for usi in pv:
            try:
                analysis_gui.advance_move(tree, usi)
            except ValueError:
                logger.warning(
                    "PV の指し手を適用できません: %s", usi
                )
                break
        return self._nav_outputs(
            view,
            tree,
            ClickState(),
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _apply_board_click(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        value: str,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """盤面クリック 1 回分を状態機械に適用する．"""
        if view is None or tree is None or not value:
            return self._nav_outputs(
                view,
                tree,
                click,
                show_arrows,
                show_pv,
                top_n,
                y_mode,
            )
        click = click or ClickState()
        node = analysis_gui.current_node(tree)
        legal = analysis_gui.node_legal_moves(tree)
        new_click, usi = analysis_gui.handle_board_click(
            legal, click, value, node.snapshot.turn
        )
        if usi is not None:
            try:
                analysis_gui.advance_move(tree, usi)
            except ValueError as e:
                raise gr.Error(f"指せません: {e}") from e
            new_click = ClickState()
        return self._nav_outputs(
            view,
            tree,
            new_click,
            show_arrows,
            show_pv,
            top_n,
            y_mode,
        )

    def _on_promotion_choice(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
        *,
        promote: bool,
    ) -> tuple[Any, ...]:
        """成/不成の確認ボタン: 保留中の指し手を確定する．"""
        click = click or ClickState()
        usi: str | None = None
        if len(click.pending_usis) == 2:
            usi = click.pending_usis[0 if promote else 1]
        return self._play_usi(
            view, tree, usi, show_arrows, show_pv, top_n, y_mode
        )

    # ------------------------------------------------------------------
    # 解析系イベントハンドラ (エンジン)
    # ------------------------------------------------------------------

    def _analyze_current(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
        budget_mode: str,
        budget_value: Any,
        *,
        force: bool,
    ) -> tuple[Any, ...]:
        """現在ノードの 1 局面解析 (キャッシュ再利用 / force で上書き)．"""
        if view is None or tree is None:
            raise gr.Error("棋譜を読み込んでください")
        node = analysis_gui.current_node(tree)
        if node.analysis is not None and not force:
            status = (
                "この局面は解析済みです (キャッシュ表示中．"
                "「再解析」で上書きできます)"
            )
        else:
            time_ms, playouts = self._budget(
                budget_mode, budget_value
            )
            try:
                record = self._analyzer.analyze_position(
                    view.document,
                    tree,
                    node.node_id,
                    time_ms=time_ms,
                    max_playouts=playouts,
                )
            except ValueError as e:
                raise gr.Error(
                    f"解析に失敗しました: {e}"
                ) from e
            node.analysis = record
            status = (
                f"解析完了: {record['playouts']} playouts / "
                f"{record['elapsed_ms']} ms"
            )
            if self._analyzer.is_mock:
                status += " ⚠️ mock 評価器 (開発検証専用)"
        return (
            *self._nav_outputs(
                view,
                tree,
                click,
                show_arrows,
                show_pv,
                top_n,
                y_mode,
            ),
            status,
        )

    def _on_analyze_all(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
        budget_mode: str,
        budget_value: Any,
    ) -> Any:
        """全局面解析 (ジェネレータ: 進捗を逐次 yield，キャンセル対応)．

        出力順: (state, *nav_outputs, move_df, summary_md,
        report_download, analyze_status)．
        """
        if view is None or tree is None:
            raise gr.Error("棋譜を読み込んでください")
        self._cancel_event.clear()
        time_ms, playouts = self._budget(
            budget_mode, budget_value
        )
        n = view.document.n_moves
        noop_nav = tuple(gr.update() for _ in range(14))

        def _progress(
            message: str,
        ) -> tuple[Any, ...]:
            return (
                gr.update(),
                *noop_nav,
                gr.update(),
                gr.update(),
                gr.update(),
                message,
            )

        yield _progress(f"全局面解析を開始します (全 {n} 局面)")
        positions: list[dict[str, Any]] = []
        for i, total, record in self._analyzer.analyze_mainline(
            view.document,
            time_ms=time_ms,
            max_playouts=playouts,
            cancel=self._cancel_event,
        ):
            positions.append(record)
            tree.nodes[tree.mainline_ids[i]].analysis = record
            yield _progress(
                f"全局面解析中 … {i + 1}/{total} 局面"
            )

        if len(positions) < n:
            status = (
                f"キャンセルしました ({len(positions)}/{n} 局面まで"
                "解析済み．結果は各局面のキャッシュに反映されています)"
            )
            yield (
                gr.update(),
                *self._nav_outputs(
                    view,
                    tree,
                    click,
                    show_arrows,
                    show_pv,
                    top_n,
                    y_mode,
                ),
                gr.update(),
                gr.update(),
                gr.update(),
                status,
            )
            return

        report = self._analyzer.build_report(
            view.document,
            positions,
            source_name=view.source_name or "uploaded",
            time_ms=time_ms,
            max_playouts=playouts,
        )
        new_view = replace(view, report=report)
        report_path = self._write_report_file(report)
        status = f"全局面解析が完了しました (全 {n} 局面)"
        if self._analyzer.is_mock:
            status += " ⚠️ mock 評価器 (開発検証専用)"
        yield (
            new_view,
            *self._nav_outputs(
                new_view,
                tree,
                click,
                show_arrows,
                show_pv,
                top_n,
                y_mode,
            ),
            analysis_gui.move_table(new_view),
            analysis_gui.summary_markdown(new_view),
            gr.update(value=report_path, visible=True),
            status,
        )

    def _on_cancel(self) -> str:
        """全局面解析のキャンセル要求 (実行中の 1 局面は完了を待つ)．"""
        self._cancel_event.set()
        return "キャンセルを要求しました (実行中の局面の完了後に停止します)"

    @staticmethod
    def _write_report_file(report: dict[str, Any]) -> str:
        """レポート JSON をダウンロード用の一時ファイルに書き出す．"""
        fd, path = tempfile.mkstemp(
            prefix="maou-analyze-report-", suffix=".json"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return path

    # ------------------------------------------------------------------
    # 読み込み
    # ------------------------------------------------------------------

    def _on_load(
        self,
        kifu_file: Any,
        report_file: Any,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """棋譜 (+ レポート) を読み込み，全出力を更新する．

        出力順: (state, *node_outputs, ply_slider, move_df,
        summary_md, load_status)．
        """
        kifu_path = _file_path(kifu_file)
        if kifu_path is None:
            raise gr.Error(
                "棋譜ファイル (.csa / .kif / .kifu) を指定してください"
            )
        report_path = _file_path(report_file)
        try:
            report_json = (
                report_path.read_text(encoding="utf-8")
                if report_path is not None
                else None
            )
            view = analysis_gui.load_session(
                kifu_path.read_bytes(),
                kifu_path.name,
                report_json,
            )
        except (ValueError, UnicodeDecodeError) as e:
            raise gr.Error(
                f"読み込みに失敗しました: {e}"
            ) from e
        tree = analysis_gui.build_variation_tree(
            view.document, view.report
        )
        status = (
            f"読み込みました: {kifu_path.name} "
            f"({view.document.n_moves} 手"
            + (
                ", 解析レポートあり"
                if view.report is not None
                else ", 解析レポートなし"
            )
            + ")"
        )
        return (
            view,
            *self._render_node(
                view,
                tree,
                ClickState(),
                show_arrows,
                show_pv,
                top_n,
                y_mode,
            ),
            gr.update(maximum=view.document.n_moves, value=0),
            analysis_gui.move_table(view),
            analysis_gui.summary_markdown(view),
            status,
        )

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------

    def create_demo(self) -> gr.Blocks:
        """gr.Blocks を構築して返す．"""
        view = self.initial_view
        tree = self.initial_tree
        initial_max = (
            view.document.n_moves if view is not None else 1
        )
        initial_top_n = min(5, self.num_candidates)
        (
            _tree,
            _click,
            initial_board,
            initial_fig,
            initial_candidates,
            initial_sfen,
            initial_position,
            initial_note,
            initial_breadcrumb,
            _dd,
            initial_click_status,
            _promo1,
            _promo2,
        ) = self._render_node(
            view,
            tree,
            ClickState(),
            True,
            False,
            initial_top_n,
            "winrate",
        )
        initial_choices: list[tuple[str, str]] = []
        if view is not None and tree is not None:
            snapshot = analysis_gui.current_node(tree).snapshot
            initial_choices = analysis_gui.legal_move_choices(
                snapshot, analysis_gui.node_legal_moves(tree)
            )
        initial_budget_mode = (
            "playouts"
            if self._default_playouts is not None
            else "time"
        )
        initial_budget_value = (
            self._default_playouts
            if self._default_playouts is not None
            else self._default_time_ms
        )

        with gr.Blocks(title="maou 棋譜解析") as demo:
            state = gr.State(view)
            tree_state = gr.State(tree)
            click_state = gr.State(ClickState())
            gr.Markdown("# maou 棋譜解析 (analyze-gui)")
            summary_md = gr.Markdown(
                analysis_gui.summary_markdown(view)
                if view is not None
                else "棋譜未読込"
            )
            with gr.Row():
                with gr.Column(scale=5):
                    board_html = gr.HTML(
                        initial_board, elem_id="board-display"
                    )
                    breadcrumb_md = gr.Markdown(
                        initial_breadcrumb
                    )
                    click_status_md = gr.Markdown(
                        initial_click_status
                    )
                    with gr.Row():
                        promote_btn = gr.Button(
                            "成る",
                            variant="primary",
                            visible=False,
                        )
                        nonpromote_btn = gr.Button(
                            "成らず", visible=False
                        )
                    with gr.Row():
                        btn_first = gr.Button("|◀ 最初")
                        btn_prev = gr.Button("◀ 前")
                        btn_next = gr.Button("次 ▶")
                        btn_last = gr.Button("最後 ▶|")
                        back_main_btn = gr.Button("本譜へ戻る")
                    ply_slider = gr.Slider(
                        minimum=0,
                        maximum=initial_max,
                        step=1,
                        value=0,
                        label="局面 (0 = 初期局面，本譜)",
                    )
                    with gr.Row():
                        move_dd = gr.Dropdown(
                            choices=initial_choices,
                            value=None,
                            label=(
                                "指し手 (盤面クリックの代替入力)"
                            ),
                        )
                        play_btn = gr.Button("指す")
                        pv_btn = gr.Button("PV を分岐で再生")
                    with gr.Row():
                        arrows_cb = gr.Checkbox(
                            value=True, label="候補手矢印"
                        )
                        pv_cb = gr.Checkbox(
                            value=False,
                            label="PV 矢印 (最善手)",
                        )
                        topn_slider = gr.Slider(
                            minimum=1,
                            maximum=self.num_candidates,
                            step=1,
                            value=initial_top_n,
                            label="候補手数",
                        )
                    sfen_box = gr.Textbox(
                        value=initial_sfen,
                        label="SFEN",
                        interactive=False,
                    )
                    position_box = gr.Textbox(
                        value=initial_position,
                        label="position 文字列",
                        interactive=False,
                    )
                    note_md = gr.Markdown(initial_note)
                with gr.Column(scale=7):
                    with gr.Tab("グラフ"):
                        y_mode = gr.Radio(
                            choices=[
                                ("勝率 (先手)", "winrate"),
                                ("評価値 (先手)", "eval_cp"),
                                (
                                    "勝率 (後手)",
                                    "winrate_gote",
                                ),
                                (
                                    "評価値 (後手)",
                                    "eval_cp_gote",
                                ),
                            ],
                            value="winrate",
                            label="縦軸 (後手視点は先手の鏡映)",
                        )
                        plot = gr.Plot(initial_fig)
                    with gr.Tab("棋譜"):
                        move_df = gr.Dataframe(
                            headers=list(
                                analysis_gui.MOVE_TABLE_HEADERS
                            ),
                            value=(
                                analysis_gui.move_table(view)
                                if view is not None
                                else []
                            ),
                            interactive=False,
                            label=(
                                "行クリックでその手の局面へ移動"
                            ),
                        )
                    with gr.Tab("候補手"):
                        cand_df = gr.Dataframe(
                            headers=list(
                                analysis_gui.CANDIDATES_TABLE_HEADERS
                            ),
                            value=initial_candidates,
                            interactive=False,
                            label=(
                                "現局面の候補手 (勝率は手番視点)．"
                                "行クリックでその手に分岐"
                            ),
                        )
                    with gr.Accordion(
                        "解析 (エンジン)", open=True
                    ):
                        gr.Markdown(self._engine_note())
                        with gr.Row():
                            budget_mode = gr.Radio(
                                choices=[
                                    ("時間 (ms)", "time"),
                                    ("playouts", "playouts"),
                                ],
                                value=initial_budget_mode,
                                label="予算の種類",
                            )
                            budget_value = gr.Number(
                                value=initial_budget_value,
                                precision=0,
                                label="予算値",
                            )
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "この局面を解析",
                                variant="primary",
                            )
                            reanalyze_btn = gr.Button(
                                "再解析 (上書き)"
                            )
                        with gr.Row():
                            analyze_all_btn = gr.Button(
                                "全局面解析"
                            )
                            cancel_btn = gr.Button("キャンセル")
                        analyze_status = gr.Markdown()
                        report_download = gr.File(
                            label=(
                                "解析レポート JSON (ダウンロード)"
                            ),
                            visible=False,
                            interactive=False,
                        )
            with gr.Accordion(
                "棋譜/レポートの読み込み",
                open=view is None,
            ):
                kifu_file = gr.File(
                    label="棋譜ファイル (.csa / .kif / .kifu)",
                    file_types=[".csa", ".kif", ".kifu"],
                )
                report_file = gr.File(
                    label=(
                        "解析レポート JSON "
                        "(maou analyze-game --output の出力，任意)"
                    ),
                    file_types=[".json"],
                )
                load_btn = gr.Button(
                    "読み込み", variant="primary"
                )
                load_status = gr.Markdown()

            # --- 盤面クリックブリッジ (server_functions) ---
            # WARNING: pending_click はクロージャとして全セッションに
            # 共有される (game_graph_server.py の _pending と同じ制約．
            # ローカル解析ツールとして単一利用者を前提とする)
            pending_click: dict[str, str] = {}

            def handle_board_click(value: str) -> bool:
                """JS から呼ばれる server_function．"""
                if not value:
                    return False
                pending_click["value"] = str(value)
                return True

            click_bridge = gr.HTML(
                value="",
                elem_id="board-click-bridge",
                elem_classes=["maou-hidden"],
                server_functions=[handle_board_click],
                js_on_load=_BOARD_CLICK_JS_ON_LOAD,
            )

            # --- イベント配線 ---

            ctx_inputs = [
                state,
                tree_state,
                click_state,
                arrows_cb,
                pv_cb,
                topn_slider,
                y_mode,
            ]
            node_outputs = [
                tree_state,
                click_state,
                board_html,
                plot,
                cand_df,
                sfen_box,
                position_box,
                note_md,
                breadcrumb_md,
                move_dd,
                click_status_md,
                promote_btn,
                nonpromote_btn,
            ]
            nav_outputs = [*node_outputs, ply_slider]

            ply_slider.release(
                self._on_slider,
                inputs=[*ctx_inputs, ply_slider],
                outputs=node_outputs,
            )
            for component in (
                arrows_cb,
                pv_cb,
                topn_slider,
                y_mode,
            ):
                component.change(
                    self._on_display,
                    inputs=ctx_inputs,
                    outputs=node_outputs,
                )

            btn_first.click(
                self._on_first,
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )
            btn_prev.click(
                self._on_prev,
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )
            btn_next.click(
                self._on_next,
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )
            btn_last.click(
                self._on_last,
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )
            back_main_btn.click(
                self._on_back_mainline,
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )
            move_df.select(
                self._on_move_select,
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )
            cand_df.select(
                self._on_candidate_select,
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )
            play_btn.click(
                self._on_play_dropdown,
                inputs=[*ctx_inputs, move_dd],
                outputs=nav_outputs,
            )
            pv_btn.click(
                self._on_pv_play,
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )
            # server_functions ブリッジ経由の .change は value 更新のみ
            # 反映され，gr.update の prop 更新 (visible / choices) が
            # 適用されない (Gradio 6 実測)．prop 更新を含む再描画を
            # 通常イベントとして .then で連鎖させて反映する
            click_bridge.change(
                lambda *args: self._apply_board_click(
                    args[0],
                    args[1],
                    args[2],
                    pending_click.pop("value", ""),
                    *args[3:],
                ),
                inputs=ctx_inputs,
                outputs=nav_outputs,
            ).then(
                self._on_display,
                inputs=ctx_inputs,
                outputs=node_outputs,
            )
            promote_btn.click(
                lambda *args: self._on_promotion_choice(
                    *args, promote=True
                ),
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )
            nonpromote_btn.click(
                lambda *args: self._on_promotion_choice(
                    *args, promote=False
                ),
                inputs=ctx_inputs,
                outputs=nav_outputs,
            )

            analyze_btn.click(
                lambda *args: self._analyze_current(
                    *args, force=False
                ),
                inputs=[*ctx_inputs, budget_mode, budget_value],
                outputs=[*nav_outputs, analyze_status],
                concurrency_limit=1,
                concurrency_id="engine",
            )
            reanalyze_btn.click(
                lambda *args: self._analyze_current(
                    *args, force=True
                ),
                inputs=[*ctx_inputs, budget_mode, budget_value],
                outputs=[*nav_outputs, analyze_status],
                concurrency_limit=1,
                concurrency_id="engine",
            )
            analyze_all_btn.click(
                self._on_analyze_all,
                inputs=[*ctx_inputs, budget_mode, budget_value],
                outputs=[
                    state,
                    *nav_outputs,
                    move_df,
                    summary_md,
                    report_download,
                    analyze_status,
                ],
                concurrency_limit=1,
                concurrency_id="engine",
            )
            cancel_btn.click(
                self._on_cancel,
                inputs=None,
                outputs=analyze_status,
            )

            load_btn.click(
                self._on_load,
                inputs=[
                    kifu_file,
                    report_file,
                    arrows_cb,
                    pv_cb,
                    topn_slider,
                    y_mode,
                ],
                outputs=[
                    state,
                    *node_outputs,
                    ply_slider,
                    move_df,
                    summary_md,
                    load_status,
                ],
            )

        return demo


def launch_analysis_gui_server(
    *,
    kifu_path: Path | None = None,
    report_path: Path | None = None,
    num_candidates: int = 5,
    engine_settings: EngineSettings | None = None,
    default_time_ms: int | None = None,
    default_playouts: int | None = None,
    port: int | None = None,
    share: bool = False,
    server_name: str = "127.0.0.1",
) -> None:
    """棋譜解析 GUI サーバーを起動する (ブロッキング)．

    Args:
        kifu_path: 起動時に読み込む棋譜ファイル．
        report_path: analyze-game の JSON レポート．
        num_candidates: 候補手表示数の上限．
        engine_settings: GUI 内解析のエンジン設定 (None は mock)．
        default_time_ms: GUI 内解析のデフォルト時間予算 (ミリ秒)．
        default_playouts: GUI 内解析のデフォルト playout 数予算．
        port: サーバーポート (None で Gradio の自動選択)．
        share: Gradio 公開リンクを作成するか．
        server_name: バインドアドレス．
    """
    server = AnalysisGuiServer(
        kifu_path=kifu_path,
        report_path=report_path,
        num_candidates=num_candidates,
        engine_settings=engine_settings,
        default_time_ms=default_time_ms,
        default_playouts=default_playouts,
    )
    demo = server.create_demo()
    logger.info(
        "Launching analysis GUI server (port=%s, share=%s)",
        port,
        share,
    )
    demo.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        head=_HEAD_SCRIPTS,
        css=_CUSTOM_CSS,
    )
