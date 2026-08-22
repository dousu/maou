"""棋譜解析 GUI (analyze-gui) の Gradio サーバー (インフラ層)．

画面は 3 カラムのワークベンチ 1 枚 (gr.HTML) で，レイアウトと表示整形は
interface 層 (:mod:`maou.interface.analysis_workbench`) が担う．本モジュールは
gr.Blocks の構築とイベント配線，およびアクション文字列の解釈だけを行う．
セッション状態 (:class:`SessionView` / 分岐木 / クリック状態 / 表示
オプション) は plain data で ``gr.State`` に保持する (ブラウザセッション
ごとに独立)．

UI 操作は ``data-action`` 文字列として JS から届く
(static/analysis_workbench.js)．レーンは 5 本
(nav / engine / engineAll / cancel / download) に分け，探索系は
``concurrency_id="engine"`` + ``concurrency_limit=1`` で直列化し，
キャンセルだけは実行中でも通るよう別レーンに置く
(docs/design/game-analysis/gui.md §11)．
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr

from maou.infra.visualization.game_graph_shared import (
    load_static_file,
)
from maou.interface import analysis_gui, analysis_workbench
from maou.interface.analysis_gui import (
    ClickState,
    EngineSettings,
    InteractiveAnalyzer,
    SessionView,
    VariationTree,
)
from maou.interface.analysis_workbench import (
    AnalysisProgress,
    WorkbenchOptions,
)

logger: logging.Logger = logging.getLogger(__name__)

_CUSTOM_CSS = ".maou-hidden {display: none !important;}"

# 各レーンの js_on_load．server / trigger を JS 側の共有オブジェクトに
# 公開する (game_graph_shared.py と同じ Gradio 6 ブリッジ方式)．
_LANES = ("nav", "engine", "engineAll", "cancel", "download")


def _js_on_load(lane: str) -> str:
    """レーン名に対応する js_on_load スニペットを返す．"""
    return (
        "window.__maou_wb = window.__maou_wb || {};"
        f"window.__maou_wb.{lane} = "
        "{server: server, trigger: trigger};"
    )


def _head_scripts() -> str:
    """ワークベンチの CSS / JS を head に注入する HTML を返す．

    gr.HTML は innerHTML で差し替わるため ``<script>`` が実行されない．
    CSS も head に置いて再描画ごとの重複注入を防ぐ (game_graph_server と
    同じ方針)．
    """
    css = load_static_file("analysis_workbench.css")
    js = load_static_file("analysis_workbench.js")
    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Archivo:wght@400;600;800&family=Noto+Sans+JP:wght@400;500;700"
        '&family=Shippori+Mincho:wght@600;700&display=swap">'
    )
    return f"{fonts}<style>{css}</style><script>{js}</script>"


def _clamp_ply(view: SessionView | None, ply: Any) -> int:
    """アクション値を有効なスナップショット番号に丸める．"""
    if view is None:
        return 0
    try:
        value = int(ply)
    except (TypeError, ValueError):
        return 0
    return max(0, min(value, view.document.n_moves))


def _file_path(file_obj: Any) -> Path | None:
    """gr.File の値 (パス文字列 or file-like) を Path にする．"""
    if file_obj is None:
        return None
    name = getattr(file_obj, "name", file_obj)
    return Path(str(name))


def _make_bridge(
    buffer: dict[str, str],
) -> Callable[[str], bool]:
    """アクション受け渡し用の server_function を作る．

    WARNING: バッファはクロージャとして全ブラウザセッションで共有される
    (game_graph_server.py の ``_pending`` と同じ制約．ローカル解析ツール
    として単一利用者を前提とする)．
    """

    def handle_action(value: str) -> bool:
        """JS から呼ばれる server_function．"""
        if not value:
            return False
        buffer["value"] = str(value)
        return True

    return handle_action


class AnalysisGuiServer:
    """棋譜解析 GUI の Gradio サーバー．

    Attributes:
        num_candidates: 候補手表示数の上限．
        initial_view: CLI 引数から構築した初期セッション (任意)．
        initial_tree: 初期セッションの分岐木 (任意)．
        initial_options: CLI 引数から決まる初期表示オプション．
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
        self._default_time_ms = (
            default_time_ms
            if default_time_ms is not None
            else analysis_gui.DEFAULT_TIME_MS
        )
        self._cancel_event = threading.Event()
        self._render_count = 0
        self.initial_options = WorkbenchOptions(
            top_n=min(5, self.num_candidates),
            budget_mode=(
                "playouts"
                if default_playouts is not None
                else "time"
            ),
            budget_value=(
                default_playouts
                if default_playouts is not None
                else self._default_time_ms
            ),
        )
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
    # 表示
    # ------------------------------------------------------------------

    def _engine_label(self) -> str:
        """ヘッダーに出すエンジン表示名を返す．"""
        if self._analyzer.is_mock:
            return "mock 評価器（開発検証専用）"
        return f"エンジン: {self._analyzer.settings.model_path}"

    def _render(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        options: WorkbenchOptions,
        progress: AnalysisProgress,
    ) -> str:
        """ワークベンチ HTML を組み立てる (再描画ごとに刻印を進める)．"""
        self._render_count += 1
        return analysis_workbench.render_workbench(
            view,
            tree,
            click or ClickState(),
            options,
            progress,
            engine_label=self._engine_label(),
            is_mock=self._analyzer.is_mock,
            max_candidates=self.num_candidates,
            render_stamp=str(self._render_count),
        )

    def _budget(
        self, options: WorkbenchOptions
    ) -> tuple[int | None, int | None]:
        """予算オプションを (time_ms, max_playouts) にする．"""
        value = int(options.budget_value)
        if value <= 0:
            return self._default_time_ms, None
        if options.budget_mode == "playouts":
            return None, value
        return value, None

    # ------------------------------------------------------------------
    # ナビゲーション・入力レーン
    # ------------------------------------------------------------------

    def _apply_option(
        self, options: WorkbenchOptions, rest: str
    ) -> WorkbenchOptions:
        """``opt:{name}:{value}`` を表示オプションに反映する．"""
        name, _, raw = rest.partition(":")
        try:
            if name == "ymode":
                if raw in ("winrate", "eval_cp"):
                    return replace(options, y_mode=raw)
            elif name == "budgetmode":
                if raw in ("time", "playouts"):
                    return replace(options, budget_mode=raw)
            elif name == "budget":
                return replace(
                    options, budget_value=max(1, int(raw))
                )
            elif name == "threshold":
                return replace(
                    options,
                    threshold=min(1.0, max(0.0, float(raw))),
                )
            elif name == "topn":
                return replace(
                    options,
                    top_n=min(
                        self.num_candidates, max(1, int(raw))
                    ),
                )
            elif name == "arrows":
                return replace(options, show_arrows=raw == "1")
            elif name == "pv":
                return replace(options, show_pv=raw == "1")
        except ValueError:
            logger.warning("不正なオプション値です: %s", rest)
        return options

    def _goto(
        self, tree: VariationTree, node_id: int | None
    ) -> None:
        """指定ノードへ移動する (None は現状維持)．"""
        if node_id is not None:
            analysis_gui.goto_node(tree, node_id)

    def _navigate(
        self,
        view: SessionView,
        tree: VariationTree,
        action: str,
        rest: str,
    ) -> None:
        """``nav:*`` / ``goto:*`` / ``blunder:*`` を分岐木に適用する．"""
        node = analysis_gui.current_node(tree)
        if action == "goto":
            self._goto(
                tree, tree.mainline_ids[_clamp_ply(view, rest)]
            )
        elif rest == "first":
            self._goto(tree, tree.mainline_ids[0])
        elif rest == "last":
            self._goto(tree, tree.mainline_ids[-1])
        elif rest == "prev":
            self._goto(tree, node.parent_id)
        elif rest == "next":
            if node.children:
                self._goto(
                    tree,
                    next(
                        (
                            cid
                            for cid in node.children
                            if tree.nodes[cid].is_mainline
                        ),
                        node.children[0],
                    ),
                )
        elif rest == "mainline":
            self._goto(
                tree,
                analysis_gui.mainline_ancestor(tree).node_id,
            )

    def _jump_blunder(
        self,
        view: SessionView,
        tree: VariationTree,
        options: WorkbenchOptions,
        forward: bool,
    ) -> str:
        """前/次の悪手に移動する．見つからなければ理由を返す．"""
        target = analysis_workbench.neighbouring_blunder(
            view,
            options.threshold,
            analysis_gui.mainline_ply(tree),
            forward=forward,
        )
        if target is None:
            return (
                "次の悪手はありません"
                if forward
                else "前の悪手はありません"
            )
        self._goto(tree, tree.mainline_ids[target])
        return ""

    def _advance(
        self, tree: VariationTree, usi: str | None
    ) -> str:
        """1 手進める (分岐の共通経路)．失敗理由を返す．"""
        if not usi:
            return ""
        try:
            analysis_gui.advance_move(tree, usi)
        except ValueError as e:
            return f"指せません: {e}"
        return ""

    def _on_action(
        self,
        action: str,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        options: WorkbenchOptions,
        progress: AnalysisProgress,
    ) -> tuple[Any, ...]:
        """ナビゲーション・入力・表示オプションのアクションを適用する．

        Returns:
            ``(tree, click, options, progress, html)``．
        """
        click = click or ClickState()
        verb, _, rest = action.partition(":")
        error = ""

        if verb == "opt":
            options = self._apply_option(options, rest)
        elif verb == "toggle" and rest == "legal":
            options = replace(
                options, legal_open=not options.legal_open
            )
        elif verb == "clear":
            click = ClickState()
        elif view is not None and tree is not None:
            if verb in ("nav", "goto"):
                self._navigate(view, tree, verb, rest)
                click = ClickState()
            elif verb == "blunder":
                error = self._jump_blunder(
                    view, tree, options, forward=rest == "next"
                )
                click = ClickState()
            elif verb == "board":
                click, error = self._board_click(
                    tree, click, rest
                )
            elif verb == "promote":
                usi = (
                    click.pending_usis[0 if rest == "1" else 1]
                    if len(click.pending_usis) == 2
                    else None
                )
                error = self._advance(tree, usi)
                click = ClickState()
            elif verb == "play":
                error = self._advance(tree, rest)
                click = ClickState()
            elif verb == "cand":
                error = self._play_candidate(
                    tree, rest, options
                )
                click = ClickState()
            elif verb == "pv":
                error = self._play_pv(tree)
                click = ClickState()

        if error:
            progress = replace(
                progress, status=error, is_error=True
            )
        elif progress.is_error:
            progress = replace(
                progress, status="", is_error=False
            )
        return (
            tree,
            click,
            options,
            progress,
            self._render(view, tree, click, options, progress),
        )

    def _board_click(
        self, tree: VariationTree, click: ClickState, value: str
    ) -> tuple[ClickState, str]:
        """盤面クリック 1 回分を状態機械に適用する．"""
        node = analysis_gui.current_node(tree)
        legal = analysis_gui.node_legal_moves(tree)
        new_click, usi = analysis_gui.handle_board_click(
            legal, click, value, node.snapshot.turn
        )
        if usi is None:
            return new_click, ""
        error = self._advance(tree, usi)
        return ClickState(), error

    def _play_candidate(
        self,
        tree: VariationTree,
        rest: str,
        options: WorkbenchOptions,
    ) -> str:
        """候補手 N 番目でその手に分岐する．"""
        node = analysis_gui.current_node(tree)
        try:
            index = int(rest)
        except ValueError:
            return ""
        usi = analysis_gui.candidate_usi(
            node.analysis, index, options.top_n
        )
        if usi is None:
            return "その順位の候補手はありません"
        return self._advance(tree, usi)

    def _play_pv(self, tree: VariationTree) -> str:
        """現在ノードの解析 PV を分岐として一括で進める．"""
        node = analysis_gui.current_node(tree)
        pv = list((node.analysis or {}).get("pv") or [])
        if not pv:
            return (
                "この局面に解析 PV がありません "
                "（先に「この局面を解析」を実行してください）"
            )
        for usi in pv:
            if self._advance(tree, usi):
                logger.warning(
                    "PV の指し手を適用できません: %s", usi
                )
                break
        return ""

    # ------------------------------------------------------------------
    # 解析レーン (エンジン)
    # ------------------------------------------------------------------

    def _on_analyze(
        self,
        action: str,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        options: WorkbenchOptions,
        progress: AnalysisProgress,
    ) -> tuple[Any, ...]:
        """1 局面解析 (キャッシュ再利用 / reanalyze で上書き)．

        Returns:
            ``(tree, progress, html)``．
        """
        force = action == "reanalyze"
        if view is None or tree is None:
            progress = replace(
                progress,
                status="棋譜を読み込んでください",
                is_error=True,
            )
            return (
                tree,
                progress,
                self._render(
                    view, tree, click, options, progress
                ),
            )
        node = analysis_gui.current_node(tree)
        if node.analysis is not None and not force:
            progress = replace(
                progress,
                status=(
                    "この局面は解析済みです"
                    "（「再解析」で上書きできます）"
                ),
                is_error=False,
            )
        else:
            time_ms, playouts = self._budget(options)
            try:
                record = self._analyzer.analyze_position(
                    view.document,
                    tree,
                    node.node_id,
                    time_ms=time_ms,
                    max_playouts=playouts,
                )
            except ValueError as e:
                progress = replace(
                    progress,
                    status=f"解析に失敗しました: {e}",
                    is_error=True,
                )
                return (
                    tree,
                    progress,
                    self._render(
                        view, tree, click, options, progress
                    ),
                )
            node.analysis = record
            progress = replace(
                progress,
                status=(
                    f"解析完了: {record['playouts']} playouts / "
                    f"{record['elapsed_ms']} ms"
                ),
                is_error=False,
            )
        return (
            tree,
            progress,
            self._render(view, tree, click, options, progress),
        )

    def _on_analyze_all(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        options: WorkbenchOptions,
        progress: AnalysisProgress,
    ) -> Any:
        """全局面解析 (ジェネレータ: 進捗を逐次 yield，キャンセル対応)．

        Yields:
            ``(view, progress, html, report_file)``．
        """
        if view is None or tree is None:
            progress = replace(
                progress,
                status="棋譜を読み込んでください",
                is_error=True,
            )
            yield (
                view,
                progress,
                self._render(
                    view, tree, click, options, progress
                ),
                gr.update(),
            )
            return

        self._cancel_event.clear()
        time_ms, playouts = self._budget(options)
        total = view.document.n_moves
        progress = AnalysisProgress(
            running=True,
            done=0,
            total=total,
            status=f"全局面解析を開始します（全 {total} 局面）",
        )
        yield (
            view,
            progress,
            self._render(view, tree, click, options, progress),
            gr.update(),
        )

        positions: list[dict[str, Any]] = []
        for i, count, record in self._analyzer.analyze_mainline(
            view.document,
            time_ms=time_ms,
            max_playouts=playouts,
            cancel=self._cancel_event,
        ):
            positions.append(record)
            tree.nodes[tree.mainline_ids[i]].analysis = record
            progress = replace(
                progress,
                done=i + 1,
                total=count,
                status=f"全局面解析中 … {i + 1}/{count} 局面",
            )
            yield (
                view,
                progress,
                self._render(
                    view, tree, click, options, progress
                ),
                gr.update(),
            )

        if len(positions) < total:
            progress = replace(
                progress,
                running=False,
                status=(
                    f"キャンセルしました（{len(positions)}/{total} "
                    "局面まで解析済み．結果は各局面のキャッシュに"
                    "残っています）"
                ),
            )
            yield (
                view,
                progress,
                self._render(
                    view, tree, click, options, progress
                ),
                gr.update(),
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
        progress = replace(
            progress,
            running=False,
            done=total,
            status=(
                f"全局面解析が完了しました（全 {total} 局面）．"
                "レポート JSON を下の読み込みパネルに置きました．"
            ),
        )
        yield (
            new_view,
            progress,
            self._render(
                new_view, tree, click, options, progress
            ),
            self._write_report_file(report),
        )

    def _on_cancel(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        options: WorkbenchOptions,
        progress: AnalysisProgress,
    ) -> tuple[Any, ...]:
        """全局面解析のキャンセル要求 (実行中の 1 局面は完了を待つ)．"""
        self._cancel_event.set()
        progress = replace(
            progress,
            status=(
                "キャンセルを要求しました"
                "（実行中の局面の完了後に停止します）"
            ),
        )
        return (
            progress,
            self._render(view, tree, click, options, progress),
        )

    def _on_save(
        self,
        view: SessionView | None,
        tree: VariationTree | None,
        click: ClickState | None,
        options: WorkbenchOptions,
        progress: AnalysisProgress,
    ) -> tuple[Any, ...]:
        """現在のレポートを JSON ファイルに書き出す．"""
        if view is None or view.report is None:
            progress = replace(
                progress,
                status=(
                    "保存できるレポートがありません"
                    "（全局面解析を実行してください）"
                ),
                is_error=True,
            )
            return (
                progress,
                self._render(
                    view, tree, click, options, progress
                ),
                gr.update(),
            )
        path = self._write_report_file(view.report)
        progress = replace(
            progress,
            status="レポート JSON を下の読み込みパネルに置きました",
            is_error=False,
        )
        return (
            progress,
            self._render(view, tree, click, options, progress),
            path,
        )

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
        options: WorkbenchOptions,
    ) -> tuple[Any, ...]:
        """棋譜 (+ レポート) を読み込み，全状態を作り直す．

        Returns:
            ``(view, tree, click, progress, html)``．
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
        progress = AnalysisProgress(
            total=view.document.n_moves,
            done=(
                view.document.n_moves
                if view.report is not None
                else 0
            ),
            status=(
                f"読み込みました: {kifu_path.name}"
                f"（{view.document.n_moves} 手"
                + (
                    "，解析レポートあり）"
                    if view.report is not None
                    else "，解析レポートなし）"
                )
            ),
        )
        click = ClickState()
        return (
            view,
            tree,
            click,
            progress,
            self._render(view, tree, click, options, progress),
        )

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------

    def create_demo(self) -> gr.Blocks:
        """gr.Blocks を構築して返す．"""
        view = self.initial_view
        tree = self.initial_tree
        options = self.initial_options
        progress = AnalysisProgress(
            total=view.document.n_moves
            if view is not None
            else 0,
            done=(
                view.document.n_moves
                if view is not None and view.report is not None
                else 0
            ),
        )

        with gr.Blocks(title="maou 棋譜解析") as demo:
            view_state = gr.State(view)
            tree_state = gr.State(tree)
            click_state = gr.State(ClickState())
            options_state = gr.State(options)
            progress_state = gr.State(progress)

            workbench = gr.HTML(
                self._render(
                    view, tree, ClickState(), options, progress
                ),
                elem_id="maou-workbench-slot",
            )

            with gr.Accordion(
                "棋譜/レポートの読み込み",
                open=view is None,
                elem_id="maou-file-panel",
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
                report_download = gr.File(
                    label="解析レポート JSON (ダウンロード)",
                    interactive=False,
                )

            buffers: dict[str, dict[str, str]] = {
                lane: {} for lane in _LANES
            }
            bridges = {
                lane: gr.HTML(
                    value="",
                    elem_id=f"maou-bridge-{lane}",
                    elem_classes=["maou-hidden"],
                    server_functions=[
                        _make_bridge(buffers[lane])
                    ],
                    js_on_load=_js_on_load(lane),
                )
                for lane in _LANES
            }

            ctx = [
                view_state,
                tree_state,
                click_state,
                options_state,
                progress_state,
            ]

            bridges["nav"].change(
                lambda *args: self._on_action(
                    buffers["nav"].pop("value", ""), *args
                ),
                inputs=ctx,
                outputs=[
                    tree_state,
                    click_state,
                    options_state,
                    progress_state,
                    workbench,
                ],
            )
            bridges["engine"].change(
                lambda *args: self._on_analyze(
                    buffers["engine"].pop("value", "analyze"),
                    *args,
                ),
                inputs=ctx,
                outputs=[tree_state, progress_state, workbench],
                concurrency_limit=1,
                concurrency_id="engine",
            )
            bridges["engineAll"].change(
                self._on_analyze_all,
                inputs=ctx,
                outputs=[
                    view_state,
                    progress_state,
                    workbench,
                    report_download,
                ],
                concurrency_limit=1,
                concurrency_id="engine",
            )
            bridges["cancel"].change(
                self._on_cancel,
                inputs=ctx,
                outputs=[progress_state, workbench],
            )
            bridges["download"].change(
                self._on_save,
                inputs=ctx,
                outputs=[
                    progress_state,
                    workbench,
                    report_download,
                ],
            )

            load_btn.click(
                self._on_load,
                inputs=[kifu_file, report_file, options_state],
                outputs=[
                    view_state,
                    tree_state,
                    click_state,
                    progress_state,
                    workbench,
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
        head=_head_scripts(),
        css=_CUSTOM_CSS,
    )
