"""棋譜解析 GUI (analyze-gui) の Gradio サーバー (インフラ層)．

gr.Blocks の構築とイベント配線のみを担い，表示整形は interface 層
(:mod:`maou.interface.analysis_gui`)，セッション状態の構築は app 層に
委譲する．セッション状態 (:class:`SessionView`) は plain data で
``gr.State`` に保持する (ブラウザセッションごとに独立)．
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.graph_objects as go

from maou.interface import analysis_gui
from maou.interface.analysis_gui import SessionView

logger: logging.Logger = logging.getLogger(__name__)

_EMPTY_BOARD_HTML = (
    "<p>棋譜が読み込まれていません．下の「棋譜/レポートの読み込み」"
    "からファイルを読み込んでください．</p>"
)


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
    """

    def __init__(
        self,
        *,
        kifu_path: Path | None = None,
        report_path: Path | None = None,
        num_candidates: int = 5,
    ) -> None:
        """CLI 引数から初期状態を構築する．

        Args:
            kifu_path: 起動時に読み込む棋譜ファイル (CSA / KIF)．
            report_path: analyze-game の JSON レポート
                (kifu_path と併せて指定)．
            num_candidates: 候補手表示数の上限．

        Raises:
            ValueError: report_path のみ指定された場合，または棋譜/
                レポートの読み込みに失敗した場合．
        """
        if report_path is not None and kifu_path is None:
            raise ValueError(
                "--report は --input-path と併せて指定してください"
            )
        self.num_candidates = max(1, num_candidates)
        self.initial_view: SessionView | None = None
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

    # ------------------------------------------------------------------
    # イベントハンドラ
    # ------------------------------------------------------------------

    def _render(
        self,
        view: SessionView | None,
        ply: Any,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[str, go.Figure, list[list[str]], str, str, str]:
        """現在の状態から表示系出力 (盤面/グラフ/候補手/局面情報) を作る．"""
        if view is None:
            return (
                _EMPTY_BOARD_HTML,
                _empty_figure(),
                [],
                "",
                "",
                "",
            )
        ply_int = _clamp_ply(view, ply)
        board = analysis_gui.board_svg(
            view,
            ply_int,
            show_candidates=bool(show_arrows),
            show_pv=bool(show_pv),
            top_n=int(top_n),
        )
        fig = analysis_gui.eval_figure(view, ply_int, y_mode)
        candidates = analysis_gui.candidates_table(
            view, ply_int, int(top_n)
        )
        sfen, position_str, note = analysis_gui.position_info(
            view, ply_int
        )
        return board, fig, candidates, sfen, position_str, note

    def _on_load(
        self,
        kifu_file: Any,
        report_file: Any,
        show_arrows: bool,
        show_pv: bool,
        top_n: Any,
        y_mode: str,
    ) -> tuple[Any, ...]:
        """棋譜 (+ レポート) を読み込み，全出力を更新する．"""
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

        board, fig, candidates, sfen, position_str, note = (
            self._render(
                view, 0, show_arrows, show_pv, top_n, y_mode
            )
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
            gr.update(maximum=view.document.n_moves, value=0),
            analysis_gui.move_table(view),
            analysis_gui.summary_markdown(view),
            status,
            board,
            fig,
            candidates,
            sfen,
            position_str,
            note,
        )

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------

    def create_demo(self) -> gr.Blocks:
        """gr.Blocks を構築して返す．"""
        view = self.initial_view
        initial_max = (
            view.document.n_moves if view is not None else 1
        )
        initial_top_n = min(5, self.num_candidates)
        (
            initial_board,
            initial_fig,
            initial_candidates,
            initial_sfen,
            initial_position,
            initial_note,
        ) = self._render(
            view, 0, True, False, initial_top_n, "winrate"
        )

        with gr.Blocks(title="maou 棋譜解析") as demo:
            state = gr.State(view)
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
                    with gr.Row():
                        btn_first = gr.Button("|◀ 最初")
                        btn_prev = gr.Button("◀ 前")
                        btn_next = gr.Button("次 ▶")
                        btn_last = gr.Button("最後 ▶|")
                    ply_slider = gr.Slider(
                        minimum=0,
                        maximum=initial_max,
                        step=1,
                        value=0,
                        label="局面 (0 = 初期局面)",
                    )
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
                            ],
                            value="winrate",
                            label="縦軸",
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
                            label="現局面の候補手 (勝率は手番視点)",
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

            render_inputs = [
                state,
                ply_slider,
                arrows_cb,
                pv_cb,
                topn_slider,
                y_mode,
            ]
            render_outputs = [
                board_html,
                plot,
                cand_df,
                sfen_box,
                position_box,
                note_md,
            ]

            for component in (
                ply_slider,
                arrows_cb,
                pv_cb,
                topn_slider,
                y_mode,
            ):
                component.change(
                    self._render,
                    inputs=render_inputs,
                    outputs=render_outputs,
                )

            btn_first.click(
                lambda: 0, inputs=None, outputs=ply_slider
            )
            btn_prev.click(
                lambda ply: max(0, int(ply) - 1),
                inputs=ply_slider,
                outputs=ply_slider,
            )
            btn_next.click(
                lambda view, ply: _clamp_ply(
                    view, int(ply) + 1
                ),
                inputs=[state, ply_slider],
                outputs=ply_slider,
            )
            btn_last.click(
                lambda view: (
                    view.document.n_moves
                    if view is not None
                    else 0
                ),
                inputs=state,
                outputs=ply_slider,
            )

            def _on_move_select(
                view: SessionView | None,
                evt: gr.SelectData,
            ) -> int:
                # 行 i = i+1 手目 → その手を指した後の局面へ
                return _clamp_ply(view, evt.index[0] + 1)

            move_df.select(
                _on_move_select,
                inputs=state,
                outputs=ply_slider,
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
                    ply_slider,
                    move_df,
                    summary_md,
                    load_status,
                    *render_outputs,
                ],
            )

        return demo


def launch_analysis_gui_server(
    *,
    kifu_path: Path | None = None,
    report_path: Path | None = None,
    num_candidates: int = 5,
    port: int | None = None,
    share: bool = False,
    server_name: str = "127.0.0.1",
) -> None:
    """棋譜解析 GUI サーバーを起動する (ブロッキング)．

    Args:
        kifu_path: 起動時に読み込む棋譜ファイル．
        report_path: analyze-game の JSON レポート．
        num_candidates: 候補手表示数の上限．
        port: サーバーポート (None で Gradio の自動選択)．
        share: Gradio 公開リンクを作成するか．
        server_name: バインドアドレス．
    """
    server = AnalysisGuiServer(
        kifu_path=kifu_path,
        report_path=report_path,
        num_candidates=num_candidates,
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
    )
