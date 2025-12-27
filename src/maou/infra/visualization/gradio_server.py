"""Gradio UIサーバー実装（インフラ層）．

将棋データ可視化のためのGradio Webインターフェースを提供する．
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from maou.domain.visualization.board_renderer import (
    BoardPosition,
    SVGBoardRenderer,
)
from maou.infra.visualization.search_index import SearchIndex
from maou.interface.visualization import VisualizationInterface

logger = logging.getLogger(__name__)


class GradioVisualizationServer:
    """Gradio可視化サーバークラス．

    将棋データの検索と視覚化のためのWebインターフェースを提供する．
    """

    def __init__(
        self,
        file_paths: List[Path],
        array_type: str,
        model_path: Optional[Path] = None,
    ) -> None:
        """サーバーを初期化．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型（hcpe, preprocessing, stage1, stage2）
            model_path: オプショナルなモデルファイルパス
        """
        self.file_paths = file_paths
        self.array_type = array_type
        self.model_path = model_path
        self.renderer = SVGBoardRenderer()

        # SearchIndexを初期化（モックデータ）
        self.search_index = SearchIndex.build(
            file_paths=file_paths,
            array_type=array_type,
            num_mock_records=1000,
        )

        # VisualizationInterfaceを初期化
        self.viz_interface = VisualizationInterface(
            search_index=self.search_index,
            file_paths=file_paths,
            array_type=array_type,
        )

        logger.info(
            f"Initializing visualization server: {len(file_paths)} files, "
            f"type={array_type}, {self.search_index.total_records()} records indexed"
        )

    def create_demo(self) -> gr.Blocks:
        """Gradio UIデモを作成．

        Returns:
            設定済みのGradio Blocksインスタンス
        """
        with gr.Blocks(
            title="Maou Shogi Data Visualizer"
        ) as demo:
            gr.Markdown("# Maou将棋データ可視化ツール")
            gr.Markdown(
                f"**データセット**: {len(self.file_paths)}ファイル，型={self.array_type}"
            )

            with gr.Row():
                # 左パネル: 検索コントロール
                with gr.Column(scale=1):
                    gr.Markdown("## 検索機能")

                    # ID検索
                    with gr.Group():
                        gr.Markdown("### ID検索")
                        id_input = gr.Textbox(
                            label="レコードID",
                            placeholder="IDを入力...",
                        )
                        id_search_btn = gr.Button(
                            "ID検索", variant="primary"
                        )

                    # 評価値範囲検索
                    with gr.Group():
                        gr.Markdown("### 評価値範囲検索")
                        min_eval = gr.Number(
                            label="最小評価値",
                            value=-1000,
                            precision=0,
                        )
                        max_eval = gr.Number(
                            label="最大評価値",
                            value=1000,
                            precision=0,
                        )
                        eval_search_btn = gr.Button(
                            "範囲検索", variant="secondary"
                        )

                    # ページネーション
                    with gr.Group():
                        gr.Markdown("### ページネーション")
                        page_size = gr.Slider(
                            label="1ページあたりの件数",
                            minimum=10,
                            maximum=100,
                            value=20,
                            step=10,
                        )
                        with gr.Row():
                            prev_btn = gr.Button("← 前へ")
                            next_btn = gr.Button("次へ →")
                        page_info = gr.Markdown("ページ 1")

                    # データセット情報
                    with gr.Group():
                        gr.Markdown("### データセット情報")
                        gr.JSON(
                            value=self.viz_interface.get_dataset_stats(),
                            label="統計情報",
                        )

                # 右パネル: 視覚化
                with gr.Column(scale=2):
                    gr.Markdown("## 盤面表示")

                    # ボード表示（SVG）
                    board_display = gr.HTML(
                        value=self._get_default_board_svg(),
                        label="盤面",
                    )

                    # レコード詳細
                    with gr.Accordion(
                        "レコード詳細", open=True
                    ):
                        record_details = gr.JSON(
                            label="全フィールド",
                        )

                    # 検索結果テーブル
                    with gr.Accordion("検索結果", open=False):
                        results_table = gr.Dataframe(
                            headers=[
                                "インデックス",
                                "ID",
                                "評価値",
                                "手数",
                            ],
                            label="結果一覧",
                            interactive=False,
                        )

            # イベントハンドラ
            current_page = gr.State(value=1)

            id_search_btn.click(
                fn=self.viz_interface.search_by_id,
                inputs=[id_input],
                outputs=[board_display, record_details],
            )

            eval_search_btn.click(
                fn=self.viz_interface.search_by_eval_range,
                inputs=[
                    min_eval,
                    max_eval,
                    current_page,
                    page_size,
                ],
                outputs=[
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                ],
            )

            # ページネーション
            next_btn.click(
                fn=lambda page: page + 1,
                inputs=[current_page],
                outputs=[current_page],
            ).then(
                fn=self.viz_interface.search_by_eval_range,
                inputs=[
                    min_eval,
                    max_eval,
                    current_page,
                    page_size,
                ],
                outputs=[
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                ],
            )

            prev_btn.click(
                fn=lambda page: max(1, page - 1),
                inputs=[current_page],
                outputs=[current_page],
            ).then(
                fn=self.viz_interface.search_by_eval_range,
                inputs=[
                    min_eval,
                    max_eval,
                    current_page,
                    page_size,
                ],
                outputs=[
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                ],
            )

        return demo

    def _get_default_board_svg(self) -> str:
        """デフォルトの盤面SVGを生成（平手初期配置）．"""
        # 平手初期配置のモック
        # 実際の実装では，標準的な初期配置を設定
        mock_board = [[0 for _ in range(9)] for _ in range(9)]
        mock_hand = [0 for _ in range(14)]

        # 簡易的な初期配置（いくつかの駒を配置）
        # 後手（白）の飛車と角
        mock_board[0][1] = 16 + 6  # 後手角（22）
        mock_board[0][7] = 16 + 7  # 後手飛車（23）
        mock_board[0][4] = 16 + 8  # 後手王（24）

        # 先手（黒）の飛車と角
        mock_board[8][7] = 6  # 先手角
        mock_board[8][1] = 7  # 先手飛車
        mock_board[8][4] = 8  # 先手王

        position = BoardPosition(
            board_id_positions=mock_board,
            pieces_in_hand=mock_hand,
        )

        return self.renderer.render(position)

    def _get_mock_stats(self) -> Dict[str, Any]:
        """インデックス統計情報を返す．"""
        return {
            "total_records": self.search_index.total_records(),
            "array_type": self.array_type,
            "num_files": len(self.file_paths),
        }

    def _search_by_id_mock(
        self, record_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """ID検索のモック実装．

        Args:
            record_id: 検索するレコードID

        Returns:
            (board_svg, record_details)のタプル
        """
        logger.info(f"Mock ID search: {record_id}")

        # モックレスポンス
        board_svg = self._get_default_board_svg()
        record_details = {
            "message": "ID検索機能は実装中です",
            "searched_id": record_id,
            "status": "mock",
        }

        return (board_svg, record_details)

    def _search_by_eval_range_mock(
        self,
        min_eval: int,
        max_eval: int,
        page: int,
        page_size: int,
    ) -> Tuple[List[List[Any]], str, str, Dict[str, Any]]:
        """評価値範囲検索のモック実装．

        Args:
            min_eval: 最小評価値
            max_eval: 最大評価値
            page: ページ番号
            page_size: ページサイズ

        Returns:
            (results_table_data, page_info, board_svg, record_details)
        """
        logger.info(
            f"Mock eval range search: [{min_eval}, {max_eval}], page={page}"
        )

        # モックテーブルデータ
        mock_results = [
            [i, f"mock_id_{i}", min_eval + i * 10, 50 + i]
            for i in range(page_size)
        ]

        page_info = f"ページ {page} （モックデータ）"
        board_svg = self._get_default_board_svg()
        record_details = {
            "message": "範囲検索機能は実装中です",
            "min_eval": min_eval,
            "max_eval": max_eval,
            "status": "mock",
        }

        return (
            mock_results,
            page_info,
            board_svg,
            record_details,
        )


def launch_server(
    file_paths: List[Path],
    array_type: str,
    port: int,
    share: bool,
    server_name: str,
    model_path: Optional[Path],
    debug: bool,
) -> None:
    """Gradioサーバーを起動．

    Args:
        file_paths: データファイルのパスリスト
        array_type: データ型
        port: サーバーポート
        share: 公開リンク作成フラグ
        server_name: サーバーバインドアドレス
        model_path: モデルファイルパス
        debug: デバッグモード
    """
    server = GradioVisualizationServer(
        file_paths=file_paths,
        array_type=array_type,
        model_path=model_path,
    )

    demo = server.create_demo()

    logger.info(
        f"Launching Gradio server on {server_name}:{port} "
        f"(share={share}, debug={debug})"
    )

    demo.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        debug=debug,
        show_error=True,
    )
