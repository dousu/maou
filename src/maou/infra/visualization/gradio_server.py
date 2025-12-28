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
        use_mock_data: bool = False,
    ) -> None:
        """サーバーを初期化．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型（hcpe, preprocessing, stage1, stage2）
            model_path: オプショナルなモデルファイルパス
            use_mock_data: Trueの場合はモックデータを使用
        """
        self.file_paths = file_paths
        self.array_type = array_type
        self.model_path = model_path
        self.use_mock_data = use_mock_data
        self.renderer = SVGBoardRenderer()

        # 評価値検索をサポートするかどうかを判定
        self.supports_eval_search = self._supports_eval_search()

        # SearchIndexを初期化
        self.search_index = SearchIndex.build(
            file_paths=file_paths,
            array_type=array_type,
            use_mock_data=use_mock_data,
            num_mock_records=1000,
        )

        # VisualizationInterfaceを初期化
        self.viz_interface = VisualizationInterface(
            search_index=self.search_index,
            file_paths=file_paths,
            array_type=array_type,
        )

        mode_msg = (
            "MOCK MODE (fake data)"
            if use_mock_data
            else "REAL MODE (actual data)"
        )
        logger.info(
            f"🎯 Visualization server initialized: {mode_msg}, "
            f"{len(file_paths)} files, type={array_type}, "
            f"{self.search_index.total_records()} records indexed"
        )

    def _supports_eval_search(self) -> bool:
        """評価値範囲検索をサポートするデータ型かどうかを判定．

        Returns:
            bool: hcpeの場合はTrue，それ以外はFalse
        """
        return self.array_type == "hcpe"

    def create_demo(self) -> gr.Blocks:
        """Gradio UIデモを作成．

        Returns:
            設定済みのGradio Blocksインスタンス
        """
        with gr.Blocks(
            title="Maou Shogi Data Visualizer"
        ) as demo:
            gr.Markdown("# Maou将棋データ可視化ツール")

            # Mode indicator
            mode_indicator = (
                "🔴 MOCK MODE (表示データは実データではありません)"
                if self.use_mock_data
                else "🟢 REAL MODE"
            )
            gr.Markdown(
                f"**{mode_indicator}** | "
                f"データセット: {len(self.file_paths)}ファイル，型={self.array_type}"
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

                    # 評価値範囲検索（HCPEデータのみ）
                    if self.supports_eval_search:
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
                    else:
                        # 評価値検索非対応の場合はダミーコンポーネント
                        min_eval = gr.Number(visible=False)
                        max_eval = gr.Number(visible=False)
                        eval_search_btn = gr.Button(
                            visible=False
                        )

                    # ページ内レコードナビゲーション（新規）
                    with gr.Group():
                        gr.Markdown(
                            "### レコードナビゲーション"
                        )
                        with gr.Row():
                            prev_record_btn = gr.Button(
                                "← 前のレコード", size="sm"
                            )
                            record_indicator = gr.Markdown(
                                "Record 0 / 0"
                            )
                            next_record_btn = gr.Button(
                                "次のレコード →", size="sm"
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
                        # Rendererから動的にヘッダーを取得
                        table_headers = self.viz_interface.get_table_columns()

                        results_table = gr.Dataframe(
                            headers=table_headers,
                            label="結果一覧",
                            interactive=False,
                        )

            # イベントハンドラとState変数
            current_page = gr.State(value=1)
            # ページ内ナビゲーション用のState
            current_page_records = gr.State(value=[])
            current_record_index = gr.State(value=0)

            # 初回表示時にページ1をロード（全データ型で実行）
            demo.load(
                fn=self._paginate_all_data,
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
                    current_page_records,  # キャッシュ
                    current_record_index,  # インデックス
                    record_indicator,  # インジケーター
                ],
            )

            id_search_btn.click(
                fn=self.viz_interface.search_by_id,
                inputs=[id_input],
                outputs=[board_display, record_details],
            )

            eval_search_btn.click(
                fn=self._search_and_cache,
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
                    current_page_records,  # キャッシュ
                    current_record_index,  # インデックス
                    record_indicator,  # インジケーター
                ],
            )

            # ページネーション（常に_search_and_cacheを使用）
            paginate_fn = (
                self._search_and_cache
                if self.supports_eval_search
                else self._paginate_all_data
            )

            next_btn.click(
                fn=lambda page: page + 1,
                inputs=[current_page],
                outputs=[current_page],
            ).then(
                fn=paginate_fn,
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
                    current_page_records,  # キャッシュ
                    current_record_index,  # インデックス
                    record_indicator,  # インジケーター
                ],
            )

            prev_btn.click(
                fn=lambda page: max(1, page - 1),
                inputs=[current_page],
                outputs=[current_page],
            ).then(
                fn=paginate_fn,
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
                    current_page_records,  # キャッシュ
                    current_record_index,  # インデックス
                    record_indicator,  # インジケーター
                ],
            )

            # ページ内レコードナビゲーション（ページ境界を跨ぐ）
            next_record_btn.click(
                fn=self._navigate_next_record,
                inputs=[
                    current_page,
                    current_record_index,
                    current_page_records,
                    page_size,
                    min_eval,
                    max_eval,
                ],
                outputs=[
                    current_page,
                    current_record_index,
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,
                    record_indicator,
                ],
            )

            prev_record_btn.click(
                fn=self._navigate_prev_record,
                inputs=[
                    current_page,
                    current_record_index,
                    current_page_records,
                    page_size,
                    min_eval,
                    max_eval,
                ],
                outputs=[
                    current_page,
                    current_record_index,
                    results_table,
                    page_info,
                    board_display,
                    record_details,
                    current_page_records,
                    record_indicator,
                ],
            )

        return demo

    def _search_and_cache(
        self,
        min_eval: Optional[int],
        max_eval: Optional[int],
        page: int,
        page_size: int,
    ) -> Tuple[
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
        int,
        str,
    ]:
        """検索を実行し，レコードをキャッシュするラッパー関数．

        ページ内ナビゲーション用にレコードをキャッシュし，
        レコードインジケーターを初期化する．

        Args:
            min_eval: 最小評価値
            max_eval: 最大評価値
            page: ページ番号
            page_size: ページサイズ

        Returns:
            (table_data, page_info, board_svg, details,
             cached_records, record_index, record_indicator)
        """
        (
            table_data,
            page_info,
            board_svg,
            details,
            cached_records,
        ) = self.viz_interface.search_by_eval_range(
            min_eval=min_eval,
            max_eval=max_eval,
            page=page,
            page_size=page_size,
        )

        # レコードインジケーター初期化
        num_records = len(cached_records)
        if num_records > 0:
            record_indicator = f"Record 1 / {num_records}"
        else:
            record_indicator = "Record 0 / 0"

        return (
            table_data,
            page_info,
            board_svg,
            details,
            cached_records,  # キャッシュ
            0,  # record_indexをリセット
            record_indicator,  # インジケーター
        )

    def _paginate_all_data(
        self,
        min_eval: Optional[int],
        max_eval: Optional[int],
        page: int,
        page_size: int,
    ) -> Tuple[
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
        int,
        str,
    ]:
        """全データをページネーション（評価値フィルタなし）．

        stage1, stage2, preprocessingなどの非HCPEデータ用．
        min_eval, max_evalパラメータは無視される（Gradio UIの互換性のため）．

        Args:
            min_eval: 無視される（互換性のため，常にNoneとして扱う）
            max_eval: 無視される（互換性のため，常にNoneとして扱う）
            page: ページ番号（1始まり）
            page_size: ページサイズ

        Returns:
            (table_data, page_info, board_svg, details,
             cached_records, record_index, record_indicator)
        """
        # 評価値パラメータを明示的にNoneにして全データを取得
        # （引数のmin_eval, max_evalは無視）
        return self._search_and_cache(
            min_eval=None,  # 評価値フィルタなし
            max_eval=None,  # 評価値フィルタなし
            page=page,
            page_size=page_size,
        )

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
    ) -> Tuple[
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
    ]:
        """評価値範囲検索のモック実装．

        Args:
            min_eval: 最小評価値
            max_eval: 最大評価値
            page: ページ番号
            page_size: ページサイズ

        Returns:
            (results_table_data, page_info, board_svg, record_details, cached_records)
        """
        logger.info(
            f"Mock eval range search: [{min_eval}, {max_eval}], page={page}"
        )

        # モックテーブルデータ
        mock_results = [
            [i, f"mock_id_{i}", min_eval + i * 10, 50 + i]
            for i in range(page_size)
        ]

        # モックレコードデータ（ナビゲーション用）
        mock_records = []
        for i in range(page_size):
            mock_board = [
                [0 for _ in range(9)] for _ in range(9)
            ]
            mock_hand = [0 for _ in range(14)]

            # 簡易的な盤面（各レコードで少し異なる配置）
            mock_board[0][4] = 16 + 8  # 後手王
            mock_board[8][4] = 8  # 先手王

            # レコードごとに駒配置を変える
            if i % 3 == 0:
                mock_board[0][1] = 16 + 6  # 後手角
                mock_board[8][7] = 6  # 先手角
            elif i % 3 == 1:
                mock_board[0][7] = 16 + 7  # 後手飛車
                mock_board[8][1] = 7  # 先手飛車
            else:
                mock_board[0][1] = 16 + 6  # 後手角
                mock_board[0][7] = 16 + 7  # 後手飛車
                mock_board[8][7] = 6  # 先手角
                mock_board[8][1] = 7  # 先手飛車

            mock_record = {
                "id": f"mock_id_{i}",
                "eval": min_eval + i * 10,
                "moves": 50 + i,
                "boardIdPositions": mock_board,
                "piecesInHand": mock_hand,
            }
            mock_records.append(mock_record)

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
            mock_records,
        )

    def _calculate_total_pages(
        self,
        min_eval: Optional[int],
        max_eval: Optional[int],
        page_size: int,
    ) -> int:
        """総ページ数を計算する．

        Args:
            min_eval: 最小評価値（HCPEのみ）
            max_eval: 最大評価値（HCPEのみ）
            page_size: ページサイズ

        Returns:
            総ページ数
        """
        if self.supports_eval_search:
            # HCPEの場合は評価値範囲でカウント
            total_records = self.search_index.count_eval_range(
                min_eval, max_eval
            )
        else:
            # その他のデータ型は全レコード
            total_records = self.search_index.total_records()

        if total_records == 0:
            return 1

        return (total_records + page_size - 1) // page_size

    def _navigate_next_record(
        self,
        current_page: int,
        current_record_index: int,
        current_page_records: List[Dict[str, Any]],
        page_size: int,
        min_eval: Optional[int],
        max_eval: Optional[int],
    ) -> Tuple[
        int,
        int,
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
        str,
    ]:
        """次のレコードへナビゲート（ページ境界を跨ぐ）．

        Args:
            current_page: 現在のページ番号
            current_record_index: 現在のレコードインデックス
            current_page_records: 現在のページのレコードキャッシュ
            page_size: ページサイズ
            min_eval: 最小評価値（HCPEのみ）
            max_eval: 最大評価値（HCPEのみ）

        Returns:
            (new_page, new_index, table_data, page_info,
             board_svg, details, cached_records, record_indicator)
        """
        num_records = len(current_page_records)
        total_pages = self._calculate_total_pages(
            min_eval, max_eval, page_size
        )

        # ページ内で次のレコードがある場合
        if current_record_index < num_records - 1:
            new_index = current_record_index + 1
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    current_page_records, new_index
                )
            )
            record_indicator = (
                f"Record {new_index + 1} / {num_records}"
            )

            # ページは変わらないので，現在のデータを返す
            table_data = [
                self.viz_interface.renderer.format_table_row(
                    i + (current_page - 1) * page_size + 1,
                    record,
                )
                for i, record in enumerate(current_page_records)
            ]
            page_info_str = (
                f"ページ {current_page} / {total_pages}"
            )

            return (
                current_page,
                new_index,
                table_data,
                page_info_str,
                board_svg,
                details,
                current_page_records,
                record_indicator,
            )

        # ページ境界：次のページへ移動
        next_page = current_page + 1
        if next_page > total_pages:
            # 最後のページなら最初のページに循環
            next_page = 1

        # 新しいページのデータを取得
        paginate_fn = (
            self._search_and_cache
            if self.supports_eval_search
            else self._paginate_all_data
        )

        (
            table_data,
            page_info_str,
            board_svg,
            details,
            cached_records,
            _,  # record_indexは0にリセットされる
            record_indicator,
        ) = paginate_fn(
            min_eval, max_eval, next_page, page_size
        )

        return (
            next_page,
            0,  # 新しいページの最初のレコード
            table_data,
            page_info_str,
            board_svg,
            details,
            cached_records,
            record_indicator,
        )

    def _navigate_prev_record(
        self,
        current_page: int,
        current_record_index: int,
        current_page_records: List[Dict[str, Any]],
        page_size: int,
        min_eval: Optional[int],
        max_eval: Optional[int],
    ) -> Tuple[
        int,
        int,
        List[List[Any]],
        str,
        str,
        Dict[str, Any],
        List[Dict[str, Any]],
        str,
    ]:
        """前のレコードへナビゲート（ページ境界を跨ぐ）．

        Args:
            current_page: 現在のページ番号
            current_record_index: 現在のレコードインデックス
            current_page_records: 現在のページのレコードキャッシュ
            page_size: ページサイズ
            min_eval: 最小評価値（HCPEのみ）
            max_eval: 最大評価値（HCPEのみ）

        Returns:
            (new_page, new_index, table_data, page_info,
             board_svg, details, cached_records, record_indicator)
        """
        num_records = len(current_page_records)
        total_pages = self._calculate_total_pages(
            min_eval, max_eval, page_size
        )

        # ページ内で前のレコードがある場合
        if current_record_index > 0:
            new_index = current_record_index - 1
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    current_page_records, new_index
                )
            )
            record_indicator = (
                f"Record {new_index + 1} / {num_records}"
            )

            # ページは変わらないので，現在のデータを返す
            table_data = [
                self.viz_interface.renderer.format_table_row(
                    i + (current_page - 1) * page_size + 1,
                    record,
                )
                for i, record in enumerate(current_page_records)
            ]
            page_info_str = (
                f"ページ {current_page} / {total_pages}"
            )

            return (
                current_page,
                new_index,
                table_data,
                page_info_str,
                board_svg,
                details,
                current_page_records,
                record_indicator,
            )

        # ページ境界：前のページへ移動
        prev_page = current_page - 1
        if prev_page < 1:
            # 最初のページなら最後のページに循環
            prev_page = total_pages

        # 新しいページのデータを取得
        paginate_fn = (
            self._search_and_cache
            if self.supports_eval_search
            else self._paginate_all_data
        )

        (
            table_data,
            page_info_str,
            board_svg,
            details,
            cached_records,
            _,  # record_indexは最後に設定される
            _,  # record_indicatorは後で更新
        ) = paginate_fn(
            min_eval, max_eval, prev_page, page_size
        )

        # 新しいページの最後のレコードを表示
        new_num_records = len(cached_records)
        if new_num_records > 0:
            new_index = new_num_records - 1
            board_svg, details = (
                self.viz_interface.navigate_within_page(
                    cached_records, new_index
                )
            )
            record_indicator = (
                f"Record {new_index + 1} / {new_num_records}"
            )
        else:
            new_index = 0
            record_indicator = "Record 0 / 0"

        return (
            prev_page,
            new_index,
            table_data,
            page_info_str,
            board_svg,
            details,
            cached_records,
            record_indicator,
        )


def launch_server(
    file_paths: List[Path],
    array_type: str,
    port: int,
    share: bool,
    server_name: str,
    model_path: Optional[Path],
    debug: bool,
    use_mock_data: bool = False,
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
        use_mock_data: Trueの場合はモックデータを使用
    """
    server = GradioVisualizationServer(
        file_paths=file_paths,
        array_type=array_type,
        model_path=model_path,
        use_mock_data=use_mock_data,
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
