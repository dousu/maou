"""可視化インターフェース層（アダプター）．

アプリケーション層とインフラ層を接続し，型変換とバリデーションを行う．
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from maou.app.visualization.board_display import (
    BoardDisplayService,
)
from maou.app.visualization.data_retrieval import DataRetriever
from maou.domain.visualization.board_renderer import (
    SVGBoardRenderer,
)
from maou.infra.visualization.search_index import SearchIndex

logger = logging.getLogger(__name__)


class VisualizationInterface:
    """可視化インターフェースアダプター．

    Gradio UIからの呼び出しを受け，アプリケーション層を経由してデータを取得・描画する．
    """

    def __init__(
        self,
        search_index: SearchIndex,
        file_paths: List[Path],
        array_type: str,
    ) -> None:
        """可視化インターフェースを初期化．

        Args:
            search_index: 検索インデックス
            file_paths: データファイルパスリスト
            array_type: データ型
        """
        self.search_index = search_index
        self.file_paths = file_paths
        self.array_type = array_type

        # アプリケーション層のサービスを初期化
        self.data_retriever = DataRetriever(
            search_index=search_index,
            file_paths=file_paths,
            array_type=array_type,
        )

        self.board_display = BoardDisplayService(
            renderer=SVGBoardRenderer()
        )

        logger.info("VisualizationInterface initialized")

    def search_by_id(
        self, record_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """IDでレコードを検索し，ボードを描画．

        Args:
            record_id: 検索するレコードID

        Returns:
            (board_svg, record_details)のタプル
        """
        if not record_id or not record_id.strip():
            return (
                self._render_empty_message(
                    "IDを入力してください"
                ),
                {},
            )

        # データ取得
        record = self.data_retriever.get_by_id(
            record_id.strip()
        )

        if record is None:
            return (
                self._render_empty_message(
                    f"ID '{record_id}' のレコードが見つかりません"
                ),
                {
                    "error": "Record not found",
                    "searched_id": record_id,
                },
            )

        # ボード描画
        board_svg = self.board_display.render_from_record(
            record
        )

        # レコード詳細
        record_details = {
            "id": record.get("id"),
            "eval": record.get("eval"),
            "moves": record.get("moves"),
        }

        logger.info(
            f"Successfully retrieved and rendered record: {record_id}"
        )

        return (board_svg, record_details)

    def search_by_eval_range(
        self,
        min_eval: int,
        max_eval: int,
        page: int,
        page_size: int,
    ) -> Tuple[List[List[Any]], str, str, Dict[str, Any]]:
        """評価値範囲で検索し，結果を表示．

        Args:
            min_eval: 最小評価値
            max_eval: 最大評価値
            page: ページ番号（1始まり）
            page_size: ページサイズ

        Returns:
            (results_table_data, page_info, first_board_svg, first_record_details)
        """
        # パラメータバリデーション
        if min_eval > max_eval:
            return (
                [],
                "エラー: 最小評価値 > 最大評価値",
                self._render_empty_message("無効な範囲です"),
                {"error": "Invalid range"},
            )

        offset = (page - 1) * page_size

        # データ取得
        records = self.data_retriever.get_by_eval_range(
            min_eval=min_eval,
            max_eval=max_eval,
            offset=offset,
            limit=page_size,
        )

        if not records:
            return (
                [],
                f"ページ {page} （結果なし）",
                self._render_empty_message(
                    "検索結果がありません"
                ),
                {},
            )

        # テーブルデータ作成
        table_data = [
            [
                i + offset + 1,  # インデックス（1始まり）
                record.get("id", ""),
                record.get("eval", 0),
                record.get("moves", 0),
            ]
            for i, record in enumerate(records)
        ]

        # 最初のレコードのボード描画
        first_record = records[0]
        first_board_svg = self.board_display.render_from_record(
            first_record
        )
        first_record_details = {
            "id": first_record.get("id"),
            "eval": first_record.get("eval"),
            "moves": first_record.get("moves"),
        }

        # ページ情報
        total_matches = self.search_index.count_eval_range(
            min_eval, max_eval
        )
        total_pages = (
            total_matches + page_size - 1
        ) // page_size
        page_info = (
            f"ページ {page} / {total_pages} "
            f"（{total_matches:,} 件中 {len(records)} 件表示）"
        )

        logger.info(
            f"Search by eval range: [{min_eval}, {max_eval}], "
            f"page={page}, found {len(records)} records"
        )

        return (
            table_data,
            page_info,
            first_board_svg,
            first_record_details,
        )

    def get_dataset_stats(self) -> Dict[str, Any]:
        """データセット統計情報を取得．

        Returns:
            統計情報の辞書
        """
        return {
            "total_records": self.search_index.total_records(),
            "array_type": self.array_type,
            "num_files": len(self.file_paths),
        }

    def _render_empty_message(self, message: str) -> str:
        """空メッセージ用のHTML．

        Args:
            message: 表示するメッセージ

        Returns:
            メッセージHTML
        """
        return f"""
        <div style="padding: 40px; text-align: center;
                    border: 2px dashed #ccc; border-radius: 10px;">
            <p style="font-size: 16px; color: #666;">{message}</p>
        </div>
        """
