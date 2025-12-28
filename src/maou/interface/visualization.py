"""可視化インターフェース層（アダプター）．

アプリケーション層とインフラ層を接続し，型変換とバリデーションを行う．
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from maou.app.visualization.data_retrieval import DataRetriever
from maou.app.visualization.record_renderer import (
    RecordRendererFactory,
)
from maou.domain.visualization.board_renderer import (
    SVGBoardRenderer,
)
from maou.domain.visualization.move_label_converter import (
    MoveLabelConverter,
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

        # RecordRendererをファクトリで生成
        self.renderer = RecordRendererFactory.create(
            array_type=array_type,
            board_renderer=SVGBoardRenderer(),
            move_converter=MoveLabelConverter(),
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

        # Rendererに委譲してボード描画と詳細抽出
        board_svg = self.renderer.render_board(record)
        record_details = self.renderer.extract_display_fields(record)

        logger.info(
            f"Successfully retrieved and rendered record: {record_id}"
        )

        return (board_svg, record_details)

    def search_by_eval_range(
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
    ]:
        """評価値範囲で検索し，結果を表示．

        Args:
            min_eval: 最小評価値（Noneで全データ取得）
            max_eval: 最大評価値（Noneで全データ取得）
            page: ページ番号（1始まり）
            page_size: ページサイズ

        Returns:
            (table_data, page_info, first_board_svg, first_details, cached_records)
            cached_recordsはページ内ナビゲーション用のレコードキャッシュ
        """
        # パラメータバリデーション（両方がNoneでない場合のみチェック）
        if (
            min_eval is not None
            and max_eval is not None
            and min_eval > max_eval
        ):
            return (
                [],
                "エラー: 最小評価値 > 最大評価値",
                self._render_empty_message("無効な範囲です"),
                {"error": "Invalid range"},
                [],  # 空のレコードキャッシュ
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
                [],  # 空のレコードキャッシュ
            )

        # Rendererを使ってテーブルデータを作成
        table_data = [
            self.renderer.format_table_row(i + offset + 1, record)
            for i, record in enumerate(records)
        ]

        # 最初のレコードをRendererで描画
        first_record = records[0]
        first_board_svg = self.renderer.render_board(first_record)
        first_record_details = self.renderer.extract_display_fields(
            first_record
        )

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
            records,  # ページ内ナビゲーション用のレコードキャッシュ
        )

    def navigate_within_page(
        self,
        cached_records: List[Dict[str, Any]],
        record_index: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """ページ内レコード間をナビゲートする．

        Gradio Stateにキャッシュされたレコードを使用して，
        ファイルI/Oなしで高速にレコード間を移動する．

        Args:
            cached_records: 現在ページのレコード（Gradio Stateから渡される）
            record_index: ページ内インデックス（0-based）

        Returns:
            (board_svg, record_details)のタプル

        Example:
            >>> interface.navigate_within_page(records, 2)
            ("<svg>...</svg>", {"id": 123, "eval": 456})
        """
        # インデックスバリデーション
        if not (0 <= record_index < len(cached_records)):
            return (
                self._render_empty_message("無効なインデックスです"),
                {},
            )

        # Rendererに委譲して描画
        record = cached_records[record_index]
        board_svg = self.renderer.render_board(record)
        details = self.renderer.extract_display_fields(record)

        logger.debug(
            f"Navigate to record {record_index} in current page"
        )

        return (board_svg, details)

    def get_table_columns(self) -> List[str]:
        """検索結果テーブルのカラム名を取得する．

        array_typeに応じたカラム名をRendererから取得する．

        Returns:
            カラム名のリスト

        Example:
            >>> interface.get_table_columns()
            ["Index", "ID", "Eval", "Moves"]  # HCPE の場合
        """
        return self.renderer.get_table_columns()

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
