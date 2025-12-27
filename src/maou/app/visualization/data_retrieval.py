"""データ検索ユースケース（アプリケーション層）．

検索インデックスとデータソースを組み合わせてデータを取得する．
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from maou.infra.file_system.file_data_source import (
    FileDataSource,
)
from maou.infra.visualization.search_index import SearchIndex

logger = logging.getLogger(__name__)


class DataRetriever:
    """データ検索オーケストレータ．

    検索インデックスでレコード位置を特定し，データソースから実データを取得する．
    """

    def __init__(
        self,
        search_index: SearchIndex,
        file_paths: List[Path],
        array_type: str,
    ) -> None:
        """データ取得サービスを初期化．

        Args:
            search_index: 検索インデックス
            file_paths: データファイルパスリスト
            array_type: データ型（hcpe, preprocessing, stage1, stage2）
        """
        self.search_index = search_index
        self.file_paths = file_paths
        self.array_type = array_type

        # FileDataSourceは後で必要に応じて初期化
        self._data_source: Optional[FileDataSource] = None

        logger.info(
            f"DataRetriever initialized: {len(file_paths)} files, "
            f"type={array_type}"
        )

    def _ensure_data_source(self) -> FileDataSource:
        """データソースを遅延初期化．

        Returns:
            FileDataSourceインスタンス
        """
        if self._data_source is None:
            self._data_source = FileDataSource(
                file_paths=self.file_paths,
                array_type=self.array_type,
                cache_mode="mmap",  # 効率的なランダムアクセス
            )
            logger.info(
                "FileDataSource initialized with mmap cache"
            )

        return self._data_source

    def get_by_id(
        self, record_id: str
    ) -> Optional[Dict[str, Any]]:
        """IDでレコードを取得．

        Args:
            record_id: 検索するレコードID

        Returns:
            レコードデータの辞書，見つからない場合はNone
        """
        # インデックスで位置を検索
        location = self.search_index.search_by_id(record_id)

        if location is None:
            logger.info(f"Record not found: {record_id}")
            return None

        file_index, row_number = location

        # データソースから実データを取得
        # 注意: 現在はモック実装のため，実際のファイル読み込みは行わない
        logger.info(
            f"Found record {record_id} at file={file_index}, row={row_number}"
        )

        # モックデータを返す
        return self._create_mock_record(record_id, row_number)

    def get_by_eval_range(
        self,
        min_eval: Optional[int],
        max_eval: Optional[int],
        offset: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """評価値範囲でレコードを取得．

        Args:
            min_eval: 最小評価値
            max_eval: 最大評価値
            offset: スキップする件数
            limit: 取得する最大件数

        Returns:
            レコードデータのリスト
        """
        # インデックスで位置を検索
        locations = self.search_index.search_by_eval_range(
            min_eval, max_eval, offset, limit
        )

        logger.info(
            f"Found {len(locations)} records in range "
            f"[{min_eval}, {max_eval}], offset={offset}, limit={limit}"
        )

        # 各レコードのデータを取得
        records = []
        for file_index, row_number in locations:
            # モックデータを生成
            record_id = f"mock_id_{row_number}"
            record = self._create_mock_record(
                record_id, row_number
            )
            records.append(record)

        return records

    def _create_mock_record(
        self, record_id: str, row_number: int
    ) -> Dict[str, Any]:
        """モックレコードデータを生成．

        Args:
            record_id: レコードID
            row_number: 行番号

        Returns:
            モックレコードデータ
        """
        # 簡単なモック盤面データ
        mock_board = [[0 for _ in range(9)] for _ in range(9)]

        # いくつかの駒を配置（row_numberに応じて変化）
        mock_board[0][4] = 16 + 8  # 後手王
        mock_board[8][4] = 8  # 先手王

        # row_numberに応じて追加の駒
        if row_number % 3 == 0:
            mock_board[0][1] = 16 + 6  # 後手角
            mock_board[8][7] = 6  # 先手角

        if row_number % 2 == 0:
            mock_board[0][7] = 16 + 7  # 後手飛車
            mock_board[8][1] = 7  # 先手飛車

        mock_hand = [0] * 14
        # row_numberに応じて持ち駒
        mock_hand[0] = row_number % 5  # 先手の歩
        mock_hand[7] = (row_number + 1) % 4  # 後手の歩

        return {
            "id": record_id,
            "eval": (row_number % 2000) - 1000,
            "moves": 50 + (row_number % 50),
            "boardIdPositions": mock_board,
            "piecesInHand": mock_hand,
        }
