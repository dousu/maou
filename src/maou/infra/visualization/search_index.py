"""Rust検索インデックスのPythonラッパー（インフラ層）．

Rustで実装された高性能インデックスをPythonから使用するためのラッパークラス．
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class SearchIndex:
    """検索インデックスクラス（Rustバックエンド）．

    現在はPure Python実装．Phase 2完了後にRustバインディングに切り替え予定．
    """

    def __init__(
        self,
        file_paths: List[Path],
        array_type: str,
    ) -> None:
        """検索インデックスを初期化．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型（hcpe, preprocessing, stage1, stage2）

        Raises:
            ValueError: 無効なarray_typeが指定された場合
        """
        # array_typeのバリデーション
        valid_types = {
            "hcpe",
            "preprocessing",
            "stage1",
            "stage2",
        }
        if array_type not in valid_types:
            raise ValueError(
                f"Invalid array_type: {array_type}. "
                f"Must be one of: {valid_types}"
            )

        self.file_paths = file_paths
        self.array_type = array_type
        self._mock_data: dict = {}

        logger.info(
            f"Initializing SearchIndex: {len(file_paths)} files, "
            f"type={array_type}"
        )

    def build_mock(self, num_records: int) -> None:
        """モックデータでインデックスを構築（テスト用）．

        Args:
            num_records: 生成するモックレコード数
        """
        logger.info(
            f"Building mock index with {num_records} records"
        )

        for i in range(num_records):
            record_id = f"mock_id_{i}"
            eval_value = (i % 2000) - 1000  # -1000 ~ 999
            location = (0, i)  # (file_index, row_number)

            self._mock_data[record_id] = {
                "eval": eval_value,
                "location": location,
            }

    def search_by_id(
        self, record_id: str
    ) -> Optional[Tuple[int, int]]:
        """IDでレコードを検索．

        Args:
            record_id: 検索するレコードID

        Returns:
            (file_index, row_number)のタプル，またはNone
        """
        if record_id in self._mock_data:
            return self._mock_data[record_id]["location"]
        return None

    def search_by_eval_range(
        self,
        min_eval: Optional[int] = None,
        max_eval: Optional[int] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> List[Tuple[int, int]]:
        """評価値範囲でレコードを検索．

        Args:
            min_eval: 最小評価値（Noneで-∞）
            max_eval: 最大評価値（Noneで+∞）
            offset: スキップする件数
            limit: 取得する最大件数

        Returns:
            [(file_index, row_number), ...]のリスト
        """
        min_val = min_eval if min_eval is not None else -32768
        max_val = max_eval if max_eval is not None else 32767

        # フィルタリング
        filtered = [
            data["location"]
            for data in self._mock_data.values()
            if min_val <= data["eval"] <= max_val
        ]

        # ページネーション
        return filtered[offset : offset + limit]

    def count_eval_range(
        self,
        min_eval: Optional[int] = None,
        max_eval: Optional[int] = None,
    ) -> int:
        """評価値範囲内のレコード総数をカウント．

        Args:
            min_eval: 最小評価値（Noneで-∞）
            max_eval: 最大評価値（Noneで+∞）

        Returns:
            レコード数
        """
        min_val = min_eval if min_eval is not None else -32768
        max_val = max_eval if max_eval is not None else 32767

        return sum(
            1
            for data in self._mock_data.values()
            if min_val <= data["eval"] <= max_val
        )

    def total_records(self) -> int:
        """総レコード数を取得．

        Returns:
            総レコード数
        """
        return len(self._mock_data)

    @classmethod
    def build(
        cls,
        file_paths: List[Path],
        array_type: str,
        num_mock_records: int = 1000,
    ) -> "SearchIndex":
        """インデックスを構築して返す（ファクトリーメソッド）．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型
            num_mock_records: モックレコード数（テスト用）

        Returns:
            構築済みSearchIndexインスタンス
        """
        index = cls(file_paths, array_type)
        index.build_mock(num_mock_records)
        return index
