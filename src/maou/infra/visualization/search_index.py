"""Rust検索インデックスのPythonラッパー（インフラ層）．

Rustで実装された高性能インデックスをPythonから使用するためのラッパークラス．
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from maou._rust.maou_io import SearchIndex as RustSearchIndex

logger = logging.getLogger(__name__)


class SearchIndex:
    """検索インデックスクラス（Rustバックエンド）．

    Rust実装の高性能インデックスをPythonから利用するためのラッパークラス．
    """

    def __init__(
        self,
        file_paths: List[Path],
        array_type: str,
        use_mock_data: bool = False,
    ) -> None:
        """検索インデックスを初期化．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型（hcpe, preprocessing, stage1, stage2）
            use_mock_data: Trueの場合はモックデータを使用

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
        self.use_mock_data = use_mock_data

        # Rustインデックスを初期化
        file_paths_str = [str(p) for p in file_paths]
        self._rust_index = RustSearchIndex(
            file_paths_str, array_type
        )

        if use_mock_data:
            logger.warning(
                "⚠️  MOCK MODE: Using generated mock data instead of reading files"
            )
        else:
            logger.info(
                "✅ REAL MODE: Will load actual data from .feather files"
            )

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
        self._rust_index.build_mock(num_records)

    def search_by_id(
        self, record_id: str
    ) -> Optional[Tuple[int, int]]:
        """IDでレコードを検索．

        Args:
            record_id: 検索するレコードID

        Returns:
            (file_index, row_number)のタプル，またはNone
        """
        result = self._rust_index.search_by_id(record_id)
        if result is None:
            return None
        # Rustから返される(u32, u32)をPythonの(int, int)に変換
        return (int(result[0]), int(result[1]))

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
        if self.array_type != "hcpe" and (
            min_eval is not None or max_eval is not None
        ):
            raise ValueError(
                f"Eval range search only supported for HCPE data, "
                f"got: {self.array_type}"
            )

        # Rustの型に合わせてi16に変換（必要に応じて）
        min_eval_i16 = (
            None if min_eval is None else int(min_eval)
        )
        max_eval_i16 = (
            None if max_eval is None else int(max_eval)
        )

        results = self._rust_index.search_by_eval_range(
            min_eval_i16, max_eval_i16, offset, limit
        )

        # Rustから返される[(u32, u32), ...]をPythonの[(int, int), ...]に変換
        return [
            (int(file_idx), int(row_num))
            for file_idx, row_num in results
        ]

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
        if self.array_type != "hcpe" and (
            min_eval is not None or max_eval is not None
        ):
            return 0

        # Rustの型に合わせてi16に変換
        min_eval_i16 = (
            None if min_eval is None else int(min_eval)
        )
        max_eval_i16 = (
            None if max_eval is None else int(max_eval)
        )

        return self._rust_index.count_eval_range(
            min_eval_i16, max_eval_i16
        )

    def _build_from_files(self) -> None:
        """実ファイルをスキャンして検索インデックスを構築．

        各.featherファイルからid/evalフィールドを読み取り，
        id → (file_index, row_number)のマッピングを構築する．
        """
        logger.info(
            f"Scanning {len(self.file_paths)} .feather files..."
        )

        try:
            self._rust_index.build_from_files()
            logger.info(
                f"✅ Index built: {self.total_records():,} total records"
            )
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise

    def total_records(self) -> int:
        """総レコード数を取得．

        Returns:
            総レコード数
        """
        return self._rust_index.total_records()

    @classmethod
    def build(
        cls,
        file_paths: List[Path],
        array_type: str,
        use_mock_data: bool = False,
        num_mock_records: int = 1000,
    ) -> "SearchIndex":
        """インデックスを構築して返す（ファクトリーメソッド）．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型
            use_mock_data: Trueの場合はモックデータを使用
            num_mock_records: モックレコード数（テスト用）

        Returns:
            構築済みSearchIndexインスタンス
        """
        index = cls(
            file_paths, array_type, use_mock_data=use_mock_data
        )

        if use_mock_data:
            index.build_mock(num_mock_records)
            logger.warning(
                f"⚠️  Built mock index: {index.total_records()} fake records"
            )
        else:
            index._build_from_files()
            logger.info(
                f"✅ Built real index: {index.total_records():,} records"
            )

        return index
