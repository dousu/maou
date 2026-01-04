"""検索インデックスのPython実装（インフラ層）．

ファイルからデータを読み込み，ID・評価値による高速検索を提供する．
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SearchIndex:
    """検索インデックスクラス（Python実装）．

    .featherファイルからデータを読み込み，ID・評価値による検索機能を提供する．
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

        # インデックス構造
        self._id_index: Dict[
            str, Tuple[int, int]
        ] = {}  # id -> (file_idx, row_idx)
        self._eval_index: Dict[int, List[Tuple[int, int]]] = (
            defaultdict(list)
        )  # eval -> [(file_idx, row_idx), ...]
        self._total_records = 0

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
        # モックデータを生成
        import random

        for i in range(num_records):
            record_id = f"mock_id_{i}"
            eval_value = (
                random.randint(-3000, 3000)
                if self.array_type == "hcpe"
                else 0
            )
            # 仮想的なfile_index=0，row_number=i
            self._id_index[record_id] = (0, i)
            if self.array_type == "hcpe":
                self._eval_index[eval_value].append((0, i))

        self._total_records = num_records
        logger.info(
            f"✅ Mock index built: {num_records:,} records"
        )

    def search_by_id(
        self, record_id: str
    ) -> Optional[Tuple[int, int]]:
        """IDでレコードを検索．

        Args:
            record_id: 検索するレコードID

        Returns:
            (file_index, row_number)のタプル，またはNone
        """
        return self._id_index.get(record_id)

    def search_by_eval_range(
        self,
        min_eval: Optional[int] = None,
        max_eval: Optional[int] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> List[Tuple[int, int]]:
        """評価値範囲でレコードを検索．

        Args:
            min_eval: 最小評価値（Noneで-∞，または全データ取得）
            max_eval: 最大評価値（Noneで+∞，または全データ取得）
            offset: スキップする件数
            limit: 取得する最大件数

        Returns:
            [(file_index, row_number), ...]のリスト

        Note:
            非HCPEデータの場合，min_eval/max_evalの両方がNoneの場合のみ
            全データのページネーションとして使用可能．
            評価値フィルタを指定するとValueErrorが発生する．
        """
        # 非HCPEデータで評価値フィルタを使おうとした場合はエラー
        if self.array_type != "hcpe" and (
            min_eval is not None or max_eval is not None
        ):
            raise ValueError(
                f"Eval range search only supported for HCPE data, "
                f"got: {self.array_type}"
            )

        # 評価値範囲で検索
        if self.array_type == "hcpe" and (
            min_eval is not None or max_eval is not None
        ):
            # HCPEデータで評価値フィルタあり
            min_v = min_eval if min_eval is not None else -32768
            max_v = max_eval if max_eval is not None else 32767

            # 該当する評価値のレコードを収集
            results = []
            for eval_value in sorted(self._eval_index.keys()):
                if min_v <= eval_value <= max_v:
                    results.extend(self._eval_index[eval_value])

            # offset/limitを適用
            return results[offset : offset + limit]
        else:
            # 非HCPEデータ，またはeval filterなし -> 全データを返す
            all_records = [
                (file_idx, row_idx)
                for file_idx, row_idx in self._id_index.values()
            ]
            return all_records[offset : offset + limit]

    def count_eval_range(
        self,
        min_eval: Optional[int] = None,
        max_eval: Optional[int] = None,
    ) -> int:
        """評価値範囲内のレコード総数をカウント．

        Args:
            min_eval: 最小評価値（Noneで-∞，または全データカウント）
            max_eval: 最大評価値（Noneで+∞，または全データカウント）

        Returns:
            レコード数

        Note:
            非HCPEデータの場合，min_eval/max_evalの両方がNoneの場合のみ
            全データのカウントとして使用可能（ページネーション用）．
            評価値フィルタを指定すると0を返す．
        """
        # 非HCPEデータで評価値フィルタを使った場合は0を返す
        # （min_eval=None, max_eval=Noneの場合は全データカウント）
        if self.array_type != "hcpe" and (
            min_eval is not None or max_eval is not None
        ):
            return 0

        # 評価値範囲でカウント
        if self.array_type == "hcpe" and (
            min_eval is not None or max_eval is not None
        ):
            # HCPEデータで評価値フィルタあり
            min_v = min_eval if min_eval is not None else -32768
            max_v = max_eval if max_eval is not None else 32767

            count = 0
            for eval_value in self._eval_index.keys():
                if min_v <= eval_value <= max_v:
                    count += len(self._eval_index[eval_value])
            return count
        else:
            # 全データカウント
            return self._total_records

    def _build_from_files(self) -> None:
        """実ファイルをスキャンして検索インデックスを構築．

        各.featherファイルからid/evalフィールドを読み取り，
        id → (file_index, row_number)のマッピングを構築する．
        """
        logger.info(
            f"Scanning {len(self.file_paths)} .feather files..."
        )

        try:
            for file_idx, file_path in enumerate(
                self.file_paths
            ):
                # Rustバックエンドで読み込み（Stream/File形式自動判定）
                from maou.domain.data.rust_io import (
                    load_hcpe_df,
                    load_preprocessing_df,
                    load_stage1_df,
                    load_stage2_df,
                )

                loader_map = {
                    "hcpe": load_hcpe_df,
                    "preprocessing": load_preprocessing_df,
                    "stage1": load_stage1_df,
                    "stage2": load_stage2_df,
                }

                load_df = loader_map[self.array_type]
                df = load_df(file_path)

                # idカラムを取得
                if "id" not in df.columns:
                    raise ValueError(
                        f"Missing 'id' column in {file_path}"
                    )

                id_column = df["id"]

                # evalカラムを取得（HCPEの場合のみ）
                eval_column = None
                if self.array_type == "hcpe":
                    if "eval" in df.columns:
                        eval_column = df["eval"]

                # インデックスに追加
                for row_idx in range(len(df)):
                    record_id = str(id_column[row_idx])
                    self._id_index[record_id] = (
                        file_idx,
                        row_idx,
                    )

                    # HCPE の場合は評価値インデックスも構築
                    if eval_column is not None:
                        eval_value = int(eval_column[row_idx])
                        self._eval_index[eval_value].append(
                            (file_idx, row_idx)
                        )

                    self._total_records += 1

            logger.info(
                f"✅ Index built: {self._total_records:,} total records"
            )
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise

    def total_records(self) -> int:
        """総レコード数を取得．

        Returns:
            総レコード数
        """
        return self._total_records

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
