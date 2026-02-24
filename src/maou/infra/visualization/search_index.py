"""検索インデックスのPython実装（インフラ層）．

ファイルからデータを読み込み，ID・評価値による高速検索を提供する．
"""

import bisect
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
        # モックモード用: 位置から評価値への逆引きマッピング
        self._position_to_eval: Dict[Tuple[int, int], int] = {}
        self._total_records = 0

        # 事前計算済みソート済みデータ (_finalize_index で初期化)
        self._sorted_eval_keys: List[int] = []
        self._eval_cumulative_counts: List[int] = []
        self._sorted_ids: List[str] = []

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

        データ型に応じて異なるインデックスを生成する:
        - HCPE: eval -3000〜3000のランダム値
        - Stage1/Stage2: eval不要，全て0に固定（IDインデックスのみ使用）
        - Preprocessing: resultValueをevalとして扱う（-100〜100にスケール）

        Args:
            num_records: 生成するモックレコード数
        """
        logger.info(
            f"Building mock index with {num_records} records "
            f"(type={self.array_type})"
        )
        # モックデータを生成
        import random

        for i in range(num_records):
            record_id = f"mock_id_{i}"
            # 仮想的なfile_index=0，row_number=i
            position = (0, i)
            self._id_index[record_id] = position

            # データ型に応じたeval値を設定
            if self.array_type == "hcpe":
                # HCPE: -3000〜3000のランダム値
                eval_value = random.randint(-3000, 3000)
                self._eval_index[eval_value].append(position)
                self._position_to_eval[position] = eval_value
            elif self.array_type == "preprocessing":
                # Preprocessing: resultValueをevalとして扱う（-100〜100）
                # row_numberに応じて-100〜100の範囲で変化
                eval_value = (i % 201) - 100  # -100〜100
                self._eval_index[eval_value].append(position)
                self._position_to_eval[position] = eval_value
            # Stage1/Stage2: eval不要（IDインデックスのみ使用）

        self._total_records = num_records
        self._finalize_index()
        logger.info(
            f"✅ Mock index built: {num_records:,} records "
            f"(type={self.array_type})"
        )

    def _finalize_index(self) -> None:
        """インデックス構築後のソートと累積カウント計算．

        eval値のソート済みキーリスト・累積カウント，
        およびIDのソート済みリストを事前計算する．
        """
        self._sorted_eval_keys = sorted(self._eval_index.keys())

        cumulative = 0
        self._eval_cumulative_counts = []
        for key in self._sorted_eval_keys:
            cumulative += len(self._eval_index[key])
            self._eval_cumulative_counts.append(cumulative)

        self._sorted_ids = sorted(self._id_index.keys())

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

    def get_eval_by_position(
        self, file_index: int, row_number: int
    ) -> Optional[int]:
        """位置から評価値を取得（モックモード用）．

        Args:
            file_index: ファイルインデックス
            row_number: 行番号

        Returns:
            インデックス時に使用した評価値，またはNone
        """
        return self._position_to_eval.get(
            (file_index, row_number)
        )

    def search_id_prefix(
        self, prefix: str, limit: int = 50
    ) -> List[str]:
        """IDプレフィックスで候補を検索．

        Args:
            prefix: 検索プレフィックス（2文字以上推奨）
            limit: 最大取得件数（デフォルト: 50）

        Returns:
            マッチするIDのリスト（ソート済み）

        Note:
            パフォーマンスのため，prefixが2文字未満の場合は空リストを返す．
        """
        if not prefix or len(prefix) < 2:
            return []

        # bisectでプレフィックスの開始位置を特定: O(log N)
        left = bisect.bisect_left(self._sorted_ids, prefix)

        # プレフィックスに一致するIDを limit 件まで収集
        results: List[str] = []
        for i in range(left, len(self._sorted_ids)):
            if self._sorted_ids[i].startswith(prefix):
                results.append(self._sorted_ids[i])
                if len(results) >= limit:
                    break
            else:
                break  # ソート済みなのでプレフィックス不一致で終了

        return results

    def get_all_ids(
        self, limit: Optional[int] = None
    ) -> List[str]:
        """全IDリストを取得（ソート済み）．

        Args:
            limit: 最大取得件数（Noneで全件）

        Returns:
            IDリスト（ソート済み）
        """
        all_ids = sorted(self._id_index.keys())
        if limit is not None:
            return all_ids[:limit]
        return all_ids

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

            # bisectで範囲内のキーインデックスを特定: O(log K)
            left = bisect.bisect_left(
                self._sorted_eval_keys, min_v
            )
            right = bisect.bisect_right(
                self._sorted_eval_keys, max_v
            )

            # 範囲内のキーに対して結果を収集 (offsetを考慮して必要分のみ)
            results: List[Tuple[int, int]] = []
            skipped = 0
            for i in range(left, right):
                eval_value = self._sorted_eval_keys[i]
                records = self._eval_index[eval_value]
                if skipped + len(records) <= offset:
                    skipped += len(records)
                    continue
                start = max(0, offset - skipped)
                remaining = limit - len(results)
                end = min(len(records), start + remaining)
                results.extend(records[start:end])
                skipped += len(records)
                if len(results) >= limit:
                    break

            return results[:limit]
        else:
            # 非HCPEデータ，またはeval filterなし -> 全データを返す
            all_records = list(self._id_index.values())
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
            if not self._sorted_eval_keys:
                return 0

            min_v = min_eval if min_eval is not None else -32768
            max_v = max_eval if max_eval is not None else 32767

            left = bisect.bisect_left(
                self._sorted_eval_keys, min_v
            )
            right = bisect.bisect_right(
                self._sorted_eval_keys, max_v
            )

            # 累積カウントから O(1) で計算
            count_right = (
                self._eval_cumulative_counts[right - 1]
                if right > 0
                else 0
            )
            count_left = (
                self._eval_cumulative_counts[left - 1]
                if left > 0
                else 0
            )
            return count_right - count_left
        else:
            # 全データカウント
            return self._total_records

    def _build_from_files(
        self,
        progress_callback: Optional[
            Callable[[int, int, str], None]
        ] = None,
    ) -> None:
        """実ファイルをスキャンして検索インデックスを構築．

        各.featherファイルからid/evalフィールドを読み取り，
        id → (file_index, row_number)のマッピングを構築する．

        Args:
            progress_callback: 進捗を通知するコールバック関数．
                (files_done, total_records, message)の引数で呼ばれる．
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

                # IDインデックスの一括構築（ベクトル化）
                import polars as pl

                id_list = id_column.cast(pl.Utf8).to_list()
                self._id_index.update(
                    {
                        record_id: (file_idx, row_idx)
                        for row_idx, record_id in enumerate(
                            id_list
                        )
                    }
                )

                # evalインデックスの一括構築
                if eval_column is not None:
                    eval_list = eval_column.to_list()
                    for row_idx, eval_value in enumerate(
                        eval_list
                    ):
                        self._eval_index[
                            int(eval_value)
                        ].append((file_idx, row_idx))

                self._total_records += len(df)

                # 進捗をコールバックで通知
                if progress_callback is not None:
                    message = f"Scanned {file_idx + 1}/{len(self.file_paths)} files"
                    progress_callback(
                        file_idx + 1,
                        self._total_records,
                        message,
                    )

            self._finalize_index()
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
        progress_callback: Optional[
            Callable[[int, int, str], None]
        ] = None,
    ) -> "SearchIndex":
        """インデックスを構築して返す（ファクトリーメソッド）．

        Args:
            file_paths: データファイルのパスリスト
            array_type: データ型
            use_mock_data: Trueの場合はモックデータを使用
            num_mock_records: モックレコード数（テスト用）
            progress_callback: 進捗を通知するコールバック関数．
                (files_done, total_records, message)の引数で呼ばれる．

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
            index._build_from_files(progress_callback)
            logger.info(
                f"✅ Built real index: {index.total_records():,} records"
            )

        return index
