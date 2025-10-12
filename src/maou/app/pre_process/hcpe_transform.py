import abc
import contextlib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Dict, Generator, Optional

import numpy as np
from tqdm.auto import tqdm

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.app.pre_process.transform import Transform
from maou.domain.data.array_io import save_preprocessing_array
from maou.domain.data.schema import (
    create_empty_preprocessing_array,
)


class FeatureStore(metaclass=abc.ABCMeta):
    """Abstract interface for storing processed features.

    Defines the contract for storing neural network training features
    in various storage backends.
    """

    @abc.abstractmethod
    def feature_store(self) -> ContextManager[None]:
        pass

    @abc.abstractmethod
    def store_features(
        self,
        *,
        name: str,
        key_columns: list[str],
        structured_array: np.ndarray,
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        pass


class DataSource:
    """Abstract interface for accessing HCPE data sources.

    Provides iteration over batches of HCPE data for processing
    into neural network training features.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        pass


class PreProcess:
    """Processes HCPE data into neural network training features.

    Takes HCPE format game data and transforms it into feature vectors
    and labels suitable for training Shogi AI neural networks.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    intermediate_dict: Dict[
        int, dict
    ]  # id -> {count, winCount, moveLabelCount, features, mask}

    def __init__(
        self,
        *,
        datasource: DataSource,
        feature_store: Optional[FeatureStore] = None,
    ):
        """Initialize pre-processor.

        Args:
            datasource: Source of HCPE data to process
            feature_store: Optional storage backend for processed features
        """
        self.__feature_store = feature_store
        self.__datasource = datasource

    @dataclass(kw_only=True, frozen=True)
    class PreProcessOption:
        output_dir: Optional[Path] = None
        output_filename: str = "transformed"
        max_workers: int

    @staticmethod
    def _process_single_array(
        data: np.ndarray,
    ) -> Dict[int, dict]:
        """Process a chunk of records (optimized: compute feature/mask only for unique boards).

        Returns:
            Dictionary mapping board hash to aggregated data
        """
        from maou.domain.board import shogi

        # 一度だけBoardオブジェクトを作成し，全ての盤面で再利用
        board = shogi.Board()

        # ハッシュ値を計算してソート
        n = len(data)
        hashs = np.empty(n, dtype=np.uint64)
        for i in range(n):
            board.set_hcp(data["hcp"][i])
            hashs[i] = board.hash()

        # ソートしてユニーク盤面を特定
        idx = np.argsort(hashs, kind="mergesort")
        sorted_hash = hashs[idx]

        # ユニーク値とその出現回数を取得
        uniq_hash, inverse_indices = np.unique(
            sorted_hash, return_inverse=True
        )

        # 各盤面のデータを事前に計算（ソート順）
        sorted_move_labels = np.empty(n, dtype=np.int32)
        sorted_wins = np.empty(n, dtype=np.float32)

        for i in range(n):
            orig_idx = idx[i]
            board.set_hcp(data["hcp"][orig_idx])
            sorted_move_labels[i] = Transform.board_move_label(
                data["hcp"][orig_idx],
                data["bestMove16"][orig_idx],
            )
            sorted_wins[i] = Transform.board_game_result(
                data["hcp"][orig_idx],
                data["gameResult"][orig_idx],
            )

        # ユニーク盤面ごとに集計
        result = {}
        start_idx = 0
        for u_idx, hash_val in enumerate(uniq_hash):
            # このハッシュ値に対応する範囲を見つける
            end_idx = start_idx
            while (
                end_idx < n and sorted_hash[end_idx] == hash_val
            ):
                end_idx += 1

            # この範囲のデータを集計
            move_counts = np.bincount(
                sorted_move_labels[start_idx:end_idx],
                minlength=MOVE_LABELS_NUM,
            )
            win_sum = np.sum(sorted_wins[start_idx:end_idx])
            count = end_idx - start_idx

            # 最初の出現位置のHCPを使って特徴量とマスクを計算
            orig_idx = idx[start_idx]
            board.set_hcp(data["hcp"][orig_idx])

            result[int(hash_val)] = {
                "count": count,
                "winCount": win_sum,
                "moveLabelCount": move_counts,
                "features": Transform.board_feature(
                    data["hcp"][orig_idx]
                ),
                "legalMoveMask": Transform.board_legal_move_mask(
                    data["hcp"][orig_idx]
                ),
            }

            start_idx = end_idx

        return result

    def merge_intermediate_data(
        self,
        batch_result: Dict[int, dict],
    ) -> None:
        """中間データをマージする（メモリ効率化版）．

        Args:
            batch_result: バッチ処理結果の辞書
        """
        for hash_id, data in batch_result.items():
            if hash_id in self.intermediate_dict:
                # 既存データを更新
                existing = self.intermediate_dict[hash_id]
                existing["count"] += data["count"]
                existing["winCount"] += data["winCount"]
                existing["moveLabelCount"] += data[
                    "moveLabelCount"
                ]
            else:
                # 新規データを追加（辞書のみで管理，配列は最後に一括作成）
                self.intermediate_dict[hash_id] = {
                    "count": data["count"],
                    "winCount": data["winCount"],
                    "moveLabelCount": data[
                        "moveLabelCount"
                    ].copy(),
                    "features": data["features"],
                    "legalMoveMask": data["legalMoveMask"],
                }

    def aggregate_intermediate_data(self) -> np.ndarray:
        """中間データを集計して最終的な前処理データを作成する（一括生成版）．

        Returns:
            前処理済みデータの構造化配列
        """
        n = len(self.intermediate_dict)
        target_data = create_empty_preprocessing_array(n)

        # 辞書から一括で配列を作成
        for i, (hash_id, data) in enumerate(
            self.intermediate_dict.items()
        ):
            target_data["id"][i] = hash_id
            target_data["features"][i] = data["features"]
            target_data["moveLabel"][i] = (
                data["moveLabelCount"] / data["count"]
            )
            target_data["resultValue"][i] = (
                data["winCount"] / data["count"]
            )
            target_data["legalMoveMask"][i] = data[
                "legalMoveMask"
            ]

        return target_data

    def transform(
        self, option: PreProcessOption
    ) -> Dict[str, str]:
        """機械学習の前処理を行う（メモリ最適化版）．

        Args:
            option: 前処理オプション

        Returns:
            処理結果の辞書
        """
        pre_process_result: Dict[str, str] = {}
        self.intermediate_dict = {}  # 辞書を初期化
        self.logger.info(
            f"前処理対象のデータ数 {len(self.__datasource)}"
        )

        # Determine number of workers
        max_workers = option.max_workers

        self.logger.info(
            f"Using {max_workers} workers for parallel processing"
        )

        with self.__context():
            if max_workers == 1:
                # シングルスレッド処理
                for dataname, data in tqdm(
                    self.__datasource.iter_batches(),
                    desc="PreProcess (single)",
                ):
                    batch_result = self._process_single_array(
                        data
                    )
                    self.merge_intermediate_data(batch_result)

                array = self.aggregate_intermediate_data()
                pre_process_result["aggregated"] = (
                    f"success {len(array)} rows"
                )

                # Store results
                if self.__feature_store is not None:
                    self.__feature_store.store_features(
                        name=option.output_filename,
                        key_columns=["id"],
                        structured_array=array,
                    )

                if option.output_dir is not None:
                    base_name = Path(option.output_filename)
                    save_preprocessing_array(
                        array,
                        option.output_dir / f"{base_name}.npy",
                        bit_pack=False,
                    )
            else:
                # 並列処理
                with ProcessPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    # Submit all jobs
                    future_to_dataname = {
                        executor.submit(
                            self._process_single_array,
                            data,
                        ): dataname
                        for dataname, data in self.__datasource.iter_batches()
                    }
                    for future in tqdm(
                        as_completed(future_to_dataname),
                        desc=f"PreProcess (parallel {max_workers} workers)",
                    ):
                        dataname = future_to_dataname[future]
                        try:
                            batch_result = future.result()
                            self.merge_intermediate_data(
                                batch_result
                            )
                        except Exception as exc:
                            self.logger.error(
                                f"{dataname} processing failed: {exc}"
                            )
                            pre_process_result[dataname] = (
                                f"Dataname {dataname} generated an exception: {exc}"
                            )

                array = self.aggregate_intermediate_data()
                pre_process_result["aggregated"] = (
                    f"success {len(array)} rows"
                )
                # Store results
                if self.__feature_store is not None:
                    self.__feature_store.store_features(
                        name=option.output_filename,
                        key_columns=["id"],
                        structured_array=array,
                    )

                if option.output_dir is not None:
                    base_name = Path(option.output_filename)
                    save_preprocessing_array(
                        array,
                        option.output_dir / f"{base_name}.npy",
                        bit_pack=False,
                    )

        return pre_process_result

    @contextlib.contextmanager
    def __context(self) -> Generator[None, None, None]:
        try:
            if self.__feature_store is not None:
                with self.__feature_store.feature_store():
                    yield
            else:
                yield
        except Exception:
            raise
