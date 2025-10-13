import abc
import contextlib
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Dict, Generator, Optional

import numpy as np
from tqdm.auto import tqdm

from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.app.pre_process.transform import Transform
from maou.domain.data.array_io import save_preprocessing_array
from maou.domain.data.intermediate_store import (
    IntermediateDataStore,
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
    intermediate_store: Optional[IntermediateDataStore]

    def __init__(
        self,
        *,
        datasource: DataSource,
        feature_store: Optional[FeatureStore] = None,
        intermediate_cache_dir: Optional[Path] = None,
        intermediate_batch_size: int = 1000,
    ):
        """Initialize pre-processor.

        Args:
            datasource: Source of HCPE data to process
            feature_store: Optional storage backend for processed features
            intermediate_cache_dir: Directory for intermediate data cache
            intermediate_batch_size: Batch size for disk writes
        """
        self.__feature_store = feature_store
        self.__datasource = datasource
        self.__intermediate_cache_dir = intermediate_cache_dir
        self.__intermediate_batch_size = intermediate_batch_size
        self.intermediate_store = None

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
        """中間データをマージする（ディスクベース版）．

        Args:
            batch_result: バッチ処理結果の辞書
        """
        if self.intermediate_store is None:
            raise RuntimeError(
                "Intermediate store not initialized"
            )

        # ディスクストアに追加/更新
        self.intermediate_store.add_or_update_batch(
            batch_result
        )

    def aggregate_intermediate_data(self) -> np.ndarray:
        """中間データを集計して最終的な前処理データを作成する（ディスクベース版）．

        Returns:
            前処理済みデータの構造化配列
        """
        if self.intermediate_store is None:
            raise RuntimeError(
                "Intermediate store not initialized"
            )

        # ディスクストアから最終配列を生成
        return self.intermediate_store.finalize_to_array()

    def transform(
        self, option: PreProcessOption
    ) -> Dict[str, str]:
        """機械学習の前処理を行う（ディスクベース版）．

        Args:
            option: 前処理オプション

        Returns:
            処理結果の辞書
        """
        pre_process_result: Dict[str, str] = {}
        self.logger.info(
            f"前処理対象のデータ数 {len(self.__datasource)}"
        )

        # Determine number of workers
        max_workers = option.max_workers

        self.logger.info(
            f"Using {max_workers} workers for parallel processing"
        )

        # 一時ディレクトリを準備
        if self.__intermediate_cache_dir is None:
            temp_dir = tempfile.mkdtemp(
                prefix="maou_preprocess_"
            )
            db_path = Path(temp_dir) / "intermediate.db"
            self.logger.info(
                f"Using temporary directory: {temp_dir}"
            )
        else:
            self.__intermediate_cache_dir.mkdir(
                parents=True, exist_ok=True
            )
            db_path = (
                self.__intermediate_cache_dir
                / "intermediate.db"
            )
            self.logger.info(
                f"Using cache directory: {self.__intermediate_cache_dir}"
            )

        # ディスクベースストアを初期化
        self.intermediate_store = IntermediateDataStore(
            db_path=db_path,
            batch_size=self.__intermediate_batch_size,
        )
        self.logger.info(
            f"Initialized disk-based intermediate store at {db_path}"
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
                        bit_pack=True,
                    )
            else:
                # 並列処理（メモリ効率化版）
                with ProcessPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    # メモリ効率化: 逐次サブミット方式
                    # 全ジョブを一度にサブミットせず、max_workers個ずつ処理
                    futures = {}
                    data_iterator = iter(
                        self.__datasource.iter_batches()
                    )

                    # バッチ数をカウント（プログレスバー用）
                    batch_count = 0

                    # 初回のmax_workers個をサブミット
                    for _ in range(max_workers):
                        try:
                            dataname, data = next(data_iterator)
                            future = executor.submit(
                                self._process_single_array,
                                data,
                            )
                            futures[future] = dataname
                            batch_count += 1
                        except StopIteration:
                            break

                    # プログレスバー初期化（バッチ数ベース）
                    pbar = tqdm(
                        desc=f"PreProcess (parallel {max_workers} workers)",
                    )

                    # ジョブが完了するたびに新しいジョブをサブミット
                    while futures:
                        # 完了したジョブを取得
                        done_futures = []
                        for future in as_completed(futures):
                            dataname = futures[future]
                            try:
                                batch_result = future.result()
                                self.merge_intermediate_data(
                                    batch_result
                                )
                                # 明示的にメモリ解放
                                del batch_result
                            except Exception as exc:
                                self.logger.error(
                                    f"{dataname} processing failed: {exc}"
                                )
                                pre_process_result[dataname] = (
                                    f"Dataname {dataname} generated an exception: {exc}"
                                )

                            done_futures.append(future)
                            pbar.update(1)

                            # 新しいジョブをサブミット
                            try:
                                new_dataname, new_data = next(
                                    data_iterator
                                )
                                new_future = executor.submit(
                                    self._process_single_array,
                                    new_data,
                                )
                                futures[new_future] = (
                                    new_dataname
                                )
                                batch_count += 1
                            except StopIteration:
                                pass

                            # 最初に完了したジョブのみ処理して次へ
                            break

                        # 完了したジョブを削除
                        for future in done_futures:
                            del futures[future]

                    pbar.close()

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
                        bit_pack=True,
                    )

        # クリーンアップ: ディスクストアを閉じて削除
        if self.intermediate_store is not None:
            self.intermediate_store.close()
            self.intermediate_store = None
            self.logger.info("Cleaned up intermediate store")

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
