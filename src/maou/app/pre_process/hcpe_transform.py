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
    create_empty_intermediate_array,
    create_empty_preprocessing_array,
    get_intermediate_dtype,
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
    intermediate_data: np.ndarray

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
    ) -> tuple[np.ndarray, np.ndarray, list, list, list]:
        """Process a chunk of records in parallel."""

        # dataの中にあるhcpからhashをだしてargsortする
        hashs = np.array(
            [Transform.board_hash(hcp) for hcp in data["hcp"]]
        )
        idx = np.argsort(
            hashs,
            kind="mergesort",
        )
        sorted_hash = hashs[idx]
        uniq, counts = np.unique(
            sorted_hash, return_counts=True
        )
        sorted_move_labels = np.array(
            [
                Transform.board_move_label(hcp, move)
                for hcp, move in zip(
                    data["hcp"], data["bestMove16"]
                )
            ]
        )[idx]
        sorted_wins = np.array(
            [
                Transform.board_game_result(hcp, game_result)
                for hcp, game_result in zip(
                    data["hcp"], data["gameResult"]
                )
            ]
        )[idx]
        sorted_hcps = data["hcp"][idx]
        moves = []
        wins = []
        hcps = []
        i = 0
        for c in counts:
            moves.append(
                np.bincount(
                    sorted_move_labels[i : i + c],
                    minlength=MOVE_LABELS_NUM,
                )
            )
            wins.append(np.sum(sorted_wins[i : i + c]))
            hcps.append(sorted_hcps[i])
            i += c

        return uniq, counts, moves, wins, hcps

    def add_intermediate_data(
        self,
        ids: np.ndarray,
        counts: np.ndarray,
        moves: list,
        wins: list,
        hcps: list,
    ) -> None:
        """中間データを追加する"""

        # hash値を持っているかを調べてなかったらレコード追加，あればマイグレーション
        for i, c, m, w, hcp in zip(
            ids, counts, moves, wins, hcps
        ):
            idx_array = np.where(
                self.intermediate_data["id"] == i
            )[0]
            if len(idx_array) == 0:
                new_record = np.zeros(
                    1, dtype=get_intermediate_dtype()
                )
                new_record["id"] = i
                new_record["count"] = c
                new_record["winCount"] = w
                new_record["moveLabelCount"] = m
                # features
                new_record["features"] = (
                    Transform.board_feature(hcp)
                )
                # legalMoveMask
                new_record["legalMoveMask"] = (
                    Transform.board_legal_move_mask(hcp)
                )
                self.intermediate_data = np.append(
                    self.intermediate_data, new_record
                )
            else:
                idx = idx_array[0]

                self.intermediate_data["count"][idx] += c
                self.intermediate_data["winCount"][idx] += w
                self.intermediate_data["moveLabelCount"][
                    idx
                ] += m

    def aggregate_intermediate_data(self) -> np.ndarray:
        """中間データを集計して最終的な前処理データを作成する"""

        target_data = create_empty_preprocessing_array(
            len(self.intermediate_data)
        )
        target_data["id"] = self.intermediate_data["id"]
        target_data["features"] = self.intermediate_data[
            "features"
        ]
        target_data["moveLabel"] = (
            self.intermediate_data["moveLabelCount"]
            / self.intermediate_data["count"][:, np.newaxis]
        )
        target_data["resultValue"] = (
            self.intermediate_data["winCount"]
            / self.intermediate_data["count"]
        )
        target_data["legalMoveMask"] = self.intermediate_data[
            "legalMoveMask"
        ]

        return target_data

    def transform(
        self, option: PreProcessOption
    ) -> Dict[str, str]:
        """機械学習の前処理を行う (並列処理版)."""

        pre_process_result: Dict[str, str] = {}
        self.intermediate_data = (
            create_empty_intermediate_array(0)
        )
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
                for dataname, data in tqdm(
                    self.__datasource.iter_batches(),
                    desc="PreProcess (single)",
                ):
                    ids, counts, moves, wins, hcps = (
                        self._process_single_array(
                            data,
                        )
                    )

                    self.add_intermediate_data(
                        ids, counts, moves, wins, hcps
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
            else:
                # Parallel processing
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
                            ids, counts, moves, wins, hcps = (
                                future.result()
                            )
                            self.add_intermediate_data(
                                ids, counts, moves, wins, hcps
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
