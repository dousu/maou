import abc
import contextlib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ContextManager, Dict, Generator, Optional

import numpy as np
from tqdm.auto import tqdm

from maou.app.pre_process.transform import Transform
from maou.domain.data.io import save_preprocessing_array
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
        max_workers: int

    @staticmethod
    def _process_single_array(
        data: np.ndarray,
        dataname: str,
    ) -> np.ndarray:
        """Process a chunk of records in parallel."""

        transform_logic = Transform()
        data_length = len(data)

        # Create output array for this chunk
        array = create_empty_preprocessing_array(data_length)

        for idx, record in enumerate(data):
            id = (
                record["id"]
                if "id" in record.dtype.names
                and record["id"] != ""
                else f"{dataname}_{idx}"
            )
            hcp = record["hcp"]
            move16 = record["bestMove16"]
            game_result = record["gameResult"]
            eval = record["eval"]

            (
                features,
                move_label,
                result_value,
                legal_move_mask,
            ) = transform_logic(
                hcp=hcp,
                move16=move16,
                game_result=game_result,
                eval=eval,
            )

            partitioning_key = (
                record["partitioningKey"]
                if "partitioningKey" in record.dtype.names
                else datetime.min.date()
            )

            np_data = array[idx]
            np_data["id"] = id
            np_data["eval"] = eval
            np_data["features"] = features
            np_data["moveLabel"] = move_label
            np_data["resultValue"] = result_value
            np_data["legalMoveMask"] = legal_move_mask
            np_data["partitioningKey"] = partitioning_key

        return array

    def transform(
        self, option: PreProcessOption
    ) -> Dict[str, str]:
        """機械学習の前処理を行う (並列処理版)."""

        pre_process_result: Dict[str, str] = {}
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
                    desc="Processing batches",
                ):
                    array = self._process_single_array(
                        data, dataname
                    )

                    pre_process_result[dataname] = (
                        f"success {len(array)} rows"
                    )
                    # Store results
                    if self.__feature_store is not None:
                        self.__feature_store.store_features(
                            name=dataname,
                            key_columns=["id"],
                            structured_array=array,
                            partitioning_key_date="partitioningKey",
                        )

                    if option.output_dir is not None:
                        base_name = Path(dataname).stem
                        save_preprocessing_array(
                            array,
                            option.output_dir
                            / f"{base_name}.npy",
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
                            dataname,
                        ): dataname
                        for dataname, data in self.__datasource.iter_batches()
                    }
                    for future in tqdm(
                        as_completed(future_to_dataname),
                        desc="Processing files",
                    ):
                        dataname = future_to_dataname[future]
                        try:
                            array = future.result()
                            pre_process_result[dataname] = (
                                f"success {len(array)} rows"
                            )
                            # Store results
                            if self.__feature_store is not None:
                                self.__feature_store.store_features(
                                    name=dataname,
                                    key_columns=["id"],
                                    structured_array=array,
                                    partitioning_key_date="partitioningKey",
                                )

                            if option.output_dir is not None:
                                base_name = Path(dataname).stem
                                save_preprocessing_array(
                                    array,
                                    option.output_dir
                                    / f"{base_name}.npy",
                                    bit_pack=False,
                                )
                        except Exception as exc:
                            self.logger.error(
                                f"{dataname} processing failed: {exc}"
                            )
                            pre_process_result[dataname] = (
                                f"Dataname {dataname} generated an exception: {exc}"
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
