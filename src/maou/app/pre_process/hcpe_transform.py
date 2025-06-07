import abc
import contextlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ContextManager, Dict, Generator, Optional

import numpy as np
from tqdm.auto import tqdm

from maou.app.pre_process.feature import FEATURES_NUM
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.app.pre_process.transform import Transform


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
        self.__transform_logic: Transform = Transform()

    @dataclass(kw_only=True, frozen=True)
    class PreProcessOption:
        output_dir: Optional[Path] = None
        max_workers: Optional[int] = None

    @staticmethod
    def _process_record_chunk(
        records_chunk: np.ndarray,
        dataname: str,
        start_idx: int,
    ) -> tuple[np.ndarray, int]:
        """Process a chunk of records in parallel."""
        from maou.app.pre_process.feature import FEATURES_NUM
        from maou.app.pre_process.label import MOVE_LABELS_NUM
        from maou.app.pre_process.transform import Transform

        transform_logic = Transform()
        chunk_length = len(records_chunk)

        # Create output array for this chunk
        array = np.zeros(
            chunk_length,
            dtype=[
                ("id", (np.unicode_, 128)),  # type: ignore[attr-defined]
                ("eval", np.int16),
                ("features", np.float32, (FEATURES_NUM, 9, 9)),
                ("moveLabel", np.uint16),
                ("resultValue", np.float32),
                ("legalMoveMask", np.float32, (MOVE_LABELS_NUM)),
                ("partitioningKey", np.dtype("datetime64[D]")),
            ],
        )

        for idx, record in enumerate(records_chunk):
            global_idx = start_idx + idx
            id = (
                record["id"]
                if "id" in record.dtype.names
                else f"{dataname}_{global_idx}"
            )
            hcp = record["hcp"]
            move16 = record["bestMove16"]
            game_result = record["gameResult"]
            eval = record["eval"]

            features, move_label, result_value, legal_move_mask = transform_logic(
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

        return array, chunk_length

    def transform(self, option: PreProcessOption) -> Dict[str, str]:
        """機械学習の前処理を行う（並列処理版）."""

        pre_process_result: Dict[str, str] = {}
        self.logger.info(f"前処理対象のデータ数 {len(self.__datasource)}")

        # Determine number of workers
        max_workers = option.max_workers
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)  # Limit to 4 for memory reasons

        self.logger.info(f"Using {max_workers} workers for parallel processing")

        with self.__context():
            for dataname, data in tqdm(
                self.__datasource.iter_batches(), desc="Processing batches"
            ):
                self.logger.debug(f"target: {dataname}")
                data_length = len(data)
                self.logger.debug(f"処理対象: {dataname}, 行数: {data_length}")

                if max_workers == 1 or data_length < 100:
                    # Sequential processing for small batches or single worker
                    array = np.zeros(
                        data_length,
                        dtype=[
                            ("id", (np.unicode_, 128)),  # type: ignore[attr-defined]
                            ("eval", np.int16),
                            ("features", np.float32, (FEATURES_NUM, 9, 9)),
                            ("moveLabel", np.uint16),
                            ("resultValue", np.float32),
                            ("legalMoveMask", np.float32, (MOVE_LABELS_NUM)),
                            ("partitioningKey", np.dtype("datetime64[D]")),
                        ],
                    )

                    for idx, record in enumerate(data):
                        id = (
                            record["id"]
                            if "id" in record.dtype.names
                            else f"{dataname}_{idx}"
                        )
                        hcp = record["hcp"]
                        move16 = record["bestMove16"]
                        game_result = record["gameResult"]
                        eval = record["eval"]
                        features, move_label, result_value, legal_move_mask = (
                            self.__transform_logic(
                                hcp=hcp,
                                move16=move16,
                                game_result=game_result,
                                eval=eval,
                            )
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

                    final_array = array
                else:
                    # Parallel processing for larger batches
                    chunk_size = max(1, data_length // max_workers)
                    chunks = []

                    # Split data into chunks
                    for i in range(0, data_length, chunk_size):
                        end_idx = min(i + chunk_size, data_length)
                        chunks.append((data[i:end_idx], dataname, i))

                    self.logger.debug(
                        f"Split {data_length} records into {len(chunks)} chunks"
                    )

                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all chunks
                        future_to_chunk = {
                            executor.submit(
                                self._process_record_chunk,
                                chunk_data,
                                chunk_dataname,
                                start_idx,
                            ): (chunk_data, start_idx)
                            for chunk_data, chunk_dataname, start_idx in chunks
                        }

                        # Collect results
                        chunk_results = []
                        for future in as_completed(future_to_chunk):
                            try:
                                chunk_array, chunk_length = future.result()
                                chunk_data, start_idx = future_to_chunk[future]
                                chunk_results.append((start_idx, chunk_array))
                            except Exception as exc:
                                self.logger.error(f"Chunk processing failed: {exc}")
                                raise

                        # Sort results by start index and combine
                        chunk_results.sort(key=lambda x: x[0])

                        # Create final array and combine chunks
                        final_array = np.zeros(
                            data_length,
                            dtype=[
                                ("id", (np.unicode_, 128)),  # type: ignore[attr-defined] # noqa: E501
                                ("eval", np.int16),
                                ("features", np.float32, (FEATURES_NUM, 9, 9)),
                                ("moveLabel", np.uint16),
                                ("resultValue", np.float32),
                                ("legalMoveMask", np.float32, (MOVE_LABELS_NUM)),
                                ("partitioningKey", np.dtype("datetime64[D]")),
                            ],
                        )

                        current_idx = 0
                        for start_idx, chunk_array in chunk_results:
                            chunk_length = len(chunk_array)
                            final_array[current_idx : current_idx + chunk_length] = (
                                chunk_array
                            )
                            current_idx += chunk_length

                # Store results
                if self.__feature_store is not None:
                    self.__feature_store.store_features(
                        name=dataname,
                        key_columns=["id"],
                        structured_array=final_array,
                        partitioning_key_date="partitioningKey",
                    )

                if option.output_dir is not None:
                    np.save(
                        option.output_dir / Path(dataname).with_suffix(".pre.npy").name,
                        final_array,
                    )
                pre_process_result[dataname] = f"success {len(final_array)} rows"

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
