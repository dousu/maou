import abc
import contextlib
import logging
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
    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        pass


class PreProcess:
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        datasource: DataSource,
        feature_store: Optional[FeatureStore] = None,
    ):
        self.__feature_store = feature_store
        self.__datasource = datasource
        self.__transform_logic: Transform = Transform()

    @dataclass(kw_only=True, frozen=True)
    class PreProcessOption:
        output_dir: Optional[Path] = None

    def transform(self, option: PreProcessOption) -> Dict[str, str]:
        """機械学習の前処理を行う."""

        pre_process_result: Dict[str, str] = {}
        self.logger.info(f"前処理対象のデータ数 {len(self.__datasource)}")
        with self.__context():
            for dataname, data in tqdm(self.__datasource.iter_batches()):
                self.logger.debug(f"target: {dataname}")
                data_length = len(data)
                self.logger.debug(f"処理対象: {dataname}, 行数: {data_length}")
                array = np.zeros(
                    data_length,
                    dtype=[
                        ("id", (np.unicode_, 128)),
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
                        # 一応Noneにしないようにしておく
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

                if self.__feature_store is not None:
                    self.__feature_store.store_features(
                        name=dataname,
                        key_columns=["id"],
                        structured_array=array[: idx + 1],
                        partitioning_key_date="partitioningKey",
                    )

                if option.output_dir is not None:
                    np.save(
                        option.output_dir / Path(dataname).with_suffix(".pre.npy").name,
                        array[: idx + 1],
                    )
                pre_process_result[dataname] = f"success {idx + 1} rows"

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
