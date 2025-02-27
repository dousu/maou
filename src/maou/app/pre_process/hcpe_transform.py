import abc
import contextlib
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ContextManager, Dict, Generator, Optional, Union

import numpy as np
import pyarrow as pa
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
        key_columns: list[str],
        arrow_table: pa.Table,
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
    ) -> Generator[tuple[str, Union[pa.Table, np.ndarray]], None, None]:
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
        self.logger.debug(f"前処理対象のデータ数 {len(self.__datasource)}")
        with self.__context():
            for dataname, data in tqdm(self.__datasource.iter_batches()):
                if isinstance(data, pa.Table):
                    data_length = data.num_rows  # type: ignore
                elif isinstance(data, np.ndarray):
                    data_length = len(data)
                self.logger.debug(f"処理対象: {dataname}, 行数: {data_length}")
                array = np.zeros(
                    data_length,
                    dtype=[
                        ("id", "U128"),
                        ("eval", "i4"),
                        ("features", "f4", (FEATURES_NUM, 9, 9)),
                        ("moveLabel", "i2"),
                        ("resultValue", "f4"),
                        ("legalMoveMask", "f4", (MOVE_LABELS_NUM)),
                    ],
                )
                arrow_features: dict[str, list[Any]] = defaultdict(list)
                for idx in range(data_length):
                    if isinstance(data, pa.Table):
                        id = data["id"][idx].as_py()
                        hcp = pickle.loads(data["hcp"][idx].as_py())
                        move16 = data["bestMove16"][idx].as_py()
                        game_result = data["gameResult"][idx].as_py()
                        eval = data["eval"][idx].as_py()
                        features, move_label, result_value, legal_move_mask = (
                            self.__transform_logic(
                                hcp=hcp,
                                move16=move16,
                                game_result=game_result,
                                eval=eval,
                            )
                        )
                        partitioning_key = data["partitioningKey"][idx].as_py()
                    elif isinstance(data, np.ndarray):
                        id = f"{dataname}_{idx}"
                        hcp = data[idx]["hcp"]
                        move16 = data[idx]["bestMove16"]
                        game_result = data[idx]["gameResult"]
                        eval = data[idx]["eval"]
                        features, move_label, result_value, legal_move_mask = (
                            self.__transform_logic(
                                hcp=hcp,
                                move16=move16,
                                game_result=game_result,
                                eval=eval,
                            )
                        )
                        partitioning_key = datetime.now().date()

                    if self.__feature_store is not None:
                        arrow_features["id"].append(id)
                        arrow_features["eval"].append(eval)
                        arrow_features["features"].append(pickle.dumps(features))
                        arrow_features["moveLabel"].append(move_label)
                        arrow_features["resultValue"].append(result_value)
                        arrow_features["legalMoveMask"].append(
                            pickle.dumps(legal_move_mask)
                        )
                        # ローカルファイルには入らない情報
                        arrow_features["partitioningKey"].append(partitioning_key)

                    if option.output_dir is not None:
                        np_data = array[idx]
                        np_data["id"] = id
                        np_data["eval"] = eval
                        np_data["features"] = features
                        np_data["moveLabel"] = move_label
                        np_data["resultValue"] = result_value
                        np_data["legalMoveMask"] = legal_move_mask

                if self.__feature_store is not None:
                    arrow_table = pa.table(arrow_features)
                    self.__feature_store.store_features(
                        key_columns=["id"],
                        arrow_table=arrow_table,
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
