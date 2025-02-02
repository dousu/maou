import abc
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyarrow as pa
from tqdm.auto import tqdm

from maou.app.pre_process.feature import FEATURES_NUM
from maou.app.pre_process.transform import Transform


class FeatureStore(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def store_features(self, *, key_columns: list[str], arrow_table: pa.Table) -> None:
        pass


class PreProcess:
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, *, feature_store: Optional[FeatureStore] = None):
        self.__feature_store = feature_store
        self.__transform_logic: Transform = Transform()

    @dataclass(kw_only=True, frozen=True)
    class PreProcessOption:
        input_paths: list[Path]
        output_dir: Path

    def transform(self, option: PreProcessOption) -> Dict[str, str]:
        """機械学習の前処理を行う."""

        pre_process_result: Dict[str, str] = {}
        self.logger.debug(f"前処理対象のファイル {option.input_paths}")
        for file in tqdm(option.input_paths):
            # 1024もあれば確保しておく局面数として十分だろう
            array = np.zeros(
                1024,
                dtype=[
                    ("id", "U128"),
                    ("eval", "i4"),
                    ("features", "f4", (FEATURES_NUM, 9, 9)),
                    ("moveLabel", "i2"),
                    ("resultValue", "f4"),
                ],
            )
            hcpes = np.load(file)
            arrow_features: dict[str, list[Any]] = defaultdict(list)
            for idx, hcpe in enumerate(hcpes):
                hcp = hcpe["hcp"]
                move16 = hcpe["bestMove16"]
                game_result = hcpe["gameResult"]
                eval = hcpe["eval"]
                features, move_label, result_value = self.__transform_logic(
                    hcp=hcp, move16=move16, game_result=game_result, eval=eval
                )

                data = array[idx]
                data["id"] = f"{file.name}_{idx}"
                data["eval"] = hcpe["eval"]
                data["features"] = features
                data["moveLabel"] = move_label
                data["resultValue"] = result_value

                if self.__feature_store is not None:
                    arrow_features["id"].append(f"{file.name}_{idx}")
                    arrow_features["eval"].append(hcpe["eval"])
                    arrow_features["features"].append(pickle.dumps(features))
                    arrow_features["moveLabel"].append(move_label)
                    arrow_features["resultValue"].append(result_value)

            np.save(
                option.output_dir / file.with_suffix(".pre.npy").name, array[: idx + 1]
            )

            if self.__feature_store is not None:
                arrow_table = pa.table(arrow_features)
                self.__feature_store.store_features(
                    key_columns=["id"],
                    arrow_table=arrow_table,
                )
            pre_process_result[str(file)] = f"success {idx + 1} rows"

        return pre_process_result
