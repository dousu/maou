import logging
import random
from collections.abc import Generator
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyarrow as pa

from maou.interface import learn, preprocess


class MissingFileDataConfig(Exception):
    pass


class FileDataSource(learn.LearningDataSource, preprocess.DataSource):
    class FileDataSourceSpliter(learn.LearningDataSource.DataSourceSpliter):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(self, *, file_paths: list[Path]) -> None:
            self.__file_manager = FileDataSource.FileManager(
                file_paths=file_paths,
            )

        def train_test_split(
            self, test_ratio: float
        ) -> tuple["FileDataSource", "FileDataSource"]:
            self.logger.info(f"test_ratio: {test_ratio}")
            input_indicies, test_indicies = self.__train_test_split(
                data=list(range(self.__file_manager.total_rows)),
                test_ratio=test_ratio,
            )
            return (
                FileDataSource(
                    file_manager=self.__file_manager,
                    indicies=input_indicies,
                ),
                FileDataSource(
                    file_manager=self.__file_manager,
                    indicies=test_indicies,
                ),
            )

        def __train_test_split(
            self,
            data: list,
            test_ratio: float = 0.25,
            seed: Optional[Union[int, float, str, bytes, bytearray]] = None,
        ) -> tuple:
            if seed is not None:
                random.seed(seed)
            random.shuffle(data)
            split_idx = int(len(data) * (1 - test_ratio))
            return data[:split_idx], data[split_idx:]

    class FileManager:
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(self, *, file_paths: list[Path]) -> None:
            """ファイルシステムから複数のファイルに入っているデータを取り出す.

            Args:
                file_paths (list[Path]): npyファイルのリスト
            """
            self.file_paths = file_paths

            self.file_row_offsets = []
            total_rows = 0
            for file in self.file_paths:
                data = np.load(file, mmap_mode="r")
                num_rows = data.shape[0]
                self.file_row_offsets.append((file, total_rows, num_rows))
                total_rows += num_rows

            self.total_rows = total_rows
            self.logger.info(f"File Data {self.total_rows} rows")

        def get_item(self, idx: int) -> np.ndarray:
            for file, start_idx, num_rows in self.file_row_offsets:
                if start_idx <= idx < start_idx + num_rows:
                    relative_idx = idx - start_idx

                    # numpy structured arrayから直接レコードを取得
                    npy_data = np.load(file, mmap_mode="r")
                    return npy_data[relative_idx]

            raise IndexError(f"Index {idx} out of range.")

        def iter_batches(self) -> Generator[tuple[str, np.ndarray], None, None]:
            for file in self.file_paths:
                data = np.load(file, mmap_mode="r")
                yield str(file), data

    def __init__(
        self,
        *,
        file_paths: Optional[list[Path]] = None,
        file_manager: Optional[FileManager] = None,
        indicies: Optional[list[int]] = None,
    ) -> None:
        """ファイルシステムから複数のファイルに入っているデータを取り出す.

        Args:
            file_paths (list[Path]): npyファイルのリスト
            schema (dict[str, str]): 各フィールドのマッピング(例: {"hcp": "hcp", "eval": "eval"})
            file_manager (Optional[FileManager]): FileManager
            indicies (Optional[list[int]]): 選択可能なインデックスのリスト
        """
        if file_manager is None:
            if file_paths is not None:
                self.__file_manager = self.FileManager(
                    file_paths=file_paths,
                )
            else:
                raise MissingFileDataConfig(
                    f"ファイル名が未設定 file_paths: {file_paths}"
                )
        else:
            self.__file_manager = file_manager

        if indicies is None:
            self.indicies = list(range(self.__file_manager.total_rows))
        else:
            self.indicies = indicies

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.indicies):
            raise IndexError(f"Index {idx} out of range.")

        return self.__file_manager.get_item(self.indicies[idx])

    def __len__(self) -> int:
        return len(self.indicies)

    def iter_batches(
        self,
    ) -> Generator[tuple[str, Union[pa.Table, np.ndarray]], None, None]:
        # indiciesを使ったランダムアクセスは無視して全体を効率よくアクセスする
        for name, batch in self.__file_manager.iter_batches():
            yield name, batch
