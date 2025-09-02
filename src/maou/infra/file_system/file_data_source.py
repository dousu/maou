import logging
import random
from collections.abc import Generator
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from maou.interface import learn, preprocess
from maou.interface.data_io import load_array, load_packed_array
from maou.interface.data_schema import (
    convert_array_from_packed_schema,
    convert_record_from_packed_schema,
)


class MissingFileDataConfig(Exception):
    pass


class FileDataSource(
    learn.LearningDataSource, preprocess.DataSource
):
    class FileDataSourceSpliter(
        learn.LearningDataSource.DataSourceSpliter
    ):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            file_paths: list[Path],
            array_type: Literal["hcpe", "preprocessing"],
            bit_pack: bool = False,
        ) -> None:
            self.__file_manager = FileDataSource.FileManager(
                file_paths=file_paths,
                array_type=array_type,
                bit_pack=bit_pack,
            )

        def train_test_split(
            self, test_ratio: float
        ) -> tuple["FileDataSource", "FileDataSource"]:
            self.logger.info(f"test_ratio: {test_ratio}")
            input_indicies, test_indicies = (
                self.__train_test_split(
                    data=list(
                        range(self.__file_manager.total_rows)
                    ),
                    test_ratio=test_ratio,
                )
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
            seed: Optional[
                Union[int, float, str, bytes, bytearray]
            ] = None,
        ) -> tuple:
            if seed is not None:
                random.seed(seed)
            random.shuffle(data)
            split_idx = int(float(len(data)) * (1 - test_ratio))
            return data[:split_idx], data[split_idx:]

    class FileManager:
        logger: logging.Logger = logging.getLogger(__name__)
        array_type: Literal["hcpe", "preprocessing"]

        def __init__(
            self,
            file_paths: list[Path],
            array_type: Literal["hcpe", "preprocessing"],
            bit_pack: bool,
        ) -> None:
            """ファイルシステムから複数のファイルに入っているデータを取り出す.

            Args:
                file_paths (list[Path]): npyファイルのリスト
                array_type (Literal["hcpe", "preprocessing"]): 配列のタイプ ("hcpe" または "preprocessing")
            """
            self.file_paths = file_paths
            self.array_type = array_type
            self.bit_pack = bit_pack
            self.memmap_arrays: list[
                tuple[str, np.ndarray]
            ] = []

            # すべてのファイルパスをmemmapで読み込む
            lengths = []
            for file_path in self.file_paths:
                try:
                    if (
                        self.bit_pack
                        and self.array_type == "preprocessing"
                    ):
                        array = load_packed_array(
                            file_path,
                            mmap_mode="r",
                            array_type=self.array_type,
                        )
                    else:
                        array = load_array(
                            file_path,
                            mmap_mode="r",
                            array_type=self.array_type,
                            bit_pack=self.bit_pack,
                        )
                    self.memmap_arrays.append(
                        (file_path.name, array)
                    )
                    lengths.append(len(array))
                except Exception as e:
                    self.logger.error(
                        f"Failed to load array {file_path}: {e}"
                    )
                    raise
            self.cum_lengths = np.cumsum([0] + lengths)

            self.total_rows = self.cum_lengths[-1]
            self.total_pages = len(self.cum_lengths) - 1

            self.logger.info(
                f"File Data {self.total_rows} rows, {self.total_pages} pages"
            )

        def get_item(self, idx: int) -> np.ndarray:
            """特定のレコードをnumpy structured arrayとして返す."""
            if idx < 0 or idx >= self.total_rows:
                raise IndexError(f"Index {idx} out of range.")
            file_idx = int(
                np.searchsorted(
                    self.cum_lengths, idx, side="right"
                )
                - 1
            )
            relative_idx = idx - self.cum_lengths[file_idx]
            if (
                self.bit_pack
                and self.array_type == "preprocessing"
            ):
                return convert_record_from_packed_schema(
                    compressed_record=self.memmap_arrays[
                        file_idx
                    ][1][relative_idx],
                    array_type=self.array_type,
                )
            else:
                return self.memmap_arrays[file_idx][1][
                    relative_idx
                ]

        def iter_batches(
            self,
        ) -> Generator[tuple[str, np.ndarray], None, None]:
            if (
                self.bit_pack
                and self.array_type == "preprocessing"
            ):
                for name, array in self.memmap_arrays:
                    yield (
                        name,
                        convert_array_from_packed_schema(
                            compressed_array=array,
                            array_type=self.array_type,
                        ),
                    )
            else:
                for name, array in self.memmap_arrays:
                    yield (
                        name,
                        array,
                    )

    def __init__(
        self,
        *,
        file_paths: Optional[list[Path]] = None,
        file_manager: Optional[FileManager] = None,
        indicies: Optional[list[int]] = None,
        array_type: Optional[
            Literal["hcpe", "preprocessing"]
        ] = None,
        bit_pack: bool = True,
    ) -> None:
        """ファイルシステムから複数のファイルに入っているデータを取り出す.

        Args:
            file_paths (list[Path]): npyファイルのリスト
            file_manager (Optional[FileManager]): FileManager
            indicies (Optional[list[int]]): 選択可能なインデックスのリスト
            array_type (Optional[Literal["hcpe", "preprocessing"]]): 配列のタイプ ("hcpe" または "preprocessing")
        """
        if file_manager is None:
            if (
                file_paths is not None
                and array_type is not None
            ):
                self.__file_manager = self.FileManager(
                    file_paths=file_paths,
                    array_type=array_type,
                    bit_pack=bit_pack,
                )
            else:
                raise MissingFileDataConfig(
                    f"ファイル名が未設定 file_paths: {file_paths}"
                )
        else:
            self.__file_manager = file_manager

        if indicies is None:
            self.indicies = list(
                range(self.__file_manager.total_rows)
            )
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
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        # indiciesを使ったランダムアクセスは無視して全体を効率よくアクセスする
        for name, batch in self.__file_manager.iter_batches():
            yield name, batch
