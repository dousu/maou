import logging
import random
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast

import numpy as np
from numpy import memmap as NpMemMap

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
    CacheMode = Literal["mmap", "memory"]

    class FileDataSourceSpliter(
        learn.LearningDataSource.DataSourceSpliter
    ):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            file_paths: list[Path],
            array_type: Literal["hcpe", "preprocessing"],
            bit_pack: bool = False,
            cache_mode: "FileDataSource.CacheMode" = "mmap",
            preprocessing_mmap_mode: Optional[
                Literal["r", "r+", "w+", "c"]
            ] = "c",
        ) -> None:
            self.__file_manager = FileDataSource.FileManager(
                file_paths=file_paths,
                array_type=array_type,
                bit_pack=bit_pack,
                cache_mode=cache_mode,
                preprocessing_mmap_mode=preprocessing_mmap_mode,
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

        @dataclass
        class _FileEntry:
            name: str
            path: Path
            dtype: np.dtype[Any]
            length: int
            memmap: Optional[NpMemMap]
            cached_array: Optional[np.ndarray]

        def __init__(
            self,
            file_paths: list[Path],
            array_type: Literal["hcpe", "preprocessing"],
            bit_pack: bool,
            cache_mode: "FileDataSource.CacheMode" = "mmap",
            preprocessing_mmap_mode: Optional[
                Literal["r", "r+", "w+", "c"]
            ] = "c",
        ) -> None:
            """ファイルシステムから複数のファイルに入っているデータを取り出す.

            Args:
                file_paths (list[Path]): npyファイルのリスト
                array_type (Literal["hcpe", "preprocessing"]): 配列のタイプ ("hcpe" または "preprocessing")
            """
            self.file_paths = file_paths
            self.array_type = array_type
            self.bit_pack = bit_pack
            self.preprocessing_mmap_mode = preprocessing_mmap_mode
            normalized_cache_mode = cast(
                FileDataSource.CacheMode, cache_mode.lower()
            )
            if normalized_cache_mode not in {"mmap", "memory"}:
                raise ValueError(
                    "cache_mode must be either 'mmap' or 'memory', "
                    f"got {cache_mode}"
                )
            self.memmap_arrays: list[tuple[str, NpMemMap]] = []
            self._file_entries: list[
                FileDataSource.FileManager._FileEntry
            ] = []
            self.cache_mode: FileDataSource.CacheMode = (
                normalized_cache_mode
            )

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
                            mmap_mode=(
                                "r"
                                if self.array_type == "hcpe"
                                else None
                            ),
                            array_type=self.array_type,
                            bit_pack=self.bit_pack,
                            preprocessing_mmap_mode=self.preprocessing_mmap_mode,
                        )
                    memmap: Optional[NpMemMap]
                    cached_array: Optional[np.ndarray] = None
                    array_length = int(len(array))
                    array_dtype = array.dtype
                    if isinstance(array, NpMemMap):
                        memmap = array
                    else:
                        memmap = None

                    if self.cache_mode == "memory":
                        array_nbytes = int(
                            getattr(
                                array,
                                "nbytes",
                                array_length
                                * array_dtype.itemsize,
                            )
                        )
                        array_size_mib = (
                            array_nbytes / (1024 * 1024)
                            if array_nbytes
                            else 0.0
                        )
                        self.logger.info(
                            "Caching %s into RAM (%.2f MiB)",
                            file_path,
                            array_size_mib,
                        )
                        try:
                            cached_array = array.copy()
                        except MemoryError:
                            self.logger.exception(
                                "Failed to allocate memory for %s (%s bytes)",
                                file_path,
                                array_nbytes,
                            )
                            raise
                        else:
                            self.logger.info(
                                "Cached %s in RAM (%.2f MiB)",
                                file_path,
                                array_size_mib,
                            )
                            if isinstance(array, NpMemMap):
                                mmap_handle = getattr(array, "_mmap", None)
                                if mmap_handle is not None:
                                    try:
                                        mmap_handle.close()
                                    except AttributeError:
                                        pass
                            memmap = None
                        array = None
                    elif isinstance(array, NpMemMap):
                        if memmap is not None:
                            self.memmap_arrays.append(
                                (file_path.name, memmap)
                            )
                    self._file_entries.append(
                        FileDataSource.FileManager._FileEntry(
                            name=file_path.name,
                            path=file_path,
                            dtype=array_dtype,
                            length=array_length,
                            memmap=memmap,
                            cached_array=cached_array,
                        )
                    )
                    lengths.append(array_length)
                    if (
                        memmap is None
                        and cached_array is None
                        and array is not None
                    ):
                        del array
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
            entry = self._file_entries[file_idx]
            if entry.cached_array is not None:
                record = entry.cached_array[relative_idx]
                if (
                    self.bit_pack
                    and self.array_type == "preprocessing"
                ):
                    return convert_record_from_packed_schema(
                        compressed_record=record,
                        array_type=self.array_type,
                    )
                return record
            if entry.memmap is None:
                record_array = np.fromfile(
                    entry.path,
                    dtype=entry.dtype,
                    count=1,
                    offset=int(
                        relative_idx * entry.dtype.itemsize
                    ),
                )
                if record_array.size == 0:
                    raise IndexError(
                        f"Index {idx} could not be loaded from {entry.path}."
                    )
                record = np.copy(record_array[0])
                del record_array
                if (
                    self.bit_pack
                    and self.array_type == "preprocessing"
                ):
                    return convert_record_from_packed_schema(
                        compressed_record=record,
                        array_type=self.array_type,
                    )
                return record
            if (
                self.bit_pack
                and self.array_type == "preprocessing"
            ):
                return convert_record_from_packed_schema(
                    compressed_record=entry.memmap[
                        relative_idx
                    ],
                    array_type=self.array_type,
                )
            else:
                return entry.memmap[relative_idx]

        def iter_batches(
            self,
        ) -> Generator[tuple[str, np.ndarray], None, None]:
            if (
                self.bit_pack
                and self.array_type == "preprocessing"
            ):
                for entry in self._file_entries:
                    if entry.cached_array is not None:
                        yield (
                            entry.name,
                            convert_array_from_packed_schema(
                                compressed_array=entry.cached_array,
                                array_type=self.array_type,
                            ),
                        )
                    elif entry.memmap is not None:
                        yield (
                            entry.name,
                            convert_array_from_packed_schema(
                                compressed_array=entry.memmap,
                                array_type=self.array_type,
                            ),
                        )
                    else:
                        loaded_array = np.fromfile(
                            entry.path,
                            dtype=entry.dtype,
                            count=entry.length,
                        )
                        unpacked_array = (
                            convert_array_from_packed_schema(
                                compressed_array=loaded_array,
                                array_type=self.array_type,
                            )
                        )
                        del loaded_array
                        yield (
                            entry.name,
                            unpacked_array,
                        )
            else:
                for entry in self._file_entries:
                    if entry.cached_array is not None:
                        yield (
                            entry.name,
                            entry.cached_array,
                        )
                    elif entry.memmap is not None:
                        yield (
                            entry.name,
                            entry.memmap,
                        )
                    else:
                        loaded_array = np.fromfile(
                            entry.path,
                            dtype=entry.dtype,
                            count=entry.length,
                        )
                        try:
                            yield (
                                entry.name,
                                loaded_array,
                            )
                        finally:
                            del loaded_array

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
        cache_mode: CacheMode = "mmap",
        preprocessing_mmap_mode: Optional[
            Literal["r", "r+", "w+", "c"]
        ] = "c",
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
                    cache_mode=cache_mode,
                    preprocessing_mmap_mode=preprocessing_mmap_mode,
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
