from __future__ import annotations

import logging
import random
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Union,
    cast,
)

import numpy as np

import maou.interface.learn as learn
import maou.interface.preprocess as preprocess
from maou.domain.data.schema import (
    convert_hcpe_df_to_numpy,
    convert_preprocessing_df_to_numpy,
    convert_stage1_df_to_numpy,
    convert_stage2_df_to_numpy,
)

if TYPE_CHECKING:
    import polars as pl

_DF_TO_NUMPY_CONVERTERS: dict[
    str, Callable[["pl.DataFrame"], np.ndarray]
] = {
    "hcpe": convert_hcpe_df_to_numpy,
    "preprocessing": convert_preprocessing_df_to_numpy,
    "stage1": convert_stage1_df_to_numpy,
    "stage2": convert_stage2_df_to_numpy,
}


class MissingFileDataConfig(Exception):
    pass


class FileDataSource(
    learn.LearningDataSource, preprocess.DataSource
):
    CacheMode = Literal["file", "memory"]

    class FileDataSourceSpliter(
        learn.LearningDataSource.DataSourceSpliter
    ):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            file_paths: list[Path],
            array_type: Literal[
                "hcpe", "preprocessing", "stage1", "stage2"
            ],
            bit_pack: bool = False,
            cache_mode: "FileDataSource.CacheMode" = "file",
        ) -> None:
            self.__file_manager = FileDataSource.FileManager(
                file_paths=file_paths,
                array_type=array_type,
                bit_pack=bit_pack,
                cache_mode=cache_mode,
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
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ]

        @dataclass
        class _FileEntry:
            name: str
            path: Path
            dtype: np.dtype[
                Any
            ]  # Placeholder (not used for DataFrames)
            length: int
            memmap: Optional[
                Any
            ]  # Placeholder (not used for DataFrames)
            cached_array: Optional[
                Any
            ]  # Can be DataFrame or ndarray

        def __init__(
            self,
            file_paths: list[Path],
            array_type: Literal[
                "hcpe", "preprocessing", "stage1", "stage2"
            ],
            bit_pack: bool,
            cache_mode: "FileDataSource.CacheMode" = "file",
        ) -> None:
            """ファイルシステムから複数のファイルに入っているデータを取り出す．

            Args:
                file_paths (list[Path]): .featherファイルのリスト
                array_type (Literal["hcpe", "preprocessing", "stage1", "stage2"]): データのタイプ
                bit_pack (bool): 未使用（後方互換性のために保持）
                cache_mode (CacheMode): キャッシュモード ("file" または "memory")
            """
            self.file_paths = file_paths
            self.array_type = array_type
            self.bit_pack = bit_pack
            normalized_cache_mode = cast(
                FileDataSource.CacheMode, cache_mode.lower()
            )
            if normalized_cache_mode not in {"file", "memory"}:
                raise ValueError(
                    "cache_mode must be either 'file' or 'memory', "
                    f"got {cache_mode}"
                )
            self.memmap_arrays: list[
                tuple[str, Any]
            ] = []  # Unused (DataFrames only)
            self._file_entries: list[
                FileDataSource.FileManager._FileEntry
            ] = []
            self.cache_mode: FileDataSource.CacheMode = (
                normalized_cache_mode
            )
            # 最適化: 最後にアクセスしたファイルインデックスをキャッシュ
            self._last_file_idx = 0
            # 最適化: cache_mode="memory"の場合、全ファイルを結合した単一numpy配列
            self._concatenated_array: Optional[np.ndarray] = (
                None
            )

            # DF→numpy変換関数を先に取得
            converter = _DF_TO_NUMPY_CONVERTERS.get(
                self.array_type
            )
            if converter is None:
                raise ValueError(
                    f"Unknown array_type: {self.array_type}"
                )

            # ファイル単位で読み込み→変換→DF解放を逐次実行
            # (ピークメモリ = 累積numpy + 1ファイル分DF + 変換中間配列)
            lengths = []
            for file_path in self.file_paths:
                try:
                    if file_path.suffix != ".feather":
                        raise ValueError(
                            f"Only .feather files are supported. Got: {file_path.suffix}"
                        )

                    try:
                        # 1. ファイル読み込み (Polars DataFrame)
                        df = self._load_feather(file_path)
                        array_length = len(df)

                        # 2. DF→numpy変換 (この間のみDFとnumpyが共存)
                        numpy_array = converter(df)

                        # 3. DF参照を即座に切る (GC対象にする)
                        del df

                        # 4. numpy arrayのみ保持
                        self._file_entries.append(
                            FileDataSource.FileManager._FileEntry(
                                name=file_path.name,
                                path=file_path,
                                dtype=numpy_array.dtype,
                                length=array_length,
                                memmap=None,
                                cached_array=numpy_array,
                            )
                        )
                        lengths.append(array_length)

                    except ImportError as e:
                        raise ImportError(
                            f"Polars and Rust backend required for .feather files: {e}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Failed to load array {file_path}: {e}"
                    )
                    raise
            self.cum_lengths = np.cumsum([0] + lengths)

            self.total_rows = self.cum_lengths[-1]
            self.total_pages = len(self.cum_lengths) - 1

            # cache_mode="memory"の場合、全ファイルを単一numpy配列に結合
            if (
                self.cache_mode == "memory"
                and self.total_pages > 1
            ):
                # メモリ見積もりの警告
                total_bytes = sum(
                    entry.cached_array.nbytes
                    for entry in self._file_entries
                    if entry.cached_array is not None
                )
                estimated_gb = total_bytes / (1024**3)
                if estimated_gb > 32:
                    self.logger.warning(
                        f"cache_mode='memory' with {self.total_rows} rows "
                        f"(estimated {estimated_gb:.1f} GB) may cause OOM. "
                        f"Consider using cache_mode='file' instead."
                    )

                self.logger.info(
                    f"Concatenating {self.total_pages} numpy arrays "
                    f"({self.total_rows} records)..."
                )

                arrays = [
                    entry.cached_array
                    for entry in self._file_entries
                    if entry.cached_array is not None
                ]
                self._concatenated_array = np.concatenate(
                    arrays
                )
                # Release individual arrays to save memory
                for entry in self._file_entries:
                    entry.cached_array = None

            self.logger.info(
                f"File Data {self.total_rows} rows, {self.total_pages} pages"
            )

        def _load_feather(
            self, file_path: Path
        ) -> "pl.DataFrame":
            """Arrow IPCファイルを読み込みPolars DataFrameとして返す．

            Args:
                file_path: .featherファイルのパス

            Returns:
                Polars DataFrame
            """
            from maou.domain.data.rust_io import (
                load_hcpe_df,
                load_preprocessing_df,
                load_stage1_df,
                load_stage2_df,
            )

            loaders = {
                "hcpe": load_hcpe_df,
                "preprocessing": load_preprocessing_df,
                "stage1": load_stage1_df,
                "stage2": load_stage2_df,
            }
            loader = loaders.get(self.array_type)
            if loader is None:
                raise ValueError(
                    f"Unsupported array_type: {self.array_type}"
                )
            return loader(file_path)

        def get_item(self, idx: int) -> np.ndarray:
            """Get single item as numpy structured array．

            Args:
                idx: Global index across all files

            Returns:
                numpy structured array (single record)
            """
            if self._concatenated_array is not None:
                return self._concatenated_array[idx]

            file_idx = np.searchsorted(
                self.cum_lengths[1:], idx, side="right"
            )
            local_idx = idx - self.cum_lengths[file_idx]
            entry = self._file_entries[file_idx]
            if entry.cached_array is None:
                raise RuntimeError(
                    f"Array not loaded for file {entry.name}"
                )
            return entry.cached_array[local_idx]

        def get_items(
            self, indices: list[int]
        ) -> list[np.ndarray]:
            """Get multiple items as numpy structured arrays．

            Converts DataFrame rows to numpy format for backward compatibility．

            Args:
                indices: List of global indices

            Returns:
                List of numpy structured arrays
            """
            return [self.get_item(idx) for idx in indices]

        def iter_batches(
            self,
        ) -> Generator[tuple[str, np.ndarray], None, None]:
            """Iterate over batches as numpy structured arrays．

            Data is already converted to numpy at initialization time．

            Yields:
                Tuple of (filename, numpy array)
            """
            # If concatenated, yield as single batch
            if self._concatenated_array is not None:
                yield ("concatenated", self._concatenated_array)
                return

            # Iterate over individual files (already numpy arrays)
            for entry in self._file_entries:
                if entry.cached_array is None:
                    continue
                yield (entry.name, entry.cached_array)

    def __init__(
        self,
        *,
        file_paths: Optional[list[Path]] = None,
        file_manager: Optional[FileManager] = None,
        indicies: Optional[Union[list[int], np.ndarray]] = None,
        array_type: Optional[
            Literal["hcpe", "preprocessing", "stage1", "stage2"]
        ] = None,
        bit_pack: bool = True,
        cache_mode: CacheMode = "file",
    ) -> None:
        """ファイルシステムから複数のファイルに入っているデータを取り出す.

        Args:
            file_paths (list[Path]): .featherファイルのリスト
            file_manager (Optional[FileManager]): FileManager
            indicies (Optional[Union[list[int], np.ndarray]]): 選択可能なインデックス
            array_type (Optional[Literal["hcpe", "preprocessing", "stage1", "stage2"]]): 配列のタイプ
            bit_pack (bool): ビットパッキングを使用するかどうか
            cache_mode (CacheMode): キャッシュモード ("file" または "memory")
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
                )
            else:
                raise MissingFileDataConfig(
                    f"ファイル名が未設定 file_paths: {file_paths}"
                )
        else:
            self.__file_manager = file_manager

        if indicies is None:
            self.indicies: np.ndarray = np.arange(
                self.__file_manager.total_rows, dtype=np.int64
            )
        else:
            self.indicies = np.asarray(indicies, dtype=np.int64)

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.indicies):
            raise IndexError(f"Index {idx} out of range.")

        return self.__file_manager.get_item(self.indicies[idx])

    def __len__(self) -> int:
        return len(self.indicies)

    def get_items(self, indices: list[int]) -> list[np.ndarray]:
        """複数のインデックスのレコードをバッチで取得する．

        Args:
            indices: 取得するインデックスのリスト（FileDataSourceのインデックス空間）

        Returns:
            レコードのリスト（入力のインデックス順）
        """
        # FileDataSourceのインデックスをFileManagerのグローバルインデックスに変換
        global_indices = [self.indicies[idx] for idx in indices]
        return self.__file_manager.get_items(global_indices)

    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """Iterate over batches as numpy structured arrays．

        Converts DataFrames to numpy format for backward compatibility．

        Yields:
            Tuple of (filename, numpy array)
        """
        return self.__file_manager.iter_batches()

    def iter_batches_df(
        self,
    ) -> Generator[tuple[str, "pl.DataFrame"], None, None]:
        """Iterate over batches as Polars DataFrames．

        Yields .feather files directly as DataFrames．
        Note: FileManager converts DataFrames to numpy at initialization
        (B1.5 method), so this method reloads each file from disk on every call．

        Yields:
            tuple[str, pl.DataFrame]: (batch_name, polars_dataframe)
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is required for DataFrame iteration. "
                "Install with: poetry add polars"
            )

        # Iterate over entries (all are .feather files)
        for entry in self.__file_manager._file_entries:
            # DataFrame already loaded and cached
            if isinstance(entry.cached_array, pl.DataFrame):
                yield entry.name, entry.cached_array
            else:
                # Load DataFrame if not cached
                from maou.domain.data.rust_io import (
                    load_hcpe_df,
                    load_preprocessing_df,
                    load_stage1_df,
                    load_stage2_df,
                )

                if self.__file_manager.array_type == "hcpe":
                    df = load_hcpe_df(entry.path)
                elif (
                    self.__file_manager.array_type
                    == "preprocessing"
                ):
                    df = load_preprocessing_df(entry.path)
                elif self.__file_manager.array_type == "stage1":
                    df = load_stage1_df(entry.path)
                elif self.__file_manager.array_type == "stage2":
                    df = load_stage2_df(entry.path)
                else:
                    raise ValueError(
                        f"Unsupported array_type: {self.__file_manager.array_type}"
                    )
                yield entry.name, df

    def total_pages(self) -> int:
        """Return the total number of pages (batches) in the data source."""
        return self.__file_manager.total_pages
