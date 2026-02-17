from __future__ import annotations

import logging
import random
import time
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
from maou.domain.data.columnar_batch import (
    ColumnarBatch,
    convert_preprocessing_df_to_columnar,
    convert_stage1_df_to_columnar,
    convert_stage2_df_to_columnar,
)
from maou.domain.data.schema import (
    convert_hcpe_df_to_numpy,
    convert_preprocessing_df_to_numpy,
    convert_stage1_df_to_numpy,
    convert_stage2_df_to_numpy,
    get_preprocessing_dtype,
    get_stage1_dtype,
    get_stage2_dtype,
)

if TYPE_CHECKING:
    import polars as pl

    from maou.infra.file_system.streaming_file_source import (
        StreamingFileSource,
    )

_DF_TO_NUMPY_CONVERTERS: dict[
    str, Callable[["pl.DataFrame"], np.ndarray]
] = {
    "hcpe": convert_hcpe_df_to_numpy,
    "preprocessing": convert_preprocessing_df_to_numpy,
    "stage1": convert_stage1_df_to_numpy,
    "stage2": convert_stage2_df_to_numpy,
}

_DF_TO_COLUMNAR_CONVERTERS: dict[
    str, Callable[["pl.DataFrame"], ColumnarBatch]
] = {
    "preprocessing": convert_preprocessing_df_to_columnar,
    "stage1": convert_stage1_df_to_columnar,
    "stage2": convert_stage2_df_to_columnar,
}

_STRUCTURED_DTYPES: dict[str, Callable[[], np.dtype]] = {
    "preprocessing": get_preprocessing_dtype,
    "stage1": get_stage1_dtype,
    "stage2": get_stage2_dtype,
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

        def file_level_split(
            self,
            test_ratio: float,
            seed: Optional[int] = None,
        ) -> tuple[
            "StreamingFileSource", "StreamingFileSource"
        ]:
            """ファイル単位のtrain/test分割．

            レコード単位ではなくファイル単位で分割し，
            それぞれ ``StreamingFileSource`` として返す．
            ストリーミングモードで使用する．

            Args:
                test_ratio: テストデータの割合(0.0 < test_ratio < 1.0)
                seed: シャッフル用のランダムシード

            Returns:
                (train_source, val_source) のタプル

            Raises:
                ValueError: ファイルが2つ未満の場合
            """
            from maou.infra.file_system.streaming_file_source import (
                StreamingFileSource,
            )

            file_paths = list(self.__file_manager.file_paths)
            n_files = len(file_paths)

            if n_files < 2:
                raise ValueError(
                    f"file_level_split requires at least 2 files, "
                    f"got {n_files}. Use train_test_split() for "
                    f"row-level splitting instead."
                )

            # Shuffle file paths
            rng = random.Random(seed)
            rng.shuffle(file_paths)

            # Split
            n_val = max(1, int(n_files * test_ratio))
            n_train = n_files - n_val

            if n_train < 1:
                n_train = 1
                n_val = n_files - 1

            train_paths = file_paths[:n_train]
            val_paths = file_paths[n_train:]

            self.logger.info(
                "File-level split: %d train files, "
                "%d val files (test_ratio=%.2f)",
                len(train_paths),
                len(val_paths),
                test_ratio,
            )

            array_type = self.__file_manager.array_type
            train_source = StreamingFileSource(
                file_paths=train_paths,
                array_type=array_type,
            )
            val_source = StreamingFileSource(
                file_paths=val_paths,
                array_type=array_type,
            )

            return train_source, val_source

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
            cached_columnar: Optional[ColumnarBatch] = (
                None  # SOA representation
            )

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
            self.logger.info(
                "Initializing FileManager with %d files, array_type=%s",
                len(file_paths),
                array_type,
            )
            t_init_start = time.perf_counter()
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
            # 最適化: cache_mode="memory"の場合、全ファイルを結合した単一配列
            self._concatenated_array: Optional[np.ndarray] = (
                None
            )
            self._concatenated_columnar: Optional[
                ColumnarBatch
            ] = None

            # SOA化対応: preprocessing/stage1/stage2はColumnarBatchで保持
            self._use_columnar = (
                array_type in _DF_TO_COLUMNAR_CONVERTERS
            )
            # structured array再構築用のdtype
            self._structured_dtype: Optional[np.dtype] = None
            if self._use_columnar:
                dtype_factory = _STRUCTURED_DTYPES.get(
                    array_type
                )
                if dtype_factory is not None:
                    self._structured_dtype = dtype_factory()

            # DF→変換関数を先に取得
            columnar_converter = _DF_TO_COLUMNAR_CONVERTERS.get(
                self.array_type
            )
            numpy_converter = _DF_TO_NUMPY_CONVERTERS.get(
                self.array_type
            )
            if (
                not self._use_columnar
                and numpy_converter is None
            ):
                raise ValueError(
                    f"Unknown array_type: {self.array_type}"
                )

            # ファイル単位で読み込み→変換→DF解放を逐次実行
            lengths = []
            n = len(self.file_paths)
            if n <= 10:
                milestone_interval = 1
            elif n <= 100:
                milestone_interval = max(1, n // 10)
            else:
                milestone_interval = max(25, n // 10)
            cumulative_rows = 0
            for idx, file_path in enumerate(self.file_paths):
                try:
                    if file_path.suffix != ".feather":
                        raise ValueError(
                            f"Only .feather files are supported. Got: {file_path.suffix}"
                        )

                    try:
                        file_size_mb = (
                            file_path.stat().st_size
                            / (1024 * 1024)
                        )
                        self.logger.debug(
                            "Loading file %d/%d: %s (%.1f MB)",
                            idx + 1,
                            len(self.file_paths),
                            file_path.name,
                            file_size_mb,
                        )

                        t0 = time.perf_counter()
                        df = self._load_feather(file_path)
                        array_length = len(df)
                        t_load = time.perf_counter()
                        self.logger.debug(
                            "Loaded %d rows in %.1fs",
                            array_length,
                            t_load - t0,
                        )

                        if (
                            self._use_columnar
                            and columnar_converter is not None
                        ):
                            columnar_batch = columnar_converter(
                                df
                            )
                            del df
                            t_convert = time.perf_counter()
                            self.logger.debug(
                                "Converted to columnar batch in %.1fs",
                                t_convert - t_load,
                            )

                            self._file_entries.append(
                                FileDataSource.FileManager._FileEntry(
                                    name=file_path.name,
                                    path=file_path,
                                    dtype=np.dtype("uint8"),
                                    length=array_length,
                                    memmap=None,
                                    cached_array=None,
                                    cached_columnar=columnar_batch,
                                )
                            )
                        elif numpy_converter is not None:
                            numpy_array = numpy_converter(df)
                            del df
                            t_convert = time.perf_counter()
                            self.logger.debug(
                                "Converted to numpy array in %.1fs",
                                t_convert - t_load,
                            )

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
                        cumulative_rows += array_length

                        if (
                            idx % milestone_interval == 0
                            or idx == n - 1
                        ):
                            self.logger.info(
                                "Progress: %d/%d files, "
                                "%d rows loaded, "
                                "%.1fs elapsed",
                                idx + 1,
                                n,
                                cumulative_rows,
                                time.perf_counter()
                                - t_init_start,
                            )

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

            # cache_mode="memory"の場合、全ファイルを単一配列に結合
            if (
                self.cache_mode == "memory"
                and self.total_pages > 1
            ):
                if self._use_columnar:
                    self._concatenate_columnar()
                else:
                    self._concatenate_numpy()

            self.logger.info(
                "FileManager initialized: %d rows from %d files in %.1fs",
                self.total_rows,
                self.total_pages,
                time.perf_counter() - t_init_start,
            )

        def _concatenate_numpy(self) -> None:
            """structured arrayを単一配列に結合する(hcpe用)."""
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
                "Concatenating %d numpy arrays (%d records)...",
                self.total_pages,
                self.total_rows,
            )

            t0 = time.perf_counter()
            arrays = [
                entry.cached_array
                for entry in self._file_entries
                if entry.cached_array is not None
            ]
            self._concatenated_array = np.concatenate(arrays)
            elapsed = time.perf_counter() - t0
            self.logger.info(
                "Concatenation completed in %.1fs",
                elapsed,
            )
            for entry in self._file_entries:
                entry.cached_array = None

        def _concatenate_columnar(self) -> None:
            """ColumnarBatchをフィールドごとに連結する(SOA化)."""
            batches = [
                entry.cached_columnar
                for entry in self._file_entries
                if entry.cached_columnar is not None
            ]

            total_bytes = sum(
                b.board_positions.nbytes
                + b.pieces_in_hand.nbytes
                + (
                    b.move_label.nbytes
                    if b.move_label is not None
                    else 0
                )
                + (
                    b.result_value.nbytes
                    if b.result_value is not None
                    else 0
                )
                + (
                    b.reachable_squares.nbytes
                    if b.reachable_squares is not None
                    else 0
                )
                + (
                    b.legal_moves_label.nbytes
                    if b.legal_moves_label is not None
                    else 0
                )
                for b in batches
            )
            estimated_gb = total_bytes / (1024**3)
            if estimated_gb > 32:
                self.logger.warning(
                    f"cache_mode='memory' with {self.total_rows} rows "
                    f"(estimated {estimated_gb:.1f} GB) may cause OOM. "
                    f"Consider using cache_mode='file' instead."
                )

            self.logger.info(
                "Concatenating %d columnar batches "
                "(%d records) field-by-field...",
                self.total_pages,
                self.total_rows,
            )

            t0 = time.perf_counter()
            self._concatenated_columnar = (
                ColumnarBatch.concatenate(batches)
            )
            elapsed = time.perf_counter() - t0
            self.logger.info(
                "Concatenation completed in %.1fs",
                elapsed,
            )
            for entry in self._file_entries:
                entry.cached_columnar = None

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
            # 従来のstructured array path (hcpe)
            if self._concatenated_array is not None:
                return self._concatenated_array[idx]

            # SOA化 path (preprocessing/stage1/stage2)
            if self._concatenated_columnar is not None:
                return self._columnar_to_structured_record(
                    self._concatenated_columnar, idx
                )

            file_idx = np.searchsorted(
                self.cum_lengths[1:], idx, side="right"
            )
            local_idx = idx - self.cum_lengths[file_idx]
            entry = self._file_entries[file_idx]

            if self._use_columnar:
                if entry.cached_columnar is None:
                    raise RuntimeError(
                        f"Columnar batch not loaded for file {entry.name}"
                    )
                return self._columnar_to_structured_record(
                    entry.cached_columnar, local_idx
                )
            else:
                if entry.cached_array is None:
                    raise RuntimeError(
                        f"Array not loaded for file {entry.name}"
                    )
                return entry.cached_array[local_idx]

        def _columnar_to_structured_record(
            self, batch: ColumnarBatch, idx: int
        ) -> np.ndarray:
            """ColumnarBatchの1レコードをstructured arrayに変換する．

            DataSource ABCの互換性を維持するため，外部I/Fは変更せず
            内部SOA表現からstructured arrayを再構築する．

            Args:
                batch: ColumnarBatch
                idx: レコードインデックス

            Returns:
                numpy structured array (single record)
            """
            assert self._structured_dtype is not None
            assert self._structured_dtype.names is not None
            dtype_names = self._structured_dtype.names
            record = np.empty(1, dtype=self._structured_dtype)

            # id field (preprocessing/stage1/stage2 dtypeに存在)
            if "id" in dtype_names:
                record["id"][0] = (
                    0  # ColumnarBatchにはid情報がない
                )

            record["boardIdPositions"][0] = (
                batch.board_positions[idx]
            )
            record["piecesInHand"][0] = batch.pieces_in_hand[
                idx
            ]

            if (
                batch.move_label is not None
                and "moveLabel" in dtype_names
            ):
                record["moveLabel"][0] = batch.move_label[idx]

            if (
                batch.result_value is not None
                and "resultValue" in dtype_names
            ):
                record["resultValue"][0] = batch.result_value[
                    idx
                ]

            if (
                batch.reachable_squares is not None
                and "reachableSquares" in dtype_names
            ):
                record["reachableSquares"][0] = (
                    batch.reachable_squares[idx]
                )

            if (
                batch.legal_moves_label is not None
                and "legalMovesLabel" in dtype_names
            ):
                record["legalMovesLabel"][0] = (
                    batch.legal_moves_label[idx]
                )

            return record[0]

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
            SOA化されたデータはstructured arrayに変換して返す．

            Yields:
                Tuple of (filename, numpy array)
            """
            # If concatenated (hcpe path), yield as single batch
            if self._concatenated_array is not None:
                yield ("concatenated", self._concatenated_array)
                return

            # If SOA concatenated, convert to structured array
            if self._concatenated_columnar is not None:
                yield (
                    "concatenated",
                    self._columnar_batch_to_structured_array(
                        self._concatenated_columnar
                    ),
                )
                return

            # Iterate over individual files
            for entry in self._file_entries:
                if self._use_columnar:
                    if entry.cached_columnar is None:
                        continue
                    yield (
                        entry.name,
                        self._columnar_batch_to_structured_array(
                            entry.cached_columnar
                        ),
                    )
                else:
                    if entry.cached_array is None:
                        continue
                    yield (entry.name, entry.cached_array)

        def _columnar_batch_to_structured_array(
            self, batch: ColumnarBatch
        ) -> np.ndarray:
            """ColumnarBatch全体をstructured arrayに変換する．

            iter_batchesなど，structured array形式が必要な箇所で使用．

            Args:
                batch: ColumnarBatch

            Returns:
                numpy structured array
            """
            assert self._structured_dtype is not None
            assert self._structured_dtype.names is not None
            dtype_names = self._structured_dtype.names
            n = len(batch)
            array = np.empty(n, dtype=self._structured_dtype)

            if "id" in dtype_names:
                array["id"] = np.zeros(n, dtype=np.uint64)

            array["boardIdPositions"] = batch.board_positions
            array["piecesInHand"] = batch.pieces_in_hand

            if (
                batch.move_label is not None
                and "moveLabel" in dtype_names
            ):
                array["moveLabel"] = batch.move_label

            if (
                batch.result_value is not None
                and "resultValue" in dtype_names
            ):
                array["resultValue"] = batch.result_value

            if (
                batch.reachable_squares is not None
                and "reachableSquares" in dtype_names
            ):
                array["reachableSquares"] = (
                    batch.reachable_squares
                )

            if (
                batch.legal_moves_label is not None
                and "legalMovesLabel" in dtype_names
            ):
                array["legalMovesLabel"] = (
                    batch.legal_moves_label
                )

            return array

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
                "Install with: uv add polars"
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
