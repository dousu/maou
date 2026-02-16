from __future__ import annotations

import logging
import random
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from tqdm.auto import tqdm

import maou.interface.learn as learn
import maou.interface.preprocess as preprocess

if TYPE_CHECKING:
    import polars as pl


class MissingObjectStorageConfig(Exception):
    pass


class ObjectStorageDataSource(
    learn.LearningDataSource, preprocess.DataSource
):
    logger: logging.Logger = logging.getLogger(__name__)

    class DataSourceSpliter(
        learn.LearningDataSource.DataSourceSpliter
    ):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            *,
            cls_ref: type["ObjectStorageDataSource"],
            bucket_name: str,
            prefix: str,
            data_name: str,
            local_cache_dir: str,
            array_type: Literal["hcpe", "preprocessing"],
            max_workers: int = 8,
            max_cached_bytes: int = 100 * 1024 * 1024,
            sample_ratio: Optional[float] = None,
            enable_bundling: bool = True,
            bundle_size_gb: float = 1.0,
        ) -> None:
            self.__page_manager = cls_ref.PageManager(
                bucket_name=bucket_name,
                prefix=prefix,
                data_name=data_name,
                local_cache_dir=local_cache_dir,
                max_workers=max_workers,
                max_cached_bytes=max_cached_bytes,
                sample_ratio=sample_ratio,
                array_type=array_type,
                enable_bundling=enable_bundling,
                bundle_size_gb=bundle_size_gb,
            )

        def train_test_split(
            self, test_ratio: float
        ) -> tuple[
            "ObjectStorageDataSource", "ObjectStorageDataSource"
        ]:
            self.logger.info(f"test_ratio: {test_ratio}")
            input_indices, test_indicies = (
                self.__train_test_split(
                    data=list(
                        range(self.__page_manager.total_rows)
                    ),
                    test_ratio=test_ratio,
                )
            )
            return (
                ObjectStorageDataSource(
                    page_manager=self.__page_manager,
                    indicies=input_indices,
                ),
                ObjectStorageDataSource(
                    page_manager=self.__page_manager,
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

    class PageManager:
        logger: logging.Logger = logging.getLogger(__name__)
        array_type: Literal["hcpe", "preprocessing"]
        bundle_cache: list[np.ndarray]
        bundle_id: int

        def __init__(
            self,
            *,
            bucket_name: str,
            prefix: str,
            data_name: str,
            local_cache_dir: str,
            array_type: Literal["hcpe", "preprocessing"],
            max_workers: int = 8,
            max_cached_bytes: int = 100 * 1024 * 1024,
            sample_ratio: Optional[float] = None,
            enable_bundling: bool = False,
            bundle_size_gb: float = 1.0,
        ) -> None:
            self.bucket_name = bucket_name
            self.prefix = prefix
            self.data_name = data_name
            self.max_workers = max_workers
            self.max_cached_bytes = max_cached_bytes
            self.array_type = array_type
            self.sample_ratio = (
                max(0.01, min(1.0, sample_ratio))
                if sample_ratio is not None
                else None
            )
            if local_cache_dir is None:
                raise ValueError(
                    "local_cache_dir must be specified"
                )
            self.local_cache_dir = Path(local_cache_dir)
            self.file_paths: list[Path] = []
            self.memmap_arrays: list[
                tuple[str, np.ndarray]
            ] = []

            # ローカルキャッシュの設定
            data_path = self.local_cache_dir / data_name
            data_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"Local cache directory: {data_path}"
            )

            self.bundle_cache = []
            self.bundle_id = 1

            # 初期化時にデータをダウンロード
            self.__download_all_to_local(
                enable_bundling=enable_bundling,
                bundle_size_gb=bundle_size_gb,
            )

            # Load all .feather files as DataFrames
            lengths = []
            for file_path in self.file_paths:
                try:
                    if file_path.suffix != ".feather":
                        raise ValueError(
                            f"Only .feather files are supported. Got: {file_path.suffix}"
                        )

                    from maou.domain.data.rust_io import (
                        load_hcpe_df,
                        load_preprocessing_df,
                    )

                    if self.array_type == "hcpe":
                        df = load_hcpe_df(file_path)
                    elif self.array_type == "preprocessing":
                        df = load_preprocessing_df(file_path)
                    else:
                        raise ValueError(
                            f"Unsupported array_type: {self.array_type}"
                        )

                    self.memmap_arrays.append(
                        (file_path.name, df)
                    )  # type: ignore
                    lengths.append(len(df))
                except Exception as e:
                    self.logger.error(
                        f"Failed to load DataFrame {file_path}: {e}"
                    )
                    raise
            self.cum_lengths = np.cumsum([0] + lengths)

            self.total_rows = self.cum_lengths[-1]
            self.total_pages = len(self.cum_lengths) - 1

            self.logger.info(
                f"bucket {self.bucket_name}, {self.total_rows} rows, {self.total_pages} pages"
            )

        def __get_data_path_prefix(self) -> str:
            """データ名からGCSのパスprefixを取得する."""
            return f"{self.prefix}/{self.data_name}/"

        def __get_local_data_path(self) -> Path:
            """データ名からローカルのパスを取得する."""
            return self.local_cache_dir / self.data_name

        def __download_all_to_local(
            self,
            *,
            enable_bundling: bool = False,
            bundle_size_gb: float = 1.0,
        ) -> None:
            """すべてのデータまたはサンプルデータをローカルにダウンロードする．

            Note: enable_bundling and bundle_size_gb are ignored for .feather files
            (kept for API compatibility but not used)
            """
            try:
                if self.sample_ratio is not None:
                    self.logger.info(
                        f"Downloading sample data ({self.sample_ratio:.1%}) "
                        f"to local cache: {self.local_cache_dir}"
                    )
                    all_objects = type(self).list_objects(
                        bucket_name=self.bucket_name,
                        data_path=self.__get_data_path_prefix(),
                    )
                    if self.sample_ratio is not None:
                        sample_count = max(
                            1,
                            int(
                                float(len(all_objects))
                                * self.sample_ratio
                            ),
                        )
                        objects = random.sample(
                            all_objects,
                            min(sample_count, len(all_objects)),
                        )
                    else:
                        objects = all_objects

                    self.logger.info(
                        f"Sampling {len(objects)} files out of "
                        f"{len(all_objects)} total files"
                    )
                else:
                    self.logger.info(
                        f"Downloading all data to local cache: {self.local_cache_dir}"
                    )
                    # list_blobsを使用してオブジェクト一覧を取得
                    objects = type(self).list_objects(  # type: ignore
                        bucket_name=self.bucket_name,
                        data_path=self.__get_data_path_prefix(),
                    )

                chunks: list[dict] = []
                current_chunk: list[str] = []
                current_sum = 0

                # Split data into chunks
                for name, size in objects:
                    if (
                        current_sum + size
                        > int(
                            self.max_cached_bytes
                            / self.max_workers
                        )
                        and current_chunk
                    ):
                        chunks.append(
                            {
                                "bucket_name": self.bucket_name,
                                "object_paths": current_chunk,
                            }
                        )
                        current_chunk.clear()
                        current_sum = 0
                    current_chunk.append(name)
                    current_sum += size
                if current_chunk:
                    chunks.append(
                        {
                            "bucket_name": self.bucket_name,
                            "object_paths": current_chunk,
                        }
                    )

                # Download files in parallel
                self.file_paths = []
                local_data_path = self.__get_local_data_path()
                local_data_path.mkdir(
                    parents=True, exist_ok=True
                )

                with ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    # Submit all download tasks
                    future_to_chunk = {
                        executor.submit(
                            type(self).download_files,
                            str(chunk["bucket_name"]),
                            list(chunk["object_paths"]),
                        ): chunk
                        for chunk in chunks
                    }

                    # Process downloads as they complete
                    for future in tqdm(
                        as_completed(future_to_chunk),
                        desc="Downloading",
                        total=len(chunks),
                    ):
                        try:
                            chunk = future_to_chunk[future]
                            byte_list = future.result()

                            # Save .feather files directly (no bundling)
                            for i, byte_data in enumerate(
                                byte_list
                            ):
                                object_path = chunk[
                                    "object_paths"
                                ][i]
                                feather_name = Path(
                                    object_path
                                ).name
                                feather_path = (
                                    local_data_path
                                    / feather_name
                                )

                                feather_path.write_bytes(
                                    byte_data
                                )
                                self.file_paths.append(
                                    feather_path
                                )

                                self.logger.debug(
                                    f"Saved .feather file: {feather_path}"
                                )

                        except Exception as exc:
                            self.logger.error(
                                f"Chunk processing failed: {exc}"
                            )
                            raise
            except Exception as e:
                self.logger.error(
                    f"Error downloading all data: {e}"
                )
                raise
            finally:
                pass

            self.logger.info(
                f"Created {len(self.file_paths)} local cache files"
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
            return self.memmap_arrays[file_idx][1][relative_idx]

        def iter_batches(
            self,
        ) -> Generator[tuple[str, np.ndarray], None, None]:
            """
            バケット全体に対して，
            ページ 単位のNumpy Structured Arrayを順次取得するジェネレータ．
            """
            for name, array in self.memmap_arrays:
                yield name, array

        @staticmethod
        def list_objects(
            bucket_name: str, data_path: str
        ) -> list[tuple[str, int]]:
            raise Exception("list_objectsが未実装")

        @staticmethod
        def download_files(
            bucket_name: str, object_paths: list[str]
        ) -> list[bytes]:
            raise Exception("download_filesが未実装")

    def __init__(
        self,
        *,
        bucket_name: Optional[str] = None,
        prefix: Optional[str] = None,
        data_name: Optional[str] = None,
        page_manager: Optional[PageManager] = None,
        indicies: Optional[Union[list[int], np.ndarray]] = None,
        local_cache_dir: Optional[str] = None,
        max_workers: int = 8,
        max_cached_bytes: int = 100 * 1024 * 1024,
        sample_ratio: Optional[float] = None,
        array_type: Optional[
            Literal["hcpe", "preprocessing"]
        ] = None,
        enable_bundling: bool = False,
        bundle_size_gb: float = 1.0,
    ) -> None:
        """
        Args:
            bucket_name (Optional[str]): GCSバケット名
            prefix (Optional[str]): GCSオブジェクトのプレフィックス
            data_name (Optional[str]): データ名
            page_manager (Optional[PageManager]): PageManager
            indicies (Optional[list[int]]): 選択可能なインデックスのリスト
            local_cache_dir (Optional[str]): ローカルキャッシュディレクトリのパス
            sample_ratio (Optional[float]): サンプリング割合 (0.01-1.0, None=全データ)
            max_workers (int): 並列ダウンロード数 (デフォルト: 8)
            max_cached_bytes (int):
              キャッシュの上限サイズ (バイト単位，デフォルト100MB)
            array_type (str): 配列タイプ ("hcpe" or "preprocessing")
            enable_bundling (bool): バンドリング機能を有効にするかどうか (デフォルト: False)
            bundle_size_gb (float): バンドルサイズ (GB) (デフォルト: 1.0)
        """
        if page_manager is None:
            if (
                bucket_name is not None
                and prefix is not None
                and data_name is not None
                and local_cache_dir is not None
                and array_type is not None
            ):
                self.__page_manager = self.PageManager(
                    bucket_name=bucket_name,
                    prefix=prefix,
                    data_name=data_name,
                    local_cache_dir=local_cache_dir,
                    max_workers=max_workers,
                    max_cached_bytes=max_cached_bytes,
                    sample_ratio=sample_ratio,
                    array_type=array_type,
                    enable_bundling=enable_bundling,
                    bundle_size_gb=bundle_size_gb,
                )
            else:
                raise MissingObjectStorageConfig(
                    f"オブジェクトストレージの設定が不足しています "
                    f"bucket_name: {bucket_name}, "
                    f"prefix: {prefix}, "
                    f"data_name: {data_name}, "
                    f"local_cache_dir: {local_cache_dir}, "
                    f"array_type: {array_type}"
                )
        else:
            self.__page_manager = page_manager

        if indicies is None:
            # 初期化を強制してtotal_rowsを取得
            self.indicies: np.ndarray = np.arange(
                self.__page_manager.total_rows, dtype=np.int64
            )
        else:
            self.indicies = np.asarray(indicies, dtype=np.int64)

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.indicies):
            raise IndexError(f"Index {idx} out of range.")

        return self.__page_manager.get_item(self.indicies[idx])

    def __len__(self) -> int:
        return len(self.indicies)

    def get_file_paths(self) -> list[Path]:
        """ローカルキャッシュのファイルパスを取得する"""
        return self.__page_manager.file_paths

    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """
        GCS のバケット全体に対して，
        ページ 単位のNumpy Structured Arrayを順次取得するジェネレータ．
        """
        for name, batch in self.__page_manager.iter_batches():
            yield name, batch

    def iter_batches_df(
        self,
    ) -> Generator[tuple[str, "pl.DataFrame"], None, None]:
        """Iterate over batches as Polars DataFrames．

        Converts numpy arrays from cloud storage to DataFrames．

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

        from maou.domain.data.schema import (
            get_hcpe_polars_schema,
            get_preprocessing_polars_schema,
        )

        schema = (
            get_hcpe_polars_schema()
            if self.__page_manager.array_type == "hcpe"
            else get_preprocessing_polars_schema()
        )

        # Convert numpy arrays to DataFrames
        for name, array in self.__page_manager.iter_batches():
            data = {}
            assert array.dtype.names is not None
            assert array.dtype.fields is not None
            for field in array.dtype.names:
                field_data = array[field]
                field_dtype = array.dtype.fields[field][0]

                # Handle binary fields (convert uint8 arrays to bytes)
                if field == "hcp" or (
                    field_dtype.shape
                    and field_dtype.base == np.dtype("uint8")
                ):
                    # Multi-dimensional uint8 field like hcp - convert to bytes
                    data[field] = [
                        bytes(row)
                        if hasattr(row, "__iter__")
                        else bytes([row])
                        for row in field_data
                    ]
                else:
                    data[field] = field_data.tolist()

            df = pl.DataFrame(data, schema=schema)
            yield name, df

    def total_pages(self) -> int:
        return self.__page_manager.total_pages
