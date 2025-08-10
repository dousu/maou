import base64
import hashlib
import logging
import random
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from maou.interface import learn, preprocess
from maou.interface.data_io import (
    load_array,
    load_array_from_bytes,
    save_array,
)


class MissingObjectStorageConfig(Exception):
    pass


class Bundler:
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        bundle_id: int,
        enable_bundling: bool,
        bundle_size_gb: float,
        array_type: Literal["hcpe", "preprocessing"],
        local_data_path: Path,
    ) -> None:
        self.bundle_id: int = bundle_id
        self.enable_bundling: bool = enable_bundling
        self.bundle_size_gb: float = bundle_size_gb
        self.array_type: Literal["hcpe", "preprocessing"] = (
            array_type
        )
        self.local_data_path: Path = local_data_path
        self.bundle_cache: list[np.ndarray] = []
        self.__bundle_cache_size: int = 0
        self.file_paths: list[Path] = []

    def bundle(self, array: np.ndarray) -> None:
        """ndarrayをキャッシュしてbundleする"""
        self.bundle_cache.append(array)
        self.__bundle_cache_size += array.nbytes
        if not self.enable_bundling or float(
            self.__bundle_cache_size
        ) >= self.bundle_size_gb * float(pow(1024, 3)):
            self.logger.debug(
                f"Bundling {len(self.bundle_cache)} arrays "
                f"({self.__bundle_cache_size / pow(1024, 3):.2f} GB)"
            )
            self.bundle_flush()

    def bundle_flush(self) -> None:
        """残っているファイルをbundleにする"""
        if not self.bundle_cache:
            self.logger.debug(
                "No arrays to bundle, skipping flush."
            )
            return
        self.__bundle_files(self.bundle_cache)
        self.bundle_cache.clear()
        self.__bundle_cache_size = 0

    def __bundle_files(self, arrays: list[np.ndarray]) -> None:
        """バンドルの作成.
        中身を読むときにmemmapで読み込みたいのでbit_pack=Falseで保存する
        """
        bundled_array = np.concatenate(arrays)
        # データが一致していたら同じファイル名にする
        size_mb = bundled_array.nbytes // pow(1024, 2)
        schema_key = str(bundled_array.dtype)
        schema_hash = self.__short_hash(schema_key)
        file_name = f"batch{self.bundle_id}_{len(arrays)}arrays_{size_mb}MB_{schema_hash}.npy"
        file_path = self.local_data_path / file_name
        if file_path.exists():
            self.logger.warning(
                f"Bundle file {file_path} already exists. Skipping save."
            )
            self.file_paths.append(file_path)
            self.bundle_id += 1
            return
        save_array(
            array=bundled_array,
            file_path=file_path,
            array_type=self.array_type,
            bit_pack=False,
        )
        self.file_paths.append(file_path)
        self.bundle_id += 1

    def get_file_paths(self) -> list[Path]:
        return self.file_paths

    def __short_hash(self, text: str, length: int = 4) -> str:
        digest = hashlib.sha256(text.encode()).digest()
        b64 = base64.urlsafe_b64encode(digest).decode()
        # 中央部分を切り出す
        start = (len(b64) - length) // 2
        return b64[start : start + length]


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
            max_workers: int = 16,
            sample_ratio: Optional[float] = None,
            enable_bundling: bool = False,
            bundle_size_gb: float = 1.0,
        ) -> None:
            self.bucket_name = bucket_name
            self.prefix = prefix
            self.data_name = data_name
            self.max_workers = max_workers
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

            # すべてのファイルパスをmemmapで読み込む
            # ここでmemmapで読み込みたいので各ローカルファイルの保存はbit_pack=Falseで行う
            lengths = []
            for file_path in self.file_paths:
                try:
                    array = load_array(
                        file_path,
                        mmap_mode="r",
                        array_type=self.array_type,
                        bit_pack=False,
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
            """すべてのデータまたはサンプルデータをローカルにダウンロードする"""
            try:
                if self.sample_ratio is not None:
                    self.logger.info(
                        f"Downloading sample data ({self.sample_ratio:.1%}) "
                        f"to local cache: {self.local_cache_dir}"
                    )
                    bundler = Bundler(
                        bundle_id=1,
                        enable_bundling=enable_bundling,
                        bundle_size_gb=bundle_size_gb,
                        array_type=self.array_type,
                        local_data_path=self.__get_local_data_path(),
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
                    bundler = Bundler(
                        bundle_id=1,
                        enable_bundling=enable_bundling,
                        bundle_size_gb=bundle_size_gb,
                        array_type=self.array_type,
                        local_data_path=self.__get_local_data_path(),
                    )
                    # list_blobsを使用してオブジェクト一覧を取得
                    objects = type(self).list_objects(  # type: ignore
                        bucket_name=self.bucket_name,
                        data_path=self.__get_data_path_prefix(),
                    )

                chunk_size = max(
                    1, len(objects) // self.max_workers
                )
                chunks = []

                # Split data into chunks
                for i in range(0, len(objects), chunk_size):
                    end_idx = min(i + chunk_size, len(objects))
                    chunks.append(
                        {
                            "bucket_name": self.bucket_name,
                            "object_paths": list(
                                objects[i:end_idx]
                            ),
                        }
                    )

                with ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    futures = [
                        executor.submit(
                            type(self).download_files,
                            str(chunk["bucket_name"]),
                            list(chunk["object_paths"]),
                        )
                        for chunk in chunks
                    ]
                    for future in as_completed(futures):
                        try:
                            byte_list = future.result()
                            for byte_data in byte_list:
                                array = load_array_from_bytes(
                                    data=byte_data,
                                    array_type=self.array_type,
                                    bit_pack=True,
                                )
                                bundler.bundle(array)
                        except Exception as exc:
                            self.logger.error(
                                f"Chunk processing failed: {exc}"
                            )
                            raise
                bundler.bundle_flush()
                self.file_paths = bundler.get_file_paths()
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
            file_idx = (
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
            GCS のバケット全体に対して，
            ページ 単位のNumpy Structured Arrayを順次取得するジェネレータ．
            """
            for name, array in self.memmap_arrays:
                yield name, array

        @staticmethod
        def list_objects(
            bucket_name: str, data_path: str
        ) -> list[str]:
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
        indicies: Optional[list[int]] = None,
        local_cache_dir: Optional[str] = None,
        max_workers: int = 16,
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
            max_workers (int): 並列ダウンロード数 (デフォルト: 16)
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
            self.indicies = list(
                range(self.__page_manager.total_rows)
            )
        else:
            self.indicies = indicies

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
