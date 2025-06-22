import logging
import random
from collections import defaultdict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import google.cloud.storage as storage
import numpy as np
from google.cloud.exceptions import NotFound
from tqdm.auto import tqdm

from maou.interface import learn, preprocess
from maou.interface.data_io import load_array


class MissingGCSConfig(Exception):
    pass


class GCSDataSource(learn.LearningDataSource, preprocess.DataSource):
    """
    GCSバケットからデータを効率的にダウンロードし，学習・前処理用データソースとして提供する．

    パフォーマンス最適化:
    - 並列ダウンロードによる高速データ転送
    - ローカルキャッシュによる重複ダウンロード回避
    - 100KB程度の小ファイル向けに最適化された並列処理設定
    - デフォルト16並列での高速データ転送

    Note:
        大量の小ファイル (1M件，100KB/file程度)でのダウンロード時間を大幅短縮．
        gsutil並列コピーに匹敵するパフォーマンスを実現．
    """

    logger: logging.Logger = logging.getLogger(__name__)

    class GCSDataSourceSpliter(learn.LearningDataSource.DataSourceSpliter):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            *,
            bucket_name: str,
            prefix: str,
            data_name: str,
            local_cache_dir: str,
            max_workers: int = 8,
            sample_ratio: Optional[float] = None,
        ) -> None:
            self.__page_manager = GCSDataSource.PageManager(
                bucket_name=bucket_name,
                prefix=prefix,
                data_name=data_name,
                local_cache_dir=local_cache_dir,
                max_workers=max_workers,
                sample_ratio=sample_ratio,
            )

        def train_test_split(
            self, test_ratio: float
        ) -> tuple["GCSDataSource", "GCSDataSource"]:
            self.logger.info(f"test_ratio: {test_ratio}")
            # 初期化を強制してtotal_rowsを取得
            self.__page_manager._ensure_initialized()
            input_indices, test_indicies = self.__train_test_split(
                data=list(range(self.__page_manager.total_rows)),
                test_ratio=test_ratio,
            )
            return (
                GCSDataSource(
                    page_manager=self.__page_manager,
                    indicies=input_indices,
                ),
                GCSDataSource(
                    page_manager=self.__page_manager,
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
            split_idx = int(float(len(data)) * (1 - test_ratio))
            return data[:split_idx], data[split_idx:]

    class PageManager:
        logger: logging.Logger = logging.getLogger(__name__)

        @dataclass
        class PruningInfo:
            start_idx: int = 0
            count: int = 0
            files: list[tuple[Path, int, int]] = field(default_factory=list)

            def append(self, file: Path, count: int) -> None:
                self.files.append((file, self.start_idx + self.count, count))
                self.count += count

        def __init__(
            self,
            *,
            bucket_name: str,
            prefix: str,
            data_name: str,
            local_cache_dir: str,
            max_workers: int = 16,
            sample_ratio: Optional[float] = None,
        ) -> None:
            # GCSクライアントは遅延初期化
            self._client = None
            self._bucket = None

            self.bucket_name = bucket_name
            self.prefix = prefix
            self.data_name = data_name
            self.max_workers = max_workers
            self.sample_ratio = (
                max(0.01, min(1.0, sample_ratio)) if sample_ratio is not None else None
            )
            if local_cache_dir is None:
                raise ValueError("local_cache_dir must be specified")
            self.local_cache_dir = Path(local_cache_dir)
            self.__pruning_info: dict[str, GCSDataSource.PageManager.PruningInfo] = (
                defaultdict(GCSDataSource.PageManager.PruningInfo)
            )
            self._initialized = False

        @property
        def client(self) -> Any:
            """GCSクライアントを遅延初期化で取得"""
            if self._client is None:
                self._client = storage.Client()
            return self._client

        @property
        def bucket(self) -> Any:
            """GCSバケットを遅延初期化で取得"""
            if self._bucket is None:
                try:
                    self._bucket = self.client.get_bucket(self.bucket_name)
                except NotFound:
                    raise ValueError(f"Bucket '{self.bucket_name}' does not exist.")
            return self._bucket

        def _ensure_initialized(self) -> None:
            """データが初期化されていることを確認し，必要に応じて初期化を実行"""
            if self._initialized:
                return

            # ローカルキャッシュの設定
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Local cache directory: {self.local_cache_dir}")

            # バケットが存在するか確認（プロパティアクセスで初期化）
            _ = self.bucket

            # 初期化時にすべてのデータをダウンロード
            self.file_paths = self.__download_all_to_local()

            # ダウンロードしたデータから__pruning_infoを作成する
            total_rows = 0
            # ディレクトリごとに処理されるようにソートする
            self.file_paths.sort()
            last_pruning_value = None
            for file in self.file_paths:
                data = load_array(file, mmap_mode="r", array_type="hcpe")
                num_rows = data.shape[0]
                # ファイルの親ディレクトリでプルーニングする
                pruning_value = file.parent.absolute().name
                if last_pruning_value != pruning_value:
                    self.__pruning_info[pruning_value].start_idx = total_rows
                    last_pruning_value = pruning_value
                self.__pruning_info[pruning_value].append(file=file, count=num_rows)
                total_rows += num_rows
            self.total_rows = total_rows
            self.total_pages = len(self.__pruning_info)

            self.logger.debug(self.__pruning_info)

            self.logger.info(
                f"GCS Data {self.total_rows} rows, {self.total_pages} pages"
            )

            # データダウンロード完了後，クライアントを破棄してPickle化の問題を回避
            self._cleanup_clients()
            self._initialized = True

        def _cleanup_clients(self) -> None:
            """GCSクライアントを破棄してDataLoader並列時のPickle化の問題を回避"""
            if self._client is not None:
                self.logger.debug("Cleaning up GCS client to enable pickling")
                self._client = None
            if self._bucket is not None:
                self.logger.debug("Cleaning up GCS bucket to enable pickling")
                self._bucket = None

        def __get_data_path(self) -> str:
            """データ名からGCSのパスを取得する."""
            return f"{self.prefix}/{self.data_name}"

        def __download_all_to_local(self) -> list[Path]:
            """すべてのデータまたはサンプルデータをローカルにダウンロードする"""
            if self.sample_ratio is not None:
                self.logger.info(
                    f"Downloading sample data ({self.sample_ratio:.1%}) "
                    f"to local cache: {self.local_cache_dir}"
                )
                return self.__download_sample_to_local()
            else:
                self.logger.info(
                    f"Downloading all data to local cache: {self.local_cache_dir}"
                )
                # gcs sync
                self.__sync_gcs_to_local(
                    bucket_name=self.bucket_name,
                    prefix=self.__get_data_path(),
                    local_path=self.local_cache_dir,
                    delete=True,
                )

            # ローカルキャッシュファイルが正しく作成されたか確認
            cache_files = list(self.local_cache_dir.glob("**/*.npy"))
            self.logger.info(f"Created {len(cache_files)} local cache files")

            if len(cache_files) == 0:
                self.logger.warning(
                    "No local cache files were created. This might indicate a problem."
                )

            return cache_files

        def __sync_gcs_to_local(
            self,
            *,
            bucket_name: str,
            prefix: str,
            local_path: Path,
            delete: bool = False,
        ) -> None:
            """GCSバケットからローカルディレクトリへ高速同期を行う.

            並列ダウンロードを使用して高速ダウンロードを実現．
            ローカルに存在しないファイルやGCS側が更新されているファイルのみをダウンロードする．

            Performance:
                - 指定された並列数での高速ダウンロード
                - 小ファイル(100KB程度)向けに最適化

            Args:
                bucket_name (str): GCSバケット名
                prefix (str): GCSバケット内のプレフィックス (フォルダパス)
                local_path (Path): ローカルの保存先ディレクトリパス
                delete (bool): GCSに存在しないローカルファイルを削除するかどうか
            """
            if not local_path.exists():
                raise ValueError("local_cache_dir does not exists")

            # prefixの末尾にスラッシュがない場合は追加
            if not prefix.endswith("/"):
                prefix = prefix + "/"

            try:
                # GCSのlist_blobsを使用してオブジェクトを取得
                blobs = list(self.bucket.list_blobs(prefix=prefix))

                # ダウンロードタスクを事前に収集
                download_tasks = []
                gcs_local_files: set[Path] = set()

                for blob in tqdm(blobs, desc="Analyzing GCS objects", leave=False):
                    if blob.name == prefix:
                        # prefix自体はスキップ
                        continue

                    local_file_path = local_path / blob.name[len(prefix) :]
                    self.logger.debug(local_path)
                    self.logger.debug(blob.name[len(prefix) :])
                    self.logger.debug(local_file_path.absolute())
                    gcs_local_files.add(local_file_path)

                    # ダウンロードするかどうかを判定する
                    download_file = False
                    if not local_file_path.exists():
                        download_file = True
                    else:
                        local_mtime = local_file_path.stat().st_mtime
                        gcs_mtime = blob.time_created.timestamp()
                        if gcs_mtime > local_mtime:
                            download_file = True

                    if download_file:
                        download_tasks.append((blob.name, local_file_path))

                # 並列ダウンロード実行
                download_count = 0
                skip_count = len(gcs_local_files) - len(download_tasks)

                if download_tasks:
                    # 概算サイズ (1.5GB/file)
                    total_size_mb = float(len(download_tasks)) * 1.5
                    desc = (
                        f"Downloading {len(download_tasks)} files "
                        f"(~{total_size_mb:.1f}GB) [{self.max_workers} workers]"
                    )

                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = []
                        for gcs_key, local_file_path in download_tasks:
                            future = executor.submit(
                                self._download_single_file,
                                bucket_name,
                                gcs_key,
                                local_file_path,
                            )
                            futures.append(future)

                        # 完了を待つ
                        for future in tqdm(
                            futures, desc=desc, unit="files", leave=True
                        ):
                            future.result()
                            download_count += 1

                delete_count = 0
                if delete:
                    # まずはファイルを削除してそのあと空のディレクトリがあれば削除する
                    gcs_local_paths = {str(file.resolve()) for file in gcs_local_files}
                    for local_file in local_path.glob("**/*"):
                        try:
                            if (
                                local_file.is_file()
                                and str(local_file.resolve()) not in gcs_local_paths
                            ):
                                local_file.unlink()
                                delete_count += 1
                                self.logger.debug(f"Deleted {local_file}")
                        except FileNotFoundError:
                            continue
                    # 空のディレクトリを削除
                    for local_dir in local_path.glob("**"):
                        try:
                            if local_dir.is_dir() and not any(local_dir.iterdir()):
                                local_dir.rmdir()
                                self.logger.debug(
                                    f"Deleted empty directory {local_dir}"
                                )
                        except FileNotFoundError:
                            continue
                result_message = (
                    f"同期完了: {download_count}ファイルをダウンロード，"
                    f"{skip_count}ファイルをスキップ"
                )
                if delete:
                    result_message += f"，{delete_count}ファイルを削除"
                self.logger.info(result_message)

            except Exception as e:
                self.logger.error(
                    "Error sync gcs object from "
                    f"'gs://{bucket_name}/{prefix}' to '{local_path}': {e}"
                )
                raise

        def __download_sample_to_local(self) -> list[Path]:
            """指定された割合のファイルのみローカルにダウンロード"""
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)

            try:
                bucket = self.client.bucket(self.bucket_name)
                prefix = self.__get_data_path()
                if not prefix.endswith("/"):
                    prefix = prefix + "/"

                # ファイル一覧を取得
                all_files = []
                for blob in bucket.list_blobs(prefix=prefix):
                    if blob.name == prefix:
                        continue  # プレフィックス自体はスキップ
                    if blob.name.endswith(".npy"):
                        all_files.append(blob.name)

                if not all_files:
                    raise ValueError(
                        f"No .npy files found in gs://{self.bucket_name}/{prefix}"
                    )

                # ファイルをサンプリング
                if self.sample_ratio is not None:
                    sample_count = max(
                        1, int(float(len(all_files)) * self.sample_ratio)
                    )
                    sampled_files = random.sample(
                        all_files, min(sample_count, len(all_files))
                    )
                else:
                    sampled_files = all_files

                self.logger.info(
                    f"Sampling {len(sampled_files)} files out of "
                    f"{len(all_files)} total files"
                )

                # ダウンロード実行
                downloaded_files = []
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for blob_name in sampled_files:
                        local_file_path = (
                            self.local_cache_dir / blob_name[len(prefix) :]
                        )
                        future = executor.submit(
                            self._download_single_file,
                            bucket,
                            blob_name,
                            local_file_path,
                        )
                        futures.append((future, local_file_path))

                    for future, local_file_path in tqdm(
                        futures, desc="Downloading sample files", unit="files"
                    ):
                        future.result()
                        downloaded_files.append(local_file_path)

                return downloaded_files

            except Exception as e:
                self.logger.error(f"Error downloading sample data: {e}")
                raise

        def _download_single_file(
            self, bucket: Any, blob_name: str, local_file_path: Path
        ) -> None:
            """単一ファイルをダウンロード"""
            try:
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Downloading {blob_name} to {local_file_path}")
                blob = bucket.blob(blob_name)
                blob.download_to_filename(str(local_file_path.absolute()))
            except Exception as e:
                self.logger.error(f"Failed to download {blob_name}: {e}")
                raise

        def get_page(self, key: str) -> np.ndarray:
            """指定されたキーのページを取得する."""
            self._ensure_initialized()
            if key not in self.__pruning_info:
                raise ValueError(f"Key {key} not found in pruning info.")

            file_paths = [path for (path, _, _) in self.__pruning_info[key].files]
            file_paths.sort()
            data = np.concatenate(
                [
                    load_array(path, mmap_mode="r", array_type="hcpe")
                    for path in file_paths
                ]
            )
            return data

        def get_item(self, idx: int) -> np.ndarray:
            """特定のレコードをnumpy structured arrayとして返す."""
            self._ensure_initialized()
            for pruning_info in self.__pruning_info.values():
                if (
                    pruning_info.start_idx
                    <= idx
                    < pruning_info.start_idx + pruning_info.count
                ):
                    for file, start_idx, num_rows in pruning_info.files:
                        if start_idx <= idx < start_idx + num_rows:
                            relative_idx = idx - start_idx
                            # numpy structured arrayから直接レコードを取得
                            npy_data = load_array(
                                file, mmap_mode="r", array_type="hcpe"
                            )
                            return npy_data[relative_idx]
            raise IndexError(f"Index {idx} out of range.")

        def iter_batches(self) -> Generator[tuple[str, np.ndarray], None, None]:
            """
            GCS のバケット全体に対して，
            ページ 単位のNumpy Structured Arrayを順次取得するジェネレータ．
            """
            self._ensure_initialized()
            for key in self.__pruning_info.keys():
                name = key
                yield name, self.get_page(key)

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
        """
        if page_manager is None:
            if (
                bucket_name is not None
                and prefix is not None
                and data_name is not None
                and local_cache_dir is not None
            ):
                self.__page_manager = self.PageManager(
                    bucket_name=bucket_name,
                    prefix=prefix,
                    data_name=data_name,
                    local_cache_dir=local_cache_dir,
                    max_workers=max_workers,
                    sample_ratio=sample_ratio,
                )
            else:
                raise MissingGCSConfig(
                    f"GCSの設定が不足しています bucket_name: {bucket_name}, prefix: {prefix}"
                )
        else:
            self.__page_manager = page_manager

        if indicies is None:
            # 初期化を強制してtotal_rowsを取得
            self.__page_manager._ensure_initialized()
            self.indicies = list(range(self.__page_manager.total_rows))
        else:
            self.indicies = indicies

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.indicies):
            raise IndexError(f"Index {idx} out of range.")

        return self.__page_manager.get_item(self.indicies[idx])

    def __len__(self) -> int:
        return len(self.indicies)

    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """
        GCS のバケット全体に対して，
        ページ 単位のNumpy Structured Arrayを順次取得するジェネレータ．
        """
        for name, batch in self.__page_manager.iter_batches():
            yield name, batch
