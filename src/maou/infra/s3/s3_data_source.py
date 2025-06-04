import logging
import random
from collections import defaultdict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import boto3
import numpy as np
from botocore.exceptions import ClientError
from tqdm.auto import tqdm

from maou.interface import learn, preprocess


class MissingS3Config(Exception):
    pass


class S3DataSource(learn.LearningDataSource, preprocess.DataSource):
    logger: logging.Logger = logging.getLogger(__name__)

    class S3DataSourceSpliter(learn.LearningDataSource.DataSourceSpliter):
        logger: logging.Logger = logging.getLogger(__name__)

        def __init__(
            self,
            *,
            bucket_name: str,
            prefix: str,
            data_name: str,
            local_cache_dir: str,
            max_workers: int = 8,
        ) -> None:
            self.__page_manager = S3DataSource.PageManager(
                bucket_name=bucket_name,
                prefix=prefix,
                data_name=data_name,
                local_cache_dir=local_cache_dir,
                max_workers=max_workers,
            )

        def train_test_split(
            self, test_ratio: float
        ) -> tuple["S3DataSource", "S3DataSource"]:
            self.logger.info(f"test_ratio: {test_ratio}")
            input_indices, test_indicies = self.__train_test_split(
                data=list(range(self.__page_manager.total_rows)),
                test_ratio=test_ratio,
            )
            return (
                S3DataSource(
                    page_manager=self.__page_manager,
                    indicies=input_indices,
                ),
                S3DataSource(
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
            split_idx = int(len(data) * (1 - test_ratio))
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
            max_workers: int = 8,
        ) -> None:
            self.s3_client = boto3.client("s3")
            self.bucket_name = bucket_name
            self.prefix = prefix
            self.data_name = data_name
            self.max_workers = max_workers
            self.__pruning_info: dict[str, S3DataSource.PageManager.PruningInfo] = (
                defaultdict(S3DataSource.PageManager.PruningInfo)
            )

            # ローカルキャッシュの設定
            if local_cache_dir is None:
                raise ValueError("local_cache_dir must be specified")
            self.local_cache_dir = Path(local_cache_dir)
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Local cache directory: {self.local_cache_dir}")

            # バケットが存在するか確認
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    raise ValueError(f"Bucket '{self.bucket_name}' does not exist.")
                else:
                    raise

            # 初期化時にすべてのデータをダウンロード
            self.file_paths = self.__download_all_to_local()

            # ダウンロードしたデータから__pruning_infoを作成する
            total_rows = 0
            # ディレクトリごとに処理されるようにソートする
            self.file_paths.sort()
            last_pruning_value = None
            for file in self.file_paths:
                data = np.load(file, mmap_mode="r")
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
                f"S3 Data {self.total_rows} rows, {self.total_pages} pages"
            )

        def __get_data_path(self) -> str:
            """データ名からS3のパスを取得する."""
            return f"{self.prefix}/{self.data_name}"

        def __download_all_to_local(self) -> list[Path]:
            """すべてのデータをローカルにダウンロードする"""
            self.logger.info(
                f"Downloading all data to local cache: {self.local_cache_dir}"
            )

            # s3 sync
            self.__sync_s3_to_local(
                bucket=self.bucket_name,
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

        def __sync_s3_to_local(
            self, *, bucket: str, prefix: str, local_path: Path, delete: bool = False
        ) -> None:
            """S3バケットからローカルディレクトリへ一方向同期 (ダウンロードのみ)を行う.
            ローカルに存在しないファイルやS3側が更新されているファイルのみをダウンロードする

            Args:
                bucket (str): S3バケット名
                prefix (str): S3バケット内のプレフィックス（フォルダパス）
                local_path (Path): ローカルの保存先ディレクトリパス
                delete (bool): S3に存在しないローカルファイルを削除するかどうか
            """
            if not local_path.exists():
                raise ValueError("local_cache_dir does not exists")

            # prefixの末尾にスラッシュがない場合は追加
            if not prefix.endswith("/"):
                prefix = prefix + "/"
            try:
                paginator = self.s3_client.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

                # ダウンロードタスクを事前に収集
                download_tasks = []
                s3_local_files: set[Path] = set()

                for page in tqdm(pages, desc="Analyzing S3 objects", leave=False):
                    if "Contents" not in page:
                        self.logger.warning("Contents not found in page")
                        continue
                    for obj in page["Contents"]:
                        if obj["Key"] == prefix:
                            # prefix自体はスキップ
                            continue

                        local_file_path = local_path / obj["Key"][len(prefix) :]
                        self.logger.debug(local_path)
                        self.logger.debug(obj["Key"][len(prefix) :])
                        self.logger.debug(local_file_path.absolute())
                        s3_local_files.add(local_file_path)
                        
                        # ダウンロードするかどうかを判定する
                        download_file = False
                        if not local_file_path.exists():
                            download_file = True
                        else:
                            local_mtime = local_file_path.stat().st_mtime
                            s3_mtime = obj["LastModified"].timestamp()
                            if s3_mtime > local_mtime:
                                download_file = True

                        if download_file:
                            download_tasks.append((obj["Key"], local_file_path))

                # 並列ダウンロード実行
                download_count = 0
                skip_count = len([f for f in s3_local_files if f.exists()]) - len(download_tasks)
                
                if download_tasks:
                    total_size_mb = len(download_tasks) * 0.08  # 概算サイズ (80KB/file)
                    desc = f"Downloading {len(download_tasks)} files (~{total_size_mb:.1f}MB) [{self.max_workers} workers]"
                    
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = []
                        for s3_key, local_file_path in download_tasks:
                            future = executor.submit(
                                self._download_single_file, bucket, s3_key, local_file_path
                            )
                            futures.append(future)

                        # 完了を待つ
                        for future in tqdm(futures, desc=desc, unit="files", leave=True):
                            future.result()
                            download_count += 1
                
                delete_count = 0
                if delete:
                    # まずはファイルを削除してそのあと空のディレクトリがあれば削除する
                    s3_local_paths = {str(file.resolve()) for file in s3_local_files}
                    for local_file in local_path.glob("**/*"):
                        try:
                            if (
                                local_file.is_file()
                                and str(local_file.resolve()) not in s3_local_paths
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

            except ClientError as e:
                self.logger.error(
                    "Error sync s3 object from "
                    f"'s3://{bucket}/{prefix}' to '{local_path}': {e}"
                )
                raise

        def _download_single_file(
            self, bucket: str, s3_key: str, local_file_path: Path
        ) -> None:
            """単一ファイルをS3からダウンロードする"""
            try:
                # 親ディレクトリ作成 (mkdir -p)
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.logger.debug(
                    f"Downloading {s3_key} to {local_file_path}"
                )
                self.s3_client.download_file(
                    Bucket=bucket,
                    Key=s3_key,
                    Filename=str(local_file_path.absolute()),
                )
            except Exception as e:
                self.logger.error(f"Failed to download {s3_key}: {e}")
                raise

        def get_page(self, key: str) -> np.ndarray:
            """指定されたキーのページを取得する."""
            if key not in self.__pruning_info:
                raise ValueError(f"Key {key} not found in pruning info.")

            file_paths = [path for (path, _, _) in self.__pruning_info[key].files]
            file_paths.sort()
            data = np.concatenate([np.load(path, mmap_mode="r") for path in file_paths])
            return data

        def get_item(self, idx: int) -> dict[str, Any]:
            """特定のレコードだけが入った辞書を返す."""
            for key, pruning_info in self.__pruning_info.items():
                if (
                    pruning_info.start_idx
                    <= idx
                    < pruning_info.start_idx + pruning_info.count
                ):
                    for file, start_idx, num_rows in pruning_info.files:
                        if start_idx <= idx < start_idx + num_rows:
                            relative_idx = idx - start_idx
                            # ここもメモリマップ使っているがファイルサイズはそれほどでもないので
                            # パフォーマンス上のデメリットがあるならなくしてもいい
                            npy_data = np.load(file, mmap_mode="r")
                            names = npy_data.dtype.names
                            return {
                                name: npy_data[relative_idx][name] for name in names
                            }
            raise IndexError(f"Index {idx} out of range.")

        def iter_batches(self) -> Generator[tuple[str, np.ndarray], None, None]:
            """
            S3 のバケット全体に対して，
            ページ 単位のNumpy Structured Arrayを順次取得するジェネレータ．
            """
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
        max_workers: int = 8,
    ) -> None:
        """
        Args:
            bucket_name (Optional[str]): S3バケット名
            prefix (Optional[str]): S3オブジェクトのプレフィックス
            data_name (Optional[str]): データ名
            page_manager (Optional[PageManager]): PageManager
            indicies (Optional[list[int]]): 選択可能なインデックスのリスト
            local_cache_dir (Optional[str]): ローカルキャッシュディレクトリのパス
            max_workers (int): 並列ダウンロード数 (デフォルト: 8)
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
                )
            else:
                raise MissingS3Config(
                    f"S3の設定が不足しています bucket_name: {bucket_name}, prefix: {prefix}"
                )
        else:
            self.__page_manager = page_manager

        if indicies is None:
            self.indicies = list(range(self.__page_manager.total_rows))
        else:
            self.indicies = indicies

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self.indicies):
            raise IndexError(f"Index {idx} out of range.")

        return self.__page_manager.get_item(self.indicies[idx])

    def __len__(self) -> int:
        return len(self.indicies)

    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """
        S3 のバケット全体に対して，
        ページ 単位のNumpy Structured Arrayを順次取得するジェネレータ．
        """
        for name, batch in self.__page_manager.iter_batches():
            yield name, batch
