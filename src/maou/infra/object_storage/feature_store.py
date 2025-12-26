import contextlib
import logging
import multiprocessing as mp
from collections.abc import Generator
from typing import Literal, Optional

import polars as pl

import maou.interface.converter as converter
import maou.interface.preprocess as preprocess

# Use 'spawn' start method to avoid fork() issues in multi-threaded environments
# This is required for compatibility with cloud storage clients (GCS, S3)
# and will be the default in Python 3.14+
_mp_context = mp.get_context("spawn")


class NotFoundKeyColumns(Exception):
    """キーカラムが対象のスキーマ内に見つからない."""

    pass


class ObjectStorageFeatureStore(
    converter.FeatureStore, preprocess.FeatureStore
):
    logger: logging.Logger = logging.getLogger(__name__)
    array_type: Literal["hcpe", "preprocessing"]

    def __init__(
        self,
        *,
        bucket_name: str,
        prefix: str,
        data_name: str,
        array_type: Literal["hcpe", "preprocessing"],
        location: str = "ASIA-NORTHEAST1",
        max_workers: int = 8,
        max_cached_bytes: int = 100 * 1024 * 1024,
        queue_timeout: int = 600,
        max_queue_size: int = 8,
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.data_name = data_name
        self.location = location
        self.array_type = array_type

        # 特徴量書き込み周りの設定
        # バッファを初期化
        self.__buffer: list[pl.DataFrame] = []
        self.__buffer_size: int = 0
        self.bundle_id: int = 1
        # upload用のワーカー数
        self.max_workers = max_workers
        # 最大値指定
        self.max_cached_bytes = max_cached_bytes
        # アップロード用のプロセスを別でたてる（spawn contextを使用）
        self.queue: mp.Queue[Optional[tuple[str, bytes]]] = (
            _mp_context.Queue(maxsize=max_queue_size)
        )
        self.uploader_processes = [
            _mp_context.Process(
                target=type(self).uploader,
                kwargs={
                    "bucket_name": self.bucket_name,
                    "queue": self.queue,
                    "queue_timeout": queue_timeout,
                    "max_workers": self.max_workers,
                },
            )
            for _ in range(self.max_workers)
        ]
        for p in self.uploader_processes:
            p.start()

    def __get_data_path(self) -> str:
        """データ名からGCSのパスを取得する."""
        return f"{self.prefix}/{self.data_name}"

    # context managerを使って特徴量ストア用の動作のflushを管理する
    @contextlib.contextmanager
    def feature_store(self) -> Generator[None, None, None]:
        try:
            yield
        except Exception:
            raise
        finally:
            self.__cleanup()

    def store_features(
        self,
        *,
        name: str,
        key_columns: list[str],
        dataframe: pl.DataFrame,
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        """GCSにデータを保存する.
        すでに同じIDが存在する場合は更新する
        """
        # オブジェクトストレージのFeatureStoreは
        # key_columnsやpruning keyの制約は関係ないのでチェックしない

        # バッファに追加
        self.__buffer.append(dataframe)
        self.__buffer_size += int(dataframe.estimated_size())
        self.logger.debug(
            f"Buffered table size: {self.__buffer_size} bytes"
        )

        # バッファが上限を超えたら一括保存
        if self.__buffer_size >= self.max_cached_bytes:
            self.flush_features(
                key_columns=key_columns,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
            )

    # シングルファイルなので並列化するのはあまり意味なさそう
    def flush_features(
        self,
        *,
        key_columns: list[str],
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        """バッファのデータを一括して保存.
        同じスキーマの配列を連結して大きなファイルにまとめることで効率化
        """
        if not self.__buffer:
            self.logger.debug(
                "Buffer is empty. Nothing to flush."
            )
            return

        try:
            from maou.domain.data.dataframe_io import (
                save_hcpe_df_to_bytes,
                save_preprocessing_df_to_bytes,
            )

            num = len(self.__buffer)
            # Concatenate DataFrames
            df = pl.concat(self.__buffer)
            size_mb = df.estimated_size() // pow(1024, 2)

            file_name = f"batch{self.bundle_id}_{num}dfs_{size_mb}MB.feather"
            object_path = (
                f"{self.__get_data_path()}/{file_name}"
            )

            # Serialize DataFrame to bytes using Arrow IPC format
            if self.array_type == "hcpe":
                byte_data = save_hcpe_df_to_bytes(df)
            else:
                byte_data = save_preprocessing_df_to_bytes(df)

            self.queue.put((object_path, byte_data))
            self.bundle_id += 1
            self.__buffer_size = 0
            self.__buffer.clear()
        except Exception as e:
            self.logger.error(f"Error flushing features: {e}")
            raise

    @staticmethod
    def uploader(
        *,
        bucket_name: str,
        queue: mp.Queue,
        queue_timeout: int,
        max_workers: int = 1,
    ) -> None:
        """S3やGCSのクライアントをこの中で使いまわしたいので各クラスに実装を任せる

        コード例

        ```
        client =
        timeout = None if max_workers == 1 else float(queue_timeout)
        while True:
          item = queue.get(timeout=timeout)
          if item is None:
            break
          object_path, byte_data = item
          client.upload_bytes(object_path, byte_data)
        ```
        """
        raise Exception("uploaderが未実装")

    def __cleanup(self) -> None:
        """store_features用のデストラクタ処理"""
        # bufferが空のときはスキップする
        if self.__buffer_size != 0:
            self.flush_features(
                key_columns=["id"],
                clustering_key=None,
                partitioning_key_date=None,
            )
        for _ in range(self.max_workers):
            self.queue.put(None)
        for p in self.uploader_processes:
            p.join()
        self.logger.debug(
            "Features successfully stored in ObjectStorage."
            " bucket:"
            f" {self.bucket_name}/{self.__get_data_path()}"
        )
