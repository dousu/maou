import base64
import contextlib
import hashlib
import logging
import multiprocessing as mp
from typing import Generator, Literal, Optional

import numpy as np

from maou.interface import converter, preprocess
from maou.interface.data_io import save_array_to_bytes


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
        self.__buffer: list[np.ndarray] = []
        self.__buffer_size: int = 0
        self.bundle_id: int = 1
        # upload用のワーカー数
        self.max_workers = max_workers
        # 最大値指定
        self.max_cached_bytes = max_cached_bytes
        # アップロード用のプロセスを別でたてる
        self.queue: mp.Queue[Optional[tuple[str, bytes]]] = (
            mp.Queue(maxsize=max_queue_size)
        )
        self.uploader_processes = [
            mp.Process(
                target=type(self).uploader,
                kwargs={
                    "bucket_name": self.bucket_name,
                    "queue": self.queue,
                    "queue_timeout": queue_timeout,
                },
            )
            for _ in range(self.max_workers)
        ]
        for p in self.uploader_processes:
            p.start()

    def __get_data_path(self) -> str:
        """データ名からGCSのパスを取得する."""
        return f"{self.prefix}/{self.data_name}"

    def __short_hash(self, text: str, length: int = 4) -> str:
        digest = hashlib.sha256(text.encode()).digest()
        b64 = base64.urlsafe_b64encode(digest).decode()
        # 中央部分を切り出す
        start = (len(b64) - length) // 2
        return b64[start : start + length]

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
        structured_array: np.ndarray,
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        """GCSにデータを保存する.
        すでに同じIDが存在する場合は更新する
        """
        # オブジェクトストレージのFeatureStoreは
        # key_columnsやpruning keyの制約は関係ないのでチェックしない

        # バッファに追加
        self.__buffer.append(structured_array)
        self.__buffer_size += structured_array.nbytes
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
            num = len(self.__buffer)
            array = np.concatenate(self.__buffer)
            size_mb = array.nbytes // pow(1024, 2)
            schema_key = str(array.dtype)
            schema_hash = self.__short_hash(schema_key)
            file_name = f"batch{self.bundle_id}_{num}arrays_{size_mb}MB_{schema_hash}.npy"
            object_path = (
                f"{self.__get_data_path()}/{file_name}"
            )
            byte_data = save_array_to_bytes(
                array=array,
                array_type=self.array_type,
                bit_pack=True,
            )
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
    ) -> None:
        """S3やGCSのクライアントをこの中で使いまわしたいので各クラスに実装を任せる

        コード例

        ```
        client =
        while True:
          item = queue.get()
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
