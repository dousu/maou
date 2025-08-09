import io
import logging
import multiprocessing as mp
from queue import Empty as QueueEmpty

import google.cloud.storage as storage

from maou.infra.object_storage.feature_store import (
    ObjectStorageFeatureStore,
)


class NotFoundKeyColumns(Exception):
    """キーカラムが対象のスキーマ内に見つからない."""

    pass


logger: logging.Logger = logging.getLogger(__name__)


class GCSFeatureStore(ObjectStorageFeatureStore):
    @staticmethod
    def uploader(
        *,
        bucket_name: str,
        queue: mp.Queue,
        queue_timeout: int,
    ) -> None:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        while True:
            try:
                item = queue.get(timeout=float(queue_timeout))
                if item is None:
                    break
                object_path, byte_data = item
                blob = bucket.blob(object_path)
                blob.upload_from_file(
                    io.BytesIO(byte_data), rewind=True
                )
            except QueueEmpty:
                logger.info("Queue is empty, closing uploader.")
                break
        queue.close()
        queue.join_thread()
