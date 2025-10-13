import io
import logging
import multiprocessing as mp
from queue import Empty as QueueEmpty

from maou.infra.object_storage.feature_store import (
    ObjectStorageFeatureStore,
)


class NotFoundKeyColumns(Exception):
    """キーカラムが対象のスキーマ内に見つからない."""

    pass


logger: logging.Logger = logging.getLogger(__name__)


class S3FeatureStore(ObjectStorageFeatureStore):
    @staticmethod
    def uploader(
        *,
        bucket_name: str,
        queue: mp.Queue,
        queue_timeout: int,
        max_workers: int = 1,
    ) -> None:
        import botocore.session

        session = botocore.session.get_session()
        client = session.create_client("s3")
        # シングルプロセスの場合はタイムアウトなし
        timeout = (
            None if max_workers == 1 else float(queue_timeout)
        )
        while True:
            try:
                item = queue.get(timeout=timeout)
                if item is None:
                    break
                object_path, byte_data = item
                client.put_object(
                    Bucket=bucket_name,
                    Key=object_path,
                    Body=io.BytesIO(byte_data),
                    ServerSideEncryption="AES256",
                )
            except QueueEmpty:
                logger.info("Queue is empty, closing uploader.")
                break
        queue.close()
        queue.join_thread()
