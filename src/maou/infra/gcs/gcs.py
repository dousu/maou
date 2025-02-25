import logging
from pathlib import Path

import google.cloud.storage as storage

from maou.interface import learn


class BucketNotFound(Exception):
    """指定されたバケット名が存在しないときのエラー."""

    pass


class GCS(learn.CloudStorage):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self, *, bucket_name: str, base_path: str, location: str = "ASIA-NORTHEAST1"
    ):
        self.bucket_name = bucket_name
        self.base_path = base_path
        client = storage.Client()

        try:
            self.bucket = client.get_bucket(self.bucket_name)
            self.logger.debug(f"Bucket '{self.bucket_name}' already exists.")
        except Exception:
            self.logger.debug(
                f"Bucket '{self.bucket_name}' does not exist. Creating it..."
            )
            self.bucket = client.create_bucket(self.bucket_name, location=location)
            self.logger.debug(f"Bucket '{bucket_name}' created.")

    def upload_from_local(self, *, local_path: Path, cloud_path: str) -> None:
        blob = self.bucket.blob(f"{self.base_path}/{cloud_path}")
        blob.upload_from_filename(local_path)
        self.logger.debug(
            f"Uploaded {local_path} to"
            f" gs://{self.bucket_name}/{self.base_path}/{cloud_path}"
        )
