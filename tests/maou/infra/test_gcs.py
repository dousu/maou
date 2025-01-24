import logging
import os
from pathlib import Path

import google_crc32c
import pytest
from google.cloud import storage

from maou.infra.gcs import GCS

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_test:
    logger.debug(f"Skip gcs.py TEST_GCP: {os.getenv("TEST_GCP", "")}")


@pytest.mark.skipif(
    skip_test,
    reason="GCPリソースを使ったテストはTEST_GCPが'true'の場合のみ実行されます",
)
class TestGCS:
    def __calculate_file_crc32c(self, filepath: Path) -> str:
        """ファイルのCRC32Cハッシュ値を計算する関数.
        ファイルの内容をもとに8文字のハッシュ値が返ってくる
        """
        if not filepath.is_file():
            raise ValueError(f"Input file `{filepath}` is not file.")

        checksum = google_crc32c.Checksum()

        # ファイルを読み取ってハッシュを計算
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):  # 8KBずつ読み取る
                checksum.update(chunk)

        # ハッシュ値を16進文字列で返す
        return checksum.digest().hex()

    def __read_file_as_string(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")
        except UnicodeDecodeError:
            print(f"Error: The file '{file_path}' could not be decoded as UTF-8.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return content

    @pytest.fixture
    def default_fixture(self) -> None:
        path = Path("src/maou/infra/gcs.py")
        self.bucket_name = "maou-test-" + self.__calculate_file_crc32c(path)
        logger.debug(f"Test Bucket: {self.bucket_name}")
        self.test_class = GCS(bucket_name=self.bucket_name, base_path="gcs")
        client = storage.Client()
        self.bucket = client.bucket(self.bucket_name)

    def test_save_object(self, default_fixture: None) -> None:
        file_path = Path("tests/maou/infra/resources/test.txt")
        cloud_path = file_path.name
        self.test_class.upload_from_local(local_path=file_path, cloud_path=cloud_path)
        blob = self.bucket.blob(cloud_path)
        assert blob.exists()
        blob_content_str = blob.download_as_bytes().decode("utf-8")
        logger.debug(f"contents of object {cloud_path}: {blob_content_str}")
        assert self.__read_file_as_string(file_path) == blob_content_str
        # clean up
        blobs = self.bucket.list_blobs()
        for blob in blobs:
            logger.debug(f"Deleting object: {blob.name}")
            blob.delete()
