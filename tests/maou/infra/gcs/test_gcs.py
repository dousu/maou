import logging
import os
from pathlib import Path

import google_crc32c
import pytest
from google.cloud import storage

from maou.infra.gcs.gcs import GCS

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_test:
    logger.debug(f"Skip gcs.py TEST_GCP: {os.getenv('TEST_GCP', '')}")


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
        path = Path("src/maou/infra/gcs/gcs.py")
        self.bucket_name = "maou-test-" + self.__calculate_file_crc32c(path)
        logger.debug(f"Test Bucket: {self.bucket_name}")
        self.test_class = GCS(bucket_name=self.bucket_name, base_path="gcs")
        client = storage.Client()
        self.bucket = client.bucket(self.bucket_name)

    def test_save_object(self, default_fixture: None) -> None:
        file_path = Path("tests/maou/infra/gcs/resources/test.txt")
        cloud_path = file_path.name
        self.test_class.upload_from_local(local_path=file_path, cloud_path=cloud_path)
        # base_pathを含めたパスでオブジェクトを確認
        blob = self.bucket.blob(f"gcs/{cloud_path}")
        assert blob.exists()
        blob_content_str = blob.download_as_bytes().decode("utf-8")
        logger.debug(f"contents of object gcs/{cloud_path}: {blob_content_str}")
        assert self.__read_file_as_string(file_path) == blob_content_str
        # clean up
        blobs = self.bucket.list_blobs()
        for blob in blobs:
            logger.debug(f"Deleting object: {blob.name}")
            blob.delete()

    def test_upload_folder(self, default_fixture: None) -> None:
        """フォルダアップロードの基本機能をテストする."""
        folder_path = Path("tests/maou/infra/gcs/resources/test_folder")
        cloud_folder = "test_upload_folder"

        # フォルダをアップロード
        self.test_class.upload_folder_from_local(
            local_folder=folder_path, cloud_folder=cloud_folder
        )

        # アップロードされたファイルを確認
        blobs = list(self.bucket.list_blobs(prefix=f"gcs/{cloud_folder}"))

        # 4つのファイルがアップロードされていることを確認
        assert len(blobs) == 4

        # ファイルパスのリストを作成
        expected_paths = [
            f"gcs/{cloud_folder}/file1.txt",
            f"gcs/{cloud_folder}/file2.csv",
            f"gcs/{cloud_folder}/subfolder/file3.txt",
            f"gcs/{cloud_folder}/subfolder/file4.json",
        ]

        # すべての期待されるパスが存在することを確認
        blob_paths = [blob.name for blob in blobs]
        for path in expected_paths:
            assert path in blob_paths

        # ファイルの内容を確認（最初のファイルのみ）
        test_file_path = folder_path / "file1.txt"
        test_blob = self.bucket.blob(f"gcs/{cloud_folder}/file1.txt")
        assert test_blob.exists()
        blob_content_str = test_blob.download_as_bytes().decode("utf-8")
        assert self.__read_file_as_string(test_file_path) == blob_content_str

        # clean up
        for blob in blobs:
            logger.debug(f"Deleting object: {blob.name}")
            blob.delete()

    def test_upload_folder_with_extensions(self, default_fixture: None) -> None:
        """拡張子フィルタリング機能をテストする."""
        folder_path = Path("tests/maou/infra/gcs/resources/test_folder")
        cloud_folder = "test_upload_folder_extensions"

        # .txtファイルのみをアップロード
        self.test_class.upload_folder_from_local(
            local_folder=folder_path, cloud_folder=cloud_folder, extensions=[".txt"]
        )

        # アップロードされたファイルを確認
        blobs = list(self.bucket.list_blobs(prefix=f"gcs/{cloud_folder}"))

        # 2つの.txtファイルのみがアップロードされていることを確認
        assert len(blobs) == 2

        # ファイルパスのリストを作成
        expected_paths = [
            f"gcs/{cloud_folder}/file1.txt",
            f"gcs/{cloud_folder}/subfolder/file3.txt",
        ]

        # すべての期待されるパスが存在することを確認
        blob_paths = [blob.name for blob in blobs]
        for path in expected_paths:
            assert path in blob_paths

        # .csvや.jsonファイルがアップロードされていないことを確認
        unexpected_paths = [
            f"gcs/{cloud_folder}/file2.csv",
            f"gcs/{cloud_folder}/subfolder/file4.json",
        ]
        for path in unexpected_paths:
            assert path not in blob_paths

        # clean up
        for blob in blobs:
            logger.debug(f"Deleting object: {blob.name}")
            blob.delete()

    def test_upload_folder_errors(self, default_fixture: None) -> None:
        """エラーケースをテストする."""
        # 存在しないフォルダを指定した場合
        with pytest.raises(ValueError, match="does not exist"):
            self.test_class.upload_folder_from_local(
                local_folder=Path("non_existent_folder"), cloud_folder="error_test"
            )

        # ファイルをフォルダとして指定した場合
        file_path = Path("tests/maou/infra/gcs/resources/test.txt")
        with pytest.raises(ValueError, match="is not a directory"):
            self.test_class.upload_folder_from_local(
                local_folder=file_path, cloud_folder="error_test"
            )
