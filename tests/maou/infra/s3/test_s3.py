import hashlib
import logging
import os
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError

from maou.infra.s3.s3 import S3

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_AWS", "").lower() != "true"

if skip_test:
    logger.debug(
        f"Skip s3.py TEST_AWS: {os.getenv('TEST_AWS', '')}"
    )


@pytest.mark.skipif(
    skip_test,
    reason="AWSリソースを使ったテストはTEST_AWSが'true'の場合のみ実行されます",
)
class TestS3:
    def __calculate_file_md5(self, filepath: Path) -> str:
        """ファイルのMD5ハッシュ値を計算する関数.
        ファイルの内容をもとに32文字のハッシュ値が返ってくる
        """
        if not filepath.is_file():
            raise ValueError(
                f"Input file `{filepath}` is not file."
            )

        hash_md5 = hashlib.md5()

        # ファイルを読み取ってハッシュを計算
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        # ハッシュ値を16進文字列で返す
        return hash_md5.hexdigest()

    def __read_file_as_string(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                return content
        except FileNotFoundError:
            print(
                f"Error: The file '{file_path}' does not exist."
            )
        except UnicodeDecodeError:
            print(
                f"Error: The file '{file_path}' could not be decoded as UTF-8."
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return ""

    @pytest.fixture
    def default_fixture(self) -> None:
        path = Path("src/maou/infra/s3/s3.py")
        self.bucket_name = (
            "maou-test-" + self.__calculate_file_md5(path)[:8]
        )
        logger.debug(f"Test Bucket: {self.bucket_name}")
        self.test_class = S3(
            bucket_name=self.bucket_name, base_path="s3"
        )
        self.s3_client = boto3.client("s3")

    def test_save_object(self, default_fixture: None) -> None:
        file_path = Path(
            "tests/maou/infra/gcs/resources/test.txt"
        )
        cloud_path = file_path.name
        self.test_class.upload_from_local(
            local_path=file_path, cloud_path=cloud_path
        )

        # base_pathを含めたパスでオブジェクトを確認
        s3_path = f"s3/{cloud_path}"

        # オブジェクトが存在するか確認
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name, Key=s3_path
            )
        except ClientError:
            assert False, (
                f"Object {s3_path} does not exist in bucket {self.bucket_name}"
            )

        # オブジェクトの内容を取得して比較
        response = self.s3_client.get_object(
            Bucket=self.bucket_name, Key=s3_path
        )
        blob_content_str = (
            response["Body"].read().decode("utf-8")
        )
        logger.debug(
            f"contents of object {s3_path}: {blob_content_str}"
        )
        assert (
            self.__read_file_as_string(file_path)
            == blob_content_str
        )

        # clean up
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix="s3/"
        )
        if "Contents" in response:
            objects = [
                {"Key": obj["Key"]}
                for obj in response["Contents"]
            ]
            if objects:
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={"Objects": objects},
                )
                for obj in objects:
                    logger.debug(
                        f"Deleting object: {obj['Key']}"
                    )

    def test_upload_folder(self, default_fixture: None) -> None:
        """フォルダアップロードの基本機能をテストする."""
        folder_path = Path(
            "tests/maou/infra/gcs/resources/test_folder"
        )
        cloud_folder = "test_upload_folder"

        # フォルダをアップロード
        self.test_class.upload_folder_from_local(
            local_folder=folder_path, cloud_folder=cloud_folder
        )

        # アップロードされたファイルを確認
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=f"s3/{cloud_folder}"
        )

        # 4つのファイルがアップロードされていることを確認
        assert "Contents" in response
        objects = response["Contents"]
        assert len(objects) == 4

        # ファイルパスのリストを作成
        expected_paths = [
            f"s3/{cloud_folder}/file1.txt",
            f"s3/{cloud_folder}/file2.csv",
            f"s3/{cloud_folder}/subfolder/file3.txt",
            f"s3/{cloud_folder}/subfolder/file4.json",
        ]

        # すべての期待されるパスが存在することを確認
        object_paths = [obj["Key"] for obj in objects]
        for path in expected_paths:
            assert path in object_paths

        # ファイルの内容を確認 (最初のファイルのみ)
        test_file_path = folder_path / "file1.txt"
        test_key = f"s3/{cloud_folder}/file1.txt"

        response = self.s3_client.get_object(
            Bucket=self.bucket_name, Key=test_key
        )
        blob_content_str = (
            response["Body"].read().decode("utf-8")
        )
        assert (
            self.__read_file_as_string(test_file_path)
            == blob_content_str
        )

        # clean up
        if objects:
            delete_objects = [
                {"Key": obj["Key"]} for obj in objects
            ]
            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": delete_objects},
            )
            for obj in objects:
                logger.debug(f"Deleting object: {obj['Key']}")

    def test_upload_folder_with_extensions(
        self, default_fixture: None
    ) -> None:
        """拡張子フィルタリング機能をテストする."""
        folder_path = Path(
            "tests/maou/infra/gcs/resources/test_folder"
        )
        cloud_folder = "test_upload_folder_extensions"

        # .txtファイルのみをアップロード
        self.test_class.upload_folder_from_local(
            local_folder=folder_path,
            cloud_folder=cloud_folder,
            extensions=[".txt"],
        )

        # アップロードされたファイルを確認
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=f"s3/{cloud_folder}"
        )

        # 2つの.txtファイルのみがアップロードされていることを確認
        assert "Contents" in response
        objects = response["Contents"]
        assert len(objects) == 2

        # ファイルパスのリストを作成
        expected_paths = [
            f"s3/{cloud_folder}/file1.txt",
            f"s3/{cloud_folder}/subfolder/file3.txt",
        ]

        # すべての期待されるパスが存在することを確認
        object_paths = [obj["Key"] for obj in objects]
        for path in expected_paths:
            assert path in object_paths

        # .csvや.jsonファイルがアップロードされていないことを確認
        unexpected_paths = [
            f"s3/{cloud_folder}/file2.csv",
            f"s3/{cloud_folder}/subfolder/file4.json",
        ]
        for path in unexpected_paths:
            assert path not in object_paths

        # clean up
        if objects:
            delete_objects = [
                {"Key": obj["Key"]} for obj in objects
            ]
            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": delete_objects},
            )
            for obj in objects:
                logger.debug(f"Deleting object: {obj['Key']}")

    def test_upload_folder_errors(
        self, default_fixture: None
    ) -> None:
        """エラーケースをテストする."""
        # 存在しないフォルダを指定した場合
        with pytest.raises(ValueError, match="does not exist"):
            self.test_class.upload_folder_from_local(
                local_folder=Path("non_existent_folder"),
                cloud_folder="error_test",
            )

        # ファイルをフォルダとして指定した場合
        file_path = Path(
            "tests/maou/infra/gcs/resources/test.txt"
        )
        with pytest.raises(
            ValueError, match="is not a directory"
        ):
            self.test_class.upload_folder_from_local(
                local_folder=file_path,
                cloud_folder="error_test",
            )
