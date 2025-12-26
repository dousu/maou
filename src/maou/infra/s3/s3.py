import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

import maou.interface.learn as learn
from maou.infra.file_system.file_system import FileSystem


class BucketNotFound(Exception):
    """指定されたバケット名が存在しないときのエラー．"""

    pass


class S3(learn.CloudStorage):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        bucket_name: str,
        base_path: str,
        region: str = "ap-northeast-1",
    ):
        self.bucket_name = bucket_name
        self.base_path = base_path
        self.region = region
        self.s3_client = boto3.client("s3")

        try:
            # バケットが存在するか確認
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.debug(
                f"Bucket '{self.bucket_name}' already exists."
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                self.logger.debug(
                    f"Bucket '{self.bucket_name}' does not exist. Creating it..."
                )
                # バケットを作成
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={
                        "LocationConstraint": self.region
                    },
                )
                self.logger.debug(
                    f"Bucket '{self.bucket_name}' created."
                )
            else:
                # その他のエラー
                raise

    def upload_from_local(
        self, *, local_path: Path, cloud_path: str
    ) -> None:
        """ローカルファイルをS3にアップロードする．

        Args:
            local_path: アップロードするローカルファイルのパス
            cloud_path: アップロード先のS3オブジェクトパス
        """
        s3_path = f"{self.base_path}/{cloud_path}"
        self.s3_client.upload_file(
            str(local_path), self.bucket_name, s3_path
        )
        self.logger.debug(
            f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_path}"
        )

    def upload_folder_from_local(
        self,
        *,
        local_folder: Path,
        cloud_folder: str,
        extensions: Optional[list[str]] = None,
    ) -> None:
        """指定されたローカルフォルダ内のファイルをS3バケットにアップロードする.

        Args:
            local_folder: アップロードするローカルフォルダのパス
            cloud_folder: アップロード先のS3フォルダパス
            extensions: アップロードするファイルの拡張子リスト (例: ['.txt', '.csv'])
                       Noneの場合はすべてのファイルをアップロード

        ローカルフォルダの構造はS3上でも維持される．
        ローカルのファイル名とオブジェクト名が重複した場合は，ローカルのデータで上書きする．
        """
        # ローカルフォルダが存在するか確認
        if not local_folder.exists():
            raise ValueError(
                f"Local folder `{local_folder}` does not exist."
            )
        if not local_folder.is_dir():
            raise ValueError(
                f"Path `{local_folder}` is not a directory."
            )

        # ファイルを収集
        files = FileSystem.collect_files(local_folder)

        # 拡張子でフィルタリング (指定されている場合)
        if extensions is not None:
            files = [
                f
                for f in files
                if any(f.suffix == ext for ext in extensions)
            ]

        # アップロード開始ログ
        self.logger.debug(
            f"Starting upload of {len(files)} files from {local_folder} to "
            f"s3://{self.bucket_name}/{self.base_path}/{cloud_folder}"
        )

        # 各ファイルをアップロード
        for file_path in files:
            # ローカルフォルダからの相対パスを計算
            relative_path = file_path.relative_to(local_folder)

            # クラウドパスを構築
            cloud_path = f"{cloud_folder}/{relative_path}"

            # ファイルをアップロード
            self.upload_from_local(
                local_path=file_path, cloud_path=cloud_path
            )

        # アップロード終了ログ
        self.logger.debug(
            f"Completed upload of {len(files)} files from {local_folder} to "
            f"s3://{self.bucket_name}/{self.base_path}/{cloud_folder}"
        )
