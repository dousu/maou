import logging
from pathlib import Path
from typing import Optional

import google.cloud.storage as storage

from maou.infra.file_system.file_system import FileSystem
from maou.interface import learn


class BucketNotFound(Exception):
    """指定されたバケット名が存在しないときのエラー."""

    pass


class GCS(learn.CloudStorage):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        bucket_name: str,
        base_path: str,
        location: str = "ASIA-NORTHEAST1",
    ):
        self.bucket_name = bucket_name
        self.base_path = base_path
        client = storage.Client()

        try:
            self.bucket = client.get_bucket(self.bucket_name)
            self.logger.debug(
                f"Bucket '{self.bucket_name}' already exists."
            )
        except Exception:
            self.logger.debug(
                f"Bucket '{self.bucket_name}' does not exist. Creating it..."
            )
            self.bucket = client.create_bucket(
                self.bucket_name, location=location
            )
            self.logger.debug(
                f"Bucket '{bucket_name}' created."
            )

    def upload_from_local(
        self, *, local_path: Path, cloud_path: str
    ) -> None:
        blob = self.bucket.blob(
            f"{self.base_path}/{cloud_path}"
        )
        blob.upload_from_filename(local_path)
        self.logger.debug(
            f"Uploaded {local_path} to"
            f" gs://{self.bucket_name}/{self.base_path}/{cloud_path}"
        )

    def upload_folder_from_local(
        self,
        *,
        local_folder: Path,
        cloud_folder: str,
        extensions: Optional[list[str]] = None,
    ) -> None:
        """指定されたローカルフォルダ内のファイルをGCSバケットにアップロードする．

        Args:
            local_folder: アップロードするローカルフォルダのパス
            cloud_folder: アップロード先のGCSフォルダパス
            extensions: アップロードするファイルの拡張子リスト (例: ['.txt', '.csv'])
                       Noneの場合はすべてのファイルをアップロード

        ローカルフォルダの構造はGCS上でも維持される．
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
            f"gs://{self.bucket_name}/{self.base_path}/{cloud_folder}"
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
            f"gs://{self.bucket_name}/{self.base_path}/{cloud_folder}"
        )
