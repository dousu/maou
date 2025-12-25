"""S3バージョンDataSource.
staticmethodはProcessPoolExecutorで新プロセス (spawn)で実行されてもいいように中でimportする
"""

import io

from maou.infra.object_storage.data_source import (
    ObjectStorageDataSource,
)


class S3DataSource(ObjectStorageDataSource):
    class PageManager(ObjectStorageDataSource.PageManager):
        @staticmethod
        def list_objects(
            bucket_name: str, data_path: str
        ) -> list[tuple[str, int]]:
            import botocore.session

            if not data_path.endswith("/"):
                data_path += "/"

            session = botocore.session.get_session()
            client = session.create_client("s3")
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=bucket_name, Prefix=data_path
            )

            return [
                (str(obj["Key"]), int(obj["Size"]))
                for page in pages
                if "Contents" in page
                for obj in page["Contents"]
                if obj["Key"].endswith(".feather")
            ]

        @staticmethod
        def download_files(
            bucket_name: str, object_paths: list[str]
        ) -> list[bytes]:
            import boto3

            client = boto3.client("s3")

            # 細かく設定する案もある
            # transfer_config = TransferConfig(
            #     max_concurrency=8,
            #     multipart_threshold=100
            #     * 1024
            #     * 1024,  # 100MB: 小ファイルは単一リクエストで処理
            # )

            data = []
            for object_path in object_paths:
                buffer = io.BytesIO()
                client.download_fileobj(
                    Bucket=bucket_name,
                    Key=object_path,
                    Fileobj=buffer,
                    # Config=transfer_config,
                )
                buffer.seek(0)
                data.append(buffer.getvalue())
            return data
