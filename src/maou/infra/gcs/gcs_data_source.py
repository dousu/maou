"""GCSバージョンDataSource.
staticmethodはProcessPoolExecutorで新プロセス (spawn)で実行されてもいいように中でimportする
"""

from maou.infra.object_storage.data_source import (
    ObjectStorageDataSource,
)


class GCSDataSource(ObjectStorageDataSource):
    class PageManager(ObjectStorageDataSource.PageManager):
        @staticmethod
        def list_objects(
            bucket_name: str, data_path: str
        ) -> list[tuple[str, int]]:
            import google.cloud.storage as storage

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            if not data_path.endswith("/"):
                data_path += "/"
            blobs = bucket.list_blobs(prefix=data_path)
            return [
                (blob.name, blob.size)
                for blob in blobs
                if blob.name.endswith(".feather")
            ]

        @staticmethod
        def download_files(
            bucket_name: str, object_paths: list[str]
        ) -> list[bytes]:
            import google.cloud.storage as storage

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            data = []
            for object_path in object_paths:
                blob = bucket.blob(object_path)
                byte_data = blob.download_as_bytes()
                data.append(byte_data)
            return data
