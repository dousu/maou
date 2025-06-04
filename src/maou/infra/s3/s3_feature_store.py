import contextlib
import logging
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional

import boto3
import numpy as np
from botocore.exceptions import ClientError
from tqdm.auto import tqdm

from maou.interface import converter, preprocess


class NotFoundKeyColumns(Exception):
    """キーカラムが対象のスキーマ内に見つからない."""

    pass


class S3FeatureStore(converter.FeatureStore, preprocess.FeatureStore):
    logger: logging.Logger = logging.getLogger(__name__)
    last_key_columns: Optional[list[str]] = None

    def __init__(
        self,
        *,
        bucket_name: str,
        prefix: str,
        region: str = "ap-northeast-1",
        data_name: str,
        max_cached_bytes: int = 50 * 1024 * 1024,
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.data_name = data_name
        self.region = region
        self.s3_client = boto3.client("s3")

        try:
            # バケットが存在するか確認
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.debug(f"Bucket '{self.bucket_name}' already exists.")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                self.logger.debug(
                    f"Bucket '{self.bucket_name}' does not exist. Creating it..."
                )
                # バケットを作成
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )
                self.logger.debug(f"Bucket '{self.bucket_name}' created.")
            else:
                raise

        # 特徴量書き込み周りの設定
        # バッファを初期化
        self.__buffer: list[tuple[str, Optional[str], np.ndarray]] = []
        self.__buffer_size: int = 0
        # 最大値指定
        self.max_cached_bytes = max_cached_bytes

    def __list_objects(self, prefix: str) -> List[Dict[str, Any]]:
        """指定されたプレフィックスのオブジェクトを一覧取得する."""
        objects = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

        for page in pages:
            if "Contents" in page:
                objects.extend(page["Contents"])

        return objects

    def __get_data_path(self) -> str:
        """データ名からS3のパスを取得する."""
        return f"{self.prefix}/{self.data_name}"

    def __get_data(self, file_key: str) -> np.ndarray:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            data = response["Body"].read()

            # NumpyのStructured Arrayファイルを読み込む
            with BytesIO(data) as buffer:
                structured_array = np.load(buffer)

        except ClientError as e:
            self.logger.error(f"Error reading file {file_key}: {e}")

        return structured_array

    def get_all_data(self) -> np.ndarray:
        """テーブルのすべてのデータを取得する"""
        data_path = self.__get_data_path()

        # テーブルに関連するすべてのnpyファイルを取得
        objects = self.__list_objects(data_path)
        data_files = [obj["Key"] for obj in objects if obj["Key"].endswith(".npy")]

        if not data_files:
            self.logger.warning(
                f"No data files found for path 's3://{self.bucket_name}/{data_path}'"
            )
            return np.array([])

        # データを結合する
        return np.concatenate([self.__get_data(file_key) for file_key in data_files])

    # context managerを使って特徴量ストア用の動作のflushを管理する
    @contextlib.contextmanager
    def feature_store(self) -> Generator[None, None, None]:
        try:
            yield
        except Exception:
            raise
        finally:
            self.__cleanup()

    def store_features(
        self,
        *,
        name: str,
        key_columns: list[str],
        structured_array: np.ndarray,
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        """S3にデータを保存する.
        すでに同じIDが存在する場合は更新する
        """

        folder = None

        # カラム指定に矛盾がないか確認
        if not (
            set(key_columns)
            <= set([name for name, _ in structured_array.dtype.fields.items()])
        ):
            self.logger.warning(
                f"キーカラムが存在しない: {key_columns}, "
                f"{[name for name, _ in structured_array.dtype.fields.items()]}"
            )
        if partitioning_key_date is not None and partitioning_key_date not in [
            name for name, _ in structured_array.dtype.fields.items()
        ]:
            self.logger.error(
                f"パーティショニングキーが存在しない: {partitioning_key_date}, "
                f"{[name for name, _ in structured_array.dtype.fields.items()]}"
            )
            raise NotFoundKeyColumns("Not found clustering key columns")
        elif partitioning_key_date is not None:
            if folder is None:
                folder = structured_array[partitioning_key_date][0]
            else:
                folder += f"/{structured_array[partitioning_key_date][0]}"
        if clustering_key is not None and clustering_key not in [
            name for name, _ in structured_array.dtype.fields.items()
        ]:
            self.logger.error(
                f"クラスタリングキーが存在しない: {clustering_key}, "
                f"{[name for name, _ in structured_array.dtype.fields.items()]}"
            )
            raise NotFoundKeyColumns("Not found clustering key columns")
        elif clustering_key is not None:
            if folder is None:
                folder = structured_array[clustering_key][0]
            else:
                folder += f"/{structured_array[clustering_key][0]}"
        if self.last_key_columns is None:
            # flush用にキーカラムを保存しておく
            self.last_key_columns = key_columns
        elif key_columns != self.last_key_columns:
            # キーカラムのリストがもし変わったらいままでのをフラッシュしてから更新する
            self.flush_features(
                key_columns=self.last_key_columns,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
            )
            self.last_key_columns = key_columns

        # バッファに追加
        self.__buffer.append((name, folder, structured_array))
        self.__buffer_size += structured_array.nbytes
        self.logger.debug(f"Buffered table size: {self.__buffer_size} bytes")

        # バッファが上限を超えたら一括保存
        if self.__buffer_size >= self.max_cached_bytes:
            self.flush_features(
                key_columns=key_columns,
                clustering_key=clustering_key,
                partitioning_key_date=partitioning_key_date,
            )

    def flush_features(
        self,
        *,
        key_columns: list[str],
        clustering_key: Optional[str] = None,
        partitioning_key_date: Optional[str] = None,
    ) -> None:
        """バッファのデータを一括して保存.
        ここでは追記しかしないので同じデータがあると重複してしまう
        """
        if not self.__buffer:
            self.logger.debug("Buffer is empty. Nothing to flush.")
            return

        # バッファをリセットする
        arrays = self.__buffer.copy()
        self.__buffer.clear()
        self.__buffer_size = 0

        try:
            for name, folder, structured_array in tqdm(
                arrays, desc="Flushing features", leave=False
            ):
                if folder is not None:
                    object_key = f"{self.__get_data_path()}/{folder}/{name}.npy"
                else:
                    object_key = f"{self.__get_data_path()}/{name}.npy"

                with BytesIO() as buffer:
                    # bufferに書き出してそれを送る仕組みにする (ローカルに保存しない)
                    np.save(buffer, structured_array)
                    buffer.seek(0)  # Reset buffer position to the beginning

                    # S3にアップロード
                    self.s3_client.put_object(
                        Bucket=self.bucket_name, Key=object_key, Body=buffer
                    )

            self.logger.debug(f"Uploaded data to s3://{self.bucket_name}/{object_key}")

        except Exception as e:
            self.logger.error(f"Error flushing features: {e}")
            raise

    def __cleanup(self) -> None:
        """store_features用のデストラクタ処理"""
        # bufferが空のときはスキップする
        if self.last_key_columns is not None and self.__buffer_size != 0:
            self.flush_features(
                key_columns=self.last_key_columns,
                clustering_key=None,
                partitioning_key_date=None,
            )
        self.logger.debug(
            "Features successfully stored in S3."
            " bucket:"
            f" {self.bucket_name}/{self.__get_data_path()}"
        )
