import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional, Tuple

import google.cloud.storage as storage
import numpy as np
from google.cloud.exceptions import NotFound
from tqdm.auto import tqdm

from maou.domain.data.io import save_array_to_buffer
from maou.interface import converter, preprocess


class NotFoundKeyColumns(Exception):
    """キーカラムが対象のスキーマ内に見つからない."""

    pass


class GCSFeatureStore(converter.FeatureStore, preprocess.FeatureStore):
    logger: logging.Logger = logging.getLogger(__name__)
    last_key_columns: Optional[list[str]] = None

    def __init__(
        self,
        *,
        bucket_name: str,
        prefix: str,
        location: str = "ASIA-NORTHEAST1",
        data_name: str,
        max_cached_bytes: int = 50 * 1024 * 1024,
        max_workers: int = 16,
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.data_name = data_name
        self.location = location

        # GCSクライアントを性能がでるように初期化
        self.client = storage.Client()

        try:
            # バケットが存在するか確認
            self.bucket = self.client.get_bucket(self.bucket_name)
            self.logger.debug(f"Bucket '{self.bucket_name}' already exists.")
        except NotFound:
            self.logger.debug(
                f"Bucket '{self.bucket_name}' does not exist. Creating it..."
            )
            # バケットを作成
            self.bucket = self.client.create_bucket(
                self.bucket_name, location=self.location
            )
            self.logger.debug(f"Bucket '{self.bucket_name}' created.")

        # 特徴量書き込み周りの設定
        # バッファを初期化
        self.__buffer: list[tuple[str, Optional[str], np.ndarray]] = []
        self.__buffer_size: int = 0
        # 最大値指定
        self.max_cached_bytes = max_cached_bytes
        # 並列アップロード用の設定
        self.max_workers = max_workers

    def __list_objects(self, prefix: str) -> List[Dict[str, Any]]:
        """指定されたプレフィックスのオブジェクトを一覧取得する."""
        objects = []

        # GCSのlist_blobsを使用してオブジェクトを取得
        blobs = self.bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            objects.append(
                {
                    "Key": blob.name,
                    "Size": blob.size,
                    "LastModified": blob.time_created,
                }
            )

        return objects

    def __get_data_path(self) -> str:
        """データ名からGCSのパスを取得する."""
        return f"{self.prefix}/{self.data_name}"

    def __get_data(self, file_key: str) -> np.ndarray:
        try:
            blob = self.bucket.blob(file_key)
            data = blob.download_as_bytes()

            # NumpyのStructured Arrayファイルを読み込む
            with BytesIO(data) as buffer:
                structured_array = np.load(buffer)

        except Exception as e:
            self.logger.error(f"Error reading file {file_key}: {e}")
            raise

        return structured_array

    def get_all_data(self) -> np.ndarray:
        """テーブルのすべてのデータを取得する"""
        data_path = self.__get_data_path()

        # テーブルに関連するすべてのnpyファイルを取得
        objects = self.__list_objects(data_path)
        data_files = [obj["Key"] for obj in objects if obj["Key"].endswith(".npy")]

        if not data_files:
            self.logger.warning(
                f"No data files found for path 'gs://{self.bucket_name}/{data_path}'"
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
        """GCSにデータを保存する.
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
        同じスキーマの配列を連結して大きなファイルにまとめることで効率化
        """
        if not self.__buffer:
            self.logger.debug("Buffer is empty. Nothing to flush.")
            return

        # スキーマとフォルダごとにグループ化
        grouped_arrays: Dict[Tuple[str, Optional[str]], List[np.ndarray]] = {}
        total_buffer_size = self.__buffer_size

        while self.__buffer:
            _, folder, structured_array = self.__buffer.pop(0)
            self.__buffer_size -= structured_array.nbytes

            # スキーマとフォルダをキーとしてグループ化
            schema_key = str(structured_array.dtype)
            group_key = (schema_key, folder)

            if group_key not in grouped_arrays:
                grouped_arrays[group_key] = []
            grouped_arrays[group_key].append(structured_array)

        try:
            upload_tasks = []

            # グループごとに配列を連結して大きなファイルを作成
            for (schema_key, folder), arrays in grouped_arrays.items():
                if len(arrays) == 1:
                    # 単一配列の場合はそのまま
                    combined_array = arrays[0]
                else:
                    # 複数配列を連結
                    combined_array = np.concatenate(arrays)

                # ファイル名に配列数とサイズ情報を含める
                size_mb = combined_array.nbytes // (1024 * 1024)
                schema_hash = hash(schema_key) % 10000
                file_name = (
                    f"batch_{len(arrays)}arrays_{size_mb}MB_{schema_hash:04d}.npy"
                )

                if folder is not None:
                    object_key = f"{self.__get_data_path()}/{folder}/{file_name}"
                else:
                    object_key = f"{self.__get_data_path()}/{file_name}"

                upload_tasks.append((object_key, combined_array))

            if not upload_tasks:
                return

            # 総データサイズを計算
            total_size_mb = total_buffer_size / (1024 * 1024)

            total_arrays = sum(len(arrays) for arrays in grouped_arrays.values())
            self.logger.info(
                f"Consolidated {total_arrays} arrays "
                f"into {len(upload_tasks)} files ({total_size_mb:.1f}MB total)"
            )

            # 並列アップロード
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for object_key, combined_array in upload_tasks:
                    future = executor.submit(
                        self._upload_single_array, object_key, combined_array
                    )
                    futures.append(future)

                # 完了を待つ (詳細な進捗情報付き)
                desc = (
                    f"Uploading {len(upload_tasks)} files ({total_size_mb:.1f}MB)"
                    f" [{self.max_workers} workers]"
                )
                for future in tqdm(futures, desc=desc, unit="files", leave=True):
                    future.result()

            # アップロード時間の統計を計算
            avg_time_per_file: float = 0.0
            if len(upload_tasks) > 0:
                # 簡易的な時間計算 (実際の時間は並列実行なので概算)
                avg_time_per_file = (
                    total_size_mb / len(upload_tasks)
                ) / 50  # 大雑把な推定

            self.logger.info(
                f"Successfully uploaded {len(upload_tasks)} files "
                f"to gs://{self.bucket_name}/{self.__get_data_path()}"
                f", ~{avg_time_per_file:.2f}s/files"
            )

        except Exception as e:
            self.logger.error(f"Error flushing features: {e}")
            raise

    def _upload_single_array(
        self, object_key: str, structured_array: np.ndarray
    ) -> None:
        """単一の配列をGCSにアップロードする (リトライ機能付き)."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # domainレイヤーのメソッドを使用してバイナリ形式で保存
                # 配列タイプを自動判定してvalidation付きで保存
                try:
                    buffer = save_array_to_buffer(
                        structured_array, validate=True, array_type="hcpe"
                    )
                except Exception:
                    try:
                        buffer = save_array_to_buffer(
                            structured_array, validate=True, array_type="preprocessing"
                        )
                    except Exception:
                        # フォールバック：検証なしで保存
                        buffer = save_array_to_buffer(structured_array, validate=False)

                blob = self.bucket.blob(object_key)
                blob.upload_from_file(buffer)

                self.logger.debug(
                    f"Uploaded {structured_array.nbytes / (1024 * 1024):.1f}MB"
                    f" to {object_key} (attempt {attempt + 1})"
                )
                return  # 成功したら終了

            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(
                        f"Failed to upload {object_key} "
                        f"after {max_retries} attempts: {e}"
                    )
                    raise
                else:
                    self.logger.warning(
                        f"Upload attempt {attempt + 1} failed for {object_key}: "
                        f"{e}, retrying..."
                    )
                    import time

                    time.sleep(2**attempt)  # exponential backoff

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
            "Features successfully stored in GCS."
            " bucket:"
            f" {self.bucket_name}/{self.__get_data_path()}"
        )
