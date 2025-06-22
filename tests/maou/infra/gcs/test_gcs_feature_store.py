import logging
import os
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Generator

import google.cloud.storage as storage
import google_crc32c
import numpy as np
import pytest

from maou.infra.gcs.gcs_feature_store import GCSFeatureStore

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_test:
    logger.debug(f"Skip {__name__} TEST_GCP: {os.getenv('TEST_GCP', '')}")


@pytest.mark.skipif(
    skip_test,
    reason="GCPリソースを使ったテストはTEST_GCPが'true'の場合のみ実行されます",
)
class TestGCSFeatureStore:
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

    def __generate_test_data(
        self, *, num_rows: int, partitioning_key: date
    ) -> np.ndarray:
        """テスト用のデータを生成する"""
        data = np.zeros(
            num_rows,
            dtype=[
                ("id", (np.unicode_, 128)),  # type: ignore[attr-defined]
                ("hcp", (np.uint8, 32)),
                ("eval", np.int16),
                ("bestMove16", np.int16),
                ("gameResult", np.int8),
                ("ratings", (np.uint16, 2)),
                ("endgameStatus", (np.unicode_, 16)),  # type: ignore[attr-defined] # noqa: E501
                ("moves", np.int16),
                ("partitioningKey", np.dtype("datetime64[D]")),
            ],
        )

        for i in range(num_rows):
            data[i]["id"] = str(uuid.uuid4())
            data[i]["hcp"] = np.zeros(32, dtype=np.uint8)
            data[i]["eval"] = np.random.randint(-1000, 1000)
            data[i]["bestMove16"] = np.random.randint(-32768, 32768)
            data[i]["gameResult"] = np.random.randint(-1, 2)
            data[i]["ratings"] = np.zeros(2)
            data[i]["endgameStatus"] = np.random.choice(
                ["WIN", "LOSE", "DRAW", "UNKNOWN"]
            )
            data[i]["moves"] = np.random.randint(0, 100)
            data[i]["partitioningKey"] = np.datetime64(partitioning_key)

        return data

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        """一時的なキャッシュディレクトリを提供するフィクスチャ"""
        cache_dir = tmp_path / "gcs_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def default_fixture(self) -> Generator[None, None, None]:
        path = Path("src/maou/infra/gcs/gcs_feature_store.py")
        self.bucket = "maou-test-bucket"
        self.prefix = "test-" + self.__calculate_file_crc32c(path)
        self.data_name = "test_data"
        yield
        # clean up
        try:
            client = storage.Client()
            bucket = client.get_bucket(self.bucket)
            blobs = list(bucket.list_blobs(prefix=self.prefix))
            for blob in blobs:
                if blob.name == self.prefix:
                    # prefix自体はスキップ
                    continue
                logger.debug(f"Deleting {blob.name}")
                blob.delete()
            logger.debug(f"Deleted all objects with prefix {self.prefix}")
        except Exception as e:
            logger.error(f"Failed to clean up test bucket: {e}")

    def test_store_features(self, default_fixture: None, temp_cache_dir: Path) -> None:
        data_name = f"{self.data_name}-1"
        feature_store = GCSFeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
        )
        rows = 3
        days = 2
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat("2022-02-02")
                + timedelta(days=float(i)),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    structured_array=structured_array,
                )

        # 中身を確認する
        array = feature_store.get_all_data()
        data_array = np.concatenate(data)
        logger.debug(data_array)
        logger.debug(array)
        logger.debug(f"Data dtype: {data_array.dtype}, GCS Data dtype: {array.dtype}")
        assert data_array.dtype == array.dtype
        assert len(array) == days * rows
        assert np.array_equal(data_array, array)
