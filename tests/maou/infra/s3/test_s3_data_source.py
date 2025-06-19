import logging
import os
import uuid
from collections.abc import Generator
from datetime import date, timedelta
from functools import reduce
from pathlib import Path

import boto3
import google_crc32c
import numpy as np
import pytest
from botocore.exceptions import ClientError

from maou.infra.s3.s3_data_source import S3DataSource
from maou.infra.s3.s3_feature_store import S3FeatureStore

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_AWS", "").lower() != "true"

if skip_test:
    logger.debug(f"Skip {__name__} TEST_AWS: {os.getenv('TEST_AWS', '')}")


@pytest.mark.skipif(
    skip_test,
    reason="AWSリソースを使ったテストはTEST_AWSが'true'の場合のみ実行されます",
)
class TestS3DataSource:
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

    def compare_records(self, r1: np.ndarray, r2: dict) -> bool:
        """Compare numpy structured array record with dict"""
        # numpy structured arrayの場合はフィールド名を取得
        if hasattr(r1, "dtype") and r1.dtype.names:
            r1_keys = set(r1.dtype.names)
        else:
            logger.debug(f"r1 is not a structured array: {type(r1)}")
            return False

        r2_keys = set(r2.keys())

        if r1_keys != r2_keys:
            logger.debug(f"keys: {r1_keys} != {r2_keys}")
            return False

        for key in r1_keys:
            r1_val = r1[key]
            r2_val = r2[key]

            if (
                isinstance(r1_val, np.memmap)
                or isinstance(r1_val, np.ndarray)
                or isinstance(r2_val, np.memmap)
                or isinstance(r2_val, np.ndarray)
            ):
                if not np.array_equal(r1_val, r2_val):
                    logger.debug(f"{key}: {r1_val} != {r2_val}")
                    return False
            else:
                # スカラー値の場合は.item()で取得
                if hasattr(r1_val, "item"):
                    r1_val = r1_val.item()
                if r1_val != r2_val:
                    logger.debug(f"{key}: {r1_val} != {r2_val}")
                    return False
        return True

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        """一時的なキャッシュディレクトリを提供するフィクスチャ"""
        cache_dir = tmp_path / "s3_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture()
    def default_fixture(self) -> Generator[None, None, None]:
        path = Path("src/maou/infra/s3/s3_data_source.py")
        self.bucket = "maou-test-bucket"
        self.prefix = "test-" + self.__calculate_file_crc32c(path)
        self.data_name = "test_data"
        yield
        # clean up
        try:
            s3_client = boto3.client("s3")
            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
            for page in pages:
                if "Contents" not in page:
                    logger.warning("Contents not found in page")
                    continue
                for obj in page["Contents"]:
                    if obj["Key"] == self.prefix:
                        # prefix自体はスキップ
                        continue
                    logger.debug(f"Deleting {obj['Key']}")
                    s3_client.delete_object(Bucket=self.bucket, Key=obj["Key"])
            logger.debug(f"Deleted all objects with prefix {self.prefix}")
        except ClientError as e:
            logger.error(f"Failed to clean up test bucket: {e}")

    def test_init_without_pruning_key(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """プルーニングキーなし"""
        data_name = f"{self.data_name}-1"
        feature_store = S3FeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
        )
        rows = 3
        days = 2
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat("2022-02-02") + timedelta(days=i),
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
        data_source = S3DataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
        )

        assert len(data_source) > 0
        assert len(data_source) == rows * days
        # データを読み込む
        read_data = [data_source[i] for i in range(len(data_source))]
        sorted_read_data = sorted(read_data, key=lambda x: x["id"])  # type: ignore
        logger.debug(sorted_read_data)
        # 読み込んだデータが正しいことを確認
        expected_data = reduce(
            lambda a, b: a + b,
            [
                [
                    {
                        "id": record["id"],
                        "hcp": record["hcp"],
                        "eval": record["eval"],
                        "bestMove16": record["bestMove16"],
                        "gameResult": record["gameResult"],
                        "ratings": record["ratings"],
                        "endgameStatus": record["endgameStatus"],
                        "moves": record["moves"],
                        "partitioningKey": record["partitioningKey"],
                    }
                    for record in daily_data
                ]
                for daily_data in data
            ],
        )
        sorted_expected_data = sorted(expected_data, key=lambda x: x["id"])
        logger.debug(sorted_expected_data)

        assert len(read_data) == len(expected_data)
        assert all(
            [
                self.compare_records(d1, d2)
                for d1, d2 in zip(sorted_read_data, sorted_expected_data)
            ]
        )

    def test_init_with_clustering_key(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """クラスタリングキーあり"""
        data_name = f"{self.data_name}-2"
        feature_store = S3FeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
        )
        rows = 3
        days = 4
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat("2022-02-02") + timedelta(days=i),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                # clustering_keyを指定する
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    structured_array=structured_array,
                    clustering_key="partitioningKey",
                )
        data_source = S3DataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
        )

        assert len(data_source) > 0
        assert len(data_source) == rows * days
        assert set(
            [
                str(date.fromisoformat("2022-02-02") + timedelta(days=i))
                for i in range(days)
            ]
        ) == set([file.parent.name for file in temp_cache_dir.glob("**/*.npy")])
        # データを読み込む
        read_data = [data_source[i] for i in range(len(data_source))]
        sorted_read_data = sorted(read_data, key=lambda x: x["id"])  # type: ignore
        logger.debug(sorted_read_data)
        # 読み込んだデータが正しいことを確認
        expected_data = reduce(
            lambda a, b: a + b,
            [
                [
                    {
                        "id": record["id"],
                        "hcp": record["hcp"],
                        "eval": record["eval"],
                        "bestMove16": record["bestMove16"],
                        "gameResult": record["gameResult"],
                        "ratings": record["ratings"],
                        "endgameStatus": record["endgameStatus"],
                        "moves": record["moves"],
                        "partitioningKey": record["partitioningKey"],
                    }
                    for record in daily_data
                ]
                for daily_data in data
            ],
        )
        sorted_expected_data = sorted(expected_data, key=lambda x: x["id"])
        logger.debug(sorted_expected_data)

        assert len(read_data) == len(expected_data)
        assert all(
            [
                self.compare_records(d1, d2)
                for d1, d2 in zip(sorted_read_data, sorted_expected_data)
            ]
        )

    def test_init_with_partitioning_key(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """パーティショニングキーあり"""
        data_name = f"{self.data_name}-3"
        feature_store = S3FeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
        )
        rows = 3
        days = 4
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat("2022-02-02") + timedelta(days=i),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                # partitioning_key_dateを指定する
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    structured_array=structured_array,
                    partitioning_key_date="partitioningKey",
                )
        data_source = S3DataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
        )

        assert len(data_source) > 0
        assert len(data_source) == rows * days
        assert set(
            [
                str(date.fromisoformat("2022-02-02") + timedelta(days=i))
                for i in range(days)
            ]
        ) == set([file.parent.name for file in temp_cache_dir.glob("**/*.npy")])
        # データを読み込む
        read_data = [data_source[i] for i in range(len(data_source))]
        sorted_read_data = sorted(read_data, key=lambda x: x["id"])  # type: ignore
        logger.debug(sorted_read_data)
        # 読み込んだデータが正しいことを確認
        expected_data = reduce(
            lambda a, b: a + b,
            [
                [
                    {
                        "id": record["id"],
                        "hcp": record["hcp"],
                        "eval": record["eval"],
                        "bestMove16": record["bestMove16"],
                        "gameResult": record["gameResult"],
                        "ratings": record["ratings"],
                        "endgameStatus": record["endgameStatus"],
                        "moves": record["moves"],
                        "partitioningKey": record["partitioningKey"],
                    }
                    for record in daily_data
                ]
                for daily_data in data
            ],
        )
        sorted_expected_data = sorted(expected_data, key=lambda x: x["id"])
        logger.debug(sorted_expected_data)

        assert len(read_data) == len(expected_data)
        assert all(
            [
                self.compare_records(d1, d2)
                for d1, d2 in zip(sorted_read_data, sorted_expected_data)
            ]
        )

    def test_train_test_split(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """train_test_splitメソッドをテストする"""
        data_name = f"{self.data_name}-4"
        feature_store = S3FeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
        )
        rows = 8
        days = 4
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat("2022-02-02") + timedelta(days=i),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                # partitioning_key_dateを指定する
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    structured_array=structured_array,
                    partitioning_key_date="partitioningKey",
                )
        spliter = S3DataSource.S3DataSourceSpliter(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
        )
        # データを4:1に分割
        train_ds, test_ds = spliter.train_test_split(test_ratio=0.2)
        total_size = len(train_ds) + len(test_ds)
        # 分割が正常に行われていることを確認する
        assert len(train_ds) / total_size == pytest.approx(0.8, abs=1.0 / rows / days)
        assert len(test_ds) / total_size == pytest.approx(0.2, abs=1.0 / rows / days)
        # 一応読み込めるかテスト
        sample_record = train_ds[0]
        assert hasattr(sample_record, "dtype") and sample_record.dtype.names
        assert "id" in sample_record.dtype.names

    def test_iter_batches(self, default_fixture: None, temp_cache_dir: Path) -> None:
        """iter_batchesメソッドをテストする"""
        data_name = f"{self.data_name}-5"
        feature_store = S3FeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
        )
        rows = 3
        days = 4
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat("2022-02-02") + timedelta(days=i),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                # partitioning_key_dateを指定する
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    structured_array=structured_array,
                    partitioning_key_date="partitioningKey",
                )
        data_source = S3DataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
        )

        # バッチを反復処理
        batches = list(data_source.iter_batches())
        assert len(batches) > 0
        assert len(batches) == days

        # 各バッチの形式を確認
        for name, batch in batches:
            assert isinstance(name, str)
            assert isinstance(batch, np.ndarray)
            assert batch.dtype.names is not None
            assert len(batch) == rows
            assert "id" in batch.dtype.names

    def test_missing_config(self, default_fixture: None, temp_cache_dir: Path) -> None:
        """設定不足の場合のエラーをテストする"""
        with pytest.raises(Exception):
            S3DataSource()

        with pytest.raises(Exception):
            S3DataSource(bucket_name=self.bucket)

        with pytest.raises(Exception):
            S3DataSource(prefix=self.prefix)

        with pytest.raises(Exception):
            S3DataSource(data_name=self.data_name)

        with pytest.raises(Exception):
            S3DataSource(local_cache_dir=str(temp_cache_dir))

        with pytest.raises(Exception):
            S3DataSource(
                bucket_name=self.bucket,
                prefix=self.prefix,
            )

        with pytest.raises(Exception):
            S3DataSource(
                bucket_name=self.bucket,
                prefix=self.prefix,
                data_name=self.data_name,
            )

        with pytest.raises(Exception):
            S3DataSource(
                bucket_name=self.bucket,
                prefix=self.prefix,
                local_cache_dir=str(temp_cache_dir),
            )
