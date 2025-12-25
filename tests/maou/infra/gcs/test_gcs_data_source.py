import logging
import os
import uuid
from collections.abc import Generator
from datetime import date, timedelta
from pathlib import Path

import google.cloud.storage as storage
import google_crc32c
import numpy as np
import polars as pl
import pytest

from maou.domain.data.schema import (
    create_empty_hcpe_array,
    get_hcpe_polars_schema,
)
from maou.infra.gcs.gcs_data_source import GCSDataSource
from maou.infra.gcs.gcs_feature_store import GCSFeatureStore

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_test:
    logger.debug(
        f"Skip {__name__} TEST_GCP: {os.getenv('TEST_GCP', '')}"
    )


@pytest.mark.skipif(
    skip_test,
    reason="GCPリソースを使ったテストはTEST_GCPが'true'の場合のみ実行されます",
)
class TestGCSDataSource:
    def __calculate_file_crc32c(self, filepath: Path) -> str:
        """ファイルのCRC32Cハッシュ値を計算する関数.
        ファイルの内容をもとに8文字のハッシュ値が返ってくる
        """
        if not filepath.is_file():
            raise ValueError(
                f"Input file `{filepath}` is not file."
            )

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
        data = create_empty_hcpe_array(num_rows)

        for idx in range(len(data)):
            data[idx]["id"] = str(uuid.uuid4())
            data[idx]["hcp"] = np.zeros(32, dtype=np.uint8)
            data[idx]["eval"] = np.random.randint(-1000, 1000)
            data[idx]["bestMove16"] = np.random.randint(
                -32768, 32768
            )
            data[idx]["gameResult"] = np.random.randint(-1, 2)
            data[idx]["ratings"] = np.zeros(2)
            data[idx]["endgameStatus"] = np.random.choice(
                ["WIN", "LOSE", "DRAW", "UNKNOWN"]
            )
            data[idx]["moves"] = np.random.randint(0, 100)
            data[idx]["partitioningKey"] = np.datetime64(
                partitioning_key
            )

        return data

    def __numpy_to_dataframe(
        self, array: np.ndarray
    ) -> pl.DataFrame:
        """numpy構造化配列をPolars DataFrameに変換する"""
        schema = get_hcpe_polars_schema()
        data_dict = {}
        assert array.dtype.names is not None
        assert array.dtype.fields is not None
        for field in array.dtype.names:
            field_data = array[field]
            field_dtype = array.dtype.fields[field][0]

            # Handle binary fields (convert uint8 arrays to bytes)
            if field == "hcp" or (
                field_dtype.shape
                and field_dtype.base == np.dtype("uint8")
            ):
                data_dict[field] = [
                    bytes(row)
                    if hasattr(row, "__iter__")
                    else bytes([row])
                    for row in field_data
                ]
            else:
                data_dict[field] = field_data.tolist()

        return pl.DataFrame(data_dict, schema=schema)

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        """一時的なキャッシュディレクトリを提供するフィクスチャ"""
        cache_dir = tmp_path / "gcs_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture()
    def default_fixture(self) -> Generator[None, None, None]:
        path = Path("src/maou/infra/gcs/gcs_data_source.py")
        self.bucket = "maou-test-bucket"
        self.prefix = "test-" + self.__calculate_file_crc32c(
            path
        )
        self.data_name = "test_data"
        yield
        # clean up
        if not skip_test:
            try:
                client = storage.Client()
                bucket = client.get_bucket(self.bucket)
                blobs = list(
                    bucket.list_blobs(prefix=self.prefix)
                )
                for blob in blobs:
                    if blob.name == self.prefix:
                        # prefix自体はスキップ
                        continue
                    logger.debug(f"Deleting {blob.name}")
                    blob.delete()
                logger.debug(
                    f"Deleted all objects with prefix {self.prefix}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to clean up test bucket: {e}"
                )

    def test_init_without_pruning_key(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """プルーニングキーなし"""
        data_name = f"{self.data_name}-1"
        feature_store = GCSFeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            array_type="hcpe",
        )
        rows = 3
        days = 2
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat(
                    "2022-02-02"
                )
                + timedelta(days=float(i)),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                df = self.__numpy_to_dataframe(structured_array)
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    dataframe=df,
                )
        data_source = GCSDataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
            array_type="hcpe",
        )

        assert len(data_source) > 0
        assert len(data_source) == rows * days
        # データを読み込む
        read_data = np.concatenate(
            [arr for _, arr in data_source.iter_batches()]
        )
        sorted_read_data = sorted(
            read_data, key=lambda x: x["id"]
        )
        # 読み込んだデータが正しいことを確認
        expected_data = np.concatenate(data)
        sorted_expected_data = sorted(
            expected_data, key=lambda x: x["id"]
        )

        assert len(sorted_read_data) == len(
            sorted_expected_data
        )
        assert np.array_equal(
            sorted_read_data, sorted_expected_data
        )

    def test_init_with_clustering_key(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """クラスタリングキーあり"""
        data_name = f"{self.data_name}-2"
        feature_store = GCSFeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            array_type="hcpe",
        )
        rows = 3
        days = 4
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat(
                    "2022-02-02"
                )
                + timedelta(days=float(i)),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                # clustering_keyを指定する
                df = self.__numpy_to_dataframe(structured_array)
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    dataframe=df,
                    clustering_key="partitioningKey",
                )
        data_source = GCSDataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
            array_type="hcpe",
        )

        assert len(data_source) > 0
        assert len(data_source) == rows * days
        # データを読み込む
        read_data = np.concatenate(
            [arr for _, arr in data_source.iter_batches()]
        )
        sorted_read_data = sorted(
            read_data, key=lambda x: x["id"]
        )
        # 読み込んだデータが正しいことを確認
        expected_data = np.concatenate(data)
        sorted_expected_data = sorted(
            expected_data, key=lambda x: x["id"]
        )

        assert len(read_data) == len(expected_data)
        assert np.array_equal(
            sorted_read_data, sorted_expected_data
        )

    def test_init_with_partitioning_key(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """パーティショニングキーあり"""
        data_name = f"{self.data_name}-3"
        feature_store = GCSFeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            array_type="hcpe",
        )
        rows = 3
        days = 4
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat(
                    "2022-02-02"
                )
                + timedelta(days=float(i)),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                # partitioning_key_dateを指定する
                df = self.__numpy_to_dataframe(structured_array)
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    dataframe=df,
                    partitioning_key_date="partitioningKey",
                )
        data_source = GCSDataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
            array_type="hcpe",
        )

        assert len(data_source) > 0
        assert len(data_source) == rows * days
        # データを読み込む
        read_data = np.concatenate(
            [arr for _, arr in data_source.iter_batches()]
        )
        sorted_read_data = sorted(
            read_data, key=lambda x: x["id"]
        )
        # 読み込んだデータが正しいことを確認
        expected_data = np.concatenate(data)
        sorted_expected_data = sorted(
            expected_data, key=lambda x: x["id"]
        )

        assert len(read_data) == len(expected_data)
        assert np.array_equal(
            sorted_read_data, sorted_expected_data
        )

    def test_train_test_split(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """train_test_splitメソッドをテストする"""
        data_name = f"{self.data_name}-4"
        feature_store = GCSFeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            array_type="hcpe",
        )
        rows = 8
        days = 4
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat(
                    "2022-02-02"
                )
                + timedelta(days=float(i)),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                # partitioning_key_dateを指定する
                df = self.__numpy_to_dataframe(structured_array)
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    dataframe=df,
                    partitioning_key_date="partitioningKey",
                )
        spliter = GCSDataSource.DataSourceSpliter(
            cls_ref=GCSDataSource,
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
            array_type="hcpe",
        )
        # データを4:1に分割
        train_ds, test_ds = spliter.train_test_split(
            test_ratio=0.2
        )
        total_size = len(train_ds) + len(test_ds)
        # 分割が正常に行われていることを確認する
        assert len(train_ds) / total_size == pytest.approx(
            0.8, abs=1.0 / float(rows) / float(days)
        )
        assert len(test_ds) / total_size == pytest.approx(
            0.2, abs=1.0 / float(rows) / float(days)
        )
        # 一応読み込めるかテスト
        sample_record = train_ds[0]
        assert (
            hasattr(sample_record, "dtype")
            and sample_record.dtype.names
        )
        assert "id" in sample_record.dtype.names

    def test_iter_batches(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """iter_batchesメソッドをテストする"""
        data_name = f"{self.data_name}-5"
        feature_store = GCSFeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            array_type="hcpe",
            # max_cached_bytesを極端に少なくすることで
            # ファイル数をarray数にする
            max_cached_bytes=1,
        )
        rows = 3
        days = 4
        data = [
            self.__generate_test_data(
                num_rows=rows,
                partitioning_key=date.fromisoformat(
                    "2022-02-02"
                )
                + timedelta(days=float(i)),
            )
            for i in range(days)
        ]
        with feature_store.feature_store():
            for i, structured_array in enumerate(data):
                # partitioning_key_dateを指定する
                df = self.__numpy_to_dataframe(structured_array)
                feature_store.store_features(
                    name=f"test-data-{i}",
                    key_columns=["id"],
                    dataframe=df,
                    partitioning_key_date="partitioningKey",
                )
        data_source = GCSDataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=data_name,
            local_cache_dir=str(temp_cache_dir),
            array_type="hcpe",
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

    def test_missing_config(
        self, default_fixture: None, temp_cache_dir: Path
    ) -> None:
        """設定不足の場合のエラーをテストする"""
        with pytest.raises(Exception):
            GCSDataSource()

        with pytest.raises(Exception):
            GCSDataSource(bucket_name=self.bucket)

        with pytest.raises(Exception):
            GCSDataSource(prefix=self.prefix)

        with pytest.raises(Exception):
            GCSDataSource(data_name=self.data_name)

        with pytest.raises(Exception):
            GCSDataSource(local_cache_dir=str(temp_cache_dir))

        with pytest.raises(Exception):
            GCSDataSource(
                bucket_name=self.bucket,
                prefix=self.prefix,
            )

        with pytest.raises(Exception):
            GCSDataSource(
                bucket_name=self.bucket,
                prefix=self.prefix,
                data_name=self.data_name,
            )

        with pytest.raises(Exception):
            GCSDataSource(
                bucket_name=self.bucket,
                prefix=self.prefix,
                local_cache_dir=str(temp_cache_dir),
            )
