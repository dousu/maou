import logging
import os
from collections.abc import Generator
from pathlib import Path

import google_crc32c
import pyarrow as pa
import pytest
from google.cloud import bigquery

from maou.infra.bigquery.bigquery import BigQuery
from maou.infra.bigquery.bq_data_source import BigQueryDataSource

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_test:
    logger.debug(f"Skip {__name__} TEST_GCP: {os.getenv("TEST_GCP", "")}")


@pytest.mark.skipif(
    skip_test,
    reason="GCPリソースを使ったテストはTEST_GCPが'true'の場合のみ実行されます",
)
class TestBigQueryDataSource:
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

    @pytest.fixture()
    def default_fixture(self) -> Generator[None, None, None]:
        path = Path("src/maou/infra/bigquery/bq_data_source.py")
        self.dataset_id = "maou_test"
        self.table_name = "test_" + self.__calculate_file_crc32c(path)
        logger.debug(f"Test table: {self.dataset_id}.{self.table_name}")
        self.bq = BigQuery(dataset_id=self.dataset_id, table_name=self.table_name)
        client = bigquery.Client()
        self.table_id = f"{client.project}.{self.dataset_id}.{self.table_name}"
        yield
        # clean up
        self.bq._BigQuery__drop_table(  # type: ignore
            dataset_id=self.dataset_id, table_name=self.table_name
        )

    def test_read_data_without_clustering_key(self, default_fixture: None) -> None:
        # クラスタリングキーが指定されていない場合にbqから正しくデータを読み込める
        # BigQueryにテストデータを投入
        data = pa.table(
            {
                "id": [1, 2, 3],
                "data": ["test1", "test2", "test3"],
            }
        )
        self.bq.store_features(key_columns=["id"], arrow_table=data)
        self.bq.flush_features(key_columns=["id"])

        # BigQueryDataSourceからデータを読み込む
        data_source = BigQueryDataSource(
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        # データを読み込む
        read_data = [data_source[i] for i in range(len(data_source))]
        sorted_read_data = sorted(read_data, key=lambda x: x["id"])
        logger.debug(sorted_read_data)
        # 読み込んだデータが正しいことを確認
        expected_data = [{"id": i, "data": f"test{i}"} for i in range(1, len(data) + 1)]
        assert len(sorted_read_data) == len(expected_data)
        assert sorted_read_data == expected_data

    def test_read_data_with_clustering_key(self, default_fixture: None) -> None:
        # クラスタリングキーが指定されている場合にbqから正しくデータを読み込める
        # BigQueryにテストデータを投入
        data = pa.table(
            {
                "id": [1, 2, 3],
                "cluster": ["A", "B", "A"],
                "data": ["test1", "test2", "test3"],
            }
        )
        self.bq.store_features(key_columns=["id"], arrow_table=data)
        self.bq.flush_features(key_columns=["id"])

        # BigQueryDataSourceからデータを読み込む
        data_source = BigQueryDataSource(
            dataset_id=self.dataset_id,
            table_name=self.table_name,
            clustering_key="cluster",
        )
        # データを読み込む
        read_data = [data_source[i] for i in range(len(data_source))]
        sorted_read_data = sorted(read_data, key=lambda x: x["id"])
        logger.debug(sorted_read_data)
        # 読み込んだデータが正しいことを確認
        expected_data = [
            {"id": 1, "cluster": "A", "data": "test1"},
            {"id": 2, "cluster": "B", "data": "test2"},
            {"id": 3, "cluster": "A", "data": "test3"},
        ]
        assert sorted_read_data == expected_data

    def test_cache_eviction(self, default_fixture: None) -> None:
        # max_chached_bytesを超えたら古いページが破棄される
        # batch_sizeより大きなレコード数の場合にキャッシュされる
        # BigQueryにテストデータを投入
        data = pa.table(
            {
                "id": [i for i in range(3)],
                "data": [f"test{i}" for i in range(3)],
            }
        )
        self.bq.store_features(key_columns=["id"], arrow_table=data)
        self.bq.flush_features(key_columns=["id"])

        # BigQueryDataSourceからデータを読み込む
        data_source = BigQueryDataSource(
            dataset_id=self.dataset_id,
            table_name=self.table_name,
            max_cached_bytes=20,
            batch_size=1,
        )
        # データを読み込む
        for i in range(len(data)):
            data_source[i]
        # キャッシュサイズが max_cached_bytes より小さいことを確認
        assert data_source.total_cached_bytes <= data_source.max_cached_bytes

    def test_batch_size_larger_than_record_count(self, default_fixture: None) -> None:
        # BigQueryにテストデータを投入
        data = pa.table(
            {
                "id": [i for i in range(5)],
                "data": [f"test{i}" for i in range(5)],
            }
        )
        self.bq.store_features(key_columns=["id"], arrow_table=data)
        self.bq.flush_features(key_columns=["id"])

        # BigQueryDataSourceからデータを読み込む
        data_source = BigQueryDataSource(
            dataset_id=self.dataset_id, table_name=self.table_name, batch_size=10
        )
        # データを読み込む
        data_source[0]
        # キャッシュサイズが 0 より大きいことを確認
        assert data_source.total_cached_bytes > 0

    def test_read_from_cache(self, default_fixture: None) -> None:
        # キャッシュされている場合にbqにアクセスせずデータを返すことができる
        # BigQueryにテストデータを投入
        data = pa.table(
            {
                "id": [1],
                "data": ["test1"],
            }
        )
        self.bq.store_features(key_columns=["id"], arrow_table=data)
        self.bq.flush_features(key_columns=["id"])

        # BigQueryDataSourceからデータを読み込む
        data_source = BigQueryDataSource(
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        # データを読み込む
        data_source[0]
        # キャッシュサイズが 0 より大きいことを確認
        assert data_source.total_cached_bytes > 0
        # キャッシュされていることを確認するために、total_cached_bytesを保存
        cached_bytes = data_source.total_cached_bytes
        # もう一度データを読み込む
        data_source[0]
        # キャッシュサイズが変わらないことを確認
        assert data_source.total_cached_bytes == cached_bytes
