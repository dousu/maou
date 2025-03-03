import logging
import os
import pickle
import re
import uuid
from collections.abc import Generator
from datetime import date, datetime
from pathlib import Path

import google_crc32c
import numpy as np
import pyarrow as pa
import pytest
from google.cloud import bigquery

from maou.infra.bigquery.bigquery import BigQuery
from maou.infra.bigquery.bq_data_source import BigQueryDataSource

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_test:
    logger.debug(f"Skip {__name__} TEST_GCP: {os.getenv('TEST_GCP', '')}")


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

    def cast_nullable_to_false(self, table: pa.Table) -> pa.Table:
        schema = pa.schema(
            [pa.field(f.name, f.type, nullable=False) for f in table.schema]
        )
        return table.cast(schema)

    def insert_partitioning_test_data(self) -> None:
        # 20MB 以下のデータを生成 (20MBが最小課金容量)
        # 1行約116バイト (36+16+4+4+4+8+4.5+4+36) × 100,000 行 = 11MB
        # これにオーバーヘッドがのって少し大きくなる
        num_rows = 100000
        partitioning_values = np.array(
            [
                date.fromisoformat("2019-12-04"),
                date.fromisoformat("2019-12-05"),
                date.fromisoformat("2019-12-07"),
                date.fromisoformat("2019-12-30"),
            ]
        )
        partitioning_keys = np.tile(
            partitioning_values, num_rows // len(partitioning_values)
        )
        num_remaining = num_rows - len(partitioning_keys)
        if num_remaining > 0:
            partitioning_keys = np.concatenate(
                [
                    partitioning_keys,
                    np.random.choice(partitioning_values, num_remaining),
                ]
            )
        np.random.shuffle(partitioning_keys)
        data = {
            "id": [str(uuid.uuid4()) for _ in range(num_rows)],
            "hcp": [pickle.dumps(np.zeros((2, 3))) for _ in range(num_rows)],
            "eval": np.random.randint(-1000, 1000, num_rows),
            "bestMove16": np.random.randint(0, 65536, num_rows),
            "gameResult": np.random.randint(-1, 2, num_rows),
            "ratings": [pickle.dumps(np.zeros(2)) for _ in range(num_rows)],
            "endgameStatus": np.random.choice(
                ["WIN", "LOSE", "DRAW", "UNKNOWN"], num_rows
            ),
            "moves": np.random.randint(0, 100, num_rows),
            "partitioningKey": partitioning_keys,
        }
        table = pa.Table.from_pydict(
            data,
            schema=pa.schema(
                [
                    pa.field("id", pa.string(), nullable=False),
                    pa.field("hcp", pa.binary(), nullable=False),
                    pa.field("eval", pa.int32(), nullable=False),
                    pa.field("bestMove16", pa.int32(), nullable=False),
                    pa.field("gameResult", pa.int32(), nullable=False),
                    pa.field("ratings", pa.binary(), nullable=False),
                    pa.field("endgameStatus", pa.string(), nullable=False),
                    pa.field("moves", pa.int32(), nullable=False),
                    pa.field("partitioningKey", pa.date64(), nullable=False),
                ]
            ),
        )
        self.bq._BigQuery__create_or_replace_table(  # type: ignore
            dataset_id=self.dataset_id,
            table_name=self.table_name,
            schema=(
                self.bq._BigQuery__generate_schema(arrow_table=table)  # type: ignore
            ),
            clustering_key=None,
            partitioning_key_date="partitioningKey",
        )
        self.bq.load_from_arrow(
            dataset_id=self.dataset_id, table_name=self.table_name, table=table
        )

        logger.debug(f"Uploaded {num_rows} rows to {self.table_id}")

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

    def test_read_data_without_pruning_key(self, default_fixture: None) -> None:
        # パーティショニングやクラスタリングキーが指定されていない場合にbqから正しくデータを読み込める
        # BigQueryにテストデータを投入
        data = self.cast_nullable_to_false(
            pa.table(
                {
                    "id": [1, 2, 3],
                    "data": ["test1", "test2", "test3"],
                }
            )
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
        data = self.cast_nullable_to_false(
            pa.table(
                {
                    "id": [1, 2, 3],
                    "cluster": ["A", "B", "A"],
                    "data": ["test1", "test2", "test3"],
                }
            )
        )
        self.bq.store_features(
            key_columns=["id"], arrow_table=data, clustering_key="cluster"
        )
        self.bq.flush_features(key_columns=["id"], clustering_key="cluster")

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

    def test_read_data_with_partitioning_key(self, default_fixture: None) -> None:
        # パーティショニングキーが指定されている場合にbqから正しくデータを読み込める
        # BigQueryにテストデータを投入
        data = self.cast_nullable_to_false(
            pa.table(
                {
                    "id": [1, 2, 3],
                    "partition_key": [
                        date.fromisoformat("2019-12-04"),
                        date.fromisoformat("2019-12-05"),
                        date.fromisoformat("2019-12-07"),
                    ],
                    "data": ["test1", "test2", "test3"],
                }
            )
        )
        self.bq.store_features(
            key_columns=["id"], arrow_table=data, partitioning_key_date="partition_key"
        )
        self.bq.flush_features(
            key_columns=["id"], partitioning_key_date="partition_key"
        )

        # BigQueryDataSourceからデータを読み込む
        data_source = BigQueryDataSource(
            dataset_id=self.dataset_id,
            table_name=self.table_name,
            partitioning_key_date="partition_key",
        )
        # データを読み込む
        read_data = [data_source[i] for i in range(len(data_source))]
        sorted_read_data = sorted(read_data, key=lambda x: x["id"])
        logger.debug(sorted_read_data)
        # 読み込んだデータが正しいことを確認
        expected_data = [
            {
                "id": 1,
                "partition_key": date.fromisoformat("2019-12-04"),
                "data": "test1",
            },
            {
                "id": 2,
                "partition_key": date.fromisoformat("2019-12-05"),
                "data": "test2",
            },
            {
                "id": 3,
                "partition_key": date.fromisoformat("2019-12-07"),
                "data": "test3",
            },
        ]
        assert sorted_read_data == expected_data

    def test_pruning(self, default_fixture: None) -> None:
        # パーティショニングキーが指定されている場合にbqで最小データ量の読み込みが処理される
        # BigQueryにテストデータを投入
        self.insert_partitioning_test_data()

        # BigQueryDataSourceからデータを読み込む
        data_source = BigQueryDataSource(
            dataset_id=self.dataset_id,
            table_name=self.table_name,
            partitioning_key_date="partitioningKey",
        )
        # データを読み込む
        start_time = datetime.now()
        _ = data_source[0]
        client = self.bq.client

        table = client.get_table(self.table_id)

        jobs = client.list_jobs(min_creation_time=start_time)
        total_bytes_processed: int
        for job in jobs:
            if job.job_type == "query":
                logger.debug(
                    f"QueryJob id: {job.job_id}, query: {job.query},"
                    f" time: {job.created},"
                    f" total_bytes_processed: {job.total_bytes_processed}"
                )
                pattern = (
                    "SELECT.*\\n"
                    f".*{re.escape(self.dataset_id)}\\.{re.escape(self.table_name)}.*"
                )
                if re.search(pattern, job.query):
                    total_bytes_processed = job.total_bytes_processed
        logger.debug(
            f"target table bytes: {table.num_bytes},"
            f" total_bytes_processed: {total_bytes_processed}"
        )
        assert total_bytes_processed is not None and table.num_bytes is not None
        assert total_bytes_processed < table.num_bytes
        # データは4等分しているので3等分よりは小さくなるはず
        assert total_bytes_processed < table.num_bytes / 3

    def test_cache_eviction(self, default_fixture: None) -> None:
        # max_chached_bytesを超えたら古いページが破棄される
        # batch_sizeより大きなレコード数の場合にキャッシュされる
        # BigQueryにテストデータを投入
        data = self.cast_nullable_to_false(
            pa.table(
                {
                    "id": [i for i in range(3)],
                    "data": [f"test{i}" for i in range(3)],
                }
            )
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
        assert (
            data_source._BigQueryDataSource__page_manager.total_cached_bytes
            <= data_source._BigQueryDataSource__page_manager.max_cached_bytes
        )

    def test_batch_size_larger_than_record_count(self, default_fixture: None) -> None:
        # BigQueryにテストデータを投入
        data = self.cast_nullable_to_false(
            pa.table(
                {
                    "id": [i for i in range(5)],
                    "data": [f"test{i}" for i in range(5)],
                }
            )
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
        assert data_source._BigQueryDataSource__page_manager.total_cached_bytes > 0

    def test_read_from_cache(self, default_fixture: None) -> None:
        # キャッシュされている場合にbqにアクセスせずデータを返すことができる
        # BigQueryにテストデータを投入
        data = self.cast_nullable_to_false(
            pa.table(
                {
                    "id": [1],
                    "data": ["test1"],
                }
            )
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
        assert data_source._BigQueryDataSource__page_manager.total_cached_bytes > 0
        # キャッシュされていることを確認するために、total_cached_bytesを保存
        cached_bytes = data_source._BigQueryDataSource__page_manager.total_cached_bytes
        # もう一度データを読み込む
        data_source[0]
        # キャッシュサイズが変わらないことを確認
        assert (
            data_source._BigQueryDataSource__page_manager.total_cached_bytes
            == cached_bytes
        )
