import logging
import os
import pickle
from pathlib import Path

import google_crc32c
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from google.cloud import bigquery

from maou.infra.bigquery.bigquery import BigQuery

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_test:
    logger.debug(f"Skip {__name__} TEST_GCP: {os.getenv("TEST_GCP", "")}")


@pytest.mark.skipif(
    skip_test,
    reason="GCPリソースを使ったテストはTEST_GCPが'true'の場合のみ実行されます",
)
class TestBigQuery:
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

    @pytest.fixture
    def default_fixture(self) -> None:
        path = Path("src/maou/infra/bigquery/bigquery.py")
        self.dataset_id = "maou_test"
        self.table_name = "test_" + self.__calculate_file_crc32c(path)
        logger.debug(f"Test table: {self.dataset_id}.{self.table_name}")
        self.test_class = BigQuery(
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        self.client = bigquery.Client()
        self.table_id = f"{self.client.project}.{self.dataset_id}.{self.table_name}"

    def test_store_features(self, default_fixture: None) -> None:
        # pyarrow tableは一定のルールでソートする
        def sort_table(table: pa.Table) -> pa.Table:
            # ソートキーとして全ての列を使う
            # カラム名をソートした順序で，列名とorderのタプルを作る
            sort_keys = [(column, "ascending") for column in sorted(table.column_names)]
            logger.debug(sort_keys)
            sorted_indices = pc.sort_indices(table, sort_keys=sort_keys)  # type: ignore
            logger.debug(sorted_indices)
            return table.take(sorted_indices)

        def cast_nullable_to_false(table: pa.Table) -> pa.Table:
            schema = pa.schema(
                [pa.field(f.name, f.type, nullable=False) for f in table.schema]
            )
            return table.cast(schema)

        data = sort_table(
            cast_nullable_to_false(
                pa.table(
                    {
                        "id": [1, 2, 3],
                        "data1": [10, 11, 21],
                        "data2": ["20".encode(), "21".encode(), "31".encode()],
                        "data3": [True, True, False],
                        "data4": [0.1, 0.2, 0.00000001],
                        "data5": [
                            pickle.dumps(np.array([0.01, 1000, 0])),
                            pickle.dumps(np.array([0.02, 2000, 0])),
                            pickle.dumps(None),
                        ],
                    }
                )
            )
        )

        schema = self.test_class._BigQuery__generate_schema(data)  # type: ignore
        logger.debug(f"BigQueryスキーマ: {schema}")
        logger.debug(f"PyArrow Table: {data}")
        key_columns = ["id"]
        self.test_class.store_features(key_columns=key_columns, arrow_table=data)
        self.test_class.flush_features(key_columns=key_columns)

        # 中身を確認する
        arrow_batches = self.test_class.select_all(
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        # 取得したデータは順序が違う場合があるのでソートしてから比較する
        bigquery_table = sort_table(
            cast_nullable_to_false(pa.Table.from_batches(arrow_batches))
        )
        logger.debug(f"Selected data from bigquery: {bigquery_table}")
        logger.debug(
            f"Data Schema: {data.schema},"
            f" Selected Data Schema: {bigquery_table.schema}"
        )
        assert data.equals(bigquery_table)

        # data2を取り出してdecodeできるか確認する
        logger.debug(bigquery_table.column("data2").to_pylist())
        assert set(
            [x.decode() for x in bigquery_table.column("data2").to_pylist()]
        ) == set(["20", "21", "31"])

        # data5を取り出して元のnumpyに戻るか確認する
        logger.debug(bigquery_table.column("data5").to_pylist())
        # np.arrayはsetにできないので少し回りくどいやり方をしている
        actual = [pickle.loads(x) for x in bigquery_table.column("data5").to_pylist()]
        expected = [
            np.array([0.01, 1000, 0]),
            np.array([0.02, 2000, 0]),
            None,
        ]
        assert all(
            (np.array_equal(a, e) if isinstance(e, np.ndarray) else a == e)
            for a, e in zip(actual, expected)
        )
        # clean up
        # 作成したテーブルを削除する
        self.test_class._BigQuery__drop_table(  # type: ignore
            dataset_id=self.dataset_id, table_name=self.table_name
        )
