import logging
import os
import pickle
from pathlib import Path

import google_crc32c
import numpy as np
import pytest
from google.cloud import bigquery

from maou.infra.bigquery.bq_feature_store import (
    BigQueryFeatureStore,
)

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
class TestBigQueryFeatureStore:
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

    @pytest.fixture
    def default_fixture(self) -> None:
        path = Path(
            "src/maou/infra/bigquery/bq_feature_store.py"
        )
        self.dataset_id = "maou_test"
        self.table_name = (
            "test_" + self.__calculate_file_crc32c(path)
        )
        logger.debug(
            f"Test table: {self.dataset_id}.{self.table_name}"
        )
        self.test_class = BigQueryFeatureStore(
            dataset_id=self.dataset_id,
            table_name=self.table_name,
        )
        self.client = bigquery.Client()
        self.table_id = f"{self.client.project}.{self.dataset_id}.{self.table_name}"

    def test_store_features(
        self, default_fixture: None
    ) -> None:
        data = np.array(
            [
                (
                    1,
                    10,
                    "20",
                    True,
                    0.1,
                    pickle.dumps(np.array([0.01, 1000, 0])),
                    np.array([0.01, 1000, 0]),
                ),
                (
                    2,
                    11,
                    "21",
                    True,
                    0.2,
                    pickle.dumps(np.array([0.02, 2000, 0])),
                    np.array([0.02, 2000, 0]),
                ),
                (
                    3,
                    21,
                    "31",
                    False,
                    0.00000001,
                    pickle.dumps(None),
                    np.array([0.03, 3000, 0]),
                ),
            ],
            dtype=[
                ("id", "i4"),
                ("data1", "i8"),
                ("data2", "U16"),
                ("data3", "b"),
                ("data4", "f8"),
                ("data5", "S512"),
                ("data6", ("f8", 3)),
            ],
        )

        schema = self.test_class._BigQueryFeatureStore__generate_schema(  # type: ignore
            data
        )
        logger.debug(f"BigQueryスキーマ: {schema}")
        logger.debug(f"Numpy Source Data: {data}")
        key_columns = ["id"]
        self.test_class.store_features(
            name="test_features",
            key_columns=key_columns,
            structured_array=data,
        )
        self.test_class.flush_features(key_columns=key_columns)

        # 中身を確認する
        numpy_array = self.test_class.select_all(
            dataset_id=self.dataset_id,
            table_name=self.table_name,
        )
        bigquery_table = np.sort(numpy_array)
        logger.debug(
            f"Selected data from bigquery: {bigquery_table}"
        )
        logger.debug(
            f"Data Schema: {data.dtype}, Selected Data Schema: {bigquery_table.dtype}"
        )
        # ソートしてから比較する
        # data6をpickle.dumpsしてから比較する
        data2 = self.test_class._BigQueryFeatureStore__numpy_flatten_nested_column(  # type: ignore
            data
        )
        assert np.array_equal(np.sort(data2), bigquery_table)

        # data5を取り出して元のnumpyに戻るか確認する
        logger.debug(bigquery_table["data5"])
        # np.arrayはsetにできないので少し回りくどいやり方をしている
        actual = [
            pickle.loads(x) for x in bigquery_table["data5"]
        ]
        expected = [
            np.array([0.01, 1000, 0]),
            np.array([0.02, 2000, 0]),
            None,
        ]
        assert all(
            (
                np.array_equal(a, e)
                if isinstance(e, np.ndarray)
                else a == e
            )
            for a, e in zip(actual, expected)
        )
        # clean up
        # 作成したテーブルを削除する
        self.test_class._BigQueryFeatureStore__drop_table(  # type: ignore
            dataset_id=self.dataset_id,
            table_name=self.table_name,
        )
