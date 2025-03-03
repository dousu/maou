import logging
import os
import re
import uuid
from collections.abc import Generator
from datetime import date, datetime
from pathlib import Path
from typing import Any

import google_crc32c
import numpy as np
import pyarrow as pa
import pytest

from maou.app.converter.hcpe_converter import HCPEConverter
from maou.infra.bigquery.bigquery import BigQuery
from maou.infra.bigquery.bq_data_source import BigQueryDataSource
from maou.infra.file_system.file_data_source import FileDataSource

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_test:
    logger.debug(f"Skip {__name__} TEST_GCP: {os.getenv('TEST_GCP', '')}")


@pytest.mark.skipif(
    skip_test,
    reason="GCPリソースを使ったテストはTEST_GCPが'true'の場合のみ実行されます",
)
class TestIntegrationHcpeConverter:
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
        path = Path("src/maou/app/converter/hcpe_converter.py")
        self.dataset_id = "maou_test"
        self.table_name = "test_" + self.__calculate_file_crc32c(path)
        logger.debug(f"Test table: {self.dataset_id}.{self.table_name}")
        self.bq = BigQuery(dataset_id=self.dataset_id, table_name=self.table_name)
        self.table_id = f"{self.bq.client.project}.{self.dataset_id}.{self.table_name}"
        yield
        # clean up
        self.bq._BigQuery__drop_table(  # type: ignore
            dataset_id=self.dataset_id, table_name=self.table_name
        )

    @pytest.fixture(autouse=True)
    def clean_up_after_test(self) -> Generator[None, Any, Any]:
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        yield
        self.clean_up_dir(output_dir)

    def clean_up_dir(self, dir: Path) -> None:
        if dir.exists() and dir.is_dir():
            for f in dir.glob("**/*"):
                if f.name != ".gitkeep":
                    f.unlink()

    def compare_dicts(self, d1: dict, d2: dict) -> bool:
        if d1.keys() != d2.keys():
            logger.debug(f"keys: {d1.keys()} != {d2.keys()}")
            return False
        for key in d1:
            if (
                isinstance(d1[key], np.memmap)
                or isinstance(d1[key], np.ndarray)
                or isinstance(d2[key], np.memmap)
                or isinstance(d2[key], np.ndarray)
            ):
                if not np.array_equal(d1[key], d2[key]):
                    logger.debug(f"{key}: {d1[key]} != {d2[key]}")
                    return False
            elif d1[key] != d2[key]:
                logger.debug(f"{key}: {d1[key]} != {d2[key]}")
                return False
        return True

    def insert_partitioning_test_data(self) -> None:
        # 20MB 以下のデータを生成 (20MBが最小課金容量)
        num_rows = 40000
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
            "hcp": [np.random.bytes(16) for _ in range(num_rows)],
            "eval": np.random.randint(-1000, 1000, num_rows),
            "bestMove16": np.random.randint(0, 65536, num_rows),
            "gameResult": np.random.randint(-1, 2, num_rows),
            "ratings": [np.random.bytes(8) for _ in range(num_rows)],
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

    def test_compare_local_and_bq_data(self, default_fixture: None) -> None:
        """ローカルファイルとBigQueryに保存されたデータが同じか確認する."""
        feature_store = BigQuery(
            dataset_id=self.dataset_id,
            table_name=self.table_name,
        )
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_3.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option = HCPEConverter.ConvertOption(
            input_paths=input_paths,
            input_format="csa",
            output_dir=output_dir,
        )
        HCPEConverter(
            feature_store=feature_store,
        ).convert(option)

        # ローカル
        output_paths = [
            option.output_dir / input_path.with_suffix(".npy").name
            for input_path in input_paths
        ]
        schema = {
            "hcp": "hcp",
            "bestMove16": "bestMove16",
            "gameResult": "gameResult",
            "eval": "eval",
        }
        local_datasource = FileDataSource(
            file_paths=output_paths,
            schema=schema,
        )
        local_data = [local_datasource[i] for i in range(len(local_datasource))]
        # ソートはhcpeに入っているデータで行わないといけない
        # hcpeには一意に決まるデータはないので各キーをbyteに変換してハッシュ値を計算してソートする
        sorted_local_data = sorted(
            local_data,
            key=lambda x: hash(
                x["hcp"].tobytes()
                + x["eval"].tobytes()
                + x["bestMove16"].tobytes()
                + x["gameResult"].tobytes()
            ),
        )
        logger.debug(sorted_local_data)

        # BigQuery
        bq_datasource = BigQueryDataSource(
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        bq_data = [
            {
                key: data
                for key, data in bq_datasource[i].items()
                if key in schema.keys()
            }
            for i in range(len(bq_datasource))
        ]
        # BQのデータはIDで一意になるがローカルに合わせてソートする
        # このときhcpeの定義を使って適切なnumpy型に変換してからbyteにする
        sorted_bq_data = sorted(
            bq_data,
            key=lambda x: hash(
                x["hcp"].tobytes()
                + np.int16(x["eval"]).tobytes()
                + np.int16(x["bestMove16"]).tobytes()
                + np.int8(x["gameResult"]).tobytes()
            ),
        )
        logger.debug(sorted_bq_data)
        assert all(
            [
                self.compare_dicts(d1, d2)
                for d1, d2 in zip(sorted_local_data, sorted_bq_data)
            ]
        )

    def test_partitioning_key_partitioning(self, default_fixture: None) -> None:
        """日付パーティショニングキーを使用した場合にmergeクエリの読み込みバイト数が減少している."""

        self.insert_partitioning_test_data()

        start_time = datetime.now()

        feature_store = BigQuery(
            dataset_id=self.dataset_id,
            table_name=self.table_name,
        )
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_3.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option = HCPEConverter.ConvertOption(
            input_paths=input_paths,
            input_format="csa",
            output_dir=output_dir,
        )
        HCPEConverter(
            feature_store=feature_store,
        ).convert(option)

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
                pattern = f"DELETE FROM.*\\n.*{re.escape(self.dataset_id)}\\.{re.escape(self.table_name)}.*"
                if re.search(pattern, job.query):
                    total_bytes_processed = job.total_bytes_processed
        logger.debug(
            f"target table bytes: {table.num_bytes},"
            f" total_bytes_processed: {total_bytes_processed}"
        )
        assert total_bytes_processed is not None and table.num_bytes is not None
        assert total_bytes_processed < table.num_bytes
        # 現在はパーティショニングだがクラスタリングだとある程度多めにかかってしまう
        # 現時点だと300kBでいいはずなのに7MBほどかかった
        # 自動の再クラスタリングが働かないこともあるのかもしれない
        # 現在のデータならデータソースは211.75kB程度になるのでほぼ最小であることを確認している
        assert total_bytes_processed < 300 * 1024
