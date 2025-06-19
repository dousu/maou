import logging
import os
import re
import uuid
from collections.abc import Generator
from datetime import date, datetime
from pathlib import Path
from typing import Any, Union

import google_crc32c
import numpy as np
import pytest

from maou.app.converter.hcpe_converter import HCPEConverter
from maou.infra.bigquery.bq_data_source import BigQueryDataSource
from maou.infra.bigquery.bq_feature_store import BigQueryFeatureStore
from maou.infra.file_system.file_data_source import FileDataSource
from maou.infra.s3.s3_data_source import S3DataSource
from maou.infra.s3.s3_feature_store import S3FeatureStore

logger: logging.Logger = logging.getLogger("TEST")


def record_to_dict(record: Union[np.ndarray, dict]) -> dict:
    """Convert numpy structured array record to dict"""
    if hasattr(record, "dtype") and record.dtype.names:  # type: ignore[union-attr]
        return {key: record[key] for key in record.dtype.names}  # type: ignore[union-attr]
    else:
        return record if isinstance(record, dict) else {}


skip_gcp_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_gcp_test:
    logger.debug(f"Skip {__name__} TEST_GCP: {os.getenv('TEST_GCP', '')}")

skip_aws_test = os.getenv("TEST_AWS", "").lower() != "true"

if skip_aws_test:
    logger.debug(f"Skip {__name__} TEST_AWS: {os.getenv('TEST_AWS', '')}")


@pytest.mark.skipif(
    skip_gcp_test and skip_aws_test,
    reason="AWSまたはGCPリソースを使ったテストはTEST_AWSまたはTEST_GCPが'true'の場合のみ実行されます",
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

    @pytest.fixture
    def temp_s3_cache_dir(self, tmp_path: Path) -> Path:
        """一時的なキャッシュディレクトリを提供するフィクスチャ"""
        cache_dir = tmp_path / "s3_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture()
    def default_fixture(self) -> Generator[None, None, None]:
        path = Path("src/maou/app/converter/hcpe_converter.py")
        self.dataset_id = "maou_test"
        self.table_name = "test_" + self.__calculate_file_crc32c(path)
        logger.debug(f"Test table: {self.dataset_id}.{self.table_name}")
        self.bq = BigQueryFeatureStore(
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        self.table_id = f"{self.bq.client.project}.{self.dataset_id}.{self.table_name}"
        self.bucket = "maou-test-bucket"
        # Add timestamp to ensure unique test data
        import time

        timestamp = str(int(time.time() * 1000))  # millisecond timestamp
        self.prefix = (
            "test-integration-" + self.__calculate_file_crc32c(path) + "-" + timestamp
        )
        self.data_name = "test_data"
        yield
        # clean up BigQuery
        self.bq._BigQueryFeatureStore__drop_table(  # type: ignore
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        # clean up S3
        try:
            import boto3

            s3_client = boto3.client("s3")
            response = s3_client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)
            if "Contents" in response:
                objects_to_delete = [
                    {"Key": obj["Key"]} for obj in response["Contents"]
                ]
                if objects_to_delete:
                    s3_client.delete_objects(
                        Bucket=self.bucket, Delete={"Objects": objects_to_delete}
                    )
                    logger.debug(
                        f"Deleted {len(objects_to_delete)} S3 objects "
                        f"with prefix {self.prefix}"
                    )
        except Exception as e:
            logger.warning(f"Failed to clean up S3 objects: {e}")

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

    def compare_records(self, r1: Union[np.ndarray, dict], r2: dict) -> bool:
        """Compare numpy structured array record or dict with dict"""
        # numpy structured arrayの場合はフィールド名を取得
        if hasattr(r1, "dtype") and r1.dtype.names:  # type: ignore[union-attr]
            r1_keys = set(r1.dtype.names)  # type: ignore[union-attr]
            is_structured_array = True
        elif isinstance(r1, dict):
            r1_keys = set(r1.keys())
            is_structured_array = False
        else:
            logger.debug(f"r1 is not a structured array or dict: {type(r1)}")
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
                if is_structured_array and hasattr(r1_val, "item"):
                    r1_val = r1_val.item()
                if r1_val != r2_val:
                    logger.debug(f"{key}: {r1_val} != {r2_val}")
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
        data = [
            (
                id,
                hcp,
                eval,
                bestMove16,
                gameResult,
                ratings,
                endgameStatus,
                moves,
                partitioningKey,
            )
            for id, hcp, eval, bestMove16, gameResult, ratings, endgameStatus, moves, partitioningKey in zip(  # noqa: E501
                [str(uuid.uuid4()) for _ in range(num_rows)],
                [np.zeros(32, dtype=np.uint8) for _ in range(num_rows)],
                np.random.randint(-1000, 1000, num_rows),
                np.random.randint(0, 65536, num_rows),
                np.random.randint(-1, 2, num_rows),
                [np.zeros(2) for _ in range(num_rows)],
                np.random.choice(["WIN", "LOSE", "DRAW", "UNKNOWN"], num_rows),
                np.random.randint(0, 100, num_rows),
                partitioning_keys,
            )
        ]
        structured_array = np.array(
            data,
            dtype=[
                ("id", (np.unicode_, 128)),  # type: ignore[attr-defined]
                ("hcp", (np.uint8, 32)),
                ("eval", np.int16),
                ("bestMove16", np.int16),
                ("gameResult", np.int8),
                ("ratings", (np.uint16, 2)),
                ("endgameStatus", (np.unicode_, 16)),  # type: ignore[attr-defined]
                ("moves", np.int16),
                ("partitioningKey", np.dtype("datetime64[D]")),
            ],
        )
        self.bq._BigQueryFeatureStore__create_or_replace_table(  # type: ignore
            dataset_id=self.dataset_id,
            table_name=self.table_name,
            schema=(
                self.bq._BigQueryFeatureStore__generate_schema(  # type: ignore
                    structured_array=structured_array
                )
            ),
            clustering_key=None,
            partitioning_key_date="partitioningKey",
        )
        self.bq.load_from_numpy_array(
            dataset_id=self.dataset_id,
            table_name=self.table_name,
            structured_array=structured_array,
        )

        logger.debug(f"Uploaded {num_rows} rows to {self.table_id}")

    @pytest.mark.skipif(
        skip_gcp_test,
        reason="GCPリソースを使ったテストはTEST_GCPが'true'の場合のみ実行されます",
    )
    def test_compare_local_and_bq_data(self, default_fixture: None) -> None:
        """ローカルファイルとBigQueryに保存されたデータが同じか確認する."""
        feature_store = BigQueryFeatureStore(
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
            max_workers=1,
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
        )
        # ローカルにはdummyが入っているので取り除く
        local_data = [
            {
                key: data
                for key, data in record_to_dict(local_datasource[i]).items()
                if key in schema.keys()
            }
            for i in range(len(local_datasource))
        ]
        # ソートはhcpeに入っているデータで行わないといけない
        # hcpeには一意に決まるデータはないので各キーをbyteに変換してハッシュ値を計算してソートする
        sorted_local_data = sorted(  # type: ignore
            local_data,
            key=lambda x: hash(
                x["hcp"].tobytes()
                + x["eval"].tobytes()
                + x["bestMove16"].tobytes()
                + x["gameResult"].tobytes()
            ),
        )

        # BigQuery
        bq_datasource = BigQueryDataSource(
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        bq_data = [
            {
                key: data
                for key, data in record_to_dict(bq_datasource[i]).items()
                if key in schema.keys()
            }
            for i in range(len(bq_datasource))
        ]
        # BQのデータはIDで一意になるがローカルに合わせてソートする
        # BQとローカルで型が違うのは許容している
        sorted_bq_data = sorted(  # type: ignore
            bq_data,
            key=lambda x: hash(
                x["hcp"].tobytes()
                + np.int16(x["eval"]).tobytes()
                + np.int16(x["bestMove16"]).tobytes()
                + np.int8(x["gameResult"]).tobytes()
            ),
        )

        logger.debug(f"local: {sorted_local_data[:10]}")
        logger.debug(f"bq: {sorted_bq_data[:10]}")
        assert all(
            [
                self.compare_records(d1, d2)
                for d1, d2 in zip(sorted_local_data, sorted_bq_data)
            ]
        )

    @pytest.mark.skipif(
        skip_gcp_test,
        reason="GCPリソースを使ったテストはTEST_GCPが'true'の場合のみ実行されます",
    )
    def test_partitioning_key_pruning(self, default_fixture: None) -> None:
        """日付パーティショニングキーを使用した場合にmergeクエリの読み込みバイト数が減少している."""

        self.insert_partitioning_test_data()

        start_time = datetime.now()

        feature_store = BigQueryFeatureStore(
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
            max_workers=1,
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
                pattern = (
                    "MERGE.*\\n"
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
        # 現在はパーティショニングだがクラスタリングだとある程度多めにかかってしまう
        # 現時点だと300kBでいいはずなのに7MBほどかかった
        # 自動の再クラスタリングが働かないこともあるのかもしれない
        # 現在のデータならデータソースは211.75kB程度になるのでほぼ最小であることを確認している
        assert total_bytes_processed < 300 * 1024

    @pytest.mark.skipif(
        skip_aws_test,
        reason="AWSリソースを使ったテストはTEST_AWSが'true'の場合のみ実行されます",
    )
    def test_compare_local_and_s3_data(
        self,
        default_fixture: None,
        temp_s3_cache_dir: Path,
    ) -> None:
        """ローカルファイルとS3に保存されたデータが同じか確認する."""
        feature_store = S3FeatureStore(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=self.data_name,
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
            max_workers=1,
        )
        HCPEConverter(
            feature_store=feature_store,
        ).convert(option)

        # ローカル
        output_paths = [
            option.output_dir / input_path.with_suffix(".npy").name
            for input_path in input_paths
        ]
        local_datasource = FileDataSource(
            file_paths=output_paths,
        )
        # ローカルにはdummyが入っているので取り除く
        local_data = [
            record_to_dict(local_datasource[i]) for i in range(len(local_datasource))
        ]
        # ソートはhcpeに入っているデータで行わないといけない
        # hcpeには一意に決まるデータはないので各キーをbyteに変換してハッシュ値を計算してソートする
        sorted_local_data = sorted(  # type: ignore
            local_data,
            key=lambda x: x["id"],
        )

        # S3
        s3_datasource = S3DataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=self.data_name,
            local_cache_dir=str(temp_s3_cache_dir),
        )
        s3_data = [record_to_dict(s3_datasource[i]) for i in range(len(s3_datasource))]
        # s3のデータはIDで一意になるがローカルに合わせてソートする
        # s3とローカルで型が違うのは許容している
        sorted_s3_data = sorted(  # type: ignore
            s3_data,
            key=lambda x: x["id"],
        )

        logger.debug(f"local: {sorted_local_data[:2]}")
        logger.debug(f"s3: {sorted_s3_data[:2]}")

        # 比較失敗の詳細を取得
        results = []
        for i, (d1, d2) in enumerate(zip(sorted_local_data, sorted_s3_data)):
            result = self.compare_records(d1, d2)
            results.append(result)
            if not result and i < 5:  # 最初の5つの失敗例のみ出力
                print(f"Comparison failed at index {i}")
                print(
                    f"  local keys: "
                    f"{set(d1.keys()) if isinstance(d1, dict) else 'not dict'}"
                )
                print(
                    f"  s3 keys: "
                    f"{set(d2.keys()) if isinstance(d2, dict) else 'not dict'}"
                )
                if isinstance(d1, dict) and isinstance(d2, dict):
                    for key in d1.keys():
                        if key in d2:
                            try:
                                if isinstance(d1[key], np.ndarray) or isinstance(
                                    d2[key], np.ndarray
                                ):
                                    if not np.array_equal(d1[key], d2[key]):
                                        print(f"  {key}: arrays differ")
                                else:
                                    if d1[key] != d2[key]:
                                        print(f"  {key}: {d1[key]} != {d2[key]}")
                            except ValueError as e:
                                print(f"  {key}: comparison error - {e}")

        assert all(results)
