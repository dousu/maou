import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import google_crc32c
import numpy as np
import pytest

from maou.app.pre_process.hcpe_transform import PreProcess
from maou.infra.bigquery.bq_data_source import BigQueryDataSource
from maou.infra.bigquery.bq_feature_store import BigQueryFeatureStore
from maou.infra.file_system.file_data_source import FileDataSource
from maou.infra.s3.s3_data_source import S3DataSource
from maou.infra.s3.s3_feature_store import S3FeatureStore

logger: logging.Logger = logging.getLogger("TEST")

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
class TestIntegrationPreProcess:
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
        path = Path("src/maou/app/pre_process/hcpe_transform.py")
        self.dataset_id = "maou_test"
        self.table_name = "test_" + self.__calculate_file_crc32c(path)
        logger.debug(f"Test table: {self.dataset_id}.{self.table_name}")
        self.bq = BigQueryFeatureStore(
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        self.bucket = "maou-test-bucket"
        self.prefix = "test-integration-" + self.__calculate_file_crc32c(path)
        self.data_name = "test_data"
        yield
        # clean up
        self.bq._BigQueryFeatureStore__drop_table(  # type: ignore
            dataset_id=self.dataset_id, table_name=self.table_name
        )

    @pytest.fixture(autouse=True)
    def clean_up_after_test(self) -> Generator[None, Any, Any]:
        output_dir = Path("tests/maou/app/pre_process/resources/test_dir/output")
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
            Path("tests/maou/app/pre_process/resources/test_dir/input/test_data_1.npy"),
            Path("tests/maou/app/pre_process/resources/test_dir/input/test_data_2.npy"),
            Path("tests/maou/app/pre_process/resources/test_dir/input/test_data_3.npy"),
        ]
        output_dir = Path("tests/maou/app/pre_process/resources/test_dir/output")
        option: PreProcess.PreProcessOption = PreProcess.PreProcessOption(
            output_dir=output_dir,
        )
        datasource = FileDataSource(file_paths=input_paths)
        PreProcess(
            datasource=datasource,
            feature_store=feature_store,
        ).transform(option)

        # ローカル
        output_paths = [
            output_dir / input_path.with_suffix(".pre.npy").name
            for input_path in input_paths
        ]
        local_datasource = FileDataSource(
            file_paths=output_paths,
        )
        local_data = [local_datasource[i] for i in range(len(local_datasource))]
        # ソートはhcpeに入っているデータで行わないといけない
        # hcpeには一意に決まるデータはないので各キーをbyteに変換してハッシュ値を計算してソートする
        sorted_local_data = sorted(
            local_data,
            key=lambda x: x["id"],
        )
        logger.debug(sorted_local_data)

        # BigQuery
        bq_datasource = BigQueryDataSource(
            dataset_id=self.dataset_id, table_name=self.table_name
        )
        bq_data = [
            {key: data for key, data in bq_datasource[i].items()}
            for i in range(len(bq_datasource))
        ]
        # BQのデータはIDで一意になるがローカルに合わせてソートする
        # このときhcpeの定義を使って適切なnumpy型に変換してからbyteにする
        sorted_bq_data = sorted(
            bq_data,
            key=lambda x: x["id"],
        )
        logger.debug(sorted_bq_data)
        assert all(
            [
                self.compare_dicts(d1, d2)
                for d1, d2 in zip(sorted_local_data, sorted_bq_data)
            ]
        )

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
            Path("tests/maou/app/pre_process/resources/test_dir/input/test_data_1.npy"),
            Path("tests/maou/app/pre_process/resources/test_dir/input/test_data_2.npy"),
            Path("tests/maou/app/pre_process/resources/test_dir/input/test_data_3.npy"),
        ]
        output_dir = Path("tests/maou/app/pre_process/resources/test_dir/output")
        option: PreProcess.PreProcessOption = PreProcess.PreProcessOption(
            output_dir=output_dir,
        )
        datasource = FileDataSource(file_paths=input_paths)
        PreProcess(
            datasource=datasource,
            feature_store=feature_store,
        ).transform(option)

        # ローカル
        output_paths = [
            output_dir / input_path.with_suffix(".pre.npy").name
            for input_path in input_paths
        ]
        local_datasource = FileDataSource(
            file_paths=output_paths,
        )
        local_data = [local_datasource[i] for i in range(len(local_datasource))]
        # ソートはhcpeに入っているデータで行わないといけない
        # hcpeには一意に決まるデータはないので各キーをbyteに変換してハッシュ値を計算してソートする
        sorted_local_data = sorted(
            local_data,
            key=lambda x: x["id"],
        )
        logger.debug(sorted_local_data)

        # S3
        s3_datasource = S3DataSource(
            bucket_name=self.bucket,
            prefix=self.prefix,
            data_name=self.data_name,
            local_cache_dir=str(temp_s3_cache_dir),
        )
        s3_data = [
            {key: data for key, data in s3_datasource[i].items()}
            for i in range(len(s3_datasource))
        ]
        # BQのデータはIDで一意になるがローカルに合わせてソートする
        # このときhcpeの定義を使って適切なnumpy型に変換してからbyteにする
        sorted_s3_data = sorted(
            s3_data,
            key=lambda x: x["id"],
        )
        logger.debug(sorted_s3_data)
        assert all(
            [
                self.compare_dicts(d1, d2)
                for d1, d2 in zip(sorted_local_data, sorted_s3_data)
            ]
        )
