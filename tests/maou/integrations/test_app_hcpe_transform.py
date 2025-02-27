import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import google_crc32c
import numpy as np
import pytest

from maou.app.pre_process.hcpe_transform import PreProcess
from maou.infra.bigquery.bigquery import BigQuery
from maou.infra.bigquery.bq_data_source import BigQueryDataSource
from maou.infra.file_system.file_data_source import FileDataSource

logger: logging.Logger = logging.getLogger("TEST")

skip_test = os.getenv("TEST_GCP", "").lower() != "true"

if skip_test:
    logger.debug(f"Skip {__name__} TEST_GCP: {os.getenv('TEST_GCP', '')}")


# TODO: HCPE Transformのテストにする
@pytest.mark.skipif(
    skip_test,
    reason="GCPリソースを使ったテストはTEST_GCPが'true'の場合のみ実行されます",
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

    @pytest.fixture()
    def default_fixture(self) -> Generator[None, None, None]:
        path = Path("src/maou/app/pre_process/hcpe_transform.py")
        self.dataset_id = "maou_test"
        self.table_name = "test_" + self.__calculate_file_crc32c(path)
        logger.debug(f"Test table: {self.dataset_id}.{self.table_name}")
        self.bq = BigQuery(dataset_id=self.dataset_id, table_name=self.table_name)
        yield
        # clean up
        self.bq._BigQuery__drop_table(  # type: ignore
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

    def test_compare_local_and_bq_data(self, default_fixture: None) -> None:
        """ローカルファイルとBigQueryに保存されたデータが同じか確認する."""
        feature_store = BigQuery(
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
        schema_datasource = {
            "hcp": "hcp",
            "bestMove16": "bestMove16",
            "gameResult": "gameResult",
            "eval": "eval",
        }
        datasource = FileDataSource(file_paths=input_paths, schema=schema_datasource)
        PreProcess(
            datasource=datasource,
            feature_store=feature_store,
        ).transform(option)

        # ローカル
        output_paths = [
            output_dir / input_path.with_suffix(".pre.npy").name
            for input_path in input_paths
        ]
        schema = {
            "id": "id",
            "eval": "eval",
            "features": "features",
            "moveLabel": "moveLabel",
            "resultValue": "resultValue",
            "legalMoveMask": "legalMoveMask",
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
            key=lambda x: x["id"],
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
            key=lambda x: x["id"],
        )
        logger.debug(sorted_bq_data)
        assert all(
            [
                self.compare_dicts(d1, d2)
                for d1, d2 in zip(sorted_local_data, sorted_bq_data)
            ]
        )
