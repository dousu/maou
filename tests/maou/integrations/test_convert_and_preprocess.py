import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import google_crc32c
import numpy as np
import pytest

from maou.app.converter.hcpe_converter import HCPEConverter
from maou.app.pre_process.hcpe_transform import PreProcess
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
class TestIntegrationConverterPreprocess:
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
        hcpe_path = Path("src/maou/app/converter/hcpe_converter.py")
        preprocess_path = Path("src/maou/app/pre_process/hcpe_transform.py")
        self.dataset_id = "maou_test"
        self.table_name_hcpe = "test_" + self.__calculate_file_crc32c(hcpe_path)
        self.table_name_preprocess = "test_" + self.__calculate_file_crc32c(
            preprocess_path
        )
        logger.debug(f"Test table (HCPE): {self.dataset_id}.{self.table_name_hcpe}")
        logger.debug(
            f"Test table (Pre Process): {self.dataset_id}.{self.table_name_preprocess}"
        )
        self.bq = BigQuery(
            dataset_id=self.dataset_id, table_name=self.table_name_preprocess
        )
        yield
        # clean up
        self.bq._BigQuery__drop_table(  # type: ignore
            dataset_id=self.dataset_id, table_name=self.table_name_hcpe
        )
        self.bq._BigQuery__drop_table(  # type: ignore
            dataset_id=self.dataset_id, table_name=self.table_name_preprocess
        )

    @pytest.fixture(autouse=True)
    def clean_up_after_test(self) -> Generator[None, Any, Any]:
        hcpe_output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        preprocess_output_dir = Path(
            "tests/maou/app/pre_process/resources/test_dir/output"
        )
        yield
        self.clean_up_dir(hcpe_output_dir)
        self.clean_up_dir(preprocess_output_dir)

    def clean_up_dir(self, dir: Path) -> None:
        if dir.exists() and dir.is_dir():
            for f in dir.glob("**/*"):
                if f.name != ".gitkeep":
                    logger.debug(f"Remove a file: {f.name}")
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

    def test_file_datasource(self, default_fixture: None) -> None:
        """
        ファイルデータソースを使って変換したデータで
        ローカルファイルとBigQueryに保存されたデータが同じか確認する.
        """
        feature_store_preprocess = BigQuery(
            dataset_id=self.dataset_id,
            table_name=self.table_name_preprocess,
        )
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_3.csa"),
        ]
        hcpe_output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        preprocess_output_dir = Path(
            "tests/maou/app/pre_process/resources/test_dir/output"
        )
        hcpe_option = HCPEConverter.ConvertOption(
            input_paths=input_paths,
            input_format="csa",
            output_dir=hcpe_output_dir,
        )
        HCPEConverter().convert(hcpe_option)
        output_paths_hcpe = [
            hcpe_option.output_dir / input_path.with_suffix(".npy").name
            for input_path in input_paths
        ]

        preprocess_option: PreProcess.PreProcessOption = PreProcess.PreProcessOption(
            output_dir=preprocess_output_dir,
        )
        datasource = FileDataSource(file_paths=output_paths_hcpe)
        transformer = PreProcess(
            datasource=datasource, feature_store=feature_store_preprocess
        )
        transformer.transform(preprocess_option)

        # ローカル
        output_paths = [
            preprocess_output_dir / input_path.with_suffix(".pre.npy").name
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
        )
        local_data = [local_datasource[i] for i in range(len(local_datasource))]
        # ソートはローカルファイルに入っているデータで行わないといけない
        # hcpeには一意に決まるデータはないので各キーをbyteに変換してハッシュ値を計算してソートする
        sorted_local_data = sorted(
            local_data,
            key=lambda x: hash(
                x["eval"].tobytes()
                + x["moveLabel"].tobytes()
                + x["resultValue"].tobytes()
                + x["features"].tobytes()
            ),
        )
        logger.debug(sorted_local_data)

        # BigQuery
        bq_datasource = BigQueryDataSource(
            dataset_id=self.dataset_id, table_name=self.table_name_preprocess
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
        sorted_bq_data = sorted(
            bq_data,
            key=lambda x: hash(
                np.int32(x["eval"]).tobytes()
                + np.int16(x["moveLabel"]).tobytes()
                + np.float32(x["resultValue"]).tobytes()
                + x["features"].tobytes()
            ),
        )
        logger.debug(sorted_bq_data)
        assert all(
            [
                self.compare_dicts(d1, d2)
                for d1, d2 in zip(sorted_local_data, sorted_bq_data)
            ]
        )

    def test_bq_datasource(self, default_fixture: None) -> None:
        """
        BigQueryデータソースを使って変換したデータで
        ローカルファイルとBigQueryに保存されたデータが同じか確認する.
        """
        feature_store_hcpe = BigQuery(
            dataset_id=self.dataset_id,
            table_name=self.table_name_hcpe,
        )
        feature_store_preprocess = BigQuery(
            dataset_id=self.dataset_id,
            table_name=self.table_name_preprocess,
        )
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_3.csa"),
        ]
        hcpe_output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        preprocess_output_dir = Path(
            "tests/maou/app/pre_process/resources/test_dir/output"
        )
        hcpe_option = HCPEConverter.ConvertOption(
            input_paths=input_paths,
            input_format="csa",
            output_dir=hcpe_output_dir,
        )
        HCPEConverter(feature_store=feature_store_hcpe).convert(hcpe_option)

        preprocess_option: PreProcess.PreProcessOption = PreProcess.PreProcessOption(
            output_dir=preprocess_output_dir,
        )
        datasource = BigQueryDataSource(
            dataset_id=self.dataset_id,
            table_name=self.table_name_hcpe,
            partitioning_key_date="partitioningKey",
        )
        transformer = PreProcess(
            datasource=datasource, feature_store=feature_store_preprocess
        )
        transformer.transform(preprocess_option)

        # ローカル
        # パーティションの値でファイル名が作られるので
        # 入力データに合わせてoutput_pathを作る
        output_paths = [
            preprocess_output_dir / p.with_suffix(".pre.npy").name
            for p in [Path("2020-12-03"), Path("2020-01-30"), Path("2020-01-31")]
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
        )
        local_data = [local_datasource[i] for i in range(len(local_datasource))]
        # ソートはローカルファイルに入っているデータで行わないといけない
        # hcpeには一意に決まるデータはないので各キーをbyteに変換してハッシュ値を計算してソートする
        sorted_local_data = sorted(
            local_data,
            key=lambda x: hash(
                x["eval"].tobytes()
                + x["moveLabel"].tobytes()
                + x["resultValue"].tobytes()
                + x["features"].tobytes()
            ),
        )
        logger.debug(sorted_local_data)

        # BigQuery
        bq_datasource = BigQueryDataSource(
            dataset_id=self.dataset_id, table_name=self.table_name_preprocess
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
        sorted_bq_data = sorted(
            bq_data,
            key=lambda x: hash(
                np.int32(x["eval"]).tobytes()
                + np.int16(x["moveLabel"]).tobytes()
                + np.float32(x["resultValue"]).tobytes()
                + x["features"].tobytes()
            ),
        )
        logger.debug(sorted_bq_data)
        assert all(
            [
                self.compare_dicts(d1, d2)
                for d1, d2 in zip(sorted_local_data, sorted_bq_data)
            ]
        )
