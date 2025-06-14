from pathlib import Path
from typing import Any, Generator

import pytest

from maou.app.pre_process import hcpe_transform
from maou.infra.file_system.file_data_source import FileDataSource


class TestHCPEConverter:
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

    def test_successfull_transformation(self) -> None:
        input_paths = [
            Path("tests/maou/app/pre_process/resources/test_dir/input/test_data_1.npy"),
            Path("tests/maou/app/pre_process/resources/test_dir/input/test_data_2.npy"),
            Path("tests/maou/app/pre_process/resources/test_dir/input/test_data_3.npy"),
        ]
        output_dir = Path("tests/maou/app/pre_process/resources/test_dir/output")
        option: hcpe_transform.PreProcess.PreProcessOption = (
            hcpe_transform.PreProcess.PreProcessOption(
                output_dir=output_dir,
            )
        )
        self.clean_up_dir(output_dir)
        datasource = FileDataSource(file_paths=input_paths)
        transformer = hcpe_transform.PreProcess(datasource=datasource)
        transformer.transform(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        for p in input_paths:
            output_file = output_dir / p.with_suffix(".pre.npy").name
            assert output_file.exists()

    def test_failed_transformation_no_input(self) -> None:
        input_paths = [
            Path("tests/maou/app/pre_process/resources/test_dir/input/not_exists.npy"),
        ]
        output_dir = Path("tests/maou/app/pre_process/resources/test_dir/output")
        option: hcpe_transform.PreProcess.PreProcessOption = (
            hcpe_transform.PreProcess.PreProcessOption(
                output_dir=output_dir,
            )
        )
        self.clean_up_dir(output_dir)
        with pytest.raises(FileNotFoundError):
            datasource = FileDataSource(file_paths=input_paths)
            transformer = hcpe_transform.PreProcess(datasource=datasource)
            transformer.transform(option)
