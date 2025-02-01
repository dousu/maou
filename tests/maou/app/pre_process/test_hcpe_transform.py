import typing
from pathlib import Path

import pytest

from maou.app.pre_process import hcpe_transform


class TestHCPEConverter:
    @pytest.fixture
    def default_fixture(self) -> None:
        self.test_class = hcpe_transform.PreProcess()

    def clean_up_dir(self, dir: Path) -> None:
        if dir.exists() and dir.is_dir():
            for f in dir.glob("**/*"):
                f.unlink()

    def test_successfull_transformation(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path(
                "tests/maou/app/pre_process/resources/test_dir/input/test_data_1.hcpe"
            ),
            Path(
                "tests/maou/app/pre_process/resources/test_dir/input/test_data_2.hcpe"
            ),
            Path(
                "tests/maou/app/pre_process/resources/test_dir/input/test_data_3.hcpe"
            ),
        ]
        output_dir = Path("tests/maou/app/pre_process/resources/test_dir/output")
        option: hcpe_transform.PreProcess.PreProcessOption = (
            hcpe_transform.PreProcess.PreProcessOption(
                input_paths=input_paths,
                output_dir=output_dir,
            )
        )
        self.clean_up_dir(output_dir)
        self.test_class.transform(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        for p in input_paths:
            output_file = output_dir / p.with_suffix(".pickle").name
            assert output_file.exists()

    def test_failed_transformation_no_input(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/pre_process/resources/test_dir/input/not_exists.hcpe"),
        ]
        output_dir = Path("tests/maou/app/pre_process/resources/test_dir/output")
        option: hcpe_transform.PreProcess.PreProcessOption = (
            hcpe_transform.PreProcess.PreProcessOption(
                input_paths=input_paths,
                output_dir=output_dir,
            )
        )
        self.clean_up_dir(output_dir)
        with pytest.raises(FileNotFoundError):
            self.test_class.transform(option)
