from pathlib import Path
from typing import Any, Generator, Literal

import numpy as np
import pytest

from maou.app.common.data_io_service import (
    load_hcpe_array,
    load_preprocessing_array,
)
from maou.app.pre_process import hcpe_transform


class TempDataSource(hcpe_transform.DataSource):
    def __init__(
        self,
        file_paths: list[Path],
        array_type: Literal["hcpe", "preprocessing"],
        bit_pack: bool,
    ) -> None:
        self.data = {}
        for file_path in file_paths:
            if file_path.exists():
                if array_type == "hcpe":
                    self.data[file_path.name] = load_hcpe_array(
                        file_path=file_path, mmap_mode="r"
                    )
                elif array_type == "preprocessing":
                    self.data[file_path.name] = (
                        load_preprocessing_array(
                            file_path=file_path,
                            bit_pack=bit_pack,
                            mmap_mode="r",
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported array type: {array_type}"
                    )
            else:
                raise FileNotFoundError(
                    f"File not found: {file_path}"
                )

    def __len__(self) -> int:
        return sum(len(arr) for arr in self.data.values())

    def iter_batches(
        self,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        for file_name, arr in self.data.items():
            yield file_name, arr


class TestHCPEConverter:
    @pytest.fixture(autouse=True)
    def clean_up_after_test(self) -> Generator[None, Any, Any]:
        output_dir = Path(
            "tests/maou/app/pre_process/resources/test_dir/output"
        )
        yield
        self.clean_up_dir(output_dir)

    def clean_up_dir(self, dir: Path) -> None:
        if dir.exists() and dir.is_dir():
            for f in dir.glob("**/*"):
                if f.name != ".gitkeep":
                    f.unlink()

    def test_successfull_transformation(self) -> None:
        input_paths = [
            Path(
                "tests/maou/app/pre_process/resources/test_dir/input/test_data_1.npy"
            ),
            Path(
                "tests/maou/app/pre_process/resources/test_dir/input/test_data_2.npy"
            ),
            Path(
                "tests/maou/app/pre_process/resources/test_dir/input/test_data_3.npy"
            ),
        ]
        output_dir = Path(
            "tests/maou/app/pre_process/resources/test_dir/output"
        )
        option: hcpe_transform.PreProcess.PreProcessOption = (
            hcpe_transform.PreProcess.PreProcessOption(
                output_dir=output_dir,
                max_workers=1,
            )
        )
        self.clean_up_dir(output_dir)
        datasource = TempDataSource(
            file_paths=input_paths,
            array_type="hcpe",
            bit_pack=False,
        )
        transformer = hcpe_transform.PreProcess(
            datasource=datasource
        )
        transformer.transform(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        for p in input_paths:
            output_file = (
                output_dir / p.with_suffix(".npy").name
            )
            assert output_file.exists()
