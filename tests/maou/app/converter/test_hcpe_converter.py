import logging
import typing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from maou.app.converter import hcpe_converter


class CustomLogFormatter(logging.Formatter):
    # 時刻はJSTを使用する
    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        tz_jst = timezone(timedelta(hours=+9), "JST")
        ct = datetime.fromtimestamp(record.created, tz=tz_jst)
        s = ct.isoformat(timespec="microseconds")

        return s


handler = logging.StreamHandler()
# jsonにフォーマットするとトレースバックとか見えないのでいったんコメントアウトしておく
# handler.setFormatter(JsonLogFormatter())
handler.setFormatter(
    CustomLogFormatter(
        "%(asctime)s | "
        "%(levelname)-5s | "
        "%(filename)15s | "
        "%(funcName)15s | "
        "%(lineno)3d | "
        "%(threadName)s | "
        "%(message)s"
    )
)
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        # logging.FileHandler("app.log"),  # ログをファイルに出力
        handler,
    ],
)

logger: logging.Logger = logging.getLogger(__name__)


class TestHCPEConverter:
    @pytest.fixture
    def default_fixture(self) -> None:
        self.test_class = hcpe_converter.HCPEConverter()

    def clean_up_dir(self, dir: Path) -> None:
        if dir.exists() and dir.is_dir():
            for f in dir.glob("**/*"):
                f.unlink()

    def test_successfull_conversion(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_3.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option: hcpe_converter.HCPEConverter.ConvertOption = (
            hcpe_converter.HCPEConverter.ConvertOption(
                input_paths=input_paths,
                input_format="csa",
                output_dir=output_dir,
                min_rating=None,
                min_moves=None,
                max_moves=None,
                allowed_endgame_status=None,
            )
        )
        self.clean_up_dir(output_dir)
        self.test_class.convert(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        for p in input_paths:
            output_file = output_dir / p.with_suffix(".hcpe").name
            assert output_file.exists()

    def test_failed_conversion_no_input(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/not_exists.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option: hcpe_converter.HCPEConverter.ConvertOption = (
            hcpe_converter.HCPEConverter.ConvertOption(
                input_paths=input_paths,
                input_format="csa",
                output_dir=output_dir,
                min_rating=None,
                min_moves=None,
                max_moves=None,
                allowed_endgame_status=None,
            )
        )
        self.clean_up_dir(output_dir)
        with pytest.raises(FileNotFoundError):
            self.test_class.convert(option)

    def test_failed_conversion_not_applicable_format(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option: hcpe_converter.HCPEConverter.ConvertOption = (
            hcpe_converter.HCPEConverter.ConvertOption(
                input_paths=input_paths,
                input_format="not applicable format",
                output_dir=output_dir,
                min_rating=None,
                min_moves=None,
                max_moves=None,
                allowed_endgame_status=None,
            )
        )
        self.clean_up_dir(output_dir)
        with pytest.raises(hcpe_converter.NotApplicableFormat):
            self.test_class.convert(option)

    def test_failed_conversion_no_output(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
        ]
        output_dir = Path(
            "tests/maou/app/converter/resources/test_dir/output_not_exists"
        )
        option: hcpe_converter.HCPEConverter.ConvertOption = (
            hcpe_converter.HCPEConverter.ConvertOption(
                input_paths=input_paths,
                input_format="csa",
                output_dir=output_dir,
                min_rating=None,
                min_moves=None,
                max_moves=None,
                allowed_endgame_status=None,
            )
        )
        self.clean_up_dir(output_dir)
        with pytest.raises(FileNotFoundError):
            self.test_class.convert(option)

    def test_conversion_filter_min_rating(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option: hcpe_converter.HCPEConverter.ConvertOption = (
            hcpe_converter.HCPEConverter.ConvertOption(
                input_paths=input_paths,
                input_format="csa",
                output_dir=output_dir,
                min_rating=2000,
                min_moves=None,
                max_moves=None,
                allowed_endgame_status=None,
            )
        )
        self.clean_up_dir(output_dir)
        self.test_class.convert(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        assert (output_dir / Path("test_data_1").with_suffix(".hcpe").name).exists()
        assert not (output_dir / Path("test_data_2").with_suffix(".hcpe").name).exists()

    def test_conversion_filter_min_moves(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option: hcpe_converter.HCPEConverter.ConvertOption = (
            hcpe_converter.HCPEConverter.ConvertOption(
                input_paths=input_paths,
                input_format="csa",
                output_dir=output_dir,
                min_rating=None,
                min_moves=113,
                max_moves=None,
                allowed_endgame_status=None,
            )
        )
        self.clean_up_dir(output_dir)
        self.test_class.convert(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        assert (output_dir / Path("test_data_1").with_suffix(".hcpe").name).exists()
        assert not (output_dir / Path("test_data_2").with_suffix(".hcpe").name).exists()

    def test_conversion_filter_max_moves(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option: hcpe_converter.HCPEConverter.ConvertOption = (
            hcpe_converter.HCPEConverter.ConvertOption(
                input_paths=input_paths,
                input_format="csa",
                output_dir=output_dir,
                min_rating=None,
                min_moves=None,
                max_moves=120,
                allowed_endgame_status=None,
            )
        )
        self.clean_up_dir(output_dir)
        self.test_class.convert(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        assert not (output_dir / Path("test_data_1").with_suffix(".hcpe").name).exists()
        assert (output_dir / Path("test_data_2").with_suffix(".hcpe").name).exists()

    def test_conversion_filter_allowed_endgame_status(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_3.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option: hcpe_converter.HCPEConverter.ConvertOption = (
            hcpe_converter.HCPEConverter.ConvertOption(
                input_paths=input_paths,
                input_format="csa",
                output_dir=output_dir,
                min_rating=None,
                min_moves=None,
                max_moves=None,
                allowed_endgame_status=["%TORYO"],
            )
        )
        self.clean_up_dir(output_dir)
        self.test_class.convert(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        assert (output_dir / Path("test_data_1").with_suffix(".hcpe").name).exists()
        assert (output_dir / Path("test_data_2").with_suffix(".hcpe").name).exists()
        assert not (output_dir / Path("test_data_3").with_suffix(".hcpe").name).exists()

    def test_conversion_composite_filter_(
        self, default_fixture: typing.Annotated[None, pytest.fixture]
    ) -> None:
        input_paths = [
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_1.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_2.csa"),
            Path("tests/maou/app/converter/resources/test_dir/input/test_data_3.csa"),
        ]
        output_dir = Path("tests/maou/app/converter/resources/test_dir/output")
        option: hcpe_converter.HCPEConverter.ConvertOption = (
            hcpe_converter.HCPEConverter.ConvertOption(
                input_paths=input_paths,
                input_format="csa",
                output_dir=output_dir,
                min_rating=2000,
                min_moves=113,
                max_moves=120,
                allowed_endgame_status=["%TORYO"],
            )
        )
        self.clean_up_dir(output_dir)
        self.test_class.convert(option)
        # 出力ファイルのチェック
        assert output_dir.exists()
        assert not (output_dir / Path("test_data_1").with_suffix(".hcpe").name).exists()
        assert not (output_dir / Path("test_data_2").with_suffix(".hcpe").name).exists()
        assert not (output_dir / Path("test_data_3").with_suffix(".hcpe").name).exists()
