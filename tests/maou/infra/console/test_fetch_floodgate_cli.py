"""fetch-floodgate CLI のテスト．

ネットワーク層 (UrllibHttpClient) はフェイクに差し替える．
"""

import json
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

import maou.infra.console.fetch_floodgate as cli_module
from maou.app.fetcher.floodgate_fetcher import (
    DEFAULT_ARCHIVE_BASE_URL,
    DEFAULT_BASE_URL,
    HttpClient,
    HttpNotFoundError,
)
from maou.infra.console.app import LAZY_COMMANDS
from maou.infra.console.fetch_floodgate import (
    fetch_floodgate,
)

CSA_NAME = "wdoor+floodgate-300-10F+A+B+20250105233006.csa"


def parse_summary(output: str) -> dict[str, Any]:
    """出力末尾の JSON サマリを取り出す (tqdm 出力を無視)．

    tqdm の進捗描画は改行で終わらないため，サマリ JSON は
    同じ行に連結されることがある．サマリはフラットな dict
    なので最後の '{' 以降をパースすれば良い．
    """
    idx = output.rfind("{")
    assert idx >= 0, f"no JSON in output: {output!r}"
    summary: dict[str, Any] = json.loads(output[idx:])
    return summary


DAY_URL = DEFAULT_BASE_URL + "2025/01/05/"
LISTING = (
    f'<html><body><a href="{CSA_NAME}">csa</a></body></html>'
).encode("utf-8")


class FakeHttpClient(HttpClient):
    def __init__(self, **kwargs: Any):
        pass

    def get(self, url: str) -> bytes:
        if url == DAY_URL:
            return LISTING
        if url == DAY_URL + CSA_NAME:
            return b"V2"
        raise HttpNotFoundError(url)

    def download(self, url: str, dest: Path) -> None:
        raise HttpNotFoundError(url)


class TestCommandRegistration:
    def test_registered_in_lazy_commands(self) -> None:
        assert "fetch-floodgate" in LAZY_COMMANDS
        spec = LAZY_COMMANDS["fetch-floodgate"]
        assert (
            spec.module_path
            == "maou.infra.console.fetch_floodgate"
        )
        assert spec.attr_name == "fetch_floodgate"
        # 標準ライブラリのみで動くので必須パッケージなし
        assert spec.required_packages == ()

    def test_is_click_command(self) -> None:
        assert isinstance(fetch_floodgate, click.Command)

    def test_option_defaults(self) -> None:
        params = {p.name: p for p in fetch_floodgate.params}
        assert params["start_date"].required
        assert params["output_dir"].required
        assert not params["end_date"].required
        assert params["strategy"].default == "auto"
        assert params["base_url"].default == DEFAULT_BASE_URL
        assert (
            params["archive_base_url"].default
            == DEFAULT_ARCHIVE_BASE_URL
        )
        assert params["delay_seconds"].default == 0.2
        assert params["timeout_seconds"].default == 30.0
        assert params["overwrite"].default is False
        assert params["dry_run"].default is False


class TestCliInvocation:
    @pytest.fixture(autouse=True)
    def fake_http(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cli_module, "UrllibHttpClient", FakeHttpClient
        )

    def test_single_day_fetch(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            fetch_floodgate,
            [
                "--start-date",
                "2025-01-05",
                "--output-dir",
                str(tmp_path),
                "--strategy",
                "daily",
                "--delay",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output
        summary = parse_summary(result.output)
        assert summary["files_downloaded"] == 1
        target = tmp_path / "2025" / "01" / "05" / CSA_NAME
        assert target.read_bytes() == b"V2"

    def test_dry_run(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            fetch_floodgate,
            [
                "--start-date",
                "2025-01-05",
                "--output-dir",
                str(tmp_path),
                "--delay",
                "0",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        summary = parse_summary(result.output)
        assert summary["files_planned"] == 1
        assert not list(tmp_path.rglob("*.csa"))

    def test_invalid_date_rejected(
        self, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            fetch_floodgate,
            [
                "--start-date",
                "2025/01/05",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0

    def test_no_games_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """全日 404 → 空成功ではなく非ゼロ終了．"""
        runner = CliRunner()
        result = runner.invoke(
            fetch_floodgate,
            [
                "--start-date",
                "2030-01-01",
                "--output-dir",
                str(tmp_path),
                "--strategy",
                "daily",
                "--delay",
                "0",
            ],
        )
        assert result.exit_code != 0

    def test_invalid_strategy_rejected(
        self, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            fetch_floodgate,
            [
                "--start-date",
                "2025-01-05",
                "--output-dir",
                str(tmp_path),
                "--strategy",
                "bogus",
            ],
        )
        assert result.exit_code != 0
