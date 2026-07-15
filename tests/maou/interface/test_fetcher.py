"""interface.fetcher のバリデーションと JSON 出力のテスト．"""

import json
from datetime import date
from pathlib import Path

import pytest

from maou.app.fetcher.floodgate_fetcher import (
    HttpClient,
    HttpNotFoundError,
    build_day_url,
)
from maou.interface import fetcher

DAY = date(2025, 1, 5)
BASE_URL = "https://example.test/x/"
CSA_NAME = "wdoor+floodgate-300-10F+A+B+20250105233006.csa"
LISTING = (
    f'<html><body><a href="{CSA_NAME}">csa</a></body></html>'
).encode("utf-8")


class FakeHttpClient(HttpClient):
    def get(self, url: str) -> bytes:
        if url == build_day_url(BASE_URL, DAY):
            return LISTING
        if url.endswith(".csa"):
            return b"V2"
        raise HttpNotFoundError(url)

    def download(self, url: str, dest: Path) -> None:
        raise HttpNotFoundError(url)


class TestFetchFloodgate:
    def test_returns_json_summary(self, tmp_path: Path) -> None:
        result = fetcher.fetch_floodgate(
            FakeHttpClient(),
            start_date=DAY,
            output_dir=tmp_path,
            strategy="daily",
            base_url=BASE_URL,
            delay_seconds=0.0,
        )
        summary = json.loads(result)
        assert summary["files_downloaded"] == 1
        assert summary["days_total"] == 1
        # end_date 省略時は start_date のみ
        assert summary["days_fetched"] == 1

    def test_end_before_start_raises(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            fetcher.fetch_floodgate(
                FakeHttpClient(),
                start_date=DAY,
                end_date=date(2025, 1, 4),
                output_dir=tmp_path,
            )

    def test_unknown_strategy_raises(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            fetcher.fetch_floodgate(
                FakeHttpClient(),
                start_date=DAY,
                output_dir=tmp_path,
                strategy="bogus",
            )

    def test_negative_delay_raises(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            fetcher.fetch_floodgate(
                FakeHttpClient(),
                start_date=DAY,
                output_dir=tmp_path,
                delay_seconds=-1.0,
            )

    def test_output_dir_is_file_raises(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "file.txt"
        target.write_text("x")
        with pytest.raises(ValueError):
            fetcher.fetch_floodgate(
                FakeHttpClient(),
                start_date=DAY,
                output_dir=target,
            )
