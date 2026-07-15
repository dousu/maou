"""floodgate 取得ユースケースのテスト．

ネットワークは FakeHttpClient で置換し，実 HTTP は行わない．
リスティング HTML は実サイト (2026-07 時点) の構造を模す．
"""

import io
import tarfile
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from maou.app.fetcher.floodgate_fetcher import (
    ArchiveToolMissingError,
    FloodgateFetcher,
    FloodgateStructureError,
    HttpClient,
    HttpNotFoundError,
    build_day_url,
    csa_filename_to_date,
    date_range,
    extract_archive_days,
    extract_csa_hrefs,
)

BASE_URL = "https://example.test/x/"
ARCHIVE_BASE_URL = "https://example.test/archive/"


def make_csa_name(
    day: date, seq: int, players: str = "A+B"
) -> str:
    stamp = f"{day.year:04d}{day.month:02d}{day.day:02d}"
    return (
        f"wdoor+floodgate-300-10F+{players}"
        f"+{stamp}{seq:06d}.csa"
    )


def make_listing_html(csa_names: list[str]) -> bytes:
    """実サイトの日別リスティング構造を模した HTML を作る．"""
    items = []
    for name in csa_names:
        stem = name.removesuffix(".csa")
        items.append(
            f'<li>2025/01/05 23:30:01<a href="{stem}.html">'
            f'{stem}</a> -- <a href="{name}">csa</a></li>'
        )
    body = "".join(items)
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        "    <title>Folders and files: x</title>\n"
        '    <link rel="StyleSheet" type="text/css" '
        'href="/shogi/shogi.css">\n'
        "</head>\n<body>\n<h1>Folders and files: x</h1>\n"
        f"    <ul>\n    {body}\n    </ul>\n"
        '    <footer><div>top: <a href="/shogi">wdoor'
        "</a></div></footer>\n</body>\n</html>\n"
    ).encode("utf-8")


class FakeHttpClient(HttpClient):
    """URL 辞書を返すテスト用 HTTP クライアント．"""

    def __init__(
        self,
        pages: dict[str, bytes] | None = None,
        downloads: dict[str, bytes] | None = None,
    ):
        self.pages = pages or {}
        self.downloads = downloads or {}
        self.get_urls: list[str] = []
        self.download_urls: list[str] = []

    def get(self, url: str) -> bytes:
        self.get_urls.append(url)
        if url not in self.pages:
            raise HttpNotFoundError(url)
        return self.pages[url]

    def download(self, url: str, dest: Path) -> None:
        self.download_urls.append(url)
        if url not in self.downloads:
            raise HttpNotFoundError(url)
        dest.write_bytes(self.downloads[url])


def make_tarxz(path: Path, entries: dict[str, bytes]) -> None:
    """テスト用の tar.xz アーカイブを作る．"""
    with tarfile.open(path, mode="w:xz") as tf:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))


def base_option(**kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "strategy": "daily",
        "base_url": BASE_URL,
        "archive_base_url": ARCHIVE_BASE_URL,
        "delay_seconds": 0.0,
    }
    defaults.update(kwargs)
    return FloodgateFetcher.FetchOption(**defaults)


class TestListingParsing:
    def test_extract_csa_hrefs(self) -> None:
        day = date(2025, 1, 5)
        names = [
            make_csa_name(day, 233006),
            make_csa_name(day, 233003, "C+D"),
        ]
        hrefs = extract_csa_hrefs(
            make_listing_html(names).decode("utf-8")
        )
        assert hrefs == names

    def test_excludes_non_csa_and_paths(self) -> None:
        html = (
            '<a href="/shogi/shogi.css">css</a>'
            '<a href="a.html">h</a>'
            '<a href="sub/dir.csa">path</a>'
            '<a href="..\\evil.csa">win</a>'
            '<a href="ok.csa">ok</a>'
            '<a href="ok.csa">dup</a>'
        )
        assert extract_csa_hrefs(html) == ["ok.csa"]

    def test_empty_html(self) -> None:
        assert extract_csa_hrefs("<html></html>") == []


class TestCsaFilenameToDate:
    def test_valid(self) -> None:
        name = make_csa_name(date(2018, 8, 10), 220004)
        assert csa_filename_to_date(name) == date(2018, 8, 10)

    def test_invalid(self) -> None:
        assert csa_filename_to_date("README.txt") is None
        assert csa_filename_to_date("foo.csa") is None
        # 不正な日付 (13月)
        assert (
            csa_filename_to_date("a+20251340120000.csa") is None
        )


class TestDateHelpers:
    def test_date_range_crosses_month(self) -> None:
        days = list(
            date_range(date(2025, 1, 31), date(2025, 2, 2))
        )
        assert days == [
            date(2025, 1, 31),
            date(2025, 2, 1),
            date(2025, 2, 2),
        ]

    def test_build_day_url(self) -> None:
        assert (
            build_day_url(BASE_URL, date(2025, 1, 5))
            == "https://example.test/x/2025/01/05/"
        )
        # 末尾スラッシュなしでも同じ
        assert (
            build_day_url(BASE_URL[:-1], date(2025, 1, 5))
            == "https://example.test/x/2025/01/05/"
        )


class TestDailyStrategy:
    def setup_day(
        self, day: date, n: int
    ) -> tuple[FakeHttpClient, list[str]]:
        names = [
            make_csa_name(day, 100000 + i) for i in range(n)
        ]
        day_url = build_day_url(BASE_URL, day)
        pages = {
            day_url: make_listing_html(names),
        }
        for name in names:
            pages[day_url + name] = f"V2:{name}".encode()
        return FakeHttpClient(pages), names

    def test_downloads_files(self, tmp_path: Path) -> None:
        day = date(2025, 1, 5)
        client, names = self.setup_day(day, 3)
        fetcher = FloodgateFetcher(http_client=client)
        summary = fetcher.fetch(
            base_option(
                start_date=day,
                end_date=day,
                output_dir=tmp_path,
            )
        )
        assert summary["files_downloaded"] == 3
        assert summary["days_fetched"] == 1
        day_dir = tmp_path / "2025" / "01" / "05"
        for name in names:
            content = (day_dir / name).read_bytes()
            assert content == f"V2:{name}".encode()
        # .part の残骸がない
        assert not list(tmp_path.rglob("*.part"))

    def test_skips_existing(self, tmp_path: Path) -> None:
        day = date(2025, 1, 5)
        client, names = self.setup_day(day, 2)
        day_dir = tmp_path / "2025" / "01" / "05"
        day_dir.mkdir(parents=True)
        (day_dir / names[0]).write_bytes(b"old")
        fetcher = FloodgateFetcher(http_client=client)
        summary = fetcher.fetch(
            base_option(
                start_date=day,
                end_date=day,
                output_dir=tmp_path,
            )
        )
        assert summary["files_skipped"] == 1
        assert summary["files_downloaded"] == 1
        # 既存はそのまま (上書きしない)
        assert (day_dir / names[0]).read_bytes() == b"old"

    def test_overwrite(self, tmp_path: Path) -> None:
        day = date(2025, 1, 5)
        client, names = self.setup_day(day, 1)
        day_dir = tmp_path / "2025" / "01" / "05"
        day_dir.mkdir(parents=True)
        (day_dir / names[0]).write_bytes(b"old")
        fetcher = FloodgateFetcher(http_client=client)
        summary = fetcher.fetch(
            base_option(
                start_date=day,
                end_date=day,
                output_dir=tmp_path,
                overwrite=True,
            )
        )
        assert summary["files_downloaded"] == 1
        assert (day_dir / names[0]).read_bytes() != b"old"

    def test_dry_run(self, tmp_path: Path) -> None:
        day = date(2025, 1, 5)
        client, _ = self.setup_day(day, 3)
        fetcher = FloodgateFetcher(http_client=client)
        summary = fetcher.fetch(
            base_option(
                start_date=day,
                end_date=day,
                output_dir=tmp_path,
                dry_run=True,
            )
        )
        assert summary["files_planned"] == 3
        assert summary["files_downloaded"] == 0
        # リスティングのみ取得しファイル本体は取らない
        assert client.get_urls == [build_day_url(BASE_URL, day)]
        assert not list(tmp_path.rglob("*.csa"))

    def test_missing_day_counted(self, tmp_path: Path) -> None:
        day1 = date(2025, 1, 5)
        day2 = date(2025, 1, 6)
        client, _ = self.setup_day(day1, 1)
        fetcher = FloodgateFetcher(http_client=client)
        summary = fetcher.fetch(
            base_option(
                start_date=day1,
                end_date=day2,
                output_dir=tmp_path,
            )
        )
        assert summary["days_fetched"] == 1
        assert summary["days_missing"] == 1

    def test_all_missing_raises(self, tmp_path: Path) -> None:
        client = FakeHttpClient()
        fetcher = FloodgateFetcher(http_client=client)
        with pytest.raises(FloodgateStructureError):
            fetcher.fetch(
                base_option(
                    start_date=date(2030, 1, 1),
                    end_date=date(2030, 1, 2),
                    output_dir=tmp_path,
                )
            )

    def test_empty_listing_raises(self, tmp_path: Path) -> None:
        """リンク 0 件 (仕様変更の兆候) は空振りで失敗する．"""
        day = date(2025, 1, 5)
        pages = {
            build_day_url(BASE_URL, day): make_listing_html([])
        }
        fetcher = FloodgateFetcher(
            http_client=FakeHttpClient(pages)
        )
        with pytest.raises(FloodgateStructureError):
            fetcher.fetch(
                base_option(
                    start_date=day,
                    end_date=day,
                    output_dir=tmp_path,
                )
            )

    def test_unknown_strategy_raises(
        self, tmp_path: Path
    ) -> None:
        fetcher = FloodgateFetcher(http_client=FakeHttpClient())
        with pytest.raises(ValueError):
            fetcher.fetch(
                base_option(
                    start_date=date(2025, 1, 5),
                    end_date=date(2025, 1, 5),
                    output_dir=tmp_path,
                    strategy="bogus",
                )
            )


class TestExtractArchive:
    def test_tarxz(self, tmp_path: Path) -> None:
        day1 = date(2011, 3, 1)
        day2 = date(2011, 3, 2)
        name1 = make_csa_name(day1, 1)
        name2 = make_csa_name(day2, 2)
        archive = tmp_path / "wdoor2011.tar.xz"
        make_tarxz(
            archive,
            {
                f"2011/{name1}": b"game1",
                f"2011/{name2}": b"game2",
                "2011/README": b"not a game",
            },
        )
        out = tmp_path / "out"
        result = extract_archive_days(
            archive_path=archive,
            wanted_days={day1},
            output_dir=out,
        )
        assert result.extracted_files == 1
        assert result.covered_days == {day1}
        target = out / "2011" / "03" / "01" / name1
        assert target.read_bytes() == b"game1"
        # wanted 外の日は展開されない
        assert not (out / "2011" / "03" / "02").exists()

    def test_tarxz_skips_existing(self, tmp_path: Path) -> None:
        day = date(2011, 3, 1)
        name = make_csa_name(day, 1)
        archive = tmp_path / "wdoor2011.tar.xz"
        make_tarxz(archive, {f"2011/{name}": b"game1"})
        out = tmp_path / "out"
        target = out / "2011" / "03" / "01" / name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"old")
        result = extract_archive_days(
            archive_path=archive,
            wanted_days={day},
            output_dir=out,
        )
        assert result.extracted_files == 0
        assert result.skipped_files == 1
        assert result.covered_days == {day}
        assert target.read_bytes() == b"old"

    def test_7z(self, tmp_path: Path) -> None:
        py7zr = pytest.importorskip("py7zr")
        day = date(2018, 8, 10)
        name = make_csa_name(day, 220004)
        src = tmp_path / name
        src.write_bytes(b"game7z")
        archive = tmp_path / "wdoor2018.7z"
        with py7zr.SevenZipFile(archive, "w") as zf:
            zf.write(src, arcname=f"2018/{name}")
        out = tmp_path / "out"
        result = extract_archive_days(
            archive_path=archive,
            wanted_days={day},
            output_dir=out,
        )
        assert result.extracted_files == 1
        target = out / "2018" / "08" / "10" / name
        assert target.read_bytes() == b"game7z"
        # 一時展開ディレクトリが残らない
        leftovers = [
            p
            for p in out.iterdir()
            if p.name.startswith(".maou-extract-")
        ]
        assert leftovers == []

    def test_unknown_format(self, tmp_path: Path) -> None:
        bogus = tmp_path / "wdoor2018.zip"
        bogus.write_bytes(b"")
        with pytest.raises(ValueError):
            extract_archive_days(
                archive_path=bogus,
                wanted_days=set(),
                output_dir=tmp_path,
            )


class TestAutoStrategy:
    def test_short_range_uses_daily(
        self, tmp_path: Path
    ) -> None:
        """32 日未満はアーカイブに触らない．"""
        day = date(2025, 1, 5)
        names = [make_csa_name(day, 1)]
        day_url = build_day_url(BASE_URL, day)
        pages = {day_url: make_listing_html(names)}
        pages[day_url + names[0]] = b"g"
        client = FakeHttpClient(pages)
        fetcher = FloodgateFetcher(http_client=client)
        summary = fetcher.fetch(
            base_option(
                start_date=day,
                end_date=day,
                output_dir=tmp_path,
                strategy="auto",
            )
        )
        assert summary["files_downloaded"] == 1
        assert summary["archives_used"] == 0
        assert client.download_urls == []

    def test_long_range_uses_archive_with_topup(
        self, tmp_path: Path
    ) -> None:
        """32 日以上はアーカイブ + 未収録日の daily 補完．"""
        start = date(2011, 1, 1)
        end = date(2011, 2, 15)  # 46 日
        day_a = date(2011, 1, 10)
        day_b = date(2011, 2, 5)
        name_a = make_csa_name(day_a, 1)
        name_b = make_csa_name(day_b, 2)
        buf = tmp_path / "src.tar.xz"
        make_tarxz(
            buf,
            {
                f"2011/{name_a}": b"ga",
                f"2011/{name_b}": b"gb",
            },
        )
        downloads = {
            ARCHIVE_BASE_URL
            + "wdoor2011.tar.xz": buf.read_bytes()
        }
        # .7z は存在しない (404) → .tar.xz へフォールバック
        client = FakeHttpClient(downloads=downloads)
        fetcher = FloodgateFetcher(http_client=client)
        out = tmp_path / "out"
        summary = fetcher.fetch(
            base_option(
                start_date=start,
                end_date=end,
                output_dir=out,
                strategy="auto",
            )
        )
        assert summary["archives_used"] == 1
        assert summary["files_downloaded"] == 2
        assert (out / "2011" / "01" / "10" / name_a).exists()
        # アーカイブでカバーされた日は daily を叩かない
        covered_url = build_day_url(BASE_URL, day_a)
        assert covered_url not in client.get_urls
        # 未収録の 44 日は daily で補完 (全部 404)
        assert len(client.get_urls) == 44
        assert summary["days_missing"] == 44
        assert summary["days_fetched"] == 2

    def test_archive_404_falls_back_to_daily(
        self, tmp_path: Path
    ) -> None:
        start = date(2031, 1, 1)
        end = date(2031, 2, 15)
        day = date(2031, 1, 5)
        names = [make_csa_name(day, 1)]
        day_url = build_day_url(BASE_URL, day)
        pages = {day_url: make_listing_html(names)}
        pages[day_url + names[0]] = b"g"
        client = FakeHttpClient(pages)
        fetcher = FloodgateFetcher(http_client=client)
        summary = fetcher.fetch(
            base_option(
                start_date=start,
                end_date=end,
                output_dir=tmp_path,
                strategy="auto",
            )
        )
        assert summary["archives_used"] == 0
        assert summary["files_downloaded"] == 1

    def test_dry_run_never_downloads_archive(
        self, tmp_path: Path
    ) -> None:
        start = date(2025, 1, 1)
        end = date(2025, 3, 1)
        day = date(2025, 1, 5)
        names = [make_csa_name(day, 1)]
        pages = {
            build_day_url(BASE_URL, day): make_listing_html(
                names
            )
        }
        client = FakeHttpClient(pages)
        fetcher = FloodgateFetcher(http_client=client)
        summary = fetcher.fetch(
            base_option(
                start_date=start,
                end_date=end,
                output_dir=tmp_path,
                strategy="auto",
                dry_run=True,
            )
        )
        assert client.download_urls == []
        assert summary["files_planned"] == 1


class TestArchiveStrategy:
    def test_leftover_days_not_topped_up(
        self, tmp_path: Path
    ) -> None:
        """明示 archive では daily 補完しない．"""
        start = date(2011, 1, 1)
        end = date(2011, 1, 3)
        day = date(2011, 1, 2)
        name = make_csa_name(day, 1)
        buf = tmp_path / "src.tar.xz"
        make_tarxz(buf, {f"2011/{name}": b"g"})
        downloads = {
            ARCHIVE_BASE_URL
            + "wdoor2011.tar.xz": buf.read_bytes()
        }
        client = FakeHttpClient(downloads=downloads)
        fetcher = FloodgateFetcher(http_client=client)
        summary = fetcher.fetch(
            base_option(
                start_date=start,
                end_date=end,
                output_dir=tmp_path / "out",
                strategy="archive",
            )
        )
        assert summary["files_downloaded"] == 1
        assert summary["days_missing"] == 2
        assert client.get_urls == []

    def test_archive_cache_reused(self, tmp_path: Path) -> None:
        day = date(2011, 1, 2)
        name = make_csa_name(day, 1)
        buf = tmp_path / "src.tar.xz"
        make_tarxz(buf, {f"2011/{name}": b"g"})
        downloads = {
            ARCHIVE_BASE_URL
            + "wdoor2011.tar.xz": buf.read_bytes()
        }
        client = FakeHttpClient(downloads=downloads)
        fetcher = FloodgateFetcher(http_client=client)
        cache = tmp_path / "cache"
        option = base_option(
            start_date=day,
            end_date=day,
            output_dir=tmp_path / "out",
            strategy="archive",
            archive_cache_dir=cache,
        )
        fetcher.fetch(option)
        assert (cache / "wdoor2011.tar.xz").exists()
        # 2 回目はキャッシュを使いダウンロードしない
        n_downloads = len(client.download_urls)
        summary = fetcher.fetch(option)
        assert len(client.download_urls) == n_downloads
        assert summary["files_skipped"] == 1

    def test_missing_py7zr_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """py7zr なし + .tar.xz も 404 → 明示エラー．"""
        import maou.app.fetcher.floodgate_fetcher as mod

        monkeypatch.setattr(
            mod, "find_spec", lambda _name: None
        )
        client = FakeHttpClient()
        fetcher = FloodgateFetcher(http_client=client)
        with pytest.raises(ArchiveToolMissingError):
            fetcher.fetch(
                base_option(
                    start_date=date(2018, 1, 1),
                    end_date=date(2018, 12, 31),
                    output_dir=tmp_path,
                    strategy="archive",
                )
            )
        # .7z は試しにも行かない (展開できないため)
        assert client.download_urls == [
            ARCHIVE_BASE_URL + "wdoor2018.tar.xz"
        ]
