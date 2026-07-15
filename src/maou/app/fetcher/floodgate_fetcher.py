"""floodgate 棋譜アーカイブからの CSA 取得ユースケース．

floodgate (wdoor.c.u-tokyo.ac.jp) から CSA 棋譜を収集する．
取得戦略は 2 系統ある:

- daily: 日別ディレクトリリスティング (`x/YYYY/MM/DD/`) を
  パースして 1 局ずつ取得する．任意の期間に使えるが，
  リクエスト数が多い (1 日あたり数百局)
- archive: 年次アーカイブ (`archive/wdoorYYYY.7z` または
  `.tar.xz`) を丸ごと取得して指定日のみ展開する．月単位
  以上の期間ではこちらが圧倒的に効率的 (年 100-350MB)

auto (デフォルト) は年内の要求日数が ``ARCHIVE_MIN_DAYS``
以上なら archive，未満なら daily を選び，アーカイブに
未収録の日 (直近日など) は daily で補完する．

警告:
    このモジュールは floodgate サイトの非公式な公開仕様
    (URL 構造・HTML リスティング形式・アーカイブ内の
    ファイル名規約) に依存する**壊れやすい**実装である．
    サイト仕様の変更で動かなくなる可能性を常に想定する
    こと (例: 年次 7z の配布 URL は過去に `x/` 直下から
    `archive/` へ移動した)．棋譜を 1 件も発見できなかった
    場合は ``FloodgateStructureError`` を送出して静かな
    空振りを防ぐ．
"""

import abc
import contextlib
import logging
import re
import shutil
import tarfile
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from html.parser import HTMLParser
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urljoin

from tqdm.auto import tqdm

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://wdoor.c.u-tokyo.ac.jp/shogi/x/"
DEFAULT_ARCHIVE_BASE_URL = (
    "https://wdoor.c.u-tokyo.ac.jp/shogi/archive/"
)

# auto 戦略で年次アーカイブを選ぶ最小日数．これ未満の区間は
# 日別クロールの方が総ダウンロード量が小さい (年次アーカイブ
# は 100-350MB 程度ある)
ARCHIVE_MIN_DAYS = 32

STRATEGIES = ("auto", "daily", "archive")

# CSA ファイル名末尾の対局開始タイムスタンプ
# (例: wdoor+floodgate-300-10F+a+b+20250105233006.csa)．
# 年次アーカイブは内部が日別ディレクトリではないため，
# 対局日はこのタイムスタンプから導出する
_CSA_TIMESTAMP_RE = re.compile(r"\+(\d{8})\d{6}\.csa$")


class HttpNotFoundError(Exception):
    """HTTP 404 (リソース不在) を表す例外．"""


class FloodgateStructureError(Exception):
    """floodgate サイト構造の想定外れを表す例外．

    指定期間から 1 件も棋譜を発見できなかった場合に送出
    される．期間指定の誤りか，floodgate のサイト仕様変更
    (リスティング HTML やファイル名規約の変化) が疑われる．
    """


class ArchiveToolMissingError(Exception):
    """アーカイブ展開に必要な py7zr がない場合の例外．"""


class HttpClient(metaclass=abc.ABCMeta):
    """Abstract interface for HTTP GET operations.

    Defines the transport port used by
    :class:`FloodgateFetcher` so the app layer stays free
    of concrete network code.
    """

    @abc.abstractmethod
    def get(self, url: str) -> bytes:
        """Fetch a URL and return the response body.

        Args:
            url: Absolute URL to fetch.

        Returns:
            Raw response body bytes.

        Raises:
            HttpNotFoundError: If the server returns 404.
        """
        pass

    @abc.abstractmethod
    def download(self, url: str, dest: Path) -> None:
        """Download a URL to a local file (streaming).

        Args:
            url: Absolute URL to fetch.
            dest: Local file path to write to.

        Raises:
            HttpNotFoundError: If the server returns 404.
        """
        pass


class _AnchorHrefParser(HTMLParser):
    """``<a href="...">`` の href 値を収集する最小パーサ．"""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value is not None:
                self.hrefs.append(value)


def extract_csa_hrefs(listing_html: str) -> list[str]:
    """日別リスティング HTML から CSA ファイル名を抽出する．

    Args:
        listing_html: ``x/YYYY/MM/DD/`` ページの HTML

    Returns:
        ``.csa`` で終わる相対 href (ファイル名) のリスト．
        出現順を保ち重複は除去する．パス区切りを含む href は
        日別ファイルではない (かつパストラバーサル対策) ため
        除外する．
    """
    parser = _AnchorHrefParser()
    parser.feed(listing_html)
    csa_names = [
        href
        for href in parser.hrefs
        if href.endswith(".csa")
        and "/" not in href
        and "\\" not in href
    ]
    # 出現順を保った重複除去
    return list(dict.fromkeys(csa_names))


def csa_filename_to_date(filename: str) -> date | None:
    """CSA ファイル名末尾のタイムスタンプから対局日を得る．

    Args:
        filename: ``wdoor+...+YYYYMMDDHHMMSS.csa`` 形式の
            ファイル名 (ディレクトリ部なし)

    Returns:
        対局日．形式が合わない場合は None
    """
    m = _CSA_TIMESTAMP_RE.search(filename)
    if m is None:
        return None
    raw = m.group(1)
    try:
        return date(int(raw[0:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return None


def date_range(start: date, end: date) -> Iterator[date]:
    """start から end までの日付を両端含みで列挙する．"""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_day_url(base_url: str, day: date) -> str:
    """日別リスティングページの URL を組み立てる．"""
    base = base_url.rstrip("/")
    return (
        f"{base}/{day.year:04d}/{day.month:02d}/{day.day:02d}/"
    )


def day_dir(output_dir: Path, day: date) -> Path:
    """出力先の日別ディレクトリ (YYYY/MM/DD) を返す．"""
    return (
        output_dir
        / f"{day.year:04d}"
        / f"{day.month:02d}"
        / f"{day.day:02d}"
    )


@dataclass
class ExtractResult:
    """アーカイブ展開の結果．"""

    extracted_files: int = 0
    skipped_files: int = 0
    covered_days: set[date] = field(default_factory=set)

    @property
    def matched_files(self) -> int:
        return self.extracted_files + self.skipped_files


def _plan_extraction(
    names: list[str],
    wanted_days: set[date],
    output_dir: Path,
    overwrite: bool,
    result: ExtractResult,
) -> dict[str, tuple[date, Path]]:
    """展開対象エントリと出力先の対応表を作る．

    既存ファイルは (overwrite でない限り) スキップとして
    result に計上する．
    """
    plan: dict[str, tuple[date, Path]] = {}
    for raw in names:
        basename = raw.rsplit("/", 1)[-1]
        day = csa_filename_to_date(basename)
        if day is None or day not in wanted_days:
            continue
        target = day_dir(output_dir, day) / basename
        if target.exists() and not overwrite:
            result.skipped_files += 1
            result.covered_days.add(day)
        else:
            plan[raw] = (day, target)
    return plan


def _extract_from_7z(
    archive_path: Path,
    wanted_days: set[date],
    output_dir: Path,
    overwrite: bool,
) -> ExtractResult:
    """7z アーカイブから指定日の CSA を展開する．"""
    try:
        import py7zr
    except ImportError as e:
        raise ArchiveToolMissingError(
            "7z アーカイブの展開には py7zr が必要．"
            "`uv sync --extra fetch` でインストールするか，"
            "--strategy daily を使うこと"
        ) from e

    result = ExtractResult()
    with py7zr.SevenZipFile(archive_path, "r") as zf:
        names = zf.getnames()
    plan = _plan_extraction(
        names, wanted_days, output_dir, overwrite, result
    )
    if not plan:
        return result

    # アーカイブ内はフラット構造 (YYYY/ファイル名) のため，
    # 一時ディレクトリへ展開してから日別レイアウトへ移す
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(
        tempfile.mkdtemp(
            prefix=".maou-extract-", dir=output_dir
        )
    )
    try:
        with py7zr.SevenZipFile(archive_path, "r") as zf:
            zf.extract(path=tmp_root, targets=list(plan))
        for raw, (day, target) in tqdm(
            plan.items(), desc="extract", leave=False
        ):
            src = tmp_root / raw
            if not src.is_file():
                logger.warning(
                    f"アーカイブ展開漏れ: {raw} (skip)"
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src.replace(target)
            result.extracted_files += 1
            result.covered_days.add(day)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return result


def _extract_from_tarxz(
    archive_path: Path,
    wanted_days: set[date],
    output_dir: Path,
    overwrite: bool,
) -> ExtractResult:
    """tar.xz アーカイブから指定日の CSA を展開する．

    2011/2012 年のアーカイブは 7z でなく tar.xz で配布
    されている．標準ライブラリのみで展開できる．
    """
    result = ExtractResult()
    with tarfile.open(archive_path, mode="r:xz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            basename = member.name.rsplit("/", 1)[-1]
            day = csa_filename_to_date(basename)
            if day is None or day not in wanted_days:
                continue
            target = day_dir(output_dir, day) / basename
            if target.exists() and not overwrite:
                result.skipped_files += 1
                result.covered_days.add(day)
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.parent / (target.name + ".part")
            with src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst)
            part.replace(target)
            result.extracted_files += 1
            result.covered_days.add(day)
    return result


def extract_archive_days(
    *,
    archive_path: Path,
    wanted_days: set[date],
    output_dir: Path,
    overwrite: bool = False,
) -> ExtractResult:
    """年次アーカイブから指定日の CSA を展開する．

    対局日はアーカイブ内エントリのファイル名タイムスタンプ
    から導出するため，アーカイブ内のディレクトリ構造には
    依存しない．出力は ``output_dir/YYYY/MM/DD/`` に置く．

    Args:
        archive_path: ローカルの .7z / .tar.xz ファイル
        wanted_days: 展開対象の日付集合
        output_dir: 出力先ルート
        overwrite: 既存ファイルを上書きするか

    Returns:
        展開結果 (展開数・スキップ数・カバーした日付)

    Raises:
        ArchiveToolMissingError: .7z なのに py7zr がない
        ValueError: 未知のアーカイブ形式
    """
    name = archive_path.name
    if name.endswith(".7z"):
        return _extract_from_7z(
            archive_path, wanted_days, output_dir, overwrite
        )
    if name.endswith(".tar.xz"):
        return _extract_from_tarxz(
            archive_path, wanted_days, output_dir, overwrite
        )
    raise ValueError(f"unknown archive format: {name}")


@dataclass
class _Counters:
    """fetch 全体の集計カウンタ．"""

    days_fetched: int = 0
    days_missing: int = 0
    days_empty: int = 0
    files_listed: int = 0
    files_downloaded: int = 0
    files_skipped: int = 0
    files_planned: int = 0
    archives_used: int = 0


class FloodgateFetcher:
    """floodgate から期間指定で CSA 棋譜を収集する．

    取得済みファイルはデフォルトでスキップするため，中断
    後の再実行で続きから取得できる (resumable)．出力は
    ``output_dir/YYYY/MM/DD/`` のミラー構造で，そのまま
    ``maou hcpe-convert --input-path`` に渡せる．
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, *, http_client: HttpClient):
        """Initialize fetcher.

        Args:
            http_client: HTTP transport implementation
        """
        self.__http_client = http_client

    @dataclass(kw_only=True, frozen=True)
    class FetchOption:
        start_date: date
        end_date: date
        output_dir: Path
        strategy: str = "auto"
        base_url: str = DEFAULT_BASE_URL
        archive_base_url: str = DEFAULT_ARCHIVE_BASE_URL
        archive_cache_dir: Path | None = None
        delay_seconds: float = 0.2
        overwrite: bool = False
        dry_run: bool = False

    def fetch(
        self, option: FetchOption
    ) -> dict[str, int | str | bool]:
        """指定期間の CSA 棋譜を取得する．

        Args:
            option: 取得条件

        Returns:
            取得結果サマリ (日数・ファイル数のカウンタ)

        Raises:
            ValueError: strategy が不正な場合
            FloodgateStructureError: 期間全体で棋譜を 1 件も
                発見できなかった場合 (期間誤りまたは
                floodgate 仕様変更の疑い)
            ArchiveToolMissingError: 7z 展開に py7zr が
                必要なのに入っていない場合
        """
        if option.strategy not in STRATEGIES:
            raise ValueError(
                f"unknown strategy '{option.strategy}' "
                f"(choose from {STRATEGIES})"
            )
        days = list(
            date_range(option.start_date, option.end_date)
        )
        counters = _Counters()

        effective_strategy = option.strategy
        if option.dry_run and effective_strategy != "daily":
            # dry-run は軽量プローブが目的なので，年次
            # アーカイブ (数百 MB) は取得せず日別リス
            # ティングだけで数える
            self.logger.info(
                "dry-run のため daily リスティングで数える"
            )
            effective_strategy = "daily"

        if effective_strategy == "daily":
            self.__fetch_days_daily(days, option, counters)
        else:
            by_year: dict[int, list[date]] = {}
            for d in days:
                by_year.setdefault(d.year, []).append(d)
            for year in sorted(by_year):
                self.__fetch_year(
                    year, by_year[year], option, counters
                )

        if counters.files_listed == 0:
            raise FloodgateStructureError(
                "指定期間から棋譜を 1 件も発見できなかった．"
                "期間指定の誤りか，floodgate のサイト仕様"
                "変更が疑われる "
                f"(days={len(days)}, "
                f"404={counters.days_missing}, "
                f"empty={counters.days_empty}, "
                f"base={option.base_url})"
            )

        summary: dict[str, int | str | bool] = {
            "output_dir": str(option.output_dir),
            "strategy": option.strategy,
            "dry_run": option.dry_run,
            "days_total": len(days),
            "days_fetched": counters.days_fetched,
            "days_missing": counters.days_missing,
            "days_empty": counters.days_empty,
            "files_listed": counters.files_listed,
            "files_downloaded": counters.files_downloaded,
            "files_skipped": counters.files_skipped,
            "files_planned": counters.files_planned,
            "archives_used": counters.archives_used,
        }
        self.logger.info(f"取得完了: {summary}")
        return summary

    def __fetch_year(
        self,
        year: int,
        year_days: list[date],
        option: FetchOption,
        counters: _Counters,
    ) -> None:
        """1 年分の取得 (archive / auto 戦略)．"""
        use_archive = (
            option.strategy == "archive"
            or len(year_days) >= ARCHIVE_MIN_DAYS
        )
        if not use_archive:
            self.__fetch_days_daily(year_days, option, counters)
            return

        covered = self.__fetch_year_via_archive(
            year, year_days, option, counters
        )
        if covered is None:
            # アーカイブ自体が存在しない
            if option.strategy == "archive":
                self.logger.warning(
                    f"wdoor{year} アーカイブが見つからない"
                    " (この年をスキップ．auto なら日別"
                    "クロールに切替わる)"
                )
                counters.days_missing += len(year_days)
            else:
                self.logger.warning(
                    f"wdoor{year} アーカイブなし．"
                    "日別クロールに切替える"
                )
                self.__fetch_days_daily(
                    year_days, option, counters
                )
            return

        leftover = [d for d in year_days if d not in covered]
        if not leftover:
            return
        if option.strategy == "auto":
            # 当年アーカイブは直近日が未収録のことがある
            self.logger.info(
                f"{year}: アーカイブ未収録の "
                f"{len(leftover)} 日を日別クロールで補完"
            )
            self.__fetch_days_daily(leftover, option, counters)
        else:
            counters.days_missing += len(leftover)

    def __fetch_year_via_archive(
        self,
        year: int,
        year_days: list[date],
        option: FetchOption,
        counters: _Counters,
    ) -> set[date] | None:
        """年次アーカイブを取得・展開する．

        Returns:
            カバーできた日付集合．アーカイブが存在しない
            場合は None
        """
        with contextlib.ExitStack() as stack:
            if option.archive_cache_dir is not None:
                cache_dir = option.archive_cache_dir
                cache_dir.mkdir(parents=True, exist_ok=True)
            else:
                cache_dir = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(
                            prefix="maou-floodgate-"
                        )
                    )
                )
            archive_path = self.__download_archive(
                year, cache_dir, option
            )
            if archive_path is None:
                return None
            counters.archives_used += 1
            result = extract_archive_days(
                archive_path=archive_path,
                wanted_days=set(year_days),
                output_dir=option.output_dir,
                overwrite=option.overwrite,
            )
        counters.files_listed += result.matched_files
        counters.files_downloaded += result.extracted_files
        counters.files_skipped += result.skipped_files
        counters.days_fetched += len(result.covered_days)
        return result.covered_days

    def __download_archive(
        self,
        year: int,
        cache_dir: Path,
        option: FetchOption,
    ) -> Path | None:
        """年次アーカイブをダウンロード (またはキャッシュ利用) する．

        Returns:
            ローカルのアーカイブパス．404 なら None

        Raises:
            ArchiveToolMissingError: py7zr がなく .tar.xz も
                見つからない場合
        """
        base = option.archive_base_url.rstrip("/") + "/"
        py7zr_available = find_spec("py7zr") is not None

        # 既存キャッシュがあればそれを使う (resumable)
        for suffix in (".7z", ".tar.xz"):
            cached = cache_dir / f"wdoor{year}{suffix}"
            if cached.exists():
                if suffix == ".7z" and not py7zr_available:
                    raise ArchiveToolMissingError(
                        f"キャッシュ {cached} の展開には "
                        "py7zr が必要．`uv sync --extra "
                        "fetch` でインストールすること"
                    )
                self.logger.info(f"キャッシュ利用: {cached}")
                return cached

        # py7zr がなければ .7z は取得しても展開できない
        # ため，(2011/2012 のみの) .tar.xz だけ試す
        suffixes = (
            (".7z", ".tar.xz")
            if py7zr_available
            else (".tar.xz",)
        )
        for suffix in suffixes:
            filename = f"wdoor{year}{suffix}"
            url = urljoin(base, filename)
            target = cache_dir / filename
            part = cache_dir / (filename + ".part")
            try:
                self.logger.info(f"アーカイブ取得: {url}")
                self.__http_client.download(url, part)
            except HttpNotFoundError:
                part.unlink(missing_ok=True)
                continue
            part.replace(target)
            if option.delay_seconds > 0:
                time.sleep(option.delay_seconds)
            return target

        if not py7zr_available:
            raise ArchiveToolMissingError(
                f"wdoor{year} の .tar.xz は見つからず，"
                ".7z の展開には py7zr が必要．`uv sync "
                "--extra fetch` でインストールするか "
                "--strategy daily を使うこと"
            )
        return None

    def __fetch_days_daily(
        self,
        days: list[date],
        option: FetchOption,
        counters: _Counters,
    ) -> None:
        """日別リスティングのクロールで取得する．"""
        for day in tqdm(days, desc="days"):
            day_url = build_day_url(option.base_url, day)
            try:
                listing = self.__get(
                    day_url, option.delay_seconds
                )
            except HttpNotFoundError:
                # 未来日や対局のない日は 404 になる
                self.logger.info(
                    f"リスティングなし (404): {day_url}"
                )
                counters.days_missing += 1
                continue
            csa_names = extract_csa_hrefs(
                listing.decode("utf-8", errors="replace")
            )
            if not csa_names:
                self.logger.warning(
                    f"{day_url} に .csa リンクが 0 件．"
                    "floodgate 仕様変更の可能性がある"
                )
                counters.days_empty += 1
                continue
            counters.files_listed += len(csa_names)
            target_dir = day_dir(option.output_dir, day)
            if not option.dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
            for name in tqdm(
                csa_names,
                desc=day.isoformat(),
                leave=False,
            ):
                target = target_dir / name
                if target.exists() and not option.overwrite:
                    counters.files_skipped += 1
                    continue
                if option.dry_run:
                    counters.files_planned += 1
                    continue
                content = self.__get(
                    urljoin(day_url, name),
                    option.delay_seconds,
                )
                # 中断時の壊れた部分ファイルを target に
                # 残さないため一時名に書いてから rename
                part = target.parent / (target.name + ".part")
                part.write_bytes(content)
                part.replace(target)
                counters.files_downloaded += 1
            counters.days_fetched += 1

    def __get(self, url: str, delay_seconds: float) -> bytes:
        """1 リクエスト取得し，礼儀として delay を挟む．"""
        body = self.__http_client.get(url)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        return body
