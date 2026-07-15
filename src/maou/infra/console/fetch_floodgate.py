from datetime import datetime
from pathlib import Path

import click

import maou.interface.fetcher as fetcher
from maou.app.fetcher.floodgate_fetcher import (
    DEFAULT_ARCHIVE_BASE_URL,
    DEFAULT_BASE_URL,
    STRATEGIES,
)
from maou.infra.console.common import handle_exception
from maou.infra.http.urllib_http_client import (
    UrllibHttpClient,
)


@click.command("fetch-floodgate")
@click.option(
    "--start-date",
    help="First day to fetch (YYYY-MM-DD, JST).",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
)
@click.option(
    "--end-date",
    help=(
        "Last day to fetch, inclusive (YYYY-MM-DD, JST). "
        "Defaults to --start-date."
    ),
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=False,
)
@click.option(
    "--output-dir",
    help=(
        "Directory for downloaded CSA files "
        "(mirrors YYYY/MM/DD structure)."
    ),
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--strategy",
    help=(
        "'daily' crawls per-day listings, 'archive' uses "
        "yearly archives (needs py7zr for .7z), 'auto' "
        "picks archives for ranges of 32+ days in a year "
        "and tops up missing days via daily crawling."
    ),
    type=click.Choice(STRATEGIES),
    default="auto",
    show_default=True,
)
@click.option(
    "--base-url",
    help="Floodgate daily listing base URL.",
    type=str,
    default=DEFAULT_BASE_URL,
    show_default=True,
)
@click.option(
    "--archive-base-url",
    help="Floodgate yearly archive base URL.",
    type=str,
    default=DEFAULT_ARCHIVE_BASE_URL,
    show_default=True,
)
@click.option(
    "--archive-cache-dir",
    help=(
        "Keep downloaded yearly archives here for reuse. "
        "Without this, archives go to a temporary "
        "directory and are deleted after extraction."
    ),
    type=click.Path(path_type=Path),
    required=False,
)
@click.option(
    "--delay",
    "delay_seconds",
    help="Politeness wait (seconds) after each request.",
    type=float,
    default=0.2,
    show_default=True,
)
@click.option(
    "--timeout",
    "timeout_seconds",
    help="HTTP timeout (seconds) per request.",
    type=float,
    default=30.0,
    show_default=True,
)
@click.option(
    "--overwrite",
    type=bool,
    is_flag=True,
    help="Re-download files that already exist locally.",
)
@click.option(
    "--dry-run",
    type=bool,
    is_flag=True,
    help=(
        "Only count games via daily listings without "
        "downloading them."
    ),
)
@handle_exception
def fetch_floodgate(
    start_date: datetime,
    end_date: datetime | None,
    output_dir: Path,
    strategy: str,
    base_url: str,
    archive_base_url: str,
    archive_cache_dir: Path | None,
    delay_seconds: float,
    timeout_seconds: float,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """Fetch CSA game records from the floodgate archive.

    Downloads shogi game records of the floodgate server
    (wdoor.c.u-tokyo.ac.jp) for the given date range and
    saves .csa files under OUTPUT_DIR in a YYYY/MM/DD
    layout, ready for `maou hcpe-convert`. Already-fetched
    files are skipped, so an interrupted run can simply be
    re-executed.

    WARNING: this command depends on the (unofficial)
    public site structure of floodgate and is fragile by
    design — it may break whenever the site changes. When
    no game is found at all, it fails loudly instead of
    producing an empty result.
    """
    http_client = UrllibHttpClient(
        timeout_seconds=timeout_seconds
    )
    result = fetcher.fetch_floodgate(
        http_client,
        start_date=start_date.date(),
        end_date=(
            end_date.date() if end_date is not None else None
        ),
        output_dir=output_dir,
        strategy=strategy,
        base_url=base_url,
        archive_base_url=archive_base_url,
        archive_cache_dir=archive_cache_dir,
        delay_seconds=delay_seconds,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    click.echo(result)
