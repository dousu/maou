"""floodgate 棋譜取得コマンドの interface アダプタ．"""

import json
import logging
from datetime import date
from pathlib import Path

from maou.app.fetcher.floodgate_fetcher import (
    DEFAULT_ARCHIVE_BASE_URL,
    DEFAULT_BASE_URL,
    STRATEGIES,
    FloodgateFetcher,
    HttpClient,
)
from maou.interface.converter import output_dir_init

logger: logging.Logger = logging.getLogger(__name__)


def fetch_floodgate(
    http_client: HttpClient,
    *,
    start_date: date,
    end_date: date | None = None,
    output_dir: Path,
    strategy: str = "auto",
    base_url: str = DEFAULT_BASE_URL,
    archive_base_url: str = DEFAULT_ARCHIVE_BASE_URL,
    archive_cache_dir: Path | None = None,
    delay_seconds: float = 0.2,
    overwrite: bool = False,
    dry_run: bool = False,
) -> str:
    """Fetch CSA game records from the floodgate archive.

    Args:
        http_client: HTTP transport implementation
        start_date: First day to fetch (inclusive)
        end_date: Last day to fetch (inclusive).
            Defaults to ``start_date``.
        output_dir: Directory for downloaded files
            (mirrors ``YYYY/MM/DD/`` structure)
        strategy: Fetch strategy ('auto', 'daily' or
            'archive')
        base_url: Daily listing base URL
        archive_base_url: Yearly archive base URL
        archive_cache_dir: Directory to keep downloaded
            yearly archives for reuse. If None, archives
            are stored in a temporary directory and
            deleted after extraction.
        delay_seconds: Politeness wait after each request
        overwrite: Re-download files that already exist
        dry_run: Only count games without downloading

    Returns:
        JSON string with fetch summary counters

    Raises:
        ValueError: If the date range, strategy or delay
            is invalid
        FloodgateStructureError: If no game is found in
            the whole range (possible site change)
        ArchiveToolMissingError: If a .7z archive is
            needed but py7zr is not installed
    """
    if end_date is None:
        end_date = start_date
    if end_date < start_date:
        raise ValueError(
            f"end_date {end_date} is before "
            f"start_date {start_date}"
        )
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy '{strategy}' "
            f"(choose from {STRATEGIES})"
        )
    if delay_seconds < 0:
        raise ValueError(
            "delay_seconds must be non-negative, "
            f"got {delay_seconds}"
        )
    output_dir_init(output_dir)
    logger.info(
        f"floodgate fetch: {start_date}..{end_date} "
        f"({strategy}) -> {output_dir}"
    )

    option = FloodgateFetcher.FetchOption(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        strategy=strategy,
        base_url=base_url,
        archive_base_url=archive_base_url,
        archive_cache_dir=archive_cache_dir,
        delay_seconds=delay_seconds,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    result = FloodgateFetcher(http_client=http_client).fetch(
        option
    )

    return json.dumps(result)
