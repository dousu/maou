# `maou fetch-floodgate`

## Overview

- Downloads CSA game records of the floodgate shogi server
  (wdoor.c.u-tokyo.ac.jp) for a given date range and saves them under
  `--output-dir` in a `YYYY/MM/DD/` layout, ready to be passed to
  `maou hcpe-convert --input-path`.
  【F:src/maou/infra/console/fetch_floodgate.py†L18-L157】
  【F:src/maou/app/fetcher/floodgate_fetcher.py†L1-L27】
- Two fetch strategies are implemented, selected via `--strategy`:
  - **daily** — crawls the per-day directory listings
    (`x/YYYY/MM/DD/`) and downloads each `.csa` file individually.
    Works for any range, but issues hundreds of requests per day of
    games.【F:src/maou/app/fetcher/floodgate_fetcher.py†L662-L725】
  - **archive** — downloads the yearly archive
    (`archive/wdoorYYYY.7z`, or `.tar.xz` for 2011/2012) once and
    extracts only the requested days. Far more efficient for ranges of
    a month or longer (a yearly archive is 100–350 MB).
    【F:src/maou/app/fetcher/floodgate_fetcher.py†L556-L660】
  - **auto** (default) — uses the yearly archive when 32+ days of a
    year are requested, otherwise crawls daily; days missing from the
    archive (e.g. the most recent days of the current year) are topped
    up by daily crawling.【F:src/maou/app/fetcher/floodgate_fetcher.py†L505-L554】
- Already-downloaded files are skipped by default, so an interrupted
  run can simply be re-executed (resumable). Files are written via a
  `.part` temp name and renamed, so no truncated `.csa` is left behind.
  【F:src/maou/app/fetcher/floodgate_fetcher.py†L699-L724】

> **⚠️ Fragile by design**: this command depends on the *unofficial*
> public site structure of floodgate (URL layout, HTML listing format,
> file-name conventions). It may break whenever the site changes — the
> yearly archive URL, for example, has moved in the past from `x/` to
> `archive/`. When no game is found at all, the command exits non-zero
> with `FloodgateStructureError` instead of silently succeeding with an
> empty result.【F:src/maou/app/fetcher/floodgate_fetcher.py†L71-L82】

## Requirements

- The `daily` strategy and `.tar.xz` archives (years 2011/2012) work
  with the Python standard library only — no extra dependencies.
- Extracting `.7z` yearly archives (2014 and later) requires `py7zr`,
  provided by the `fetch` extra: `uv sync --extra fetch`. Without it,
  `--strategy daily` still works and a clear error explains the
  alternative.【F:src/maou/app/fetcher/floodgate_fetcher.py†L252-L268】

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--start-date YYYY-MM-DD` | ✅ | First day to fetch (JST).【F:src/maou/infra/console/fetch_floodgate.py†L19-L24】 |
| `--end-date YYYY-MM-DD` | default `--start-date` | Last day to fetch, inclusive.【F:src/maou/infra/console/fetch_floodgate.py†L25-L33】 |
| `--output-dir PATH` | ✅ | Output root. Files land in `OUTPUT_DIR/YYYY/MM/DD/*.csa`.【F:src/maou/infra/console/fetch_floodgate.py†L34-L42】 |
| `--strategy [auto\|daily\|archive]` | default `auto` | Fetch strategy (see Overview).【F:src/maou/infra/console/fetch_floodgate.py†L43-L54】 |
| `--base-url URL` | default floodgate `x/` | Daily listing base URL. Override if the site moves (or for testing).【F:src/maou/infra/console/fetch_floodgate.py†L55-L61】 |
| `--archive-base-url URL` | default floodgate `archive/` | Yearly archive base URL.【F:src/maou/infra/console/fetch_floodgate.py†L62-L68】 |
| `--archive-cache-dir PATH` | optional | Keep downloaded yearly archives here and reuse them on re-runs. Without it, archives go to a temp directory and are deleted after extraction. Do **not** point this inside `--output-dir` (hcpe-convert would pick the archives up as input).【F:src/maou/infra/console/fetch_floodgate.py†L69-L78】 |
| `--delay FLOAT` | default `0.2` | Politeness wait (seconds) after each HTTP request.【F:src/maou/infra/console/fetch_floodgate.py†L79-L86】 |
| `--timeout FLOAT` | default `30.0` | HTTP timeout (seconds) per request.【F:src/maou/infra/console/fetch_floodgate.py†L87-L94】 |
| `--overwrite` | flag | Re-download files that already exist locally.【F:src/maou/infra/console/fetch_floodgate.py†L95-L100】 |
| `--dry-run` | flag | Only count games via daily listings; never downloads files or archives.【F:src/maou/infra/console/fetch_floodgate.py†L101-L109】 |

## Execution flow

1. **CLI validation** -- click parses dates and the strategy choice,
   then hands off to the interface adapter with a
   `UrllibHttpClient` (stdlib `urllib`, explicit User-Agent,
   http/https only).【F:src/maou/infra/console/fetch_floodgate.py†L110-L157】
   【F:src/maou/infra/http/urllib_http_client.py†L28-L92】
2. **Interface hand-off** -- `maou.interface.fetcher.fetch_floodgate`
   validates the date range, strategy and delay, initializes the
   output directory and builds the
   `FloodgateFetcher.FetchOption`.【F:src/maou/interface/fetcher.py†L20-L105】
3. **Fetch use case** -- `FloodgateFetcher.fetch` groups the range by
   year, picks the strategy per year, downloads/extracts, and returns
   a JSON summary (days fetched/missing, files downloaded/skipped,
   archives used) that the CLI echoes.
   【F:src/maou/app/fetcher/floodgate_fetcher.py†L427-L503】
4. **Archive extraction** -- game days are derived from the trailing
   `+YYYYMMDDHHMMSS.csa` timestamp of each entry name, so extraction
   does not depend on the directory structure inside the archive
   (the yearly 7z is flat: `YYYY/<file>.csa`).
   【F:src/maou/app/fetcher/floodgate_fetcher.py†L165-L182】
   【F:src/maou/app/fetcher/floodgate_fetcher.py†L343-L380】

## Validation and guardrails

- Unknown `--strategy` values are rejected by `click.Choice` and again
  by the interface layer.【F:src/maou/interface/fetcher.py†L73-L77】
- `end_date < start_date` and negative `--delay` raise `ValueError`
  before any network access.【F:src/maou/interface/fetcher.py†L66-L83】
- A day with no listing (HTTP 404 — future dates, no games) is counted
  and skipped; a listing with zero `.csa` links is logged as a possible
  site change. If the *whole range* yields zero games, the command
  fails loudly (`FloodgateStructureError`, exit code 1).
  【F:src/maou/app/fetcher/floodgate_fetcher.py†L477-L488】
- `.7z` archives without `py7zr` installed produce
  `ArchiveToolMissingError` with the exact `uv sync --extra fetch`
  hint; `.tar.xz` (2011/2012) extracts with the standard library.
  【F:src/maou/app/fetcher/floodgate_fetcher.py†L599-L660】
- Downloads stream to disk in 1 MiB chunks (yearly archives are never
  held in memory).【F:src/maou/infra/http/urllib_http_client.py†L152-L192】

## Example invocation

```bash
# One day of games (289 games on 2025-01-05), daily crawl via auto
maou fetch-floodgate --start-date 2025-01-05 \
    --output-dir ./floodgate

# Probe first: how many games would be fetched?
maou fetch-floodgate --start-date 2025-01-01 --end-date 2025-01-31 \
    --output-dir ./floodgate --dry-run

# A whole year via the yearly archive (needs: uv sync --extra fetch),
# keeping the archive for later re-use
maou fetch-floodgate --start-date 2025-01-01 --end-date 2025-12-31 \
    --output-dir ./floodgate --archive-cache-dir ./floodgate-archives

# Then convert to HCPE
maou hcpe-convert --input-path ./floodgate --input-format csa \
    --output-dir ./hcpe
```

## Implementation references

- CLI definition and flag parsing --
  `src/maou/infra/console/fetch_floodgate.py`.
  【F:src/maou/infra/console/fetch_floodgate.py†L1-L157】
- Interface adapter -- `src/maou/interface/fetcher.py`.
  【F:src/maou/interface/fetcher.py†L1-L105】
- Fetch use case (strategies, listing parsing, archive extraction) --
  `src/maou/app/fetcher/floodgate_fetcher.py`.
  【F:src/maou/app/fetcher/floodgate_fetcher.py†L1-L725】
- HTTP transport (stdlib urllib) --
  `src/maou/infra/http/urllib_http_client.py`.
  【F:src/maou/infra/http/urllib_http_client.py†L1-L192】
