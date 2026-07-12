# `maou hcpe-convert`

## Overview

- Converts CSA/KIF game records into HCPE `.feather` shards and optionally
  streams them to BigQuery, GCS, or S3. Every flag defined in
  `src/maou/infra/console/hcpe_convert.py` maps directly to the interface layer,
  so operators can drive filtering, concurrency, and output destinations without
  touching Python code.【F:src/maou/infra/console/hcpe_convert.py†L1-L150】
- The interface gathers the requested filters and worker counts into
  `HCPEConverter.ConvertOption`, then `HCPEConverter.convert` delegates the whole
  file set to the `maou_convert` Rust pipeline, emitting per-file status strings
  plus optional feature store uploads.【F:src/maou/interface/converter.py†L60-L117】【F:src/maou/app/converter/hcpe_converter.py†L79-L170】

## Requirements

- CSA/KIF → HCPE conversion runs entirely in the in-house Rust pipeline
  (`maou_convert`, exposed as `maou._rust.maou_convert.convert_hcpe_files`).
  Files are read, decoded, replayed, and written to `.feather` on the Rust
  side (rayon-parallel, GIL released); no extra dependency is required —
  `hcpe-convert` works on a base install. The output is parity-verified
  bit-exact against the previous Python/cshogi implementation
  (`tests/maou/app/converter/test_rust_convert_parity.py`,
  `rust/maou_shogi/tests/kifu_parity.rs`).
  The legacy `hcpe` extra (cshogi) is retained only for fixture
  regeneration; production code does not import it.
- **Multi-game CSA files are fully converted.** A file containing several
  games (separated by `/`) yields rows for every game; game 0 keeps the
  legacy id form `{stem}.hcpe_{ply}` and games ≥ 1 use
  `{stem}.hcpe_g{game}_{ply}`. (The previous implementation converted only
  the first game.)
- **Shift_JIS (cp932) `.kif` files are supported.** The Rust decoder tries
  UTF-8 first and falls back to cp932, so `.kif` exports from ShogiGUI etc.
  are read without manual re-encoding. (The previous `read_text()` was
  UTF-8-only.)

## CLI options

### Input and filtering

| Flag | Required | Description |
| --- | --- | --- |
| `--input-path PATH` | ✅ | File or directory containing CSA/KIF records. Directories are walked recursively via `FileSystem.collect_files`.【F:src/maou/infra/console/hcpe_convert.py†L16-L72】【F:src/maou/interface/converter.py†L60-L89】 |
| `--input-format {kif,csa}` | ✅ | Accepted formats enforced by `InputFormat`; anything else raises `ValueError`.【F:src/maou/infra/console/hcpe_convert.py†L22-L34】【F:src/maou/interface/converter.py†L33-L58】 |
| `--output-dir PATH` | ✅ | Destination directory for `.npy` shards. Created automatically when missing; must already be a directory if it exists.【F:src/maou/infra/console/hcpe_convert.py†L34-L46】【F:src/maou/interface/converter.py†L58-L75】 |
| `--min-rating/--min-moves/--max-moves` | optional | Filter out games below rating thresholds or outside the move-count window before conversion.【F:src/maou/infra/console/hcpe_convert.py†L46-L90】【F:src/maou/app/converter/hcpe_converter.py†L90-L179】 |
| `--allowed-endgame-status` | repeatable | Restrict CSA/KIF terminal markers (e.g., `%TORYO`). An empty list means "any".【F:src/maou/infra/console/hcpe_convert.py†L72-L90】【F:src/maou/app/converter/hcpe_converter.py†L90-L158】 |
| `--exclude-moves` | repeatable | Skip specific move IDs even inside accepted games.【F:src/maou/infra/console/hcpe_convert.py†L90-L102】【F:src/maou/app/converter/hcpe_converter.py†L158-L214】 |
| `--chunk-size INT` | default `500000` | Number of rows per chunked output file. After individual files are converted, they are merged into chunked files of this size. Set to `0` to disable chunking and keep one file per game.【F:src/maou/infra/console/hcpe_convert.py†L143-L150】 |
| `--process-max-workers INT` | default `4` | Number of rayon threads used by the Rust conversion pipeline. `1` runs single-threaded; negative values raise before work starts; omitting the flag defaults to `min(4, cpu_count)`.【F:src/maou/infra/console/hcpe_convert.py†L150-L158】【F:src/maou/interface/converter.py†L75-L110】 |

### Cloud outputs and batching

| Flag group | Required | Description |
| --- | --- | --- |
| `--output-bigquery` + `--dataset-id` + `--table-name` | optional | Streams converted features directly into BigQuery via `BigQueryFeatureStore`. Honors `--max-cached-bytes` for batching.【F:src/maou/infra/console/hcpe_convert.py†L102-L134】 |
| `--output-gcs` + `--bucket-name` + `--prefix` + `--data-name` | optional | Uploads `.npy` shards to GCS with configurable worker counts, queue sizes, and cache limits. Arrays are tagged as `array_type="hcpe"`.【F:src/maou/infra/console/hcpe_convert.py†L134-L210】 |
| `--output-s3` + bucket flags | optional | Mirrors the GCS contract using `S3FeatureStore`, requiring the AWS optional extra.【F:src/maou/infra/console/hcpe_convert.py†L134-L232】 |
| `--max-cached-bytes` | default `524,288,000` | Upper bound for feature-store batching buffers.【F:src/maou/infra/console/hcpe_convert.py†L118-L150】 |
| `--output-max-workers` / `--output-max-queue-size` | default `4` | Concurrency knobs for object-storage uploads; ignored by BigQuery.【F:src/maou/infra/console/hcpe_convert.py†L118-L150】 |

## Execution flow

1. **Validation and setup** – `input_format_validation` accepts only `csa` or
   `kif`, and `output_dir_init` ensures the target directory exists before any
   parsing begins.【F:src/maou/interface/converter.py†L33-L75】
2. **File discovery** – `FileSystem.collect_files` enumerates the requested file
   or directory, handing a full list of paths to the converter options so both
   sequential and parallel execution modes know what to process.【F:src/maou/interface/converter.py†L89-L107】
3. **Option assembly** – The interface stores rating, move-count, and
   endgame/move filters inside `HCPEConverter.ConvertOption`, along with the
   worker count and destination directory.【F:src/maou/interface/converter.py†L96-L117】
4. **Feature-store selection** – The CLI enforces that at most one cloud
   provider is active, then builds the appropriate feature store. Missing
   optional extras trigger warnings instead of crashes so local-only runs keep
   going.【F:src/maou/infra/console/hcpe_convert.py†L150-L232】
5. **Conversion** – `HCPEConverter.convert` hands the discovered paths (in
   batches, for progress reporting) to `maou._rust.maou_convert.convert_hcpe_files`,
   which reads/decodes/replays each file and writes an individual `.feather`
   on the Rust side using `--process-max-workers` rayon threads, returning
   per-file status strings.【F:src/maou/app/converter/hcpe_converter.py†L79-L170】
6. **Chunking and uploads** – After all files are converted, individual `.feather`
   files are merged into chunked files via `merge_hcpe_feather_files` (controlled
   by `--chunk-size`). When a feature store is configured, each chunk is loaded
   and pushed to the cloud. Individual pre-merge files are cleaned
   up.【F:src/maou/app/converter/hcpe_converter.py†L370-L440】

## Validation and guardrails

- Input format typos surface as `ValueError("Input \"kif\" or \"csa\".")` so
  incorrect flags never reach the parser layer.【F:src/maou/interface/converter.py†L33-L58】
- `output_dir_init` raises when the path already exists but is not a directory,
  preventing accidental overwrites of files.【F:src/maou/interface/converter.py†L58-L75】
- Negative `--process-max-workers` values are rejected with
  `ValueError("max_workers must be non-negative"...)` before any files are
  touched.【F:src/maou/interface/converter.py†L75-L110】
- Only one of `--output-bigquery`, `--output-gcs`, or `--output-s3` may be set.
  The CLI counts enabled flags and raises when multiple providers are
  requested.【F:src/maou/infra/console/hcpe_convert.py†L150-L168】
- When optional extras are missing the CLI logs a warning explaining which
  `poetry install -E ...` flag to run instead of crashing outright.【F:src/maou/infra/console/hcpe_convert.py†L168-L232】

## Outputs and usage

- Each processed file yields a status string such as `"success 128 rows"`,
  `"skipped"`, or `"error: ..."`. The CLI prints the aggregated JSON blob, making
  it easy to audit which source files produced data.【F:src/maou/interface/converter.py†L96-L117】【F:src/maou/app/converter/hcpe_converter.py†L180-L236】
- When `--chunk-size` is set (default `500000`), individual `.feather` files are
  merged into chunked outputs named `hcpe_chunk0000.feather`,
  `hcpe_chunk0001.feather`, etc. The original per-game files are removed after
  merging.【F:src/maou/app/converter/hcpe_converter.py†L370-L440】
- When a feature store is configured, each chunk is loaded and uploaded via
  `store_features` with `key_columns=["id"]` and
  `partitioning_key_date="partitioningKey"`, ensuring downstream tables retain
  consistent keys.【F:src/maou/app/converter/hcpe_converter.py†L370-L440】

### Example invocation

```bash
uv run maou hcpe-convert \
  --input-path data/csa_games \
  --input-format csa \
  --output-dir artifacts/hcpe \
  --min-rating 2400 \
  --chunk-size 500000 \
  --process-max-workers 8 \
  --output-gcs --bucket-name my-bucket --prefix hcpe --data-name training
```

## Implementation references

- CLI definition, feature-store wiring, and output printing –
  `src/maou/infra/console/hcpe_convert.py`.【F:src/maou/infra/console/hcpe_convert.py†L1-L232】
- Interface adapter and validation helpers – `src/maou/interface/converter.py`.【F:src/maou/interface/converter.py†L33-L117】
- Conversion logic, filtering, and optional uploads –
  `src/maou/app/converter/hcpe_converter.py`.【F:src/maou/app/converter/hcpe_converter.py†L58-L236】
