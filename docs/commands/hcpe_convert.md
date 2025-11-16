# `maou hcpe-convert`

This document explains how the `hcpe-convert` CLI command orchestrates Shogi
record ingestion and hand-off to the application layer. It summarizes every
flag defined in `src/maou/infra/console/hcpe_convert.py`, the validation and
queuing logic inside `src/maou/interface/converter.py`, and how work eventually
lands in the app-layer `HCPEConverter`.

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--input-path PATH` | ✅ | File or directory that already exists. Directories are walked recursively. |
| `--input-format {kif,csa}` | ✅ | Forwarded to the interface validator so only CSA/KIF inputs are accepted. |
| `--output-dir PATH` | ✅ | Destination directory. Created automatically when missing, but must be a directory if it already exists. |
| `--min-rating INT` |  | Lower bound for either player's rating before a game is considered. |
| `--min-moves INT` |  | Filters out games shorter than the threshold. |
| `--max-moves INT` |  | Filters out games longer than the threshold. |
| `--allowed-endgame-status VALUE` | repeatable | Allow-list of CSA/KIF terminal markers (e.g. `%TORYO`). An empty list means "any". |
| `--exclude-moves INT` | repeatable | Individual move IDs to skip even inside accepted games. |
| `--process-max-workers INT` | default `4` | Maximum number of CPU workers dedicated to parsing and conversion. Negative values are rejected inside the interface layer. |
| `--output-bigquery` + `--dataset-id` + `--table-name` |  | Stream converted features directly into BigQuery when the optional `gcp` extra is installed. |
| `--output-gcs` + `--bucket-name` + `--prefix` + `--data-name` |  | Upload `.npy` shards to Google Cloud Storage via the GCS feature store helper. |
| `--output-s3` + `--bucket-name` + `--prefix` + `--data-name` |  | Same as GCS but routed through the S3 feature store helper. Requires the optional `aws` dependencies. |
| `--max-cached-bytes INT` | default `524,288,000` | Upper bound for the batching buffer used by every cloud feature store. |
| `--output-max-workers INT` | default `4` | Number of asynchronous uploader workers for GCS/S3 transfers. Ignored by BigQuery. |
| `--output-max-queue-size INT` | default `4` | Size of the producer queue feeding each object-storage worker pool. Ignored by BigQuery. |

The command forbids mixing providers: it counts the `--output-bigquery`,
`--output-gcs`, and `--output-s3` flags and raises an error when more than one
backend is enabled at once. When optional extras are missing, the CLI logs a
warning instead of crashing so that local file-only runs remain usable.

## Data path inside `converter.transform`

1. **Format and output validation** – `input_format_validation` accepts only
   `csa` or `kif`, and `output_dir_init` either creates the target directory or
   verifies that an existing path is a directory.
2. **File discovery** – `FileSystem.collect_files` receives the validated
   `input_path`. When the path is a directory, it walks recursively and returns
   every file. When it is a file, the returned list contains just that path.
3. **Worker determination** – The CLI passes `--process-max-workers` to the
   interface layer. If the flag is omitted the interface selects the minimum of
   4 and the current CPU count; if the flag is negative a `ValueError` is raised
   before any files are touched.
4. **Filtering rules** – The interface packages all filtering knobs
   (`min_rating`, `min_moves`, `max_moves`, `allowed_endgame_status`,
   `exclude_moves`) into `HCPEConverter.ConvertOption`. The app layer enforces
   these rules per game: files that fail validation are marked as `skipped`
   and never produce output.
5. **Feature store selection** – When no cloud flag is provided the converter
   writes only to the local filesystem. Otherwise the CLI injects a feature
   store object that wraps cloud-specific concurrency and caching behavior.

## Cloud feature stores

All cloud integrations share the `feature_store` interface so the app layer can
blindly call `store_features`, yet each backend has unique requirements:

- **BigQuery** – Requires `--dataset-id` and `--table-name`. It accepts only a
  `max_cached_bytes` argument because uploads are performed through streaming
  inserts that rely on batching inside `BigQueryFeatureStore`.
- **GCS** – Needs `--bucket-name`, `--prefix`, and `--data-name`. Besides
  `max_cached_bytes`, the CLI forwards `--output-max-workers` and
  `--output-max-queue-size` so the uploader can multiplex chunk uploads to the
  bucket. The array type is pinned to `hcpe` to keep schema handling consistent.
- **S3** – Mirrors the GCS contract (bucket, prefix, data name) and uses the
  same worker-count and queue-size knobs when instantiating `S3FeatureStore`.

Because the CLI enforces mutual exclusivity, at most one feature store object is
passed down per run. When extras are missing the command logs a warning but
keeps running so that `.npy` files still land locally.

## Work hand-off and outputs

1. The CLI builds the feature store (if any) and then calls
   `converter.transform(FileSystem(), ...)`.
2. The interface layer wraps the gathered options into
   `HCPEConverter.ConvertOption` and instantiates `HCPEConverter` with the
   optional feature store.
3. `HCPEConverter.convert` fans out over the discovered files, using either a
   single-process loop or a `ProcessPoolExecutor` configured with
   `max_workers`. Each successful conversion writes an `.npy` file whose name
   matches the source file with the `.hcpe` extension swapped for `.npy`.
4. When a feature store exists, the converter immediately reloads the saved
   `.npy` shard and calls `store_features` with `key_columns=["id"]` and
   `partitioning_key_date="partitioningKey"`, letting the backend handle
   concurrency or caching.
5. The conversion status for every file is aggregated into a dict that maps
   source paths to strings such as `"success 128 rows"` or `"skipped"`.
   `converter.transform` serializes this mapping to JSON and the CLI prints the
   JSON blob to STDOUT. The local `.npy` files remain in `--output-dir`, while
   cloud stores receive the same data via their respective `store_features`
   implementations.
