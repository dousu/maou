# `maou pre-process`

## Overview

- Bridges HCPE datasources (local folders, BigQuery, GCS, S3) to feature stores
  by converting raw `.hcpe` inputs into preprocessed `.npy` shards and optional
  BigQuery/GCS/S3 uploads. All CLI flags are defined in
  `src/maou/infra/console/pre_process.py` and feed directly into the interface
  adapter.【F:src/maou/infra/console/pre_process.py†L1-L400】
- `maou.interface.preprocess` validates the destination directory, worker count,
  and caching knobs before instantiating the `PreProcess` use case, which
  orchestrates batching, intermediate caches, and feature-store writes.【F:src/maou/interface/preprocess.py†L1-L89】【F:src/maou/app/pre_process/hcpe_transform.py†L1-L147】

## CLI options

### Input selection (HCPE only)

| Source | Required flags | Notes |
| --- | --- | --- |
| Local filesystem | `--input-path PATH` (file or directory), optional `--input-file-packed` | Walks recursively via `FileSystem.collect_files` and decodes bit-packed numpy payloads when requested.【F:src/maou/infra/console/pre_process.py†L16-L66】 |
| BigQuery | `--input-dataset-id` + `--input-table-name` | Streams HCPE rows with configurable batch size, cache limits, clustering, and partition hints. Requires the `gcp` optional extra.【F:src/maou/infra/console/pre_process.py†L66-L200】 |
| GCS | `--input-gcs` + `--input-bucket-name` + `--input-prefix` + `--input-data-name` + `--input-local-cache-dir` | Downloads `.npy` shards tagged `array_type="hcpe"`. Supports worker counts, bundling (`--input-enable-bundling`, `--input-bundle-size-gb`), and optional local caching.【F:src/maou/infra/console/pre_process.py†L200-L360】 |
| S3 | `--input-s3` + bucket metadata | Mirrors the GCS contract using `S3DataSource`. Requires the `aws` optional extra.【F:src/maou/infra/console/pre_process.py†L318-L360】 |

Exactly one datasource may be active; the CLI raises a `ValueError` when multiple
providers are requested.【F:src/maou/infra/console/pre_process.py†L360-L420】

### Feature-store outputs

| Destination | Required flags | Data layout |
| --- | --- | --- |
| Local only | `--output-dir PATH` | Writes `.npy` shards tagged `array_type="preprocessing"` to disk. Directory is created automatically when missing.【F:src/maou/infra/console/pre_process.py†L16-L120】【F:src/maou/interface/preprocess.py†L17-L47】 |
| BigQuery | `--output-bigquery` + `--dataset-id` + `--table-name` | Streams feature rows via `BigQueryFeatureStore` with cache limits governed by `--output-max-cached-bytes`.【F:src/maou/infra/console/pre_process.py†L360-L487】 |
| GCS | `--output-gcs` + `--output-bucket-name` + `--output-prefix` + `--output-data-name` | Uploads `.npy` shards as `array_type="preprocessing"` with configurable worker counts and queue sizes.【F:src/maou/infra/console/pre_process.py†L420-L520】 |
| S3 | `--output-s3` + bucket metadata | Same contract as GCS, requiring the AWS optional extra.【F:src/maou/infra/console/pre_process.py†L420-L540】 |

Only one feature store may be selected per run; the CLI enforces mutual
exclusion and warns when extras are missing.【F:src/maou/infra/console/pre_process.py†L360-L540】

### Intermediate caching & workers

| Flag | Default | Description |
| --- | --- | --- |
| `--process-max-workers INT` | `3` | Caps CPU workers dedicated to preprocessing. Negative values raise `ValueError`; omitting the flag defaults to `min(3, cpu_count)`．【F:src/maou/infra/console/pre_process.py†L540-L575】【F:src/maou/interface/preprocess.py†L47-L71】 |
| `--intermediate-cache-dir PATH` | optional | Directory where the app layer spills batched numpy arrays before uploading or finalizing outputs.【F:src/maou/infra/console/pre_process.py†L575-L620】【F:src/maou/app/pre_process/hcpe_transform.py†L93-L147】 |
| `--intermediate-batch-size INT` | `50000` | DuckDBへのフラッシュ前に蓄積するレコード数．バッチ蓄積バッファにより，複数の小さなDataFrameを結合してからDuckDBにupsertすることでトランザクションオーバーヘッドを削減．A100 High Memory (83GB RAM) 向けに最適化．【F:src/maou/infra/console/pre_process.py†L575-L620】【F:src/maou/app/pre_process/hcpe_transform.py†L93-L147】 |
| `--win-rate-threshold INT` | `2` | 指し手別勝率計算の最小出現回数閾値．局面の出現回数(`count`)がこの値未満の場合，`moveWinRate` を合法手への均等配分(1/N)に置換する．推奨範囲: 2〜4．【F:src/maou/infra/console/pre_process.py†L251-L258】 |
| `--input-split-rows INT` | `500000` | 入力ファイルをこの行数にリサイズして並列処理を改善する．大きなファイルは分割し，小さなファイル（目標行数の半分未満）はチャンクにまとめる．Rustバックエンドを使用してLZ4圧縮を維持したまま高速に処理．0を指定すると無効化．A100 High Memory 向けに最適化．【F:src/maou/infra/console/pre_process.py†L230-L245】【F:src/maou/interface/preprocess.py†L50-L180】 |

## Execution flow

1. **Datasource resolution** – The CLI checks provider exclusivity, instantiates
   a `DataSource` (local `FileDataSource`, `BigQueryDataSource`, `GCSDataSource`,
   or `S3DataSource`), and pins `array_type="hcpe"` so only HCPE tensors enter
   the workflow.【F:src/maou/infra/console/pre_process.py†L66-L360】
2. **Interface normalization** – `maou.interface.preprocess.transform` ensures
   the optional `--output-dir` is a directory, validates the worker count, and
   builds `PreProcess.PreProcessOption` plus the optional feature store.
   Intermediate cache hints are passed through untouched.【F:src/maou/interface/preprocess.py†L17-L89】
3. **Preprocessing and batching** – `PreProcess.transform` iterates over the HCPE
   batches, deduplicates board states, writes intermediate bundles to disk or a
   temporary directory, and emits final `.npy` shards. When a feature store is
   configured, each shard is reloaded and uploaded with consistent metadata.
   【F:src/maou/app/pre_process/hcpe_transform.py†L1-L205】
4. **Result reporting** – The CLI prints the JSON mapping returned by the
   interface, showing per-batch status strings so operators can audit how many
   samples were processed locally vs. sent to cloud stores.【F:src/maou/interface/preprocess.py†L47-L89】

## Validation and guardrails

- Datasource and feature-store flags are mutually exclusive; selecting more than
  one provider raises a descriptive error before any work begins.【F:src/maou/infra/console/pre_process.py†L360-L540】
- All inputs must already be HCPE tensors. The CLI hard-codes `array_type="hcpe"`
  for every datasource, so preprocessing never attempts to parse CSA/KIF
  directly.【F:src/maou/infra/console/pre_process.py†L200-L360】
- `output_dir_init` raises when the provided path exists but is not a directory,
  preventing accidental overwrites.【F:src/maou/interface/preprocess.py†L17-L47】
- Negative `--process-max-workers` values raise immediately with
  `ValueError("max_workers must be non-negative"...)`, keeping the pipeline from
  spawning invalid executor settings.【F:src/maou/interface/preprocess.py†L47-L71】
- When optional extras (GCP/AWS) are missing the CLI logs warnings instead of
  crashing, so local-only runs remain usable.【F:src/maou/infra/console/pre_process.py†L200-L520】

## Outputs and usage

- Local runs write `.npy` shards derived from HCPE inputs into `--output-dir`
  (if provided); otherwise the app layer uses a temporary directory purely for
  uploading to feature stores.【F:src/maou/app/pre_process/hcpe_transform.py†L93-L205】
- Cloud-enabled runs push structured arrays with `key_columns=["id"]` and
  `partitioning_key_date="partitioningKey"` to the selected feature store so
  downstream pipelines can join on consistent keys.【F:src/maou/infra/console/pre_process.py†L360-L520】
- The JSON summary lists counts per batch, making it easy to script health checks
  or store logs alongside converted artifacts.【F:src/maou/interface/preprocess.py†L47-L89】

### Example invocation

```bash
poetry run maou pre-process \
  --input-gcs \
  --input-bucket-name maou-hcpe \
  --input-prefix prod/shards \
  --input-data-name training \
  --input-local-cache-dir /tmp/cache \
  --output-dir artifacts/preprocess \
  --output-bigquery --dataset-id features --table-name preprocessing \
  --process-max-workers 3 \
  --intermediate-cache-dir /tmp/pre-cache \
  --intermediate-batch-size 2000
```

## Implementation references

- CLI definition and provider wiring – `src/maou/infra/console/pre_process.py`.【F:src/maou/infra/console/pre_process.py†L1-L540】
- Interface adapter, directory/worker validation – `src/maou/interface/preprocess.py`.【F:src/maou/interface/preprocess.py†L17-L89】
- Preprocessing workflow, batching, and feature-store uploads –
  `src/maou/app/pre_process/hcpe_transform.py`.【F:src/maou/app/pre_process/hcpe_transform.py†L1-L205】
