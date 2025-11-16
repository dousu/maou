# `maou pre-process`

This guide documents how the `pre-process` CLI command wires HCPE data sources
into feature stores. It summarizes every decision the CLI makes inside
`src/maou/infra/console/pre_process.py` and how those options reach the
interface-layer adapter in `src/maou/interface/preprocess.py`.

## Input data-source decision tree

```mermaid
flowchart TD
    start([CLI invocation])
    local{`--input-path` provided?}
    bigquery{BigQuery IDs present?}
    gcs{`--input-gcs` + bucket + prefix + data-name + cache dir?}
    s3{`--input-s3` + bucket + prefix + data-name + cache dir?}
    success[Datasource object created]
    error([Abort with validation error])

    start --> bigquery
    bigquery -- yes --> bq[BigQueryDataSource (HCPE arrays)] --> success
    bigquery -- no --> gcs
    gcs -- yes --> gcsds[GCSDataSource (HCPE arrays with optional bundling)] --> success
    gcs -- no --> s3
    s3 -- yes --> s3ds[S3DataSource (HCPE arrays with optional bundling)] --> success
    s3 -- no --> local
    local -- yes --> files[FileDataSource collects `.hcpe` files; `--input-file-packed` toggles bit-pack] --> success
    local -- no --> error
```

The CLI guards against mutually exclusive cloud options before any objects are
instantiated: it counts whether BigQuery identifiers, `--input-gcs`, or
`--input-s3` are supplied and raises `ValueError` when more than one provider is
selected. The `array_type` is hard-coded to `"hcpe"` for every datasource, so all
inputs must already be encoded in HCPE format.

| Source | Required flags | Expected data format | Notes |
| --- | --- | --- | --- |
| Local filesystem | `--input-path PATH` (file or directory) | `.hcpe` files; use `--input-file-packed` when the numpy payloads are bit-packed | Walks directories recursively via `FileSystem.collect_files`. |
| BigQuery | `--input-dataset-id` + `--input-table-name` | HCPE rows stored in BigQuery, streamed in batches (`--input-batch-size`) | Optional `--input-clustering-key`, `--input-partitioning-key-date`, and `--input-max-cached-bytes` tune fetches; requires `poetry install -E gcp`. |
| Google Cloud Storage | `--input-gcs` + `--input-bucket-name` + `--input-prefix` + `--input-data-name` + `--input-local-cache-dir` | `.npy` shards containing HCPE arrays | Supports background download workers (`--input-max-workers`) and optional bundling via `--input-enable-bundling`/`--input-bundle-size-gb`. |
| Amazon S3 | `--input-s3` + `--input-bucket-name` + `--input-prefix` + `--input-data-name` + `--input-local-cache-dir` | `.npy` shards containing HCPE arrays | Same contract as GCS but requires the `aws` extra. |

## Feature-store selection

Output destinations follow the same mutually exclusive rule: at most one of
`--output-bigquery`, `--output-gcs`, or `--output-s3` may be passed. When none is
chosen the CLI only writes the generated `.npy` files to `--output-dir`.

| Feature store | Required flags | Data layout | Extra knobs |
| --- | --- | --- | --- |
| BigQuery | `--output-bigquery` + `--dataset-id` + `--table-name` | Feature rows streamed via `BigQueryFeatureStore` | Uses `--output-max-cached-bytes` for batching. |
| Google Cloud Storage | `--output-gcs` + `--output-bucket-name` + `--output-prefix` + `--output-data-name` | `.npy` files tagged as `array_type="preprocessing"` | Honors `--output-max-workers`, `--output-max-queue-size`, and `--output-max-cached-bytes`. |
| Amazon S3 | `--output-s3` + `--output-bucket-name` + `--output-prefix` + `--output-data-name` | `.npy` files tagged as `array_type="preprocessing"` | Shares the same worker, queue, and cache knobs as GCS. |

## Intermediate caching and interface hand-off

The CLI exposes two knobs for staging data between the datasource and feature
store:

- `--intermediate-cache-dir` – Optional directory where the app layer can spill
  pre-batched numpy arrays. When omitted, `PreProcess` falls back to a temporary
  directory.
- `--intermediate-batch-size` – Controls how many samples are written in each
  disk flush (default `1000`).

These options, together with `--process-max-workers`, are forwarded to the
interface layer through `preprocess.transform`. The adapter in
`src/maou/interface/preprocess.py` initializes a `PreProcess.PreProcessOption`
with the normalized worker count, then instantiates `PreProcess` with the
selected datasource, optional feature store, and the intermediate caching
arguments. Every `PreProcess.transform(...)` call therefore receives the same
settings that were provided on the CLI, ensuring deterministic cache placement
and batch sizing regardless of whether the workload is local-only or involves a
cloud feature store.

## Quick reference

- Data ingestion always expects HCPE tensors, whether they arrive from files,
  BigQuery, GCS, or S3.
- Exactly one input provider and at most one feature store may be selected per
  run. Violations raise a `ValueError` before any processing occurs.
- Intermediate caching flags control the on-disk staging layer that bridges the
  datasource (`HCPE` arrays) and the downstream feature stores (`preprocessing`
  arrays).
