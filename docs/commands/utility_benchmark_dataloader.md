# `maou utility benchmark-dataloader`

## Overview

- Benchmarks DataLoader throughput across worker counts, pin-memory settings, and
  batch sizes before launching long training jobs. The CLI lives in
  `src/maou/infra/console/utility.py` and wraps the interface helper in
  `src/maou/interface/utility_interface.py`.【F:src/maou/infra/console/utility.py†L217-L413】【F:src/maou/interface/utility_interface.py†L44-L146】
- `DataLoaderBenchmark` iterates through worker/prefetch combinations, records
  wall-clock timings, and surfaces optimal configurations plus insights that you
  can paste into `maou learn-model`.【F:src/maou/app/utility/dataloader_benchmark.py†L16-L392】

## CLI options

### Input selection & caching

| Flag | Required | Description |
| --- | --- | --- |
| `--input-dir PATH` | one of the sources | Reads local `.npy` tensors via `FileDataSource.FileDataSourceSpliter` with optional bit-packed decoding (`--input-file-packed`). `--sample-ratio` is ignored because the files already reside on disk.【F:src/maou/infra/console/utility.py†L217-L274】 |
| `--input-dataset-id` + `--input-table-name` | pair | Streams directly from BigQuery with the same batching, caching, clustering, and partition knobs as `learn-model`. Enabling `--sample-ratio` logs that `TABLESAMPLE` is active before constructing the datasource.【F:src/maou/infra/console/utility.py†L260-L325】 |
| `--input-gcs` / `--input-s3` + bucket metadata (`--input-bucket-name`, `--input-prefix`, `--input-data-name`, `--input-local-cache-dir`) | pair | Downloads shards via `GCSDataSource` or `S3DataSource` splitters. Supports worker counts (`--input-max-workers`), bundling (`--input-enable-bundling`, `--input-bundle-size-gb`), and optional sampling ratios. Requires the corresponding optional extras.【F:src/maou/infra/console/utility.py†L306-L374】 |
| `--input-format {hcpe,preprocess}` | default `hcpe` | Drives both CLI validation and the interface `datasource_type`. Unsupported strings raise `ValueError`.【F:src/maou/infra/console/utility.py†L217-L247】【F:src/maou/interface/utility_interface.py†L50-L79】 |
| BigQuery cache knobs (`--input-batch-size`, `--input-max-cached-bytes`, `--input-local-cache`, `--input-local-cache-dir`, `--input-clustering-key`, `--input-partitioning-key-date`) | optional | Forwarded directly to the datasource constructors to mimic production ingestion pressure.【F:src/maou/infra/console/utility.py†L57-L205】 |

Only **one** remote provider can be enabled at a time; the CLI counts enabled
flags and aborts when multiple cloud inputs are selected.【F:src/maou/infra/console/utility.py†L230-L247】

### Benchmark knobs

| Flag | Default | Description |
| --- | --- | --- |
| `--gpu DEVICE` | optional | Chooses the PyTorch device string passed into the benchmark config (e.g., `cuda:0`, `cpu`).【F:src/maou/infra/console/utility.py†L397-L403】【F:src/maou/interface/utility_interface.py†L50-L60】 |
| `--batch-size INT` | `256` | Overrides the batch size used during benchmarking; must be positive.【F:src/maou/infra/console/utility.py†L164-L169】【F:src/maou/interface/utility_interface.py†L61-L68】 |
| `--pin-memory` | auto | The CLI forwards the flag and the interface defaults to `True` on CUDA and `False` on CPU.【F:src/maou/infra/console/utility.py†L170-L176】【F:src/maou/interface/utility_interface.py†L69-L72】 |
| `--num-batches INT` | `100` | Number of minibatches each configuration processes; must be positive.【F:src/maou/infra/console/utility.py†L177-L183】【F:src/maou/interface/utility_interface.py†L73-L79】 |
| `--sample-ratio FLOAT` | full dataset | Restricts BigQuery/GCS/S3 datasources to `0.01–1.0` of the dataset. The interface enforces the range and logs the selection; the CLI warns when the flag is redundant for local folders.【F:src/maou/infra/console/utility.py†L251-L325】【F:src/maou/interface/utility_interface.py†L81-L95】 |

## Execution flow

1. **Datasource resolution** – The CLI enforces provider exclusivity, instantiates
   the requested datasource, and surfaces explicit errors when the required GCP/AWS
   extras are missing.【F:src/maou/infra/console/utility.py†L217-L374】
2. **Interface normalization** – `utility_interface.benchmark_dataloader` validates
   the datasource type, materializes a PyTorch device (bumping matmul precision on
   CUDA), and calls `datasource.train_test_split(test_ratio=0.1)` to isolate the
   training subset that feeds the benchmark.【F:src/maou/interface/utility_interface.py†L44-L100】
3. **Benchmark orchestration** – `DataLoaderBenchmark` builds a `KifDataset`, tests
   worker counts up to the CPU core count (capped at 16), sweeps prefetch factors,
   and records timings plus throughput metrics for each configuration.【F:src/maou/interface/utility_interface.py†L101-L146】【F:src/maou/app/utility/dataloader_benchmark.py†L16-L320】
4. **Result packaging** – The interface returns JSON with `summary`,
   `recommendations`, `insights`, `optimal_config`, and raw `all_results`. The CLI
   prints the human-readable sections in order (Summary → Recommendations →
   Insights).【F:src/maou/interface/utility_interface.py†L120-L146】【F:src/maou/app/utility/dataloader_benchmark.py†L323-L392】【F:src/maou/infra/console/utility.py†L406-L413】

## Validation and guardrails

- Input providers are mutually exclusive. Attempting to enable more than one cloud
  source triggers a `ValueError` before any network calls occur.【F:src/maou/infra/console/utility.py†L230-L247】
- `--sample-ratio` must live inside `[0.01, 1.0]` for remote datasources; the
  interface raises descriptive errors otherwise.【F:src/maou/interface/utility_interface.py†L81-L95】
- Batch size and num-batches must be positive integers, preventing zero-division
  or empty loops.【F:src/maou/infra/console/utility.py†L164-L183】【F:src/maou/interface/utility_interface.py†L61-L79】
- Pin-memory defaults follow the selected device automatically, so forgetting the
  flag will not silently hurt GPU throughput.【F:src/maou/interface/utility_interface.py†L69-L72】

## Outputs and usage

- The console displays the Summary table, paste-ready Recommendations (e.g.,
  `--dataloader-workers 8 --pin-memory`), and qualitative Insights so you can act
  on the results immediately.【F:src/maou/infra/console/utility.py†L406-L413】【F:src/maou/app/utility/dataloader_benchmark.py†L323-L392】
- The JSON payload also includes `optimal_config`, `all_results`, `device`,
  `batch_size`, and `num_batches`, making it straightforward to archive or script
  benchmark comparisons.【F:src/maou/interface/utility_interface.py†L120-L146】
- Use the recommendations to tune `maou learn-model` flags (workers, pin-memory,
  prefetch factor) before expensive training runs.

### Example invocation

```bash
poetry run maou utility benchmark-dataloader \
  --input-dir datasets/hcpe \
  --input-format hcpe \
  --batch-size 512 \
  --num-batches 200 \
  --gpu cuda:0
```

## Implementation references

- CLI definition and datasource wiring – `src/maou/infra/console/utility.py`.【F:src/maou/infra/console/utility.py†L217-L413】
- Interface adapter and config assembly – `src/maou/interface/utility_interface.py`.【F:src/maou/interface/utility_interface.py†L44-L146】
- Benchmark runner and reporting – `src/maou/app/utility/dataloader_benchmark.py`.【F:src/maou/app/utility/dataloader_benchmark.py†L16-L392】
