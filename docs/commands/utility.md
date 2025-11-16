# `maou utility benchmark-dataloader`

`benchmark-dataloader` lives in `src/maou/infra/console/utility.py` and wraps the
interface entry point in `src/maou/interface/utility_interface.py`. Use it to
stress-test DataLoader combinations (workers, pin-memory, batch size) **before**
running long training jobs so you can discover the fastest configuration for a
given datasource.

## CLI options

### Input selection & caching

| Flag | Required | Description |
| --- | --- | --- |
| `--input-dir PATH` | one of the inputs | Reads local `.npy` tensors via `FileDataSource.FileDataSourceSpliter` with optional bit-packed decoding. `--sample-ratio` is ignored here because all files are already on disk.【F:src/maou/infra/console/utility.py†L230-L259】 |
| `--input-dataset-id` + `--input-table-name` | pair | Streams directly from BigQuery using the same batching, caching, and optional clustering/partition metadata as `learn-model`. When `--sample-ratio` is present the CLI enables BigQuery `TABLESAMPLE` logging before constructing the datasource.【F:src/maou/infra/console/utility.py†L260-L305】 |
| `--input-gcs` / `--input-s3` + bucket options | pair | Download shards from Google Cloud Storage or Amazon S3 through their `DataSourceSpliter` implementations. You must also supply `--input-local-cache-dir` for staging, with optional bundling (`--input-enable-bundling`, `--input-bundle-size-gb`) and worker count (`--input-max-workers`). `--sample-ratio` throttles remote sampling on these providers.【F:src/maou/infra/console/utility.py†L306-L374】 |
| `--input-file-packed` | optional | Unpacks bit-packed HCPE blobs when reading from disk.【F:src/maou/infra/console/utility.py†L29-L36】 |
| `--input-format {hcpe,preprocess}` | default `hcpe` | Drives both CLI validation and the interface `datasource_type`. Any other string raises a `ValueError`.【F:src/maou/infra/console/utility.py†L217-L229】 |
| BigQuery caching knobs (`--input-batch-size`, `--input-max-cached-bytes`, `--input-local-cache`, `--input-local-cache-dir`, `--input-clustering-key`, `--input-partitioning-key-date`) | optional | Forwarded untouched to the datasource constructor so you can mimic production ingestion pressure while benchmarking.【F:src/maou/infra/console/utility.py†L57-L205】 |

Only **one** remote provider can be enabled at a time; the CLI counts enabled
cloud flags and aborts if multiple providers are selected.【F:src/maou/infra/console/utility.py†L230-L247】

### Benchmark knobs

| Flag | Default | Description |
| --- | --- | --- |
| `--gpu DEVICE` | | Chooses the PyTorch device string (`cuda:0`, `cpu`, etc.) that the interface passes into the benchmark config.【F:src/maou/infra/console/utility.py†L397-L403】【F:src/maou/interface/utility_interface.py†L50-L60】 |
| `--batch-size INT` | `256` | Overrides the batch size used during benchmarking (validated to be positive).【F:src/maou/infra/console/utility.py†L164-L169】【F:src/maou/interface/utility_interface.py†L61-L68】 |
| `--pin-memory` | auto | The CLI forwards the flag and the interface defaults to `True` when benchmarking on CUDA and `False` on CPU.【F:src/maou/infra/console/utility.py†L170-L176】【F:src/maou/interface/utility_interface.py†L69-L72】 |
| `--num-batches INT` | `100` | Controls how many minibatches each worker/prefetch combination will process; validated to be positive.【F:src/maou/infra/console/utility.py†L177-L183】【F:src/maou/interface/utility_interface.py†L73-L79】 |
| `--sample-ratio FLOAT` | full dataset | Lets BigQuery/GCS/S3 datasources pull a subset (between `0.01` and `1.0`). The interface enforces the range, logs the percentage, and the CLI warns when the flag is redundant for local folders.【F:src/maou/infra/console/utility.py†L251-L325】【F:src/maou/interface/utility_interface.py†L81-L95】 |

## How the CLI hands off to the interface

1. The CLI chooses a datasource based on the mutually-exclusive input flags and
   emits helpful error messages when required extras (GCP/AWS) are missing.
2. It passes the `datasource` object, the normalized `input_format`, GPU/batch
   hints, and sampling preferences into `utility_interface.benchmark_dataloader`.
3. The interface re-validates the datasource type, materializes a PyTorch device
   (and bumps matmul precision when CUDA is used), then calls
   `datasource.train_test_split(test_ratio=0.1)` to isolate the training subset
   that feeds the benchmark.【F:src/maou/interface/utility_interface.py†L44-L100】
4. A `BenchmarkConfig` is assembled with the normalized defaults (batch size 256,
   100 batches, pin-memory inferred from device) and handed to
   `DataLoaderBenchmark` for execution.【F:src/maou/interface/utility_interface.py†L101-L119】

### What the benchmark measures

`DataLoaderBenchmark` builds a `KifDataset` using the requested transform
(`Transform()` for HCPE, `None` for preprocessed tensors) and times every
combination of worker counts (up to your CPU core count, capped at 16) and
prefetch factors (`[1,2,4,8]` for multi-worker runs). Each configuration reports:

- `num_workers`
- `prefetch_factor` (skipped for synchronous loads)
- `time_taken` (wall-clock seconds for the sampled batches)
- `avg_batch_time`
- `batches_processed`
- the `pin_memory` flag that was applied globally

These metrics are stored in `BenchmarkResult` objects and later serialized into
`all_results` (one dictionary per configuration) plus a dedicated
`optimal_config` entry containing the fastest average batch time.【F:src/maou/interface/utility_interface.py†L101-L146】【F:src/maou/app/utility/dataloader_benchmark.py†L16-L205】【F:src/maou/app/utility/dataloader_benchmark.py†L270-L320】

## Console & JSON output

The interface returns a JSON blob with three key sections under
`benchmark_results`:

1. **Summary** – A sorted table showing worker/prefetch combinations, their total
   time, and average batch latency. The optimal row is annotated with `← Optimal`.
2. **Recommendations** – Concrete CLI arguments (`--dataloader-workers`,
   `--pin-memory`, optional `--prefetch-factor`) that you can paste into
   `maou learn-model` to reproduce the best-performing configuration.
3. **Insights** – Narrative tips chosen from heuristics (e.g., `num_workers==0`
   implies CPU limits, `num_workers>=8` implies you should ensure enough cores)
   and the overall speedup between the slowest and fastest trials.【F:src/maou/app/utility/dataloader_benchmark.py†L323-L392】

`src/maou/infra/console/utility.py` parses the JSON and prints the Summary,
Recommendations, and Insights sections in order, separated by blank lines so the
most actionable information stands out in the terminal.【F:src/maou/infra/console/utility.py†L406-L413】

Outside of `benchmark_results`, the JSON payload also includes:

- `optimal_config`: `{num_workers, prefetch_factor, pin_memory, avg_batch_time,
  total_time}` for downstream automation.
- `all_results`: The raw measurements for every tested configuration.
- `device`, `batch_size`, and `num_batches`: echoing the environment so you can
  tie reports back to specific hardware choices.【F:src/maou/interface/utility_interface.py†L120-L146】

## When to run this command

- **Before long training jobs** – Quickly surface how many DataLoader workers
  your machine can sustain without starving the GPU. Apply the recommended worker
  count and pin-memory guidance to `maou learn-model` for better throughput.
- **When changing datasources** – Switching from local HCPE files to GCS/S3 or
  altering sample ratios changes the IO profile; rerun the benchmark so the
  summary reflects the new bottlenecks.
- **While tuning batch size/pin-memory** – Pair the CLI flags here with the
  recommendations to verify if a larger batch or pinned host buffers meaningfully
  shrink per-batch latency.

## Reading the console sections

- **Summary**: Scan the ascending list of configurations to understand how
  quickly performance improves as workers/prefetch factors increase. Use the
  `avg_batch=` numbers to estimate how much GPU idle time you can eliminate.
- **Recommendations**: Copy/paste the listed flags directly into your training
  command. This section is purpose-built for `maou learn-model`, so you can trust
  the syntax without additional translation.
- **Insights**: Treat these as qualitative tips. For example, a "Single-threaded
  loading performed best" insight suggests either the dataset is tiny or the CPU
  is saturated, while "High worker count optimal" nudges you to provision more
  cores or memory when scaling up. The reported `X.x` speedup quantifies the
  impact of following the recommendations.

By following the workflow above—selecting the appropriate datasource, optionally
sampling remote data, and applying the Summary/Recommendations/Insights—you can
confidently size DataLoader worker pools and pin-memory usage before launching
expensive learning runs.
