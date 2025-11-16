# `maou utility benchmark-training`

Use `maou utility benchmark-training` to run a timed, single-epoch dry run of the
learning stack (DataLoader → network → optimizer) without touching your
production checkpoints. The CLI lives in
`src/maou/infra/console/utility.py` and wires into
`utility_interface.benchmark_training`, which drives the full training stack and
produces console-friendly summaries plus machine-readable metrics.

If you only need a quick inference sanity check instead of a full benchmark,
run [`maou evaluate`](./evaluate.md) to score an arbitrary SFEN board position
with ONNX or TensorRT backends and inspect the policy/evaluation readout.

## Input selection & caching flags

The training benchmark shares its datasource options with
`benchmark-dataloader`, so you can reproduce the exact I/O profile of your
future training job. Only one provider can be active at a time, and the CLI
raises when multiple cloud inputs are requested simultaneously.【F:src/maou/infra/console/utility.py†L520-L803】

| Flag(s) | Required | Description |
| --- | --- | --- |
| `--input-dir PATH` + optional `--input-file-packed` | one of the sources | Streams local `.npy` shards and can unpack bit-packed HCPE tensors before benchmarking. Supplying `--sample-ratio` here triggers a warning because every file is already on disk.【F:src/maou/infra/console/utility.py†L22-L117】【F:src/maou/infra/console/utility.py†L804-L821】 |
| `--input-dataset-id` + `--input-table-name` | pair | Pulls directly from BigQuery with the same batching, cache sizing, clustering, and partition controls as `learn-model`. When `--sample-ratio` is set the CLI logs that `TABLESAMPLE` is enabled before constructing `BigQueryDataSourceSpliter`. Missing GCP extras cause a hard error.【F:src/maou/infra/console/utility.py†L37-L117】【F:src/maou/infra/console/utility.py†L824-L868】 |
| `--input-gcs` or `--input-s3` + bucket metadata (`--input-bucket-name`, `--input-prefix`, `--input-data-name`, `--input-local-cache-dir`) | provider-specific | Downloads tensors from Google Cloud Storage or Amazon S3 via their `DataSourceSpliter` helpers. You can throttle concurrent downloads (`--input-max-workers`), enable bundling to reduce small-file churn (`--input-enable-bundling`, `--input-bundle-size-gb`), and honor remote sampling ratios. The CLI validates that the appropriate extras (`poetry install -E gcp` / `-E aws`) are installed before instantiation.【F:src/maou/infra/console/utility.py†L104-L156】【F:src/maou/infra/console/utility.py†L869-L951】 |
| `--input-format {hcpe,preprocess}` | required | Ensures the CLI knows how to instantiate transforms and rejects anything but `hcpe` or `preprocess`. The value is forwarded directly into the interface `datasource_type` check.【F:src/maou/infra/console/utility.py†L50-L55】【F:src/maou/interface/utility_interface.py†L213-L216】 |
| BigQuery cache knobs (`--input-batch-size`, `--input-max-cached-bytes`, `--input-clustering-key`, `--input-partitioning-key-date`, `--input-local-cache`, `--input-local-cache-dir`) | optional | Passed unchanged to the BigQuery datasource so you can mimic production ingestion patterns while benchmarking.【F:src/maou/infra/console/utility.py†L57-L117】【F:src/maou/infra/console/utility.py†L824-L848】 |

## Training, dataloader, and optimization flags

After the datasource is resolved the CLI forwards every training knob to the
interface, where defaults and validation guards live. The table below groups the
flags by purpose and calls out the downstream behavior.

| Flag(s) | Purpose | Interface behavior |
| --- | --- | --- |
| `--gpu DEVICE` | Device placement | Passed into the shared `TrainingSetup` so the benchmark runs on CPU or CUDA exactly like a real training job.【F:src/maou/infra/console/utility.py†L552-L557】【F:src/maou/app/utility/training_benchmark.py†L436-L456】 |
| `--compilation` | Enable `torch.compile` | When true, `TrainingBenchmarkUseCase` wraps the model with `compile_module` before running the epoch, letting you compare ahead-of-time compilation overhead vs. throughput.【F:src/maou/infra/console/utility.py†L558-L564】【F:src/maou/app/utility/training_benchmark.py†L460-L467】 |
| `--detect-anomaly` | Autograd debugging | Propagates into `TrainingSetup.setup_training_components`, enabling anomaly detection for both benchmarking and any profiling run.【F:src/maou/infra/console/utility.py†L565-L571】【F:src/maou/app/utility/training_benchmark.py†L436-L458】 |
| `--test-ratio FLOAT` | Train/validation split | Validated to live inside `(0,1)` and consumed by `datasource.train_test_split(test_ratio)` so the training loader and optional validation loader match your real experiment split.【F:src/maou/infra/console/utility.py†L573-L579】【F:src/maou/interface/utility_interface.py†L218-L224】【F:src/maou/app/utility/training_benchmark.py†L422-L427】 |
| `--batch-size INT` | Mini-batch size | Defaults to 256 at the CLI, but the interface enforces positivity and would fall back to 1000 if invoked without a CLI value. The normalized number feeds both loaders via `TrainingSetup`.【F:src/maou/infra/console/utility.py†L580-L586】【F:src/maou/interface/utility_interface.py†L225-L231】【F:src/maou/app/utility/training_benchmark.py†L436-L456】 |
| `--dataloader-workers INT`, `--pin-memory`, `--prefetch-factor INT` | Input pipeline | Validated to be non-negative/positive, then passed into `TrainingSetup`. `pin_memory` defaults to `False` unless you explicitly flip it on, while `prefetch_factor` defaults to `2`.【F:src/maou/infra/console/utility.py†L587-L607】【F:src/maou/interface/utility_interface.py†L232-L249】【F:src/maou/app/utility/training_benchmark.py†L436-L456】 |
| `--cache-transforms/--no-cache-transforms` | Dataset transform caching | If unset, HCPE datasources cache transforms automatically. Otherwise the interface honours your override and forwards the flag to `TrainingSetup`.【F:src/maou/infra/console/utility.py†L608-L616】【F:src/maou/interface/utility_interface.py†L250-L254】 |
| Loss & optimizer knobs (`--gce-parameter`, `--policy-loss-ratio`, `--value-loss-ratio`, `--learning-ratio`, `--momentum`, `--optimizer`, `--optimizer-beta1`, `--optimizer-beta2`, `--optimizer-eps`) | Fine-tune optimization math | Each flag is range-checked (e.g., `0 < momentum ≤ 1`, `optimizer` limited to AdamW or SGD, `beta2 > beta1`) before populating `TrainingBenchmarkConfig`, ensuring the benchmark loop matches a real training invocation without silently accepting unstable hyperparameters.【F:src/maou/infra/console/utility.py†L617-L683】【F:src/maou/interface/utility_interface.py†L255-L329】 |
| `--warmup-batches INT`, `--max-batches INT` | Scope the benchmark | Warmup batches are excluded from the averages via the `TimingCallback`, and `max_batches` caps both the training and validation loops if you only want a slice of the epoch.【F:src/maou/infra/console/utility.py†L684-L697】【F:src/maou/app/utility/training_benchmark.py†L122-L205】【F:src/maou/app/utility/training_benchmark.py†L176-L310】 |
| `--enable-profiling` | PyTorch profiler toggle | Flips the `enable_profiling` argument on `TrainingLoop.run_epoch`, so you can capture autograd timelines while still receiving timing summaries.【F:src/maou/infra/console/utility.py†L698-L704】【F:src/maou/app/utility/training_benchmark.py†L176-L185】 |
| `--run-validation` | Optional validation pass | Adds a post-training inference-only benchmark that reuses the same callbacks (minus warmup and profiling) and injects a `ValidationSummary` plus `validation_metrics` into the JSON output.【F:src/maou/infra/console/utility.py†L705-L711】【F:src/maou/app/utility/training_benchmark.py†L492-L699】 |
| `--sample-ratio FLOAT` | Remote sampling | Enforced to `[0.01, 1.0]`. When set, the CLI configures the datasource to fetch only that percentage of remote data and later scales the measured epoch time and batch count to estimate the full epoch (see “Sampling estimates” below).【F:src/maou/infra/console/utility.py†L713-L848】【F:src/maou/interface/utility_interface.py†L353-L359】【F:src/maou/app/utility/training_benchmark.py†L501-L545】 |
| `--enable-resource-monitoring` | System telemetry | Appends a `ResourceMonitoringCallback` during both training and validation benchmarks so CPU, RAM, and GPU utilization end up in the summaries and JSON metrics.【F:src/maou/infra/console/utility.py†L718-L724】【F:src/maou/app/utility/training_benchmark.py†L152-L207】【F:src/maou/app/utility/training_benchmark.py†L532-L605】 |

## What the benchmark does under the hood

1. **Datasource split** – The use case splits the incoming datasource into
   training and validation subsets via `train_test_split(test_ratio)` so both
   loaders mirror your desired evaluation split.【F:src/maou/app/utility/training_benchmark.py†L422-L427】
2. **Training setup** – `TrainingSetup.setup_training_components` builds the
   PyTorch `DataLoader` objects, initializes the network, optimizer, and losses,
   and enables transform caching, pin-memory, and prefetch settings based on the
   CLI flags (including anomaly detection).【F:src/maou/app/utility/training_benchmark.py†L429-L458】
3. **Warmup & batch caps** – `SingleEpochBenchmark.benchmark_epoch` wraps the
   standard training loop with a `TimingCallback`, discards the first
   `warmup_batches` from the averages, and optionally halts early after
   `max_batches` iterations to keep test runs short.【F:src/maou/app/utility/training_benchmark.py†L122-L205】
4. **Profiling hooks** – When `--enable-profiling` is passed, the benchmark hands
   `enable_profiling=True` to `TrainingLoop.run_epoch`, allowing PyTorch’s
   profiler to collect traces without altering the summary formatting.【F:src/maou/app/utility/training_benchmark.py†L176-L185】
5. **Resource monitoring** – If requested, a `ResourceMonitoringCallback`
   attaches alongside the timing callback so the final summary can report CPU,
   memory, and GPU saturation in addition to timing data.【F:src/maou/app/utility/training_benchmark.py†L152-L207】【F:src/maou/app/utility/training_benchmark.py†L532-L605】
6. **Optional validation run** – `--run-validation` executes an additional
   inference-only pass with its own timing callback (no warmup, no profiler)
   before packaging the validation summary and raw metrics into the JSON
   response.【F:src/maou/app/utility/training_benchmark.py†L492-L699】

## Sampling estimates

When you throttle remote inputs via `--sample-ratio`, the benchmark records the
actual run time and batch count, divides by the ratio, and reports the scaled
full-epoch estimates (seconds, minutes, and total batches) alongside the raw
measurements. These values flow into the `estimation` section of the JSON payload
and the CLI prints them under a `=== Data Sampling Estimation ===` header.【F:src/maou/app/utility/training_benchmark.py†L501-L545】【F:src/maou/infra/console/utility.py†L1000-L1017】

## Reading the output

The interface returns JSON that the CLI renders in three distinct blocks:

1. **Timing summary** – `TrainingBenchmarkUseCase` generates a formatted summary
   (and an optional validation summary) that lists total time, average batch
   time, throughput, per-stage breakdowns, loss values, and resource utilization
   when enabled.【F:src/maou/app/utility/training_benchmark.py†L532-L609】【F:src/maou/app/utility/training_benchmark.py†L666-L699】
2. **Recommendations** – Heuristics explain what to tweak next (e.g., raise batch
   size, add workers, enable pin-memory). The CLI prints this block after the
   summaries so you can immediately reuse the suggestions in `maou learn-model`.【F:src/maou/app/utility/training_benchmark.py†L614-L684】【F:src/maou/infra/console/utility.py†L991-L1020】
3. **Sampling estimates (optional)** – If `--sample-ratio` was provided, the CLI
   appends the estimated full-epoch timing block that reports the percentage
   sampled, actual batches processed, projected total batches, and projected full
   epoch duration.【F:src/maou/app/utility/training_benchmark.py†L501-L545】【F:src/maou/infra/console/utility.py†L1000-L1017】

### Sample console excerpt

```
=== Training Performance Benchmark Results ===

Training Performance Summary:
  Total Time: 38.21s
  Average Batch Time: 0.1520s
  ...

Validation Performance Summary:
  Total Time: 6.04s
  Average Batch Time: 0.0604s
  ...

=== Data Sampling Estimation ===
Sample Ratio Used: 10.0%
Actual Batches Processed: 1,280
Estimated Total Batches: 12,800
Estimated Full Epoch Time: 6.4 minutes

Performance Recommendations:
- Data loading is a bottleneck - consider increasing DataLoader workers or enabling prefetch
- GPU transfer is slow - ensure pin_memory=True and consider larger batch sizes
```

The headers and block order shown above match the CLI’s output logic: timing
summaries first (training plus optional validation), estimation when sampling is
active, then the Recommendations narrative.【F:src/maou/infra/console/utility.py†L991-L1020】 The ellipses represent the rest of the
per-stage timing breakdown emitted by the formatter.【F:src/maou/app/utility/training_benchmark.py†L532-L605】
