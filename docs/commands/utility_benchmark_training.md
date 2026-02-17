# `maou utility benchmark-training`

## Overview

- Runs a timed, single-epoch dry run of the learning stack (datasource →
  DataLoader → network → optimizer) without touching production checkpoints.
  The CLI and flag definitions live in `src/maou/infra/console/utility.py` and
  fan into `utility_interface.benchmark_training`.【F:src/maou/infra/console/utility.py†L520-L1020】【F:src/maou/interface/utility_interface.py†L213-L359】
- `TrainingBenchmarkUseCase` sets up the real training components, executes a
  warmup plus capped number of batches, and optionally profiles, validates, and
  tracks resource utilization before reporting actionable summaries and
  recommendations.【F:src/maou/app/utility/training_benchmark.py†L122-L699】

## CLI options

### Input selection & caching

| Flag(s) | Required | Description |
| --- | --- | --- |
| `--input-path PATH` + optional `--input-file-packed` | one of the sources | Streams local `.npy` shards and can unpack bit-packed HCPE tensors. Supplying `--sample-ratio` here logs a warning because every file is already on disk.【F:src/maou/infra/console/utility.py†L520-L821】 |
| `--input-dataset-id` + `--input-table-name` | pair | Pulls from BigQuery with the same batching, cache sizing, clustering, and partition controls as `learn-model`. Missing GCP extras cause a hard error.【F:src/maou/infra/console/utility.py†L824-L868】 |
| `--input-gcs` / `--input-s3` + bucket metadata (`--input-bucket-name`, `--input-prefix`, `--input-data-name`, `--input-local-cache-dir`) | provider-specific | Downloads tensors via `GCSDataSource`/`S3DataSource` splitters. Supports worker counts, bundling (`--input-enable-bundling`, `--input-bundle-size-gb`), and optional sampling ratios; requires the respective optional extras.【F:src/maou/infra/console/utility.py†L869-L951】 |
| `--input-format {hcpe,preprocess}` | required | Determines transforms and informs the interface `datasource_type`. Any other value raises `ValueError`.【F:src/maou/infra/console/utility.py†L520-L550】【F:src/maou/interface/utility_interface.py†L213-L216】 |
| BigQuery cache knobs (`--input-batch-size`, `--input-max-cached-bytes`, `--input-clustering-key`, `--input-partitioning-key-date`, `--input-local-cache`, `--input-local-cache-dir`) | optional | Forwarded untouched to mimic production ingestion behavior while benchmarking.【F:src/maou/infra/console/utility.py†L824-L848】 |
| `--input-cache-mode {file,memory,mmap}` | default `file` | Cache strategy for local inputs. `file` uses standard file I/O, `memory` copies into RAM. `mmap` is **deprecated** and internally converted to `file`.【F:src/maou/infra/console/utility.py†L469-L475】 |

Only one datasource may be active at a time; the CLI raises when multiple cloud
inputs are requested.【F:src/maou/infra/console/utility.py†L520-L803】

### Training & benchmarking knobs

| Flag(s) | Purpose | Behavior |
| --- | --- | --- |
| `--gpu DEVICE`, `--compilation`, `--detect-anomaly` | Device placement & runtime toggles | Passed into the shared `TrainingSetup` so the benchmark mirrors real training (including optional `torch.compile` and autograd anomaly detection).【F:src/maou/infra/console/utility.py†L552-L571】【F:src/maou/app/utility/training_benchmark.py†L436-L467】 |
| `--test-ratio FLOAT` | Train/validation split | Validated to sit inside `(0,1)` and used by `datasource.train_test_split(test_ratio)` so loaders mirror production splits.【F:src/maou/infra/console/utility.py†L573-L579】【F:src/maou/interface/utility_interface.py†L218-L224】 |
| `--batch-size INT` | Mini-batch size | Defaults to `256` at the CLI, with the interface enforcing positivity and falling back to `1000` if invoked directly. Applied to both loaders via `TrainingSetup`.【F:src/maou/infra/console/utility.py†L580-L586】【F:src/maou/interface/utility_interface.py†L225-L231】 |
| `--dataloader-workers INT`, `--pin-memory`, `--prefetch-factor INT` | Input pipeline | Validated to be non-negative/positive, then injected into the training setup so benchmarking reflects the target configuration.【F:src/maou/infra/console/utility.py†L587-L607】【F:src/maou/interface/utility_interface.py†L232-L249】 |
| `--cache-transforms/--no-cache-transforms` | Dataset transform caching | Defaults to `True` for HCPE datasources and `False` otherwise unless overridden.【F:src/maou/infra/console/utility.py†L608-L616】【F:src/maou/interface/utility_interface.py†L250-L254】 |
| Loss & optimizer knobs (`--gce-parameter`, `--policy-loss-ratio`, `--value-loss-ratio`, `--learning-ratio`, `--momentum`, `--optimizer`, `--optimizer-beta1`, `--optimizer-beta2`, `--optimizer-eps`) | Fine-tune optimization math | Range-checked before populating the benchmark config so unstable hyperparameters never reach the loop.【F:src/maou/infra/console/utility.py†L617-L683】【F:src/maou/interface/utility_interface.py†L255-L329】 |
| `--warmup-batches INT`, `--max-batches INT` | Scope the run | Warmup batches are excluded from averages and `max_batches` caps how many iterations execute, keeping dry runs short.【F:src/maou/infra/console/utility.py†L684-L697】【F:src/maou/app/utility/training_benchmark.py†L122-L205】 |
| `--enable-profiling` | PyTorch profiler toggle | Enables the profiler inside `TrainingLoop.run_epoch` for deeper diagnostics.【F:src/maou/infra/console/utility.py†L698-L704】【F:src/maou/app/utility/training_benchmark.py†L176-L185】 |
| `--run-validation` | Optional validation pass | Adds an inference-only benchmark with its own timing summary and metrics.【F:src/maou/infra/console/utility.py†L705-L711】【F:src/maou/app/utility/training_benchmark.py†L492-L699】 |
| `--sample-ratio FLOAT` | Remote sampling | Enforced to `[0.01, 1.0]`. When set, the benchmark scales time/batch metrics to estimate a full epoch (printed under “Data Sampling Estimation”).【F:src/maou/infra/console/utility.py†L713-L868】【F:src/maou/app/utility/training_benchmark.py†L501-L545】 |
| `--enable-resource-monitoring` | System telemetry | Attaches a `ResourceMonitoringCallback` so CPU/RAM/GPU stats appear in the summaries and JSON output.【F:src/maou/infra/console/utility.py†L718-L724】【F:src/maou/app/utility/training_benchmark.py†L152-L207】 |

## Execution flow

1. **Datasource split** – The use case invokes `datasource.train_test_split(test_ratio)` so training and validation loaders match
   your intended split, including sampling limits when `--sample-ratio` is used.【F:src/maou/app/utility/training_benchmark.py†L422-L427】【F:src/maou/interface/utility_interface.py†L218-L359】
2. **Training setup** – `TrainingSetup.setup_training_components` prepares
   DataLoaders, networks, optimizer, scheduler, and callbacks using the normalized
   options (cache transforms, pin-memory, anomaly detection, etc.).【F:src/maou/app/utility/training_benchmark.py†L429-L458】
3. **Warmup & batch caps** – `SingleEpochBenchmark.benchmark_epoch` wraps the
   training loop with a `TimingCallback`, discards the first `warmup_batches` from
   averages, and optionally halts after `max_batches`.【F:src/maou/app/utility/training_benchmark.py†L122-L205】
4. **Profiling & monitoring** – When requested, the benchmark enables PyTorch’s
   profiler and the resource-monitoring callback to capture CPU/GPU utilization in
   addition to throughput.【F:src/maou/app/utility/training_benchmark.py†L152-L207】【F:src/maou/app/utility/training_benchmark.py†L176-L185】
5. **Optional validation & sampling estimates** – `--run-validation` triggers an
   inference-only pass, and `--sample-ratio` adds the estimation block that scales
   timings back to a full epoch.【F:src/maou/app/utility/training_benchmark.py†L492-L699】【F:src/maou/app/utility/training_benchmark.py†L501-L545】
6. **Result packaging** – The CLI renders training and validation summaries,
   optional sampling estimation, and heuristic recommendations describing what to
   tweak next.【F:src/maou/app/utility/training_benchmark.py†L532-L699】【F:src/maou/infra/console/utility.py†L991-L1020】

## Validation and guardrails

- Datasource flags are mutually exclusive; selecting more than one provider raises
  before any network calls occur.【F:src/maou/infra/console/utility.py†L520-L803】
- `--sample-ratio` must be between `0.01` and `1.0`. Violations surface as explicit
  errors in both the CLI and interface.【F:src/maou/infra/console/utility.py†L713-L848】【F:src/maou/interface/utility_interface.py†L353-L359】
- `--test-ratio`, `--batch-size`, worker counts, warmup/max batches, and optimizer
  parameters are range-checked before the benchmark runs, preventing undefined
  training states.【F:src/maou/infra/console/utility.py†L573-L697】【F:src/maou/interface/utility_interface.py†L218-L329】
- Missing optional extras (GCP/AWS) cause descriptive errors so operators know to
  install the required dependency groups.【F:src/maou/infra/console/utility.py†L824-L951】

## Outputs and usage

- Console output begins with the training summary (total time, average batch time,
  throughput, per-stage breakdowns, losses, and optional resource stats) followed
  by the validation summary when enabled.【F:src/maou/app/utility/training_benchmark.py†L532-L699】
- If `--sample-ratio` is set, the CLI prints the “Data Sampling Estimation” block
  showing actual vs. estimated batches and full-epoch durations.【F:src/maou/app/utility/training_benchmark.py†L501-L545】【F:src/maou/infra/console/utility.py†L1000-L1017】
- Recommendations list actionable CLI flags (e.g., increase workers, enable
  pin-memory) so you can immediately adjust `maou learn-model`. Additional fields
  in the JSON payload (`training_metrics`, `validation_metrics`, `estimation`,
  `recommendations`) make it easy to archive benchmark results.【F:src/maou/app/utility/training_benchmark.py†L614-L699】【F:src/maou/infra/console/utility.py†L991-L1020】

### Example invocation

```bash
poetry run maou utility benchmark-training \
  --input-s3 --input-bucket-name shogi-data --input-prefix hcpe \
  --input-data-name training --input-local-cache-dir /tmp/cache \
  --input-format hcpe \
  --gpu cuda:0 \
  --batch-size 512 \
  --dataloader-workers 8 \
  --pin-memory \
  --run-validation \
  --enable-resource-monitoring
```

## Implementation references

- CLI definition, datasource wiring, and console rendering –
  `src/maou/infra/console/utility.py`.【F:src/maou/infra/console/utility.py†L520-L1020】
- Interface adapter and config assembly – `src/maou/interface/utility_interface.py`.【F:src/maou/interface/utility_interface.py†L213-L359】
- Benchmark orchestration, timing, validation, and recommendations –
  `src/maou/app/utility/training_benchmark.py`.【F:src/maou/app/utility/training_benchmark.py†L122-L699】
