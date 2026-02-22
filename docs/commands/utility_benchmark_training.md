# `maou utility benchmark-training`

## Overview

- Runs a timed, single-epoch dry run of the learning stack (datasource →
  DataLoader → network → optimizer) without touching production checkpoints.
  Supports all three training stages: Stage 1 (Reachable Squares), Stage 2
  (Legal Moves), and Stage 3 (Policy+Value).
  The CLI and flag definitions live in `src/maou/infra/console/utility.py` and
  fan into `utility_interface.benchmark_training`.【F:src/maou/infra/console/utility.py†L520-L1020】【F:src/maou/interface/utility_interface.py†L213-L359】
- `TrainingBenchmarkUseCase` sets up the real training components, executes a
  warmup plus capped number of batches, and optionally profiles, validates, and
  tracks resource utilization before reporting actionable summaries and
  recommendations.【F:src/maou/app/utility/training_benchmark.py†L122-L699】
- When local `.feather` files are provided via `--stage3-data-path` and there are
  2 or more files, streaming mode is used by default. Streaming mode uses
  `StreamingKifDataset` and `DataLoaderFactory.create_streaming_dataloaders()` for
  memory-efficient, file-level data loading. Use `--no-streaming` to force
  map-style dataset loading.

## CLI options

### Input selection & caching

| Flag(s) | Required | Description |
| --- | --- | --- |
| `--stage INT` (1, 2, 3) | optional (default 3) | Selects the training stage to benchmark: 1 = Reachable Squares, 2 = Legal Moves, 3 = Policy+Value. Stage 1/2 use adapter patterns to reuse the existing `SingleEpochBenchmark` infrastructure with stage-specific models and loss functions. |
| `--stage1-data-path PATH` | required when `--stage=1` | Path to Stage 1 (reachable squares) training data directory. Uses map-style dataset loading only. |
| `--stage2-data-path PATH` | required when `--stage=2` | Path to Stage 2 (legal moves) training data directory. Supports both map-style and streaming modes (streaming enabled by default when 2+ files are provided). |
| `--stage3-data-path PATH` + optional `--input-file-packed` | one of the sources | Streams local `.npy` shards for Stage 3 (policy+value) benchmarking and can unpack bit-packed HCPE tensors. Supplying `--sample-ratio` here logs a warning because every file is already on disk.【F:src/maou/infra/console/utility.py†L520-L821】 |
| `--stage12-lr-scheduler CHOICE` | optional | Learning rate scheduler for Stage 1/2 benchmarks. Choices: `warmup_cosine_decay`, `cosine_annealing`, `step`. Overrides `--lr-scheduler` for Stage 1/2. |
| `--stage12-compilation/--no-stage12-compilation` | optional (default off) | Enable/disable `torch.compile` for Stage 1/2 benchmarks independently from `--compilation` (which applies to Stage 3). |
| `--input-dataset-id` + `--input-table-name` | pair | Pulls from BigQuery with the same batching, cache sizing, clustering, and partition controls as `learn-model`. Missing GCP extras cause a hard error.【F:src/maou/infra/console/utility.py†L824-L868】 |
| `--input-gcs` / `--input-s3` + bucket metadata (`--input-bucket-name`, `--input-prefix`, `--input-data-name`, `--input-local-cache-dir`) | provider-specific | Downloads tensors via `GCSDataSource`/`S3DataSource` splitters. Supports worker counts, bundling (`--input-enable-bundling`, `--input-bundle-size-gb`), and optional sampling ratios; requires the respective optional extras.【F:src/maou/infra/console/utility.py†L869-L951】 |
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
| Stage 1/2 loss & head knobs (`--stage1-pos-weight`, `--stage2-pos-weight`, `--stage2-gamma-pos`, `--stage2-gamma-neg`, `--stage2-clip`, `--stage2-hidden-dim`, `--stage2-head-dropout`, `--stage2-test-ratio`) | Multi-stage training params | Accepted for CLI compatibility with `learn-model`. Includes ASL (Asymmetric Loss) parameters, Stage 2 head architecture, and per-stage loss weights. |
| `--no-streaming` | Data loading mode | Disables streaming mode and forces map-style dataset loading. Streaming is enabled by default when 2+ local files are provided via `--stage3-data-path`. With fewer than 2 files, map-style is used automatically regardless of this flag. |
| `--warmup-batches INT`, `--max-batches INT` | Scope the run | Warmup batches (default 10) are excluded from averages and `max_batches` caps how many iterations execute, keeping dry runs short.【F:src/maou/infra/console/utility.py†L684-L697】【F:src/maou/app/utility/training_benchmark.py†L122-L205】 |
| `--enable-profiling` | PyTorch profiler toggle | Enables the profiler inside `TrainingLoop.run_epoch` for deeper diagnostics.【F:src/maou/infra/console/utility.py†L698-L704】【F:src/maou/app/utility/training_benchmark.py†L176-L185】 |
| `--run-validation` | Optional validation pass | Adds an inference-only benchmark with its own timing summary and metrics.【F:src/maou/infra/console/utility.py†L705-L711】【F:src/maou/app/utility/training_benchmark.py†L492-L699】 |
| `--sample-ratio FLOAT` | Remote sampling | Enforced to `[0.01, 1.0]`. When set, the benchmark scales time/batch metrics to estimate a full epoch (printed under “Data Sampling Estimation”).【F:src/maou/infra/console/utility.py†L713-L868】【F:src/maou/app/utility/training_benchmark.py†L501-L545】 |
| `--enable-resource-monitoring` | System telemetry | Attaches a `ResourceMonitoringCallback` so CPU/RAM/GPU stats appear in the summaries and JSON output.【F:src/maou/infra/console/utility.py†L718-L724】【F:src/maou/app/utility/training_benchmark.py†L152-L207】 |

## Execution flow

1. **Data setup** – The CLI routes to stage-specific setup based on `--stage`:
   Stage 1 creates `Stage1ModelAdapter` + `ReachableSquaresLoss` (map-style only);
   Stage 2 creates `Stage2ModelAdapter` + `LegalMovesLoss` (map-style or streaming
   via `Stage2StreamingAdapter`); Stage 3 uses the existing full-model path. In
   streaming mode, files are split at the file level; in map-style mode,
   `datasource.train_test_split(test_ratio)` is used for sample-level splitting.
2. **Training setup** – In streaming mode,
   `TrainingBenchmarkUseCase._setup_streaming_components` creates streaming
   DataLoaders via `DataLoaderFactory.create_streaming_dataloaders`. In map-style
   mode, `TrainingSetup.setup_training_components` prepares standard DataLoaders.
   Both paths set up the network, optimizer, scheduler, and callbacks.
3. **Warmup & batch caps** – `SingleEpochBenchmark.benchmark_epoch` wraps the
   training loop with a `TimingCallback`, discards the first `warmup_batches` from
   averages, and optionally halts after `max_batches`. When the dataset is small
   (e.g., Stage 1 with ~1,105 samples), warmup is automatically clamped to
   `max(0, estimated_batches - 2)` to prevent zero-division errors.【F:src/maou/app/utility/training_benchmark.py†L122-L205】
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
- The JSON payload includes a `data_load_method` field (`"streaming"` or
  `"map-style"`) in both `training_metrics` and `validation_metrics`, indicating
  which data loading strategy was used for the benchmark run.

### Example invocations

```bash
# Stage 1 benchmark (reachable squares, map-style only)
uv run maou utility benchmark-training \
  --stage 1 --stage1-data-path /data/stage1 \
  --gpu cuda:0 --batch-size 256 --max-batches 10 --warmup-batches 1

# Stage 2 benchmark (legal moves, streaming)
uv run maou utility benchmark-training \
  --stage 2 --stage2-data-path /data/stage2 \
  --gpu cuda:0 --batch-size 512 --max-batches 50

# Stage 3 benchmark (policy+value, default)
uv run maou utility benchmark-training \
  --stage3-data-path /data/stage3 \
  --gpu cuda:0 --batch-size 512 --max-batches 100

# Stage 3 via cloud (S3)
uv run maou utility benchmark-training \
  --input-s3 --input-bucket-name shogi-data --input-prefix hcpe \
  --input-data-name training --input-local-cache-dir /tmp/cache \
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
