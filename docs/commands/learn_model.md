# `maou learn-model`

## Overview

- Ingests HCPE or preprocessing datasets from local folders, BigQuery, GCS, or
  S3, then normalizes every CLI flag before wiring them into the training
  pipeline defined in `src/maou/infra/console/learn_model.py`. The CLI exposes
  mutually exclusive datasource selectors, cache controls, and cloud upload
  toggles so operators can mirror production setups during experiments.【F:src/maou/infra/console/learn_model.py†L1-L639】
- The interface (`maou.interface.learn`) converts the parsed flags into a
  `Learning.LearningOption`, instantiates the requested datasource, and then
  hands control to the `Learning` app layer, which prepares DataLoaders, models,
  optimizers, checkpoints, and optional cloud uploads.【F:src/maou/interface/learn.py†L101-L266】【F:src/maou/app/learning/dl.py†L94-L209】

## CLI options

### Input sources and caching

| Flag | Required | Description |
| --- | --- | --- |
| `--input-path PATH` | one of the sources | Recursively collects `.npy` shards through `FileSystem.collect_files`. Works with either `hcpe` or `preprocess` tensors, and can unpack bit-packed data via `--input-file-packed`.【F:src/maou/infra/console/learn_model.py†L1-L120】 |
| `--input-format {hcpe,preprocess}` | default `hcpe` | Drives both CLI validation and the interface `datasource_type`. Any other string raises a `ValueError`.【F:src/maou/infra/console/learn_model.py†L96-L156】【F:src/maou/interface/learn.py†L101-L120】 |
| `--input-dataset-id` + `--input-table-name` | pair | Streams from BigQuery when the optional `gcp` extra is installed. Requires `--input-format` to select the array schema and supports batching/cache knobs (below).【F:src/maou/infra/console/learn_model.py†L245-L330】 |
| `--input-gcs` / `--input-s3` + bucket metadata | pair | Downloads shards via `GCSDataSource` or `S3DataSource` splitters. Both providers need `--input-local-cache-dir` and honor optional bundling (`--input-enable-bundling`, `--input-bundle-size-gb`) and worker counts (`--input-max-workers`).【F:src/maou/infra/console/learn_model.py†L330-L399】 |
| `--input-max-workers`, `--input-batch-size`, `--input-max-cached-bytes`, `--input-local-cache`, `--input-local-cache-dir`, `--input-clustering-key`, `--input-partitioning-key-date` | optional | Fine-tune remote datasource caching and streaming. Forwarded directly to the datasource constructors and into the interface options.【F:src/maou/infra/console/learn_model.py†L122-L399】【F:src/maou/interface/learn.py†L198-L210】 |
| `--input-file-packed` | optional | Tells file-based datasources to unpack bit-packed numpy blobs. Ignored for cloud providers.【F:src/maou/infra/console/learn_model.py†L96-L130】 |
| `--input-cache-mode {file,memory,mmap}` | default `file` | Cache strategy for local inputs. `file` uses standard file I/O, `memory` copies into RAM. `mmap` is **deprecated** and internally converted to `file`.【F:src/maou/infra/console/learn_model.py†L106-L113】【F:src/maou/interface/learn.py†L198-L210】 |
| `--input-enable-bundling` + `--input-bundle-size-gb` | optional | Bundle remote shards (default 1 GB) before caching to reduce metadata churn. Applies to GCS/S3 datasources.【F:src/maou/infra/console/learn_model.py†L330-L399】 |

**Input exclusivity.** Only one provider (local, BigQuery, GCS, or S3) may be
active. The CLI counts enabled sources and raises when more than one set of
flags is present.【F:src/maou/infra/console/learn_model.py†L568-L639】

### Training hardware and performance knobs

| Flag | Default | Description |
| --- | --- | --- |
| `--gpu DEVICE` | | Chooses the PyTorch device string (`cuda:0`, `cpu`, etc.).【F:src/maou/infra/console/learn_model.py†L400-L430】 |
| `--compilation` | `false` | Enables `torch.compile` prior to the training loop for ahead-of-time graph optimization.【F:src/maou/infra/console/learn_model.py†L400-L470】 |
| `--detect-anomaly` | `false` | Wraps the loop with `torch.autograd.set_detect_anomaly` for debugging gradients.【F:src/maou/infra/console/learn_model.py†L400-L470】 |
| `--epoch INT` | interface default `10` | Number of passes over the training loader; must be positive.【F:src/maou/interface/learn.py†L132-L147】 |
| `--batch-size INT` | interface default `1000` | Minibatch size shared by train/test loaders; must be positive.【F:src/maou/interface/learn.py†L142-L156】 |
| `--dataloader-workers INT` | interface default `0` | Worker processes for PyTorch DataLoaders. Negative values raise `ValueError`.【F:src/maou/interface/learn.py†L158-L177】 |
| `--pin-memory` | `false` | Toggles pinned host memory for faster GPU transfers.【F:src/maou/interface/learn.py†L158-L177】 |
| `--prefetch-factor INT` | interface default `2` | Number of batches prefetched per worker; must be positive.【F:src/maou/interface/learn.py†L158-L177】 |
| `--cache-transforms/--no-cache-transforms` | format-dependent | HCPE datasources cache transforms by default; preprocessed tensors do not. Flags override the heuristic.【F:src/maou/interface/learn.py†L226-L239】 |
| `--test-ratio FLOAT` | interface default `0.2` | Portion of the dataset reserved for validation. Must satisfy `0 < ratio < 1`.【F:src/maou/interface/learn.py†L132-L140】 |
| `--tensorboard-histogram-frequency INT` + `--tensorboard-histogram-module PATTERN` | default `0` | Controls how often histogram dumps occur and which parameter names qualify.【F:src/maou/interface/learn.py†L233-L244】 |
| `--no-streaming` | `false` | Disable streaming mode for file input; uses map-style dataset instead. Streaming is the default for multi-file inputs.【F:src/maou/infra/console/learn_model.py†L520-L524】 |

### Model architecture (ViT)

| Flag | Default | Description |
| --- | --- | --- |
| `--vit-embed-dim INT` | `None` (model default: 512) | ViT: embedding dimension. Only applies when `--model-architecture vit`.【F:src/maou/infra/console/learn_model.py†L216-L220】 |
| `--vit-num-layers INT` | `None` (model default: 6) | ViT: number of encoder layers.【F:src/maou/infra/console/learn_model.py†L221-L225】 |
| `--vit-num-heads INT` | `None` (model default: 8) | ViT: number of attention heads.【F:src/maou/infra/console/learn_model.py†L226-L230】 |
| `--vit-mlp-ratio FLOAT` | `None` (model default: 4.0) | ViT: MLP hidden dimension ratio.【F:src/maou/infra/console/learn_model.py†L231-L236】 |
| `--vit-dropout FLOAT` | `None` (model default: 0.1) | ViT: dropout rate.【F:src/maou/infra/console/learn_model.py†L237-L244】 |

### Multi-stage training

| Flag | Default | Description |
| --- | --- | --- |
| `--stage {1,2,3,all}` | `3` | Training stage: 1=Reachable Squares, 2=Legal Moves, 3=Policy+Value, all=Sequential.【F:src/maou/infra/console/learn_model.py†L453-L460】 |
| `--stage1-data-path PATH` | optional | File or directory path for Stage 1 (reachable squares) training data.【F:src/maou/infra/console/learn_model.py†L462-L466】 |
| `--stage2-data-path PATH` | optional | File or directory path for Stage 2 (legal moves) training data.【F:src/maou/infra/console/learn_model.py†L467-L471】 |
| `--stage3-data-path PATH` | optional | File or directory path for Stage 3 (policy+value) training data.【F:src/maou/infra/console/learn_model.py†L472-L477】 |
| `--stage1-threshold FLOAT` | `0.99` | Accuracy threshold for Stage 1 (99%). Training advances to the next stage once this threshold is reached.【F:src/maou/infra/console/learn_model.py†L480-L485】 |
| `--stage2-threshold FLOAT` | `0.95` | Accuracy threshold for Stage 2 (95%).【F:src/maou/infra/console/learn_model.py†L486-L492】 |
| `--stage1-max-epochs INT` | `10` | Maximum epochs for Stage 1.【F:src/maou/infra/console/learn_model.py†L494-L499】 |
| `--stage2-max-epochs INT` | `10` | Maximum epochs for Stage 2.【F:src/maou/infra/console/learn_model.py†L500-L506】 |
| `--stage1-batch-size INT` | `None` (inherits `--batch-size`) | Batch size for Stage 1. When unset, inherits the global `--batch-size`. For small datasets (~1,000 positions), `32` is recommended.【F:src/maou/infra/console/learn_model.py†L507-L513】 |
| `--stage2-batch-size INT` | `None` (inherits `--batch-size`) | Batch size for Stage 2. When unset, inherits the global `--batch-size`.【F:src/maou/infra/console/learn_model.py†L514-L520】 |
| `--freeze-backbone` | `false` | Freeze backbone parameters (embedding, backbone, pool, hand projection).【F:src/maou/infra/console/learn_model.py†L437-L441】 |
| `--trainable-layers INT` | `None` | Number of trailing backbone layer groups to keep trainable. `0` = freeze all backbone layers. Unset = all layers trainable.【F:src/maou/infra/console/learn_model.py†L443-L451】 |
| `--resume-reachable-head-from PATH` | optional | Reachable squares head parameter file to resume training (Stage 1).【F:src/maou/infra/console/learn_model.py†L508-L512】 |
| `--resume-legal-moves-head-from PATH` | optional | Legal moves head parameter file to resume training (Stage 2).【F:src/maou/infra/console/learn_model.py†L514-L518】 |

### Loss, optimizer, and scheduler controls

| Flag | Default | Description |
| --- | --- | --- |
| `--gce-parameter FLOAT` | CLI default `0.1`, interface clamps to `(0,1]` with default `0.7` | Sets the generalized cross-entropy loss parameter.【F:src/maou/interface/learn.py†L179-L204】 |
| `--policy-loss-ratio FLOAT` / `--value-loss-ratio FLOAT` | default `1.0` | Relative head weights; must be positive.【F:src/maou/interface/learn.py†L179-L204】 |
| `--learning-ratio FLOAT` | default `0.01` | Base learning rate supplied to the optimizer. Must be positive.【F:src/maou/interface/learn.py†L179-L204】 |
| `--optimizer {adamw,sgd}` | default `adamw` | Normalized to lowercase and validated against supported names.【F:src/maou/interface/learn.py†L206-L221】 |
| `--momentum FLOAT` | default `0.9` | Applies to SGD setups and must live inside `[0,1]`.【F:src/maou/interface/learn.py†L206-L221】 |
| `--optimizer-beta1 FLOAT` / `--optimizer-beta2 FLOAT` / `--optimizer-eps FLOAT` | defaults `0.9`, `0.999`, `1e-8` | AdamW parameters validated to satisfy `0 < beta1 < beta2 < 1` and `eps > 0`.【F:src/maou/interface/learn.py†L223-L247】 |
| `--lr-scheduler {Warmup+CosineDecay,CosineAnnealingLR}` | default `Warmup+CosineDecay` | Friendly labels that map to canonical scheduler keys through `normalize_lr_scheduler_name`. Unsupported names raise `ValueError`.【F:src/maou/interface/learn.py†L12-L55】 |
| `--model-architecture` | default `resnet` | Must be part of `BACKBONE_ARCHITECTURES`. Case-insensitive at the CLI, enforced by the interface.【F:src/maou/interface/learn.py†L122-L129】 |

### Logging, checkpoints, and uploads

| Flag | Default | Description |
| --- | --- | --- |
| `--resume-from PATH` | optional | Must point to an existing checkpoint file before training resumes.【F:src/maou/interface/learn.py†L249-L266】 |
| `--resume-backbone-from PATH` | optional | Backbone parameter file to resume training from.【F:src/maou/infra/console/learn_model.py†L418-L422】 |
| `--resume-policy-head-from PATH` | optional | Policy head parameter file to resume training from.【F:src/maou/infra/console/learn_model.py†L423-L428】 |
| `--resume-value-head-from PATH` | optional | Value head parameter file to resume training from.【F:src/maou/infra/console/learn_model.py†L429-L434】 |
| `--start-epoch INT` | default `0` | Lets you offset the epoch counter while still completing `--epoch` total passes. Must be non-negative.【F:src/maou/interface/learn.py†L226-L244】 |
| `--log-dir PATH` / `--model-dir PATH` | defaults `./logs`, `./models` | Created automatically when missing so TensorBoard and checkpoints always have a target directory.【F:src/maou/interface/learn.py†L249-L266】 |
| `--output-gcs` + `--gcs-bucket-name` + `--gcs-base-path` | optional | Uploads checkpoints and TensorBoard runs to Google Cloud Storage when the `gcp` extra is installed.【F:src/maou/infra/console/learn_model.py†L416-L520】 |
| `--output-s3` + `--s3-bucket-name` + `--s3-base-path` | optional | Same behavior for AWS S3, gated behind the `aws` extra.【F:src/maou/infra/console/learn_model.py†L416-L520】 |

Only one cloud output provider can be active; the CLI warns when extras are
missing and continues with local-only writes.【F:src/maou/infra/console/learn_model.py†L416-L520】

## Execution flow

1. **Datasource selection** – The CLI enforces provider exclusivity, instantiates
   the requested datasource (File/BigQuery/GCS/S3), and passes it to the
   interface along with cache/bundling hints.【F:src/maou/infra/console/learn_model.py†L122-L399】【F:src/maou/infra/console/learn_model.py†L568-L639】
2. **Option normalization** – `learn.learn` validates ratios, batch sizes,
   worker counts, optimizer parameters, cache settings, and scheduler names
   before building `Learning.LearningOption`. Defaults such as
   `epoch=10`, `batch_size=1000`, `test_ratio=0.2`, and format-specific
   `cache_transforms` are applied here.【F:src/maou/interface/learn.py†L101-L247】
3. **Training setup** – The app-layer `Learning` object prepares DataLoaders,
   networks (`BACKBONE_ARCHITECTURES`), optimizers, schedulers, and callbacks
   (TensorBoard, checkpointing, optional cloud uploads) via
   `TrainingSetup.setup_training_components`.【F:src/maou/app/learning/dl.py†L94-L209】
4. **Execution and persistence** – `TrainingLoop` drives the epochs, writes
   TensorBoard runs under `log_dir`, saves PyTorch/ONNX checkpoints in
   `model_dir`, and mirrors artifacts to the configured cloud storage when
   enabled.【F:src/maou/app/learning/dl.py†L94-L209】【F:src/maou/app/learning/model_io.py†L1-L86】

## Validation and guardrails

- Selecting multiple input or output providers triggers an early `ValueError`
  that lists the conflicting flags, preventing accidental double uploads or
  hybrid datasources.【F:src/maou/infra/console/learn_model.py†L416-L639】
- Scheduler/optimizer typos surface through `normalize_lr_scheduler_name` and
  the optimizer guard, so unsupported names fail fast.【F:src/maou/interface/learn.py†L12-L221】
- Ratios, worker counts, epochs, and batch sizes must be positive (or within
  `(0,1)` for ratios); the interface raises descriptive errors before GPUs spin
  up.【F:src/maou/interface/learn.py†L132-L210】
- Missing extras for BigQuery/GCS/S3 inputs or outputs produce explicit warning
  messages instructing operators to run `poetry install -E gcp` or `-E aws`.【F:src/maou/infra/console/learn_model.py†L400-L520】
- Checkpoint resume paths are validated to exist, and log/model directories are
  created automatically to avoid runtime `FileNotFoundError` issues.【F:src/maou/interface/learn.py†L249-L266】

## Outputs and usage

- Training produces TensorBoard logs under `log_dir/<arch>_training_log_*`, best
  checkpoints and ONNX exports inside `model_dir`, and optional mirrors in the
  configured cloud bucket via `CloudStorage`.【F:src/maou/app/learning/dl.py†L94-L209】【F:src/maou/app/learning/model_io.py†L1-L86】
- The CLI prints progress and returns once the requested epochs complete; any
  warnings about datasources or uploads surface before the training loop starts.
- Pair this command with `maou utility benchmark-dataloader` or
  `maou utility benchmark-training` to pre-tune worker counts and hyperparameters
  before long runs.

### Example invocation

```bash
poetry run maou learn-model \
  --input-path datasets/hcpe \
  --input-format hcpe \
  --epoch 20 \
  --batch-size 2048 \
  --gpu cuda:0 \
  --output-gcs --gcs-bucket-name my-lab --gcs-base-path experiments/latest
```

## Implementation references

- CLI definition, datasource wiring, and cloud upload hooks –
  `src/maou/infra/console/learn_model.py`.【F:src/maou/infra/console/learn_model.py†L1-L639】
- Interface normalization helpers and scheduler/optimizer guards –
  `src/maou/interface/learn.py`.【F:src/maou/interface/learn.py†L12-L266】
- Training setup, checkpoints, and logging –
  `src/maou/app/learning/dl.py`, `src/maou/app/learning/model_io.py`.【F:src/maou/app/learning/dl.py†L94-L209】【F:src/maou/app/learning/model_io.py†L1-L86】
