# `maou learn-model`

`maou learn-model` loads large Shogi datasets, wires them into the training
pipeline, and optionally ships checkpoints/logs to cloud storage. This document
covers every flag in `src/maou/infra/console/learn_model.py`, explains how the
interface layer (`src/maou/interface/learn.py`) normalizes the options, and
shows how data moves through the pipeline.

## CLI options

### Input source & caching

| Flag | Required | Description |
| --- | --- | --- |
| `--input-dir PATH` | one of the input sources | Recursively collects `.npy` files through `FileSystem.collect_files`. Works with either `hcpe` or `preprocess` tensors. |
| `--input-file-packed` |  | Instructs the local datasource to unpack bit-packed numpy blobs. Only applies to `--input-dir` runs. |
| `--input-format {hcpe,preprocess}` | default `hcpe` | Drives both CLI validation and the interface `datasource_type`. Anything else raises a `ValueError`. |
| `--input-dataset-id` + `--input-table-name` | pair | Streams from BigQuery when the optional `gcp` extra is installed. Requires `--input-format` to choose `array_type`. |
| `--input-gcs` + `--input-bucket-name` + `--input-prefix` + `--input-data-name` + `--input-local-cache-dir` | pair | Downloads shards from Google Cloud Storage via `GCSDataSource.DataSourceSpliter`. Enables optional bundling/local caching controls. |
| `--input-s3` + `--input-bucket-name` + `--input-prefix` + `--input-data-name` + `--input-local-cache-dir` | pair | Same contract as GCS but backed by `S3DataSource.DataSourceSpliter`. Requires the `aws` extra. |
| `--input-max-workers INT` | default `8` | Concurrency for S3/GCS downloads. |
| `--input-batch-size INT` | default `10000` | BigQuery streaming batch size. |
| `--input-max-cached-bytes INT` | default `524,288,000` | Upper bound for cache/memory pressure while staging BigQuery/GCS/S3 transfers. |
| `--input-cache-mode {mmap,memory}` | default `mmap` | Forwarded to file-based datasources so tensors are memory-mapped or copied eagerly. The interface revalidates the value. |
| `--input-local-cache` | flag | Tells BigQuery to cache downloaded batches locally when combined with `--input-local-cache-dir`. |
| `--input-local-cache-dir PATH` | required for GCS/S3 | Root directory for cached shards and dataset bundles. |
| `--input-enable-bundling` + `--input-bundle-size-gb FLOAT` | defaults `false`, `1.0` | Bundle arrays into ~1 GB packages before caching cloud inputs to reduce metadata churn. |
| `--input-clustering-key` / `--input-partitioning-key-date` |  | Optional clustering/partition hints for BigQuery tables. |
| `--input-file-packed` |  | Enables unpacking local `.npy` bundles containing bit-packed planes. |

**Input exclusivity.** The CLI forbids mixing BigQuery, GCS, and S3 flags at the
same time and throws a `ValueError` when more than one provider is enabled. When
no input source is configured the command aborts with "Please specify an input
directory, a BigQuery table, a GCS bucket, or an S3 bucket."【F:src/maou/infra/console/learn_model.py†L568-L639】

### Training hardware & performance knobs

| Flag | Default | Description |
| --- | --- | --- |
| `--gpu DEVICE` | | Chooses the PyTorch device string (e.g. `cuda:0`, `cpu`). |
| `--compilation` | `false` | Enables `torch.compile` before entering the training loop. |
| `--detect-anomaly` | `false` | Wraps training with `torch.autograd.set_detect_anomaly` for debugging. |
| `--epoch INT` | interface default `10` | Number of full passes over the training loader. Must be `> 0`. |
| `--batch-size INT` | interface default `1000` | Minibatch size, validated to be positive. |
| `--dataloader-workers INT` | interface default `0` | DataLoader worker processes. Negative values raise `ValueError`. |
| `--pin-memory` | `false` | Enables pinned host memory for faster GPU transfers. |
| `--prefetch-factor INT` | interface default `2` | Number of batches prefetched per worker; must be positive. |
| `--cache-transforms/--no-cache-transforms` | inferred | When omitted, the interface caches transforms only for `hcpe` datasources. |
| `--test-ratio FLOAT` | interface default `0.2` | Portion of the dataset reserved for validation. Must be `0 < ratio < 1`. |
| `--tensorboard-histogram-frequency INT` | CLI default `0` | 0 disables histogram dumps, otherwise controls per-epoch frequency. Negative values are rejected by the interface. |
| `--tensorboard-histogram-module PATTERN` | repeatable | Glob filters so only specific parameter names produce histograms. |
| `--cache-transforms` |  | Lets CLI force-enable/disable dataset-level caching regardless of datasource type. |

### Loss, optimizer, and scheduler controls

| Flag | Default | Description |
| --- | --- | --- |
| `--gce-parameter FLOAT` | CLI default `0.1`, interface clamps to `(0,1]` and defaults to `0.7` | Controls the generalized cross-entropy loss. |
| `--policy-loss-ratio FLOAT` | default `1.0` | Relative weight for the policy head. Must be positive. |
| `--value-loss-ratio FLOAT` | default `1.0` | Relative weight for the value head. Must be positive. |
| `--learning-ratio FLOAT` | default `0.01` | Base learning rate, validated to be positive. |
| `--optimizer {adamw,sgd}` | default `adamw` | Chooses optimizer family; normalized to lowercase in the interface. |
| `--momentum FLOAT` | default `0.9` | Used by SGD and the optimizer setup code. Must sit inside `[0,1]`. |
| `--optimizer-beta1 FLOAT` | default `0.9` | AdamW β₁, constrained to `(0,1)` and must be less than β₂. |
| `--optimizer-beta2 FLOAT` | default `0.999` | AdamW β₂, constrained to `(0,1)` and must be greater than β₁. |
| `--optimizer-eps FLOAT` | default `1e-8` | AdamW ε, must be positive. |
| `--lr-scheduler {Warmup+CosineDecay,CosineAnnealingLR}` | default `Warmup+CosineDecay` | Friendly labels that map onto canonical keys via `normalize_lr_scheduler_name`. Unsupported names raise `ValueError`. |
| `--model-architecture` | default `resnet` | One of `BACKBONE_ARCHITECTURES`. Case-insensitive in the CLI; the interface enforces membership. |

### Logging, checkpoints, and uploads

| Flag | Default | Description |
| --- | --- | --- |
| `--resume-from PATH` |  | Validated to be an existing file before training resumes. |
| `--start-epoch INT` | interface default `0` | Lets operators skip straight to a later epoch counter while still training for `--epoch` total iterations. Must be non-negative. |
| `--log-dir PATH` | default `./logs` | Created automatically (if missing) before TensorBoard starts writing. |
| `--model-dir PATH` | default `./models` | Created automatically before checkpoints and ONNX exports are stored. |
| `--output-gcs`, `--gcs-bucket-name`, `--gcs-base-path` |  | Uploads both model artifacts and TensorBoard logs to Google Cloud Storage. Requires `poetry install -E gcp`. |
| `--output-s3`, `--s3-bucket-name`, `--s3-base-path` |  | Same behavior for AWS S3, gated behind `poetry install -E aws`. |

Only one cloud provider may be selected for outputs. The CLI warns when the
necessary extras are missing and continues with local-only writes.

## How options map into `learn.learn`

### Input handling

- `datasource_type` is forced to either `hcpe` or `preprocess`. The interface
  rejects any other string before the app layer sees it.【F:src/maou/interface/learn.py†L101-L120】
- `learn.learn` also verifies `model_architecture` is part of
  `BACKBONE_ARCHITECTURES`, ensuring command aliases remain case-insensitive but
  consistent internally.【F:src/maou/interface/learn.py†L122-L129】
- When the datasource flag set includes BigQuery, GCS, or S3, the CLI instantiates
  the matching `DataSourceSpliter` object. Every constructor forwards cache
  parameters and local cache directories so later iterator calls can memory-map
  bundles or throttle caching pressure. Input cache mode values are revalidated in
  `learn.learn` before being stored in `LearningOption`.【F:src/maou/interface/learn.py†L198-L210】

### Training defaults & validation

The interface normalizes every runtime hyperparameter before building the
`Learning.LearningOption` dataclass:

- `test_ratio` defaults to `0.2` and must be between `0` and `1`.【F:src/maou/interface/learn.py†L132-L140】
- `epoch` (default `10`) and `batch_size` (default `1000`) must be positive
  integers.【F:src/maou/interface/learn.py†L142-L156】
- `dataloader_workers` defaults to `0` (synchronous loading) but can be increased
  to any non-negative integer. `pin_memory` defaults to `False` and `prefetch`
  defaults to `2` (must be positive).【F:src/maou/interface/learn.py†L158-L177】
- Loss weights and learning rate fall back to `gce=0.7`, `policy=1`, `value=1`,
  `learning_ratio=0.01`; each must remain positive and the GCE parameter cannot
  exceed `1`.【F:src/maou/interface/learn.py†L179-L204】
- `cache_transforms` is `True` whenever `datasource_type == "hcpe"` (because
  HCPE tensors benefit from repeated reuse) and `False` otherwise unless the CLI
  overrides it. `tensorboard_histogram_frequency` must be non-negative, while the
  optional module filters are either `None` or the tuple passed in. 【F:src/maou/interface/learn.py†L226-L239】

### Optimizer & scheduler normalization

- Scheduler names accept canonical keys (`warmup_cosine_decay`,
  `cosine_annealing_lr`) or human-readable labels (`Warmup+CosineDecay`). The
  helper strips whitespace/punctuation and raises a `ValueError` if no alias
  matches.【F:src/maou/interface/learn.py†L12-L55】
- `momentum` defaults to `0.9` but must be between `0` and `1`. Optimizer choices
  are normalized to lowercase and restricted to `adamw` or `sgd`. Any other value
  raises immediately.【F:src/maou/interface/learn.py†L206-L221】
- AdamW parameters inherit CLI defaults when omitted and are validated to be in
  `(0,1)` with `beta2 > beta1` and `eps > 0`.【F:src/maou/interface/learn.py†L223-L247】

### Logging, checkpoints, and uploads

- `resume_from` is validated via `file_validation`, which throws if the path does
  not exist or is not a file. `log_dir` and `model_dir` default to `./logs` and
  `./models` and are automatically created using `dir_init`.【F:src/maou/interface/learn.py†L249-L266】
- `start_epoch` defaults to `0` and must be non-negative. The CLI-provided
  `tensorboard_histogram_frequency` and module filters are stored verbatim in the
  `LearningOption` object, enabling the app layer callbacks to determine when to
  emit histograms.【F:src/maou/interface/learn.py†L233-L244】
- `cloud_storage` is injected directly into `Learning`, enabling
  `ModelIO.save_model` and the TensorBoard upload hook to mirror artifacts to GCS
  or S3 whenever they exist.【F:src/maou/infra/console/learn_model.py†L416-L520】【F:src/maou/app/learning/dl.py†L94-L209】

## Error cases to expect

1. **Input/provider mistakes** – invalid `--input-format`, missing provider
   credentials, or trying to enable more than one input or output cloud provider
   trigger `ValueError`/`ImportError` before training begins.【F:src/maou/infra/console/learn_model.py†L400-L470】
2. **Scheduler/optimizer typos** – any spelling that fails the alias map causes
   `normalize_lr_scheduler_name` to throw, and unsupported optimizers raise their
   own `ValueError`.【F:src/maou/interface/learn.py†L12-L221】
3. **Range validations** – ratios, batch sizes, and worker counts are checked for
   positivity; violating these constraints produces descriptive error messages in
   the interface layer before any GPU work starts.【F:src/maou/interface/learn.py†L132-L210】
4. **Missing extras** – requesting BigQuery, GCS, or S3 inputs/outputs without the
   corresponding optional dependencies causes explicit `ImportError` messages so
   operators know to rerun `poetry install -E gcp` or `-E aws`.【F:src/maou/infra/console/learn_model.py†L473-L517】

## Training data flow

```mermaid
sequenceDiagram
    participant CLI as CLI (`maou learn-model`)
    participant DS as DataSourceSpliter
    participant Interface as `learn.learn`
    participant App as `Learning`
    participant Loop as TrainingLoop/ModelIO
    participant Local as log_dir & model_dir
    participant Cloud as Optional cloud storage

    CLI->>DS: Instantiate File/BigQuery/GCS/S3 datasource
    CLI->>Interface: Call learn.learn(datasource, options)
    Interface->>DS: train_test_split(test_ratio)
    Interface->>App: Build LearningOption & instantiate Learning
    App->>Loop: Setup loaders, optimizer, scheduler, callbacks
    Loop->>Local: Write TensorBoard runs, checkpoints, ONNX exports
    alt cloud uploads enabled
        Loop->>Cloud: Upload models + TensorBoard folders
    end
```

The datasource streams batches into `TrainingSetup.setup_training_components`,
which prepares loaders on the requested GPU, builds the backbone architecture,
and packages logging callbacks. During training the best-performing epoch is
persisted under `model_dir` (PyTorch weights plus ONNX/FP16 variants) and
TensorBoard runs are written beneath `log_dir/<arch>_training_log_<timestamp>`.
Whenever `--output-gcs` or `--output-s3` is configured the `CloudStorage`
adapter uploads both the checkpoints and tensorboard folders so artifacts stay in
sync with local disk.【F:src/maou/app/learning/dl.py†L98-L209】【F:src/maou/app/learning/model_io.py†L1-L86】
