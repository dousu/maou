# `maou pretrain`

`maou pretrain` drives the masked autoencoder (MAE) warm‑up that produces an
encoder checkpoint usable by the downstream learning jobs. The flags exposed by
the CLI are defined in
`src/maou/infra/console/pretrain_cli.py` and are forwarded to the interface layer
(`src/maou/interface/pretrain.py`).【F:src/maou/infra/console/pretrain_cli.py†L1-L146】【F:src/maou/interface/pretrain.py†L1-L49】

## CLI options and behaviors

| Flag | Required | Description |
| --- | --- | --- |
| `--input-dir PATH` | ✅ | Folder of local `.npz` or `.npy` files containing the structured records emitted by the preprocessing pipeline. The CLI walks the directory recursively via `FileSystem.collect_files` and aborts when no files are found.【F:src/maou/infra/console/pretrain_cli.py†L16-L60】 |
| `--input-format {preprocess,hcpe}` | default `preprocess` | Used twice: it decides which parser the `FileDataSourceSpliter` should use (`preprocessing` vs `hcpe`) and feeds the interface so the dataset logic can toggle HCPE-specific defaults.【F:src/maou/infra/console/pretrain_cli.py†L21-L66】【F:src/maou/interface/pretrain.py†L13-L38】 |
| `--input-file-packed/--no-input-file-packed` | default `--no-input-file-packed` | Flags whether the local files were bit-packed during preprocessing. The bit-pack hint is passed straight into `FileDataSource.FileDataSourceSpliter`.【F:src/maou/infra/console/pretrain_cli.py†L22-L66】 |
| `--input-cache-mode {mmap,memory}` | default `mmap` | Chooses how the on-disk arrays are cached (memory mapped or eager load). The value is lower-cased before being passed to the datasource builder.【F:src/maou/infra/console/pretrain_cli.py†L31-L70】 |
| `--config-path FILE` | optional | JSON or TOML file whose fields override the CLI values (everything except the datasource). This path is handed off to the interface and ultimately consumed by `MaskedAutoencoderPretraining._apply_config_overrides`.【F:src/maou/infra/console/pretrain_cli.py†L34-L83】【F:src/maou/app/learning/masked_autoencoder.py†L259-L325】 |
| `--output-path FILE` | optional | Destination for the serialized encoder `state_dict`. When omitted the workflow writes `masked_autoencoder_state.pt` in the current working directory.【F:src/maou/infra/console/pretrain_cli.py†L36-L99】【F:src/maou/app/learning/masked_autoencoder.py†L342-L356】 |
| `--epochs INT` | default `5` | Total passes over the dataset. Values ≤0 trigger a validation error before the training loop begins.【F:src/maou/infra/console/pretrain_cli.py†L41-L107】【F:src/maou/app/learning/masked_autoencoder.py†L210-L236】 |
| `--batch-size INT` | default `64` | Mini-batch size supplied to the PyTorch `DataLoader`. Must be positive.【F:src/maou/infra/console/pretrain_cli.py†L46-L107】【F:src/maou/app/learning/masked_autoencoder.py†L210-L236】 |
| `--learning-rate FLOAT` | default `1e-3` | Adam optimizer learning rate.【F:src/maou/infra/console/pretrain_cli.py†L51-L107】【F:src/maou/app/learning/masked_autoencoder.py†L264-L273】 |
| `--mask-ratio FLOAT` | default `0.75` | Portion of flattened board features that get zeroed before reconstruction. The runner enforces the inclusive `[0,1]` range.【F:src/maou/infra/console/pretrain_cli.py†L56-L114】【F:src/maou/app/learning/masked_autoencoder.py†L210-L236】 |
| `--device STRING` | optional | Explicit device identifier, e.g. `cuda:0` or `cpu`. When omitted, the interface lets the runner auto-detect CUDA availability and default to CPU otherwise.【F:src/maou/infra/console/pretrain_cli.py†L61-L121】【F:src/maou/app/learning/masked_autoencoder.py†L357-L382】 |
| `--compilation/--no-compilation` | default `--no-compilation` | Enables PyTorch 2.x compilation (`torch.compile`) for the MAE model via `compile_module`. Useful when benchmarking the new backend.【F:src/maou/infra/console/pretrain_cli.py†L66-L132】【F:src/maou/app/learning/masked_autoencoder.py†L180-L203】 |
| `--dataloader-workers / --num-workers INT` | default `0` | Worker processes for the `DataLoader`. The CLI advertises `None`, but before hitting the interface `None` is resolved to `0` so single-process loading remains the default.【F:src/maou/infra/console/pretrain_cli.py†L71-L146】【F:src/maou/infra/console/pretrain_cli.py†L158-L177】 |
| `--prefetch-factor INT` | default `2` | How many batches each worker keeps buffered when workers are enabled. The CLI and interface both replace `None` with `2` to preserve deterministic defaults.【F:src/maou/infra/console/pretrain_cli.py†L74-L146】【F:src/maou/interface/pretrain.py†L28-L49】 |
| `--pin-memory/--no-pin-memory` | default `device dependent` | Passing the flag overrides the Tensor pinning behavior. Otherwise the runner mirrors the selected device (pin memory on CUDA, off on CPU).【F:src/maou/infra/console/pretrain_cli.py†L79-L146】【F:src/maou/app/learning/masked_autoencoder.py†L383-L425】 |
| `--cache-transforms/--no-cache-transforms` | default `format dependent` | Allows caching the flattened feature tensors in RAM. When omitted the interface auto-enables caching for HCPE inputs (which already come as flattened tensors) and disables it for preprocessing arrays to avoid duplicated work.【F:src/maou/infra/console/pretrain_cli.py†L86-L153】【F:src/maou/interface/pretrain.py†L37-L65】【F:src/maou/app/learning/masked_autoencoder.py†L21-L84】 |
| `--hidden-dim INT` | default `512` | Size of the decoder MLP bottleneck inside `_MaskedAutoencoder`. Must be positive.【F:src/maou/infra/console/pretrain_cli.py†L91-L153】【F:src/maou/app/learning/masked_autoencoder.py†L93-L147】 |
| `--forward-chunk-size INT` | default `auto` | Caps the number of samples processed per forward micro-batch. When omitted the runner uses `batch-size` (no additional chunking).【F:src/maou/infra/console/pretrain_cli.py†L97-L153】【F:src/maou/app/learning/masked_autoencoder.py†L300-L337】 |

## How the CLI hands off to the interface layer

1. **Local datasource assembly** – After validating that `--input-dir` exists the
   CLI enumerates the files and instantiates
   `FileDataSource.FileDataSourceSpliter` with the selected format, pack mode,
   and cache mode. No remote backends participate yet, so all training data must
   reside on the local filesystem.【F:src/maou/infra/console/pretrain_cli.py†L16-L83】
2. **Worker and cache defaults** – The CLI normalizes `None` to `0` workers and
   `2` prefetched batches before calling the interface. The interface repeats the
   guard to ensure defaults are preserved even when contributors call it directly
   (for example from tests). Pin memory remains `None` until the app layer can
   align it with the actual device.【F:src/maou/infra/console/pretrain_cli.py†L132-L177】【F:src/maou/interface/pretrain.py†L25-L66】
3. **Interface normalization** – `maou.interface.pretrain.pretrain` verifies that
   `datasource_type` is either `preprocess` or `hcpe`, picks the cache-transforms
   default (`True` for HCPE, `False` otherwise), and builds
   `MaskedAutoencoderPretraining.Options`. Forward chunking is only included when
   explicitly set so the runner can fall back to adaptive chunk sizes.【F:src/maou/interface/pretrain.py†L25-L72】
4. **Workflow execution** – `MaskedAutoencoderPretraining.run` performs the
   training loop and saves the encoder `state_dict`. Progress bars, worker init
   hooks, caching of flattened tensors, and encoder-state persistence all live in
   `src/maou/app/learning/masked_autoencoder.py`. The CLI simply echoes the
   returned status message.【F:src/maou/infra/console/pretrain_cli.py†L146-L185】【F:src/maou/app/learning/masked_autoencoder.py†L165-L466】

### What remains stubbed?

- **Datasets beyond local files** – Even though `maou.infra.console.common`
  exposes optional imports for GCS/S3/BigQuery, the pretraining CLI currently
  instantiates `FileDataSource` unconditionally. Remote storage selectors and the
  ability to stream directly from object stores are still TODOs.【F:src/maou/infra/console/common.py†L1-L135】【F:src/maou/infra/console/pretrain_cli.py†L16-L83】
- **Train/validation split logic** – `MaskedAutoencoderPretraining` requests a
  train/test split with `test_ratio=0.0`, meaning there is no validation loop or
  checkpoint comparison yet. Hooks for evaluation, early stopping, and metric
  publishing will be added once the MAE is integrated into the end-to-end
  training schedule.【F:src/maou/app/learning/masked_autoencoder.py†L201-L224】

## Prerequisites and future integration points

1. **Local files only (for now)** – Supply an `--input-dir` that points to
   preprocessed `.npz`/`.npy` artifacts containing structured arrays with the
   `boardIdPositions` field; `_FeatureDataset` validates the shape at runtime and
   aborts otherwise. Large runs benefit from `--input-cache-mode mmap` and the
   optional `--cache-transforms` flag to reduce repeated decoding work.【F:src/maou/infra/console/pretrain_cli.py†L16-L70】【F:src/maou/app/learning/masked_autoencoder.py†L1-L84】
2. **Configuration overrides** – Use `--config-path` to capture repeatable MAE
   settings in JSON/TOML files. Every CLI option except the datasource can be
   overridden there, so future integrations (e.g., alternative optimizers or
   telemetry toggles) should add matching fields inside `Options` and the config
   coercion helper.【F:src/maou/infra/console/pretrain_cli.py†L34-L83】【F:src/maou/app/learning/masked_autoencoder.py†L259-L337】
3. **Planned remote backends** – The infrastructure layer already exposes
   placeholders for BigQuery/GCS/S3 datasources; once those connectors gain MAE
   support you can add mutually exclusive CLI flags mirroring the patterns used
   by other commands (see `maou utility`). Each backend should still feed a
   `LearningDataSource.DataSourceSpliter` so the interface contract remains
   unchanged.【F:src/maou/infra/console/common.py†L1-L135】【F:src/maou/app/learning/dl.py†L1-L120】
4. **Extended workflow hooks** – There are natural insertion points for
   experiment tracking (after `_log_model_summary`), validation metrics (inside
   `_run_epoch`), and alternate encoders (via `ModelFactory`). Contributors adding
   those capabilities should wire the extra flags through the CLI → interface →
   `Options` pipeline so downstream tooling automatically documents them here.
   【F:src/maou/app/learning/masked_autoencoder.py†L95-L466】

