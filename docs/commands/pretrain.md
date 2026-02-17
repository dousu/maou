# `maou pretrain`

## Overview

- Runs the masked autoencoder (MAE) warm-up loop that produces encoder weights
  for downstream training. The CLI, defined in
  `src/maou/infra/console/pretrain_cli.py`, only ingests local `.npz`/`.npy`
  datasets and exposes all MAE hyperparameters plus optional config-file
  overrides.【F:src/maou/infra/console/pretrain_cli.py†L1-L185】
- `maou.interface.pretrain` normalizes cache modes, worker counts, and transform
  caching before instantiating `MaskedAutoencoderPretraining`, which executes the
  PyTorch training loop and writes the encoder `state_dict`.【F:src/maou/interface/pretrain.py†L13-L72】【F:src/maou/app/learning/masked_autoencoder.py†L165-L466】

## CLI options

### Dataset and configuration

| Flag | Required | Description |
| --- | --- | --- |
| `--input-path PATH` | ✅ | Folder containing `.npz`/`.npy` artifacts from the preprocessing pipeline. The CLI walks recursively and aborts when no files are found.【F:src/maou/infra/console/pretrain_cli.py†L16-L66】 |
| `--input-format {preprocess,hcpe}` | default `preprocess` | Selects how `FileDataSourceSpliter` parses the files and informs the interface so HCPE-specific defaults can apply.【F:src/maou/infra/console/pretrain_cli.py†L21-L83】【F:src/maou/interface/pretrain.py†L13-L38】 |
| `--input-file-packed/--no-input-file-packed` | default `--no-input-file-packed` | Flags whether local shards were bit-packed. Passed directly to `FileDataSource`.【F:src/maou/infra/console/pretrain_cli.py†L22-L66】 |
| `--input-cache-mode {file,memory,mmap}` | default `file` | Cache strategy for local inputs. `file` uses standard file I/O, `memory` copies into RAM. `mmap` is **deprecated** and internally converted to `file`.【F:src/maou/infra/console/pretrain_cli.py†L37-L44】 |
| `--config-path FILE` | optional | JSON/TOML file whose fields override CLI hyperparameters (excluding the datasource). Consumed later by `_apply_config_overrides`.【F:src/maou/infra/console/pretrain_cli.py†L34-L83】【F:src/maou/app/learning/masked_autoencoder.py†L259-L325】 |
| `--output-path FILE` | optional | Destination for the serialized encoder `state_dict`. Defaults to `masked_autoencoder_state.pt` in the current directory.【F:src/maou/infra/console/pretrain_cli.py†L34-L99】【F:src/maou/app/learning/masked_autoencoder.py†L342-L356】 |

### Training hyperparameters and runtime knobs

| Flag | Default | Description |
| --- | --- | --- |
| `--epochs INT` | `5` | Number of passes over the dataset. Values ≤0 raise before training starts.【F:src/maou/infra/console/pretrain_cli.py†L41-L114】【F:src/maou/app/learning/masked_autoencoder.py†L210-L236】 |
| `--batch-size INT` | `64` | Minibatch size for the PyTorch DataLoader. Must be positive.【F:src/maou/infra/console/pretrain_cli.py†L46-L114】 |
| `--learning-rate FLOAT` | `1e-3` | Adam learning rate.【F:src/maou/infra/console/pretrain_cli.py†L51-L114】【F:src/maou/app/learning/masked_autoencoder.py†L264-L273】 |
| `--mask-ratio FLOAT` | `0.75` | Portion of flattened board features masked per sample. Constrained to `[0,1]`.【F:src/maou/infra/console/pretrain_cli.py†L56-L121】【F:src/maou/app/learning/masked_autoencoder.py†L210-L236】 |
| `--device STRING` | auto | Explicit device identifier (`cuda:0`, `cpu`). When omitted the runner auto-detects CUDA and falls back to CPU.【F:src/maou/infra/console/pretrain_cli.py†L61-L132】【F:src/maou/app/learning/masked_autoencoder.py†L357-L425】 |
| `--compilation/--no-compilation` | default `--no-compilation` | Enables `torch.compile` for the MAE model via `compile_module`. Useful for benchmarking PyTorch 2.x backends.【F:src/maou/infra/console/pretrain_cli.py†L66-L146】【F:src/maou/app/learning/masked_autoencoder.py†L180-L203】 |
| `--dataloader-workers/--num-workers INT` | default `0` | Number of DataLoader workers. `None` resolves to `0` before reaching the interface.【F:src/maou/infra/console/pretrain_cli.py†L71-L153】 |
| `--prefetch-factor INT` | default `2` | Batches prefetched per worker when workers are enabled. `None` resolves to `2`.【F:src/maou/infra/console/pretrain_cli.py†L74-L146】【F:src/maou/interface/pretrain.py†L28-L49】 |
| `--pin-memory/--no-pin-memory` | device-dependent | Overrides Tensor pinning. When unset, the runner mirrors the selected device (pin on CUDA, off on CPU).【F:src/maou/infra/console/pretrain_cli.py†L79-L146】【F:src/maou/app/learning/masked_autoencoder.py†L383-L425】 |
| `--cache-transforms/--no-cache-transforms` | format-dependent | When omitted, caching defaults to `True` for HCPE inputs and `False` for preprocess arrays. Flags override the heuristic.【F:src/maou/infra/console/pretrain_cli.py†L86-L153】【F:src/maou/interface/pretrain.py†L37-L65】 |
| `--hidden-dim INT` | `512` | Size of the decoder bottleneck. Must be positive.【F:src/maou/infra/console/pretrain_cli.py†L91-L153】【F:src/maou/app/learning/masked_autoencoder.py†L93-L147】 |
| `--forward-chunk-size INT` | auto | Caps the number of samples processed per forward micro-batch. When omitted the runner uses the batch size as-is.【F:src/maou/infra/console/pretrain_cli.py†L97-L153】【F:src/maou/app/learning/masked_autoencoder.py†L300-L337】 |

## Execution flow

1. **Datasource assembly** – After validating `--input-path`, the CLI enumerates
   the files and instantiates `FileDataSource.FileDataSourceSpliter` with the
   selected format, pack mode, and cache mode. Remote datasources are not yet
   supported, so all training data must live locally.【F:src/maou/infra/console/pretrain_cli.py†L16-L83】
2. **Default normalization** – Worker counts (`None → 0`), prefetch factors
   (`None → 2`), and optional pin-memory overrides are normalized before calling
   the interface to ensure consistent defaults in tests and scripts.【F:src/maou/infra/console/pretrain_cli.py†L132-L177】【F:src/maou/interface/pretrain.py†L25-L49】
3. **Interface validation** – `maou.interface.pretrain.pretrain` re-validates the
   datasource type (`preprocess` or `hcpe`), enforces cache-transforms defaults,
   clamps worker counts to non-negative values, and builds the MAE options
   dataclass with optional forward chunking or config overrides.【F:src/maou/interface/pretrain.py†L13-L72】
4. **Training & persistence** – `MaskedAutoencoderPretraining.run` performs the
   training loop, applies configuration overrides, logs model summaries, and saves
   the encoder `state_dict` to the requested path. Progress bars and device logic
   live entirely inside the app layer, and the CLI simply prints the returned
   status message.【F:src/maou/app/learning/masked_autoencoder.py†L165-L466】【F:src/maou/infra/console/pretrain_cli.py†L146-L185】

## Validation and guardrails

- `--input-path` must exist and contain files; the CLI aborts early if the folder
  is empty so GPU resources are never allocated unnecessarily.【F:src/maou/infra/console/pretrain_cli.py†L16-L66】
- Cache mode strings are lower-cased and validated before building the
  datasource; invalid values raise descriptive errors.【F:src/maou/infra/console/pretrain_cli.py†L31-L70】
- Worker counts and prefetch factors must be non-negative/positive. The interface
  raises `ValueError` when those invariants are violated.【F:src/maou/interface/pretrain.py†L25-L49】
- Mask ratio, epochs, and batch size are checked for valid ranges; invalid values
  raise before the training loop begins.【F:src/maou/infra/console/pretrain_cli.py†L41-L121】【F:src/maou/app/learning/masked_autoencoder.py†L210-L236】
- TensorRT/GPU extras are not required, but the runner auto-detects CUDA and uses
  CPU when unavailable so contributors can run the MAE locally.【F:src/maou/app/learning/masked_autoencoder.py†L357-L425】

## Outputs and usage

- Successful runs produce `masked_autoencoder_state.pt` (or your custom
  `--output-path`) containing the encoder `state_dict`. Downstream training jobs
  can load this file to initialize the backbone before supervised learning.
- Config files supplied via `--config-path` override matching CLI flags, making
  it easy to share reproducible MAE setups across machines.【F:src/maou/app/learning/masked_autoencoder.py†L259-L325】
- The CLI prints whatever message the app layer returns (e.g., "MAE pretraining
  completed"), so automation scripts can treat success as a zero-exit run.

### Example invocation

```bash
poetry run maou pretrain \
  --input-path artifacts/preprocess \
  --input-format preprocess \
  --epochs 10 \
  --batch-size 128 \
  --learning-rate 5e-4 \
  --mask-ratio 0.8 \
  --device cuda:0 \
  --compilation \
  --cache-transforms
```

## Implementation references

- CLI definition, dataset loading, and config overrides –
  `src/maou/infra/console/pretrain_cli.py`.【F:src/maou/infra/console/pretrain_cli.py†L1-L185】
- Interface adapter and normalization rules – `src/maou/interface/pretrain.py`.【F:src/maou/interface/pretrain.py†L13-L72】
- Masked autoencoder training loop and persistence –
  `src/maou/app/learning/masked_autoencoder.py`.【F:src/maou/app/learning/masked_autoencoder.py†L165-L466】
