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
| `--input-format {hcpe,preprocess}` | default `hcpe` | Drives both CLI validation and the interface `datasource_type`. Any other string raises a `ValueError`.【F:src/maou/infra/console/learn_model.py†L96-L156】【F:src/maou/interface/learn.py†L101-L120】<br><br>**注意**: マルチステージ学習（`--stage all`または`--stage 3`）において、Stage 3は常に前処理済みデータ（`array_type="preprocessing"`）を使用します。このため、`--input-format`オプションはStage 3には影響しません。 |
| `--input-dataset-id` + `--input-table-name` | pair | Streams from BigQuery when the optional `gcp` extra is installed. Requires `--input-format` to select the array schema and supports batching/cache knobs (below).【F:src/maou/infra/console/learn_model.py†L245-L330】 |
| `--input-gcs` / `--input-s3` + bucket metadata | pair | Downloads shards via `GCSDataSource` or `S3DataSource` splitters. Both providers need `--input-local-cache-dir` and honor optional bundling (`--input-enable-bundling`, `--input-bundle-size-gb`) and worker counts (`--input-max-workers`).【F:src/maou/infra/console/learn_model.py†L330-L399】 |
| `--input-max-workers`, `--input-batch-size`, `--input-max-cached-bytes`, `--input-local-cache`, `--input-local-cache-dir`, `--input-clustering-key`, `--input-partitioning-key-date` | optional | Fine-tune remote datasource caching and streaming. Forwarded directly to the datasource constructors and into the interface options.【F:src/maou/infra/console/learn_model.py†L122-L399】【F:src/maou/interface/learn.py†L198-L210】 |
| `--input-file-packed` | optional | Tells file-based datasources to unpack bit-packed numpy blobs. Ignored for cloud providers.【F:src/maou/infra/console/learn_model.py†L96-L130】 |
| `--input-cache-mode {file,memory,mmap}` | default `file` | Cache strategy for local inputs. `file` uses standard file I/O, `memory` copies into RAM. `mmap` is **deprecated** and internally converted to `file`.【F:src/maou/infra/console/learn_model.py†L106-L113】【F:src/maou/interface/learn.py†L198-L210】 |
| `--input-enable-bundling` + `--input-bundle-size-gb` | optional | Bundle remote shards (default 1 GB) before caching to reduce metadata churn. Applies to GCS/S3 datasources.【F:src/maou/infra/console/learn_model.py†L330-L399】 |

**Input exclusivity.** Only one provider (local, BigQuery, GCS, or S3) may be
active. The CLI counts enabled sources and raises when more than one set of
flags is present.【F:src/maou/infra/console/learn_model.py†L568-L639】

#### `--input-cache-mode` 使い分けガイド

##### 動作の仕組み

両モードとも**全データをメモリに載せる**点は共通である．`file`は「ディスクから逐次読み込み」ではない．

- **`file`モード(デフォルト)**: 各ファイルのデータを個別の配列としてメモリに保持する．アクセス時に`np.searchsorted()`でO(log F)（F=ファイル数）のファイル境界探索を行い，該当ファイルの配列から直接取得する．【F:src/maou/infra/file_system/file_data_source.py†L573-L611】
- **`memory`モード**: 初期化時に全ファイルのデータを1つの配列に結合(`np.concatenate`)してメモリに保持する．O(1)の直接インデックスアクセスが可能．結合完了後に個別配列は解放されるため，定常状態のメモリ使用量は`file`と同等になる．【F:src/maou/infra/file_system/file_data_source.py†L446-L480】

**注意**: `memory`モードの初期化時には，個別配列と結合後の配列が同時に存在するため，一時的にデータサイズの最大**2倍**のメモリを消費する．この2倍ピークは一時的であり，結合完了後に個別配列が解放される．

##### 比較表

| 観点 | `file` | `memory` |
|------|--------|----------|
| メモリ使用量 | データサイズ分 | 初期化時に最大2倍（定常状態はデータサイズ分） |
| アクセス速度 | O(log F) `searchsorted`（F=ファイル数） | O(1) 直接アクセス |
| 実効速度差 | ベースライン | <0.2%改善（実用上無視可能） |
| OOMリスク | 低 | 高（データ>32GBで警告を出力）【F:src/maou/infra/file_system/file_data_source.py†L457-L464】 |
| Stage 1/2 | そのまま使用 | `file`に強制変更【F:src/maou/infra/console/learn_model.py†L986-L992】 |

##### 推奨ガイドライン

- **大半のケース → `file`（デフォルト）推奨．** 速度差が<0.2%と実用上無視できるため，OOMリスクの低い`file`が安全な選択である．`memory`モードを選択しても速度改善は<0.2%であるため，メモリに十分な余裕がある場合でも積極的に`memory`を選ぶ理由は薄い．
- `memory`の使用目安: データサイズが搭載メモリの1/4以下の場合（32GB警告閾値の半分=16GB相当が目安）．
- ADR-003のベースライン構成（`file`モード）がパフォーマンス検証で最適と確認されている．`__getitems__()`によるバッチ一括取得も検証されたが却下（115%の性能劣化）されており，現状の`file`モードによる個別アクセス(`__getitem__` + PyTorch DataLoaderのC++バッチング)が既に最速の実用構成である．詳細は[ADR-003](../adr-003-training-performance-optimization-attempts.md)を参照．

**注意**: 実際のプロセスメモリにはモデルパラメータ，DataLoaderワーカー，CUDAコンテキスト等も含まれるため，データサイズだけでメモリの余裕を判断できない．

##### `mmap`モードについて

`mmap`は**deprecated**である．CLIで指定した場合は`file`に自動変換される．【F:src/maou/infra/console/learn_model.py†L689-L698】

### Training hardware and performance knobs

| Flag | Default | Description |
| --- | --- | --- |
| `--gpu DEVICE` | | Chooses the PyTorch device string (`cuda:0`, `cpu`, etc.).【F:src/maou/infra/console/learn_model.py†L400-L430】 |
| `--compilation` | `false` | Enables `torch.compile` prior to the training loop for ahead-of-time graph optimization.【F:src/maou/infra/console/learn_model.py†L400-L470】 |
| `--detect-anomaly` | `false` | Wraps the loop with `torch.autograd.set_detect_anomaly` for debugging gradients.【F:src/maou/infra/console/learn_model.py†L400-L470】 |
| `--epoch INT` | interface default `10` | Number of passes over the training loader; must be positive.【F:src/maou/interface/learn.py†L132-L147】 |
| `--batch-size INT` | interface default `1000` | Minibatch size shared by train/test loaders; must be positive. Training batch size. Recommended by GPU memory: 512 (8GB), 1024 (16GB), 2048 (24GB), 4096 (40-80GB). Use `--gradient-accumulation-steps` to simulate larger batches.【F:src/maou/interface/learn.py†L142-L156】 |
| `--dataloader-workers INT` | interface default `0` | Worker processes for PyTorch DataLoaders. Negative values raise `ValueError`.【F:src/maou/interface/learn.py†L158-L177】 |
| `--pin-memory` | `false` | Toggles pinned host memory for faster GPU transfers.【F:src/maou/interface/learn.py†L158-L177】 |
| `--prefetch-factor INT` | interface default `4` | Number of batches prefetched per worker; must be positive.【F:src/maou/interface/learn.py†L158-L177】 |
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
| `--gradient-checkpointing` | `False` | Enable gradient checkpointing to reduce activation memory. Recommended for large batch sizes with ViT. |

### Multi-stage training

| Flag | Default | Description |
| --- | --- | --- |
| `--stage {1,2,3,all}` | `3` | Training stage: 1=Reachable Squares, 2=Legal Moves, 3=Policy+Value (前処理済みデータのみ使用), all=Sequential.【F:src/maou/infra/console/learn_model.py†L453-L460】 |
| `--stage1-data-path PATH` | optional | File or directory path for Stage 1 (reachable squares) training data.【F:src/maou/infra/console/learn_model.py†L462-L466】 |
| `--stage2-data-path PATH` | optional | File or directory path for Stage 2 (legal moves) training data.【F:src/maou/infra/console/learn_model.py†L467-L471】 |
| `--stage3-data-path PATH` | optional | Stage 3（policy+value）の学習データパスを指定します。**Stage 3は前処理済みデータ（`array_type="preprocessing"`）専用です。** hcpe形式のデータは使用できません。【F:src/maou/infra/console/learn_model.py†L472-L477】 |
| `--stage1-threshold FLOAT` | `0.99` | Accuracy threshold for Stage 1 (99%). Training advances to the next stage once this threshold is reached.【F:src/maou/infra/console/learn_model.py†L480-L485】 |
| `--stage2-threshold FLOAT` | `0.85` | F1 threshold for Stage 2 (85%).【F:src/maou/infra/console/learn_model.py†L486-L492】 |
| `--stage1-max-epochs INT` | `10` | Maximum epochs for Stage 1.【F:src/maou/infra/console/learn_model.py†L494-L499】 |
| `--stage2-max-epochs INT` | `10` | Maximum epochs for Stage 2.【F:src/maou/infra/console/learn_model.py†L500-L506】 |
| `--stage1-batch-size INT` | `None` (inherits `--batch-size`) | Batch size for Stage 1. When unset, inherits the global `--batch-size`. For small datasets (~1,000 positions), `32` is recommended.【F:src/maou/infra/console/learn_model.py†L507-L513】 |
| `--stage2-batch-size INT` | `None` (inherits `--batch-size`) | Batch size for Stage 2. When unset, inherits the global `--batch-size`.【F:src/maou/infra/console/learn_model.py†L514-L520】 |
| `--stage1-learning-rate FLOAT` | `None` (inherits `--learning-ratio`) | Learning rate for Stage 1. When unset, inherits the global `--learning-ratio`.【F:src/maou/infra/console/learn_model.py†L521-L527】 |
| `--stage2-learning-rate FLOAT` | `None` (inherits `--learning-ratio`) | Learning rate for Stage 2. When unset, inherits the global `--learning-ratio`.【F:src/maou/infra/console/learn_model.py†L528-L534】 |
| `--stage12-lr-scheduler {auto,none,Warmup+CosineDecay,CosineAnnealingLR}` | `auto` | Learning rate scheduler for Stage 1/2. `auto` enables Warmup+CosineDecay when batch_size > 256. `none` disables the scheduler (fixed LR). |
| `--stage12-compilation` / `--no-stage12-compilation` | `False` | Stage 1/2でtorch.compileを有効化．A100で10-30%高速化 |
| `--stage1-pos-weight FLOAT` | `1.0` | Stage 1損失関数の正例重み．値 > 1.0 でrecall向上，< 1.0 でprecision向上． |
| `--stage2-pos-weight FLOAT` | `1.0` | Stage 2 BCE損失の正例重み．ASL有効時は通常1.0のまま(ASLが不均衡を処理)． |
| `--stage2-gamma-pos FLOAT` | `0.0` | ASL正例focusing parameter．0.0 = 正例損失を軽視しない(推奨)． |
| `--stage2-gamma-neg FLOAT` | `0.0` | ASL負例focusing parameter．0.0 = 標準BCE，2.0 = 不均衡データ推奨．ASL有効化には > 0.0 を指定する． |
| `--stage2-clip FLOAT` | `0.0` | ASL負例確率クリッピングマージン．0.0 = 無効，0.02 = 推奨．容易な負例を完全に無視する閾値． |
| `--stage2-hidden-dim INT` | `None` | Stage 2 headの隠れ層次元．None = 単一線形層．512推奨(hidden層使用時)． |
| `--stage2-head-dropout FLOAT` | `0.0` | Stage 2 headのDropout率(0.0-1.0)．`--stage2-hidden-dim`指定時のみ有効． |
| `--stage2-test-ratio FLOAT` | `0.0` | Stage 2検証データ分割比率．0.0で分割なし，0.1で10%を検証用に使用．ストリーミングモードでは未対応． |
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
