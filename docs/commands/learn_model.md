# `maou learn-model`

## Overview

- Ingests preprocessing datasets from local folders via stage-specific file
  paths, then normalizes every CLI flag before wiring them into the training
  pipeline defined in `src/maou/infra/console/learn_model.py`. The CLI exposes
  cloud upload toggles so operators can mirror production
  setups during experiments.【F:src/maou/infra/console/learn_model.py†L1-L639】
- The interface (`maou.interface.learn`) converts the parsed flags into a
  `Learning.LearningOption`, instantiates the requested datasource, and then
  hands control to the `Learning` app layer, which prepares DataLoaders, models,
  optimizers, checkpoints, and optional cloud uploads.【F:src/maou/interface/learn.py†L101-L266】【F:src/maou/app/learning/dl.py†L94-L209】

## CLI options

### Training hardware and performance knobs

| Flag | Default | Description |
| --- | --- | --- |
| `--gpu DEVICE` | | Chooses the PyTorch device string (`cuda:0`, `cpu`, etc.).【F:src/maou/infra/console/learn_model.py†L400-L430】 |
| `--compilation` | `false` | Enables `torch.compile` prior to the training loop for ahead-of-time graph optimization.【F:src/maou/infra/console/learn_model.py†L400-L470】 |
| `--detect-anomaly` | `false` | Wraps the loop with `torch.autograd.set_detect_anomaly` for debugging gradients.【F:src/maou/infra/console/learn_model.py†L400-L470】 |
| `--epoch INT` | interface default `10` | Number of passes over the training loader; must be positive.【F:src/maou/interface/learn.py†L132-L147】 |
| `--batch-size INT` | interface default `1000` | Minibatch size shared by train/test loaders; must be positive. Training batch size. Recommended by GPU memory: 512 (8GB), 1024 (16GB), 2048 (24GB), 4096 (40-80GB). Use `--gradient-accumulation-steps` to simulate larger batches.【F:src/maou/interface/learn.py†L142-L156】 |
| `--gradient-accumulation-steps INT` | `1` | Number of gradient accumulation steps. Effective batch size = batch_size × gradient_accumulation_steps. Use when GPU memory is insufficient for the desired effective batch size. Consider scaling learning rate proportionally (linear scaling rule) when using large accumulation steps. Ignored when `--adaptive-batch` is enabled. |
| `--adaptive-batch` | `false` | Enable adaptive batch size based on Gradient Noise Scale (GNS). Dynamically adjusts gradient accumulation steps during training to maintain optimal effective batch size near the Critical Batch Size (CBS). |
| `--adaptive-batch-min-steps INT` | `2` | Minimum gradient accumulation steps for adaptive batch. Must be >= 2 for GNS estimation. |
| `--adaptive-batch-max-steps INT` | `8` | Maximum gradient accumulation steps for adaptive batch. |
| `--adaptive-batch-interval INT` | `50` | Number of optimizer steps between adaptive batch size adjustments. |
| `--adaptive-batch-smoothing FLOAT` | `0.1` | EMA smoothing factor for GNS estimates. 0 に近いほど安定，1 に近いほど追従性が高い． |
| `--dataloader-workers INT` | interface default `0` | Worker processes for PyTorch DataLoaders. Negative values raise `ValueError`.【F:src/maou/interface/learn.py†L158-L177】 |
| `--pin-memory` | `false` | Toggles pinned host memory for faster GPU transfers.【F:src/maou/interface/learn.py†L158-L177】 |
| `--prefetch-factor INT` | interface default `4` | Number of batches prefetched per worker; must be positive.【F:src/maou/interface/learn.py†L158-L177】 |
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

#### `--gradient-checkpointing` の影響

ViTエンコーダのGPU活性化メモリを約93%削減するオプション．
大きなバッチサイズでのCUDA OOMを回避できる．

| 項目 | 影響 |
|------|------|
| GPU活性化メモリ | 約93%削減（例: batch_size=4096で ~30GB → ~2GB） |
| 学習速度 (Stage 1/2) | 20〜30%低下（forward passの再計算） |
| 学習速度 (Stage 3, trainable_layers=2) | 7〜10%低下 |
| 学習精度 | 影響なし（数学的に等価） |
| CPU/ホストメモリ | 影響なし |

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
| `--trainable-layers INT` | `None` | Number of trailing backbone layer groups to keep trainable. `0` = freeze all backbone layers. Unset = all layers trainable. **マルチステージ時の層分離**: 設定すると Stage 1/2 では最初の `(total - N)` グループのみを訓練し，Stage 3 では末尾 N グループのみを訓練する．ResNet は投射層 (Pool+Linear) 経由，MLP-Mixer/ViT は LayerNorm+Mean Pooling 経由で出力次元を合わせる．【F:src/maou/infra/console/learn_model.py†L443-L451】 |
| `--resume-reachable-head-from PATH` | optional | Reachable squares head parameter file to resume training (Stage 1).【F:src/maou/infra/console/learn_model.py†L508-L512】 |
| `--resume-legal-moves-head-from PATH` | optional | Legal moves head parameter file to resume training (Stage 2).【F:src/maou/infra/console/learn_model.py†L514-L518】 |

### Loss, optimizer, and scheduler controls

| Flag | Default | Description |
| --- | --- | --- |
| `--gce-parameter FLOAT` | CLI default `0.1`, interface clamps to `(0,1]` with default `0.7` | Sets the generalized cross-entropy loss parameter.【F:src/maou/interface/learn.py†L179-L204】 |
| `--policy-loss-ratio FLOAT` / `--value-loss-ratio FLOAT` | default `1.0` | Relative head weights; must be positive.【F:src/maou/interface/learn.py†L179-L204】 |
| `--policy-target-mode {move-label,win-rate,weighted}` | `win-rate` | Policy教師信号モード．`move-label`=棋譜中の選択頻度(moveLabel)，`win-rate`=指し手別勝率(moveWinRate)を正規化，`weighted`=moveLabel×moveWinRateを正規化．`win-rate`/`weighted`モードにはpreprocessパイプラインで`moveWinRate`カラムが生成されたデータが必要．旧データでは`move-label`のみ使用可能．|
| `--learning-ratio FLOAT` | default `0.01` | Base learning rate supplied to the optimizer. Must be positive.【F:src/maou/interface/learn.py†L179-L204】 |
| `--optimizer {adamw,sgd}` | default `adamw` | Normalized to lowercase and validated against supported names.【F:src/maou/interface/learn.py†L206-L221】 |
| `--momentum FLOAT` | default `0.9` | Applies to SGD setups and must live inside `[0,1]`.【F:src/maou/interface/learn.py†L206-L221】 |
| `--optimizer-beta1 FLOAT` / `--optimizer-beta2 FLOAT` / `--optimizer-eps FLOAT` | defaults `0.9`, `0.999`, `1e-8` | AdamW parameters validated to satisfy `0 < beta1 < beta2 < 1` and `eps > 0`.【F:src/maou/interface/learn.py†L223-L247】 |
| `--lr-scheduler {Warmup+CosineDecay,CosineAnnealingLR}` | default `Warmup+CosineDecay` | Friendly labels that map to canonical scheduler keys through `normalize_lr_scheduler_name`. Unsupported names raise `ValueError`.【F:src/maou/interface/learn.py†L12-L55】 |
| `--model-architecture` | default `resnet` | Must be part of `BACKBONE_ARCHITECTURES`. Case-insensitive at the CLI, enforced by the interface.【F:src/maou/interface/learn.py†L122-L129】 |

### Logging, checkpoints, and uploads

| Flag | Default | Description |
| --- | --- | --- |
| `--save-split-params` | `false` | Save backbone and head parameters as separate `.pt` files (for mix-and-match loading via `--resume-backbone-from` etc.). |
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

1. **Datasource selection** – The CLI collects stage-specific file paths,
   then passes them to the multi-stage training
   interface.【F:src/maou/infra/console/learn_model.py†L122-L399】
2. **Option normalization** – `learn.learn` validates ratios, batch sizes,
   worker counts, optimizer parameters, and scheduler names
   before building `Learning.LearningOption`. Defaults such as
   `epoch=10`, `batch_size=1000`, `test_ratio=0.2` are applied
   here.【F:src/maou/interface/learn.py†L101-L247】
3. **Training setup** – The app-layer `Learning` object prepares DataLoaders,
   networks (`BACKBONE_ARCHITECTURES`), optimizers, schedulers, and callbacks
   (TensorBoard, checkpointing, optional cloud uploads) via
   `TrainingSetup.setup_training_components`.【F:src/maou/app/learning/dl.py†L94-L209】
4. **Execution and persistence** – `TrainingLoop` drives the epochs, writes
   TensorBoard runs under `log_dir`, saves PyTorch/ONNX checkpoints in
   `model_dir`, and mirrors artifacts to the configured cloud storage when
   enabled.【F:src/maou/app/learning/dl.py†L94-L209】【F:src/maou/app/learning/model_io.py†L1-L86】

## Validation and guardrails

- Selecting multiple output providers triggers an early `ValueError`
  that lists the conflicting flags, preventing accidental double
  uploads.【F:src/maou/infra/console/learn_model.py†L416-L639】
- Scheduler/optimizer typos surface through `normalize_lr_scheduler_name` and
  the optimizer guard, so unsupported names fail fast.【F:src/maou/interface/learn.py†L12-L221】
- Ratios, worker counts, epochs, and batch sizes must be positive (or within
  `(0,1)` for ratios); the interface raises descriptive errors before GPUs spin
  up.【F:src/maou/interface/learn.py†L132-L210】
- Missing extras for GCS/S3 outputs produce explicit warning messages
  instructing operators to run `uv sync --extra gcp` or `--extra aws`.【F:src/maou/infra/console/learn_model.py†L400-L520】
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
  --stage3-data-path datasets/preprocessed \
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

## 変更履歴

- **2026-02-22**: `--cache-transforms`，`--input-cache-mode`，`--input-file-packed` を削除
  - `--cache-transforms`: learn-model では transform=None 固定のため実質無効
  - `--input-cache-mode`: Stage 1/2 で "file" 強制，Stage 3 streaming で無視．内部で "file" 固定
  - `--input-file-packed`: Arrow IPC 移行に伴い不要
- **2026-02-23**: `--resume-from` を削除，`--save-split-params` を追加
  - `--resume-from`: 未使用のレガシーオプション．分割パラメータファイルによる再開は `--resume-backbone-from` 等で対応
  - `--save-split-params`: backbone/head パラメータを個別 `.pt` ファイルとして保存するオプション
- **2026-03-06**: `--gradient-accumulation-steps` を追加
  - GPU メモリ不足時に有効バッチサイズを擬似的に拡大するオプション
- **2026-03-06**: Adaptive Batch Size (GNS-Based) を追加
  - `--adaptive-batch`: GNS に基づく adaptive batch size の有効化
  - `--adaptive-batch-min-steps`，`--adaptive-batch-max-steps`: accumulation steps 範囲
  - `--adaptive-batch-interval`: 調整間隔
  - `--adaptive-batch-smoothing`: EMA 平滑化係数
  - 訓練中に Gradient Noise Scale をオンライン計測し，gradient accumulation steps を動的に調整
  - `benchmark-training --estimate-cbs` の出力に adaptive batch 推奨設定を追加
