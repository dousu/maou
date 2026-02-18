# Maouマルチステージ学習パイプライン調査報告

## 1. ファイル構成 (`src/maou/app/learning/`)

### キーファイル
- **multi_stage_training.py**: マルチステージ学習の中核実装
  - `TrainingStage`: ステージ列挙型 (REACHABLE_SQUARES=1, LEGAL_MOVES=2, POLICY_VALUE=3)
  - `StageConfig`: 各ステージの設定（バッチサイズ、閾値、DataLoader、最大エポック等）
  - `StageResult`: 実行結果（精度、損失、エポック数、成功判定）
  - `SingleStageTrainingLoop`: 単一ステージの訓練ループ
  - `MultiStageTrainingOrchestrator`: マルチステージ管理・自動進行

- **training_loop.py**: 汎用訓練ループ（Policy+Value学習用）
  - バッチサイズの解決・グラデーション蓄積・混合精度対応

- **dl.py**: Learning クラス（メイン学習ユースケース）
  - `LearningOption` データクラス：全訓練パラメータを集約
  - バッチサイズはここに格納（config.batch_size）

- **setup.py**: ファクトリクラス群
  - `DataLoaderFactory.create_dataloaders()`: バッチサイズをDataLoaderに渡す
  - `DataLoaderFactory.create_streaming_dataloaders()`: ストリーミング用（batch_size=None）

- **dataset.py**: データセット実装
  - `KifDataset`: 通常の学習用（Policy+Value）
  - `Stage1Dataset`: Stage1用（reachable squares）
  - `Stage2Dataset`: Stage2用（legal moves）

- **learn.py** (interface): CLI から呼ばれる学習実行
  - `learn()`: 単ステージ訓練
  - `learn_multi_stage()`: マルチステージオーケストレーション
  - `_run_stage1()`, `_run_stage2()`: ステージ1/2実行関数
  - `_run_stage1_streaming()`, `_run_stage2_streaming()`: ストリーミング版

---

## 2. マルチステージ学習実装（特にステージ分割・バッチサイズ・閾値）

### 2.1 ステージ概要

| Stage | 名称 | 目的 | 出力ヘッド | 目標精度 | 最大エポック |
|-------|------|------|-----------|---------|------------|
| 1 | Reachable Squares | 駒が到達できる升を学習 | ReachableSquaresHead | 99% (default) | 10 (default) |
| 2 | Legal Moves | 合法手を学習 | LegalMovesHead | 95% (default) | 10 (default) |
| 3 | Policy+Value | 最適手・評価値を学習 | PolicyHead+ValueHead | なし | 任意 |

### 2.2 ステージ分割の仕組み

**データの分割:**
- **Stage 1/2**: `train_test_split(test_ratio=0.0)` → 全データを訓練に使用
  ```python
  # learn.py の _run_stage1/_run_stage2
  train_ds, _ = datasource.train_test_split(test_ratio=0.0)
  ```

- **Stage 3**: `train_test_split(test_ratio=指定値)` → 訓練/検証に分割
  ```python
  # dl.py の _setup_streaming_components
  test_ratio = config.test_ratio  # デフォルト 0.2
  ```

**データ量の違い:**
- Stage 1/2: **全データを訓練に使用** → より多くのデータで高精度に学習
- Stage 3: **test_ratioで分割** → デフォルトは80:20分割

### 2.3 バッチサイズの使われ方

**CLI定義** (`learn_model.py`)
```python
@click.option(
    "--batch-size",
    type=int,
    help="Training batch size.",
    required=False,
)
```

**パラメータフロー:**
1. CLI: `--batch-size <N>` 入力
2. learn_model.py: `batch_size: Optional[int]` パラメータ
3. learn_multi_stage(): `batch_size=batch_size or 256` (デフォルト256)
4. _run_stage1/_run_stage2:
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,  # ← Stage1/2はここで使用
       shuffle=True,
       num_workers=0,
       pin_memory=(device.type == "cuda"),
   )
   ```
5. StreamingKifDataset (Stage 3):
   ```python
   train_dataset = StreamingKifDataset(
       streaming_source=...,
       batch_size=config.batch_size,  # ← Stage3でも同じbatch_size
       shuffle=True,
   )
   dataloader = DataLoader(
       dataset,
       batch_size=None,  # ← 注意：ストリーミングではNone（dataset内でバッチ作成）
       shuffle=False,
   )
   ```

### 2.4 閾値チェック（Threshold Logic）

**multi_stage_training.py の SingleStageTrainingLoop.run()**

```python
for epoch in range(self.config.max_epochs):
    # ...
    epoch_loss, epoch_accuracy = self._train_epoch(epoch)

    # 閾値チェック（エポック終了直後）
    if epoch_accuracy >= self.config.accuracy_threshold:
        logger.info(
            f"Stage {self.config.stage} threshold achieved! "
            f"({epoch_accuracy:.2%} >= {self.config.accuracy_threshold:.2%})"
        )
        return StageResult(
            stage=self.config.stage,
            achieved_accuracy=epoch_accuracy,
            final_loss=final_loss,
            epochs_trained=epoch + 1,
            threshold_met=True,  # ← 早期停止
        )

# 最大エポック到達 → 閾値確認
threshold_met = best_accuracy >= self.config.accuracy_threshold
return StageResult(
    threshold_met=threshold_met,  # ← 成功/失敗判定
)
```

**エラーハンドリング** (multi_stage_training.py の run_all_stages)

```python
if not result.threshold_met:
    raise RuntimeError(
        f"Stage {self.config.stage} failed to meet accuracy threshold "
        f"after {result.epochs_trained} epochs.\n"
        f"  achieved: {result.achieved_accuracy:.2%}"
        f" / required: {stage1_config.accuracy_threshold:.2%}"
        f" (reached {result.achieved_accuracy / stage1_config.accuracy_threshold:.1%}"
        f" of target)\n"
        f"  Hint: try increasing --stage1-max-epochs"
        f" or lowering --stage1-threshold"
    )
```

---

## 3. CLI の `--batch-size` オプション定義と使われ方

### 3.1 定義位置
**File**: `/workspaces/maou/src/maou/infra/console/learn_model.py`
- Line: `@click.option("--batch-size", type=int, ...)`
- タイプ: `int`
- 必須: `required=False`
- デフォルト: `None` (後で解決)

### 3.2 解決フロー

```
learn_model CLI
    ↓ batch_size=int or None
learn_multi_stage() (learn.py)
    ↓ batch_size=batch_size or 256 (デフォルト256)
_run_stage1/_run_stage2 (learn.py)
    ↓ DataLoader(batch_size=batch_size, ...)
SingleStageTrainingLoop._train_epoch()
    ↓ for batch_idx, (inputs, targets) in enumerate(dataloader)
```

**Stage 3 (ストリーミング)**:
```
learn() (learn.py)
    ↓ Learning(cloud_storage).learn(config)
dl.py: Learning._setup_streaming_components()
    ↓ StreamingKifDataset(batch_size=config.batch_size)
    ↓ DataLoader(batch_size=None)  # 自動バッチングOFF
```

---

## 4. Stage1と Stage2のデータ量の違い

### 4.1 データ分割方式

**Stage 1/2**: 全データを訓練用に使用
```python
# learn.py _run_stage1/_run_stage2
datasource = data_config.create_datasource()
train_ds, _ = datasource.train_test_split(test_ratio=0.0)  # 100%訓練
```

**Stage 3**: test_ratioで分割
```python
# learn.py: learn_multi_stage()で Stage3を処理
stage3_datasource = stage3_data_config.create_datasource()
# learn() に渡され、Learning.learn() 内で分割
training_datasource, validation_datasource = (
    config.datasource.train_test_split(test_ratio=config.test_ratio)
)
# デフォルト: test_ratio=0.2 → 80:20分割
```

### 4.2 異なるデータセットの場合

Stage 1/2/3 に異なるファイルパスを指定可能:
```bash
maou learn-model \
    --stage all \
    --stage1-data-path /path/to/stage1_data \
    --stage2-data-path /path/to/stage2_data \
    --stage3-data-path /path/to/stage3_data
```

各ステージは **独立したデータソース** を使用:
- `--stage1-data-path` → Stage1専用データ
- `--stage2-data-path` → Stage2専用データ
- `--stage3-data-path` → Stage3専用データ

---

## 5. 学習ループの実装

### 5.1 エポック数

**設定値**:
- CLI: `--stage1-max-epochs 10` (デフォルト10)
- CLI: `--stage2-max-epochs 10` (デフォルト10)
- CLI: `--epoch <N>` (Stage3用、デフォルトなし)

### 5.2 勾配更新（Gradient Update）

**multi_stage_training.py: SingleStageTrainingLoop._train_epoch()**

```python
for batch_idx, (inputs, targets) in enumerate(self.config.dataloader):
    # データをデバイスに転送
    board_tensor = board_tensor.to(self.device, non_blocking=True)
    hand_tensor = hand_tensor.to(...) if ... else None
    targets = targets.to(self.device, non_blocking=True)

    # 勾配クリア
    self.config.optimizer.zero_grad()

    # 順伝播
    with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
        features = self.model.forward_features((board_tensor, hand_tensor))
        logits = self.head(features)
        loss = self.config.loss_fn(logits, targets)

    # 逆伝播
    if self.scaler is not None:
        self.scaler.scale(loss).backward()
        self.scaler.step(self.config.optimizer)
        self.scaler.update()
    else:
        loss.backward()
        self.config.optimizer.step()

    # 精度計算
    with torch.no_grad():
        predictions = torch.sigmoid(logits) > 0.5
        correct = (predictions == targets.bool()).float().sum()
```

### 5.3 閾値到達判定ロジック

```python
# SingleStageTrainingLoop.run()
best_accuracy = 0.0

for epoch in range(self.config.max_epochs):
    epoch_loss, epoch_accuracy = self._train_epoch(epoch)

    best_accuracy = max(best_accuracy, epoch_accuracy)

    # 【早期停止】: 各エポック後に判定
    if epoch_accuracy >= self.config.accuracy_threshold:
        return StageResult(
            achieved_accuracy=epoch_accuracy,
            epochs_trained=epoch + 1,
            threshold_met=True,  # ← 成功
        )

# 【最大エポック到達】: 最良精度で判定
threshold_met = best_accuracy >= self.config.accuracy_threshold
return StageResult(
    achieved_accuracy=best_accuracy,
    epochs_trained=self.config.max_epochs,
    threshold_met=threshold_met,  # ← 成功/失敗
)
```

---

## 6. DataLoaderの構成

### 6.1 Map-style Dataset (Stage 1/2 デフォルト)

**setup.py: DataLoaderFactory.create_dataloaders()**

```python
training_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,  # ← バッチサイズ指定
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=pin_memory,
    persistent_workers=dataloader_workers > 0,
    prefetch_factor=prefetch_factor if dataloader_workers > 0 else None,
    drop_last=drop_last_train,  # Train: True
    timeout=120 if dataloader_workers > 0 else 0,
    worker_init_fn=worker_init_fn,
)
```

**バッチサイズの渡され方**:
1. `batch_size=256` (デフォルト) → CLI または プログラムで指定
2. DataLoader が dataset から `batch_size` 個のサンプルを取得
3. collate_fn で default batching（テンソルをスタック）

### 6.2 Streaming Dataset (Stage 1/2/3 オプション)

**setup.py: DataLoaderFactory.create_streaming_dataloaders()**

```python
training_loader = DataLoader(
    train_dataset,
    batch_size=None,  # ← 注意: 自動バッチングOFF
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=pin_memory,
    persistent_workers=dataloader_workers > 0,
    prefetch_factor=prefetch_factor if dataloader_workers > 0 else None,
)
```

**StreamingKifDataset (Stage 3)**:
```python
class StreamingKifDataset(IterableDataset):
    def __init__(self, streaming_source, batch_size, shuffle):
        self.batch_size = batch_size  # ← Dataset内で保持
        self.shuffle = shuffle

    def __iter__(self):
        # ストリーミング = バッチ単位でデータを読込・yield
        for batch in self._fetch_batches():
            # batch は既に batch_size でまとめられたテンソル
            yield batch
```

### 6.3 バッチサイズ解決ロジック

**training_loop.py: TrainingLoop._resolve_batch_size()**

```python
@staticmethod
def _resolve_batch_size(inputs: ModelInputs) -> int:
    """バッチサイズを inputs の形状から推定"""
    if isinstance(inputs, torch.Tensor):
        return int(inputs.size(0))
    if isinstance(inputs, Sequence):
        for element in inputs:
            if isinstance(element, torch.Tensor):
                return int(element.size(0))
            # ネストされた Sequence の場合は再帰
            if isinstance(element, Sequence):
                return TrainingLoop._resolve_batch_size(element)
```

**使用箇所**:
```python
# training_loop.py: TrainingLoop.run_epoch()
batch_size = self._resolve_batch_size(inputs)
context = TrainingContext(
    batch_idx=batch_idx,
    batch_size=batch_size,  # ← コンテキストに格納
    ...
)
```

---

## 7. 重要な実装詳細

### 7.1 GPU Prefetch (training_loop.py)

```python
# バッチサイズに応じた GPU prefetch buffer 自動計算
effective_batch_size = (
    dataloader.batch_size or 256
)
buffer_size = calculate_recommended_buffer_size(effective_batch_size)
prefetcher = DataPrefetcher(
    dataloader,
    device=self.device,
    buffer_size=buffer_size,
)
```

### 7.2 Mixed Precision + Gradient Accumulation (training_loop.py)

```python
# Gradient accumulation step calculation
accumulation_step = context.batch_idx % self.gradient_accumulation_steps

# 蓄積ステップの最後でのみオプティマイザを実行
if not is_accumulation_step:
    context.loss = (
        self.policy_loss_ratio * policy_loss
        + self.value_loss_ratio * value_loss
    ) / self.gradient_accumulation_steps
    self.scaler.step(self.optimizer)
```

### 7.3 ストリーミングモード最適化 (learn.py)

- FileDataSourceSpliter の OOM 回避
- StreamingFileSource で遅延ファイルロード
- Stage 1/2 では cache_mode を強制的に "file" に変更
  ```python
  # Stage1/2では cache_mode を強制的に "file" に（メモリ効率）
  stage12_cache_mode = "file"
  if input_cache_mode.lower() == "memory":
      logger.info("Stage 1/2: cache_mode='memory' is ignored...")
  ```

---

## 8. 精度・損失計算

### 8.1 Stage 1/2 (二値分類)

```python
# SingleStageTrainingLoop._train_epoch()
predictions = torch.sigmoid(logits) > 0.5  # 二値化
correct = (predictions == targets.bool()).float().sum()
accuracy = correct / targets.numel()  # サンプル単位の精度
```

### 8.2 Stage 3 (Policy + Value)

```python
# training_loop.py TrainingLoop._compute_policy_loss()
policy_log_probs = torch.nn.functional.log_softmax(
    context.outputs_policy, dim=1
)
policy_loss = self.loss_fn_policy(
    policy_log_probs, policy_targets
)

# Value loss (BCEWithLogitsLoss)
value_loss = self.loss_fn_value(
    context.outputs_value, context.labels_value
)

total_loss = (
    self.policy_loss_ratio * policy_loss
    + self.value_loss_ratio * value_loss
) / self.gradient_accumulation_steps
```

---

## 9. チェックポイント保存

**multi_stage_training.py: MultiStageTrainingOrchestrator._save_stage_checkpoint()**

```python
if stage == TrainingStage.REACHABLE_SQUARES:
    head_filename = f"stage1_reachable_head_{timestamp}.pt"
    backbone_filename = f"stage1_backbone_{timestamp}.pt"
elif stage == TrainingStage.LEGAL_MOVES:
    head_filename = f"stage2_legal_moves_head_{timestamp}.pt"
    backbone_filename = f"stage2_backbone_{timestamp}.pt"

torch.save(head.state_dict(), model_dir / head_filename)
torch.save(self.backbone.state_dict(), model_dir / backbone_filename)
```

**Stage 3 で再利用**:
```python
# learn.py learn_multi_stage()
saved_backbone = _find_latest_backbone_checkpoint(model_dir)
if saved_backbone is not None:
    stage3_resume_backbone = saved_backbone
    learn(..., resume_backbone_from=stage3_resume_backbone)
```

---

## 関連ファイル一覧

| ファイル | 役割 |
|---------|------|
| `/workspaces/maou/src/maou/app/learning/multi_stage_training.py` | マルチステージ中核実装 |
| `/workspaces/maou/src/maou/app/learning/training_loop.py` | 汎用訓練ループ |
| `/workspaces/maou/src/maou/app/learning/dl.py` | Learning ユースケース |
| `/workspaces/maou/src/maou/app/learning/setup.py` | ファクトリクラス群 |
| `/workspaces/maou/src/maou/app/learning/dataset.py` | データセット (KifDataset, Stage1/2Dataset) |
| `/workspaces/maou/src/maou/interface/learn.py` | learn(), learn_multi_stage() 実装 |
| `/workspaces/maou/src/maou/infra/console/learn_model.py` | CLI エントリポイント |
