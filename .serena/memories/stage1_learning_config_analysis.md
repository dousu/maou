# Stage1 Learning Configuration Analysis

## 1. Stage1 Learning のバッチサイズ設定

### CLIオプション
- **ファイル**: `/workspaces/maou/src/maou/infra/console/learn_model.py`
- **オプション**: `--batch-size` (行 273-277)
  - 型: `int`
  - 説明: "Training batch size."
  - 必須: False

**重要**: Stage1用の専用バッチサイズオプションは存在しない。

### バッチサイズの経路（CLI → Interface → App → Domain）

1. **CLI層** (`learn_model.py` 行 1086):
   ```python
   batch_size=batch_size or 256,  # デフォルト 256
   ```

2. **Interface層** (`learn.py` 行 792-804):
   - `learn_multi_stage()` の `batch_size` パラメータ (型: int, デフォルト: 256)

3. **Stage1実行関数** (`learn.py` 行 522-583):
   - `_run_stage1()`: map-style dataset用
   - `_run_stage1_streaming()`: streaming dataset用
   - どちらも同一の `batch_size` を使用

4. **DataLoader構成**:

   **Map-style** (`learn.py` 行 555-561):
   ```python
   DataLoader(
       dataset,
       batch_size=batch_size,      # 指定されたバッチサイズ
       shuffle=True,                # シャッフル有効
       num_workers=0,               # ワーカーなし
       pin_memory=(device.type == "cuda"),  # GPU時はピン留め
   )
   ```

   **Streaming** (`learn.py` 行 681-687):
   ```python
   DataLoader(
       dataset,
       batch_size=None,             # 自動バッチング OFF
       shuffle=False,               # Dataset内部でシャッフル
       num_workers=0,               # ワーカーなし
       pin_memory=(device.type == "cuda"),  # GPU時はピン留め
   )
   ```

### Multi-stage training での独立性
- **バッチサイズ**: Stage1, Stage2, Stage3 で共通の `batch_size` を使用
- **学習率**: Stage1, Stage2, Stage3 で共通の `learning_rate` を使用
- **エポック数**: 各ステージで独立設定可能
  - `--stage1-max-epochs` (デフォルト: 10)
  - `--stage2-max-epochs` (デフォルト: 10)
  - `--epoch` (Stage 3用, デフォルト: 10)

---

## 2. Stage1 の学習データ量

### データ生成
- **ファイル**: `/workspaces/maou/src/maou/domain/data/stage1_generator.py`
- **合計パターン**: 約1,105件

### パターン構成
- **ボード上のパターン**: 約1,098件
  - 通常の駒(8種類): FU, KY, KE, GI, KI, KA, HI, OU
  - 成った駒(6種類): TO, NKY, NKE, NGI, UMA, RYU
  - 各駒の配置: 9×9ボード上の合法的な位置

- **手持ちのパターン**: 7件
  - 7種類の駒 (FU, KY, KE, GI, KI, KA, HI)
  - 成った駒は手持ちできない

### データロード方法
1. **生成**: `Stage1DataGenerator.generate_all_stage1_data()` (stage1_generator.py 行 228)
   - 結果: Polars DataFrame (~1,105行)

2. **保存**: `save_stage1_df()` で `.feather` ファイル (stage1_data_generation.py 行 56)

3. **学習時**: `FileDataSource.FileDataSourceSpliter` または `StreamingFileSource` で読込

---

## 3. 閾値99%の判定ロジック

### CLI オプション
- **ファイル**: `/workspaces/maou/src/maou/infra/console/learn_model.py`
- **オプション**: `--stage1-threshold` (行 480-485)
  - 型: `float`
  - デフォルト: 0.99 (99%)
  - 説明: "Accuracy threshold for Stage 1 (default: 0.99 = 99%)"

### メトリクスの種類
- **メトリクス**: Accuracy (精度)
- **計算方法**: (`learn.py` 行 240-249, `multi_stage_training.py` 行 240-249)
  ```python
  # Binary classification accuracy (threshold 0.5)
  predictions = torch.sigmoid(logits) > 0.5
  correct = (predictions == targets.bool()).float().sum()
  total_correct += correct.item()
  total_samples += targets.numel()

  accuracy = total_correct / total_samples
  ```

### 判定タイミング
- **タイミング**: エポック終了時
- **ファイル**: `/workspaces/maou/src/maou/app/learning/multi_stage_training.py`
- **行**: 134-167

判定ロジック:
```python
for epoch in range(self.config.max_epochs):
    # ... training ...
    epoch_loss, epoch_accuracy = self._train_epoch(epoch)

    # Check if threshold met (早期停止)
    if epoch_accuracy >= self.config.accuracy_threshold:  # 行 156
        logger.info(f"Stage {self.config.stage} threshold achieved!")
        return StageResult(
            stage=self.config.stage,
            achieved_accuracy=epoch_accuracy,
            final_loss=final_loss,
            epochs_trained=epoch + 1,
            threshold_met=True,
        )
```

### 閾値達成時の振る舞い
1. **閾値達成**: 即座にそのステージを終了し、次ステージに進行
2. **閾値未達成 (max_epochs到達)**:
   - 最高精度を記録
   - RuntimeError を発生させる (multi_stage_training.py 行 342-353)
   ```python
   if not result.threshold_met:
       raise RuntimeError(
           f"Stage 1 failed to meet accuracy threshold "
           f"after {result.epochs_trained} epochs.\n"
           f"  achieved: {result.achieved_accuracy:.2%}"
           f" / required: {stage1_config.accuracy_threshold:.2%}"
           f"  Hint: try increasing --stage1-max-epochs"
           f" or lowering --stage1-threshold"
       )
   ```

---

## 4. Stage1 固有のバッチサイズ設定の有無

**結論**: Stage1用の専用バッチサイズオプションは存在しない。

### 調査結果
- CLI検索: `grep -n "stage1.*batch\|batch.*stage1"` → 結果なし
- Stage1, Stage2, Stage3 は共通の `--batch-size` オプションを使用
- CLIから `batch_size` の指定がない場合、デフォルト値 256 が使用される (learn_model.py 行 1086)

---

## 5. DataLoaderの構成

### Map-style Dataset (ファイル少数またはストリーミング無効時)
**場所**: `/workspaces/maou/src/maou/interface/learn.py` 行 555-561

```python
DataLoader(
    dataset,
    batch_size=batch_size,              # Stage1ではCLI指定値またはデフォルト256
    shuffle=True,                        # ファイル順序でシャッフル
    num_workers=0,                       # マルチプロセッシング不使用
    pin_memory=(device.type == "cuda"),  # CUDA利用時にメモリピン留め
)
```

**パラメータ説明**:
- `batch_size`: 1バッチあたりのサンプル数 (デフォルト: 256)
- `shuffle=True`: エポックごとにデータをシャッフル
- `num_workers=0`: メインプロセスでデータ読み込み
- `pin_memory`: GPU転送の高速化

### Streaming Dataset (ファイル複数かつストリーミング有効時)
**場所**: `/workspaces/maou/src/maou/interface/learn.py` 行 681-687, streaming_dataset.py 行 156-162

```python
DataLoader(
    dataset,                            # StreamingStage1Dataset
    batch_size=None,                    # 自動バッチング OFF
    shuffle=False,                      # Dataset内部でシャッフル
    num_workers=0,                      # ワーカーなし
    pin_memory=(device.type == "cuda"), # CUDA利用時にメモリピン留め
)
```

**StreamingStage1Dataset** (`streaming_dataset.py` 行 156-162):
```python
dataset = StreamingStage1Dataset(
    streaming_source=streaming_source,
    batch_size=batch_size,              # バッチサイズをDatasetに指定
    shuffle=True,                        # ファイル内レコードをシャッフル
)
```

**パラメータ説明**:
- バッチサイズはStreamingDataset内部で管理
- DataLoaderは `batch_size=None` で自動バッチング無効化
- Dataset内部で `batch_size` 分のレコードを集め、バッチをyield

### drop_last パラメータ
- Map-style: 指定なし (デフォルト: False) → 最後の不完全なバッチも処理
- Streaming: 指定なし (Dataset内部で管理)

---

## 関連ファイル一覧

| ファイル | 行 | 説明 |
|---------|-----|------|
| learn_model.py | 273-277 | `--batch-size` CLI オプション |
| learn_model.py | 480-485 | `--stage1-threshold` CLI オプション |
| learn_model.py | 1086 | `batch_size` デフォルト値: 256 |
| learn.py | 522-583 | `_run_stage1()` 関数 (map-style) |
| learn.py | 650-709 | `_run_stage1_streaming()` 関数 (streaming) |
| learn.py | 792-847 | `learn_multi_stage()` 関数 |
| learn.py | 555-561 | Map-style DataLoader構成 |
| learn.py | 681-687 | Streaming DataLoader構成 |
| multi_stage_training.py | 38-52 | StageConfig dataclass |
| multi_stage_training.py | 115-180 | SingleStageTrainingLoop.run() |
| multi_stage_training.py | 134-167 | エポック終了時の閾値判定 |
| multi_stage_training.py | 240-256 | 精度計算 (_train_epoch) |
| multi_stage_training.py | 342-353 | 閾値未達成時のエラーハンドリング |
| dataset.py | 295-369 | Stage1Dataset クラス |
| streaming_dataset.py | 149-220 | StreamingStage1Dataset クラス |
| stage1_generator.py | 7, 228-244 | Stage1データ生成 (~1,105件) |
