# 学習率チューニングガイド

## 概要

`learn-model` コマンドの学習率(LR)設定は，学習の安定性と最終的なモデル性能に大きく影響する．
本ドキュメントでは，ログから学習率の問題を診断する方法と，アーキテクチャ・ステージごとの推奨設定を示す．

---

## 学習率の構造

### ステージごとの指定

| CLIオプション | 対象 | デフォルト |
|-------------|------|-----------|
| `--stage1-learning-rate` | Stage 1 (到達可能マス) | `--learning-ratio` の値を継承 |
| `--stage2-learning-rate` | Stage 2 (合法手) | `--learning-ratio` の値を継承 |
| `--learning-ratio` | Stage 3 (Policy+Value) | 0.01 |

### Sqrt Batch Scaling (Stage 1/2)

Stage 1/2では，バッチサイズが `base_batch_size`(256)を超えると自動的にsqrt scalingが適用される:

```
effective_lr = learning_rate × √(actual_batch_size / 256)
```

| batch_size | スケール倍率 | LR=0.0001 の実効値 |
|-----------|-------------|------------------|
| 256 | 1.0x | 0.0001 |
| 1024 | 2.0x | 0.0002 |
| 2048 | 2.83x | 0.000283 |
| 4096 | 4.0x | 0.0004 |

**重要**: `--stage2-learning-rate 0.0001 --stage2-batch-size 4096` を指定した場合，
実際にoptimizerに設定されるLRは0.0004である．ログの `LR sqrt scaling:` 行で確認できる．

### Warmup + Cosine Decay スケジューラ

デフォルトのスケジューラはバッチ単位(per-step)で動作し，以下のスケジュールを適用する:

1. **Warmup期間** (全ステップの10%): LRが0から `effective_lr` まで線形に増加
2. **Cosine Decay期間** (残り90%): LRが `effective_lr` から0にcosine曲線で減衰

---

## ログからの診断: 兆候と対処法

### 1. Lossの急激な増大(発散)

**ログの兆候**:
```
Epoch 1 Batch 100/5000: Loss=0.0300, F1=75.00%
Epoch 1 Batch 200/5000: Loss=0.1500, F1=40.00%   ← Lossが5倍に増大
Epoch 1 Batch 300/5000: Loss=0.8000, F1=10.00%   ← さらに悪化
Epoch 1 Batch 400/5000: Loss=nan                   ← 発散
```

**原因**: 学習率が高すぎて勾配更新がモデルを不安定化している．

**対処**:
- LRを1/2〜1/5に下げる
- Stage 2で発散する場合: `--stage2-learning-rate` を下げる(batch scalingが大きい可能性)
- Stage 3で発散する場合: `--learning-ratio` を下げる

### 2. Lossの停滞(学習が進まない)

**ログの兆候**:
```
Epoch 1/50: Loss=0.0500, F1=60.00%
Epoch 5/50: Loss=0.0490, F1=60.50%   ← 5エポックでほとんど変化なし
Epoch 10/50: Loss=0.0485, F1=60.80%  ← 進歩が極めて遅い
```

**原因**: 学習率が低すぎるか，warmup期間のLRが低すぎて効果的な学習が始まっていない．

**対処**:
- LRを2〜5倍に上げる
- warmup期間中の場合は，エポック数を増やしてwarmup完了後のLRを確認する
- Stage 1でこの症状が出る場合: `--stage1-learning-rate 0.001` を試す

### 3. Lossの振動(上下を繰り返す)

**ログの兆候**:
```
Batch 100: Loss=0.0300, F1=75.00%
Batch 200: Loss=0.0350, F1=72.00%   ← 悪化
Batch 300: Loss=0.0280, F1=76.00%   ← 改善
Batch 400: Loss=0.0370, F1=71.00%   ← また悪化
```

**原因**: 学習率がやや高く，最適解の周りを振動している．

**対処**:
- LRを1/2〜1/3に下げる
- Optimizerの `--optimizer-beta2` を0.999→0.998に微調整(Adamの適応性を上げる)
- バッチサイズが小さすぎる場合は増やす(勾配の分散を低減)

### 4. エポック境界でのメトリクス急落と回復

**ログの兆候**:
```
Epoch 1 Batch 10000/10088: Loss=0.0242, F1=80.21%
Epoch 1/50: Loss=0.0242, F1=80.22%
Epoch 2 Batch 65/10088:    Loss=0.0403, F1=70.98%   ← 急落
Epoch 2 Batch 131/10088:   Loss=0.0638, F1=46.02%   ← さらに悪化
...
Epoch 2 Batch 988/10088:   Loss=0.0285, F1=75.16%   ← 回復
```

**原因**: 2つの要因の組み合わせ．

1. **メトリクスの累積平均リセット**: 表示されるF1/Accuracyはエポック開始からの累積平均．
   エポック境界でアキュムレータがリセットされるため，少数バッチの平均が表示される．
2. **LRの急激な変化**: per-epochスケジューラではwarmup中にLRが離散的にジャンプする
   (例: 2倍に増大)．これはper-stepスケジューラへの移行で解消済み(後述)．

**対処**:
- per-stepスケジューラが適用されていることを確認する(現在のデフォルト)
- 回復パターンが見られる場合は，warmup比率の問題ではなくメトリクス表示の特性

### 5. Warmup期間のLRが高すぎる

**ログの兆候**:
```
Stage LEGAL_MOVES Epoch 1: LR = 0.000400   ← Warmup完了時点でLRが高い
Epoch 2 Batch 100/10088: Loss=0.0500, F1=65.00%
Epoch 2 Batch 200/10088: Loss=0.0600, F1=60.00%   ← 悪化が続く
```

**原因**: sqrt scalingにより `effective_lr` が想定より高い．

**対処**:
- `LR sqrt scaling:` ログで実効LRを確認する
- `--stage2-learning-rate` をsqrt scaling込みの値を意識して設定する
- 例: batch_size=4096でLR=0.0001の場合，実効LR=0.0004になる

### 6. 過学習(Training精度は上がるがValidation精度が下がる)

**ログの兆候** (Stage 2で `--stage2-test-ratio 0.1` 使用時):
```
Epoch 5: Train Loss=0.0100, Val Loss=0.0200, Val F1=82.00%
Epoch 10: Train Loss=0.0050, Val Loss=0.0300, Val F1=79.00%  ← Val悪化
Epoch 15: Train Loss=0.0020, Val Loss=0.0500, Val F1=75.00%  ← Val急落
```

**対処**:
- LRを下げる(更新幅を小さくして緩やかに学習)
- `--stage2-head-dropout 0.1` を追加(正則化)
- エポック数を減らす(`--stage2-max-epochs`)
- 閾値に到達した時点で自動停止させる(`--stage2-threshold`)

---

## アーキテクチャ別推奨学習率

### ResNet (デフォルト)

比較的安定したアーキテクチャであり，広い学習率範囲で動作する．

| ステージ | batch_size | 推奨LR | 探索範囲 | 備考 |
|---------|-----------|--------|---------|------|
| Stage 1 | 16-32 | 0.001 | 0.0005-0.01 | タスクが単純なため高LRで高速収束 |
| Stage 2 | 256 | 0.0001 | 0.00005-0.001 | sqrt scaling無し |
| Stage 2 | 4096 | 0.0001 | 0.00005-0.0005 | 実効LR = 0.0004 (4x scaling) |
| Stage 3 | 256 | 0.001 | 0.0001-0.01 | sqrt scaling無し |
| Stage 3 | 1024 | 0.001 | 0.0001-0.005 | 実効LR = 0.002 (2x scaling) |

### Vision Transformer (ViT)

Transformerは学習率に敏感であり，ResNetより低い学習率が必要．
特にLayerNormとattention機構の組み合わせにより，高LRで発散しやすい．

| ステージ | batch_size | 推奨LR | 探索範囲 | 備考 |
|---------|-----------|--------|---------|------|
| Stage 1 | 16-32 | 0.0005 | 0.0001-0.001 | ResNetの半分程度 |
| Stage 2 | 256 | 0.0001 | 0.00005-0.0005 | |
| Stage 2 | 4096 | 0.00005-0.0001 | 0.00002-0.0002 | 実効LR = 0.0002-0.0004 |
| Stage 3 | 256 | 0.0001 | 0.00005-0.001 | ResNetの1/10程度が安定 |
| Stage 3 | 1024 | 0.0001 | 0.00005-0.0005 | 実効LR = 0.0002 |

**ViT固有の注意点**:
- `--vit-embed-dim` が大きい(768以上)場合はLRをさらに下げる
- `--vit-num-layers` が深い(8以上)場合もLRを下げる方向に調整
- `--vit-dropout` を0.1に設定すると安定性が向上
- `--gradient-checkpointing` でメモリ節約可能(学習速度は低下)

### MLP-Mixer

ResNetとViTの中間的な特性．Self-attentionがないためViTよりは安定するが，
全結合の混合層があるためResNetよりは敏感．

| ステージ | batch_size | 推奨LR | 探索範囲 | 備考 |
|---------|-----------|--------|---------|------|
| Stage 1 | 16-32 | 0.001 | 0.0005-0.005 | ResNetと同程度 |
| Stage 2 | 256 | 0.0001 | 0.00005-0.001 | |
| Stage 2 | 4096 | 0.0001 | 0.00005-0.0003 | 実効LR = 0.0004 |
| Stage 3 | 256 | 0.0005 | 0.0001-0.005 | ResNetよりやや低め |
| Stage 3 | 1024 | 0.0005 | 0.0001-0.002 | 実効LR = 0.001 |

---

## Sqrt Scaling を考慮したLR設定の実践

### 設定時の思考プロセス

1. **アーキテクチャとステージから推奨LRを選ぶ** (上表参照)
2. **batch_sizeからsqrt scalingの倍率を計算する**
3. **CLI指定値 = 推奨実効LR / scaling倍率 で指定する**

**例: ViT + Stage 2 + batch_size=4096**
- 推奨実効LR: 0.0002
- scaling倍率: √(4096/256) = 4.0
- CLI指定値: 0.0002 / 4.0 = 0.00005
- コマンド: `--stage2-learning-rate 0.00005 --stage2-batch-size 4096`

### Stage 3の注意

Stage 3では`TrainingSetup`内でoptimizerが作成され，sqrt scalingは適用されない．
`--learning-ratio` で指定した値がそのままoptimizerのLRになる．

---

## スケジューラの動作

### Warmup+CosineDecay (デフォルト)

バッチ単位でステップし，エポック境界でのLRジャンプが発生しない．

```
LR
 ^
 |       /‾‾‾‾‾\
 |      /        \
 |     /          \
 |    /            \
 |   /              \
 |  /                \___
 | /
 +----------------------------> step
   |  warmup |   cosine decay  |
   |  (10%)  |     (90%)       |
```

- **Warmup期間**: 全ステップの10%，最低1エポック分
- **活性化条件** (Stage 1/2): `--stage12-lr-scheduler auto` かつ `batch_size > 256` で自動有効

### CosineAnnealingLR

Warmupなしのcosine減衰．学習率が初期値から0まで滑らかに減衰する．
Warmupが不要な場合(小さいバッチサイズ，安定した学習)に適する．

---

## 学習率チューニングのワークフロー

### Step 1: ベースライン実行

アーキテクチャごとの推奨値で最初の実行を行う:

```bash
# ViT + 大バッチの例
uv run maou learn-model \
  --stage all \
  --model-architecture vit \
  --stage2-learning-rate 0.00005 \
  --stage2-batch-size 4096 \
  --learning-ratio 0.0001 \
  --batch-size 1024 \
  --lr-scheduler Warmup+CosineDecay \
  ...
```

### Step 2: ログの確認ポイント

実行後，以下のログ行を確認する:

1. **`LR sqrt scaling:`** — 実効LRが想定通りか
2. **`Stage X Epoch 1:`** — 初期のLossレベル
3. **バッチログのLoss推移** — 単調に減少しているか，振動していないか
4. **エポック終了時のLR** — warmup/decayが期待通りか

### Step 3: 問題があれば調整

| 症状 | LRの方向 | 調整幅 |
|-----|---------|-------|
| Loss発散 / nan | 下げる | 1/5〜1/2 |
| Loss振動 | 下げる | 1/3〜1/2 |
| Loss停滞 | 上げる | 2〜5倍 |
| 過学習 | 下げる + 正則化追加 | 1/2 + dropout |

### Step 4: 微調整

大まかな範囲が決まったら，その範囲内で2-3値を試してベストを選ぶ．
例: LR=0.0001が良い場合，0.00007と0.00015も試す．

---

## コマンド例集

### 保守的な設定(安定性重視)

```bash
uv run maou learn-model \
  --model-architecture vit \
  --stage2-learning-rate 0.00003 \
  --stage2-batch-size 4096 \
  --learning-ratio 0.00005 \
  --batch-size 1024 \
  --lr-scheduler Warmup+CosineDecay \
  --optimizer adamw
```

### 攻撃的な設定(速度重視)

```bash
uv run maou learn-model \
  --model-architecture resnet \
  --stage2-learning-rate 0.0003 \
  --stage2-batch-size 4096 \
  --learning-ratio 0.005 \
  --batch-size 1024 \
  --lr-scheduler Warmup+CosineDecay \
  --optimizer adamw
```

---

## Reference

- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", 2017 — Sqrt/Linear scaling rule
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT), ICLR 2021 — ViT LR sensitivity
