# トレーニング問題調査レポート

## 実行日時
2025-11-01

## 問題の概要

大規模データでのトレーニング時に以下の問題が発生:
- **Value accuracyが全エポックで0.5418で完全に固定**
- **Policy accuracyが1.4%程度と非常に低い**
- **Lossの減少が極めて遅い** (1.657 → 1.641)

## 調査結果

### 1. データ分布の極端な偏り

```
Value label distribution (n=1000):
  Mean: 0.501418
  [0.0, 0.1]: 426 samples (42.6%) ← 敗北
  [0.1, 0.9]: 144 samples (14.4%) ← 中間値
  [0.9, 1.0]: 430 samples (43.0%) ← 勝利
```

**重大な発見:**
- データの85.6%が0-0.1または0.9-1.0に集中（二峰性分布）
- 平均値: 0.501418 ≈ **実際の学習時のvalue accuracy 0.5418**
- これはモデルが学習していないのではなく，**MSE lossに対する最適解**に収束している

### 2. MSE Lossの問題

```python
# Value head structure:
value_head = nn.Sequential(
    nn.Linear(input_dim, 1),
    nn.Sigmoid()  # Output range: [0, 1]
)

# Loss function:
loss_fn_value = torch.nn.MSELoss()
```

**問題点:**
- 二峰性分布（0と1に集中）に対してMSE lossを使うと，モデルは平均値を予測するのが最適戦略になる
- 0を予測すると勝利データ（1.0）に対して大きな誤差
- 1を予測すると敗北データ（0.0）に対して大きな誤差
- **平均値（~0.54）を予測すると両方に対して誤差が最小化される**

### 3. ハイパーパラメータの問題

元のコマンドのパラメータ:
```bash
--learning-ratio 0.05      # 非常に高い (通常は0.001-0.01)
--batch-size 15000         # 非常に大きい
--gce-parameter 0.7        # Policy lossのパラメータ
--momentum 0.9
```

**問題点:**
- 学習率0.05はSGDにとって極めて高く，不安定な学習を引き起こす
- バッチサイズ15000は非常に大きく，勾配の分散が小さくなりすぎる
- これらの組み合わせにより，モデルが局所最適解（平均値予測）から抜け出せない

### 4. 小規模テストでの検証

```python
# 10ステップの学習テスト結果:
Initial loss: 0.240517 → Final loss: 0.120539  (50%減少)
Initial pred: 0.600849 → Final pred: 0.529712  (平均値に接近)
```

**結論:**
- モデル自体は正常に動作し，勾配も流れている
- 小規模テストでは学習が進行している
- 問題はデータ分布とハイパーパラメータの組み合わせ

## 根本原因のまとめ

1. **データの二峰性分布** + **MSE loss** = 平均値予測が最適解
2. **高すぎる学習率** (0.05) = 不安定な学習と局所最適解への収束
3. **大きすぎるバッチサイズ** (15000) = 勾配の分散低下
4. **Pre-trained model** (masked-autoencoder.pt) = 不適切な初期化の可能性

## 解決策

### 即効性の高い解決策（優先度順）

#### 1. 学習率の大幅な削減【最重要】
```bash
--learning-ratio 0.001  # 50分の1に削減（元: 0.05）
# または
--learning-ratio 0.0001 # より安全な初期値
```

**理由:**
- SGDで0.05は極めて高く，loss landscapeを大きく飛び越えてしまう
- Value headの学習が不安定になり，平均値から動けなくなる

#### 2. バッチサイズの削減【重要】
```bash
--batch-size 1024  # または 2048（元: 15000）
```

**理由:**
- 大きすぎるバッチは勾配の分散を減らし，局所最適解に陥りやすい
- 小さめのバッチでノイジーな勾配を得ることで，平均値予測から脱出できる

#### 3. Pre-trained modelを使わない【推奨】
```bash
# --resume-from artifacts/masked-autoencoder.pt を削除
```

**理由:**
- Masked autoencoderの事前学習は，supervised learningと目的が異なる
- ランダム初期化からの学習の方が良い結果を得られる可能性が高い

#### 4. Learning rate scheduleの導入【推奨】
現在はコードに実装されていないが，以下を検討:
- Warm-up: 最初の数エポックで学習率を徐々に上昇
- Cosine annealing: エポックとともに学習率を減衰

### 中長期的な改善策

#### 1. Loss functionの変更
```python
# Option 1: Binary Cross Entropy (より適切)
loss_fn_value = torch.nn.BCELoss()

# Option 2: Focal Loss（不均衡データ向け）
# 中間値を重視する重み付けを追加

# Option 3: Smooth L1 Loss
loss_fn_value = torch.nn.SmoothL1Loss()
```

**理由:**
- BCEは0/1の二値分類に適している
- 二峰性分布では，MSEより適切な損失関数

#### 2. Value label ratioの調整
```bash
--policy-loss-ratio 1.0
--value-loss-ratio 2.0  # Value lossの重みを増加
```

**理由:**
- 現在はvalue lossが相対的に小さい可能性
- Value headの学習を促進

#### 3. データの前処理
- Value labelを[-1, 1]から[0, 1]への変換を確認
- データのバランシング（オプション）

## 推奨される修正コマンド

### 最小限の修正（即効性重視）
```bash
poetry run maou learn-model \
  --input-format preprocess \
  --input-dir preprocess/floodgate/2020 \
  --input-file-packed \
  --test-ratio 0.2 \
  --output-gcs \
  --gcs-bucket-name maou-test-dousu \
  --gcs-base-path large_test_fixed \
  --batch-size 1024 \
  --epoch 10 \
  --gpu cuda:0 \
  --compilation true \
  --dataloader-workers 12 \
  --prefetch-factor 1 \
  --pin-memory \
  --learning-ratio 0.001 \      # 修正: 0.05 → 0.001
  --gce-parameter 0.7 \
  --momentum 0.9
  # --resume-from を削除        # 修正: Pre-trainedモデルを使わない
```

### より保守的な設定
```bash
poetry run maou learn-model \
  --input-format preprocess \
  --input-dir preprocess/floodgate/2020 \
  --input-file-packed \
  --test-ratio 0.2 \
  --output-gcs \
  --gcs-bucket-name maou-test-dousu \
  --gcs-base-path large_test_conservative \
  --batch-size 512 \             # より小さく
  --epoch 20 \
  --gpu cuda:0 \
  --compilation true \
  --dataloader-workers 12 \
  --prefetch-factor 1 \
  --pin-memory \
  --learning-ratio 0.0005 \      # より低く
  --gce-parameter 0.7 \
  --momentum 0.9 \
  --value-loss-ratio 2.0         # Value lossの重みを増加
```

## 期待される改善

### 学習率0.001の場合
- Value accuracy: 0.54 → 0.60-0.70 (10-20エポック後)
- Policy accuracy: 0.014 → 0.05-0.10
- Loss減少: より安定した減少曲線

### 学習率0.0001の場合
- より緩やかだが安定した学習
- 収束まで時間がかかるが，より良い最終性能

## 次のステップ

1. **まず学習率0.001で10エポック試す**
   - Value accuracyが0.54から動くか確認
   - Lossが安定して減少するか確認

2. **結果を見て調整**
   - 学習が速すぎる → 0.0005に下げる
   - 学習が遅すぎる → 0.002に上げる
   - Value headが動かない → Loss functionの変更を検討

3. **TensorBoardで監視**
   - Value predictionの分布を確認
   - 平均値から脱出できているか確認

## 技術的詳細

### なぜMSE lossは二峰性分布に不適切か

数学的説明:
```
データ: 50% が 0.0, 50% が 1.0
予測値をpとすると，MSE loss:

L(p) = 0.5 * (p - 0)² + 0.5 * (p - 1)²
     = 0.5 * p² + 0.5 * (p² - 2p + 1)
     = p² - p + 0.5

dL/dp = 2p - 1 = 0
→ p = 0.5 (最適解)
```

つまり，**データ平均を予測するのが数学的に最適**であり，モデルは正しく動作している．

### 小さいバッチサイズが有効な理由

- 大きいバッチ: 勾配 ≈ 期待値（分散小）→ 局所最適解に収束
- 小さいバッチ: 勾配にノイズ → 局所最適解から脱出可能
- Mini-batch SGDのノイズは正則化効果を持つ

## まとめ

**問題:**
- データの二峰性分布とMSE lossの組み合わせ
- 高すぎる学習率（0.05）
- 大きすぎるバッチサイズ（15000）

**解決策:**
1. 学習率を0.001に削減 【最重要】
2. バッチサイズを1024以下に削減
3. Pre-trainedモデルを使わない
4. （オプション）Loss functionの変更を検討

**期待される結果:**
- Value accuracy: 0.54 → 0.60-0.70
- 安定した学習曲線
- Policy accuracyの改善
