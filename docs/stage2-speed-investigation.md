# Stage 2 速度改善 調査報告

## 問題の概要

Stage 2(合法手学習，~40M レコード)の学習スループットが Stage 3(棋譜学習)の約 1/7 にとどまる．

| 項目 | Stage 2 | Stage 3 |
|------|---------|---------|
| スループット | ~3 it/s | ~20 it/s |
| データ規模 | ~40M レコード | ~37M レコード |
| ファイル数 | 404 | 37 |
| 行/ファイル | ~100K | ~1M |
| バッチサイズ | 1024 | 1024 |

同一設定(`--batch-size 1024 --dataloader-workers 8 --pin-memory --epochs 1`)で 6-7 倍の差が発生する．

## 実行環境

- **GPU**: NVIDIA A100
- **DataLoader workers**: 8(ただし workers=0 でも速度変わらず)
- **batch_size**: 1024(Stage 2・Stage 3 とも同一)
- **pin_memory**: True
- **persistent_workers**: True
- **multiprocessing context**: spawn
- **PyTorch**: DataLoader with IterableDataset, `batch_size=None`

## 決定的発見: DataLoader はボトルネックではない

ユーザーの計測により，**workers 数を変えても速度が変わらない**ことが確認された:

| 設定 | Stage 2 | Stage 3 |
|------|---------|---------|
| workers=0 | **3 it/s** | **19 it/s** |
| workers=8 | **3 it/s** | **20 it/s** |

**結論**: ボトルネックは **100% コンピュート側**(モデル forward/backward，loss 計算，データ変換等)にある．DataLoader/I/O 系の最適化は効果がない．

## 実施した改善と結果

4 つの改善を実施したが，**すべて効果なし**(3 it/s のまま)．すべて I/O またはデータ転送に関する改善であり，コンピュートがボトルネックだったため効果がなかった．

### 1. マルチワーカー DataLoader 化

**コミット**: `b5dde92 perf(learning): enable multi-worker DataLoader for Stage 2 streaming`

Stage 2 を `workers=0`(メインプロセスのみ)から `workers=N` のマルチワーカー DataLoader に移行．ワーカーファイル分割，spawn コンテキスト対応，persistent_workers を追加．

**perf-engineer 評価**: I/O 並列化の改善．ボトルネックが I/O ではないため効果なし．

**結果**: 効果なし（3 it/s のまま）

### 2. TrainingLoop 統合(CUDA stream overlap)

**コミット**: `3142063 feat(learning): integrate Stage 2 with TrainingLoop for CUDA stream overlap`

Stage 2 を `SingleStageTrainingLoop` から `TrainingLoop`(Stage 3 と同一)に移行．`_iterate_cuda_overlap()` による非同期 H2D 転送，tqdm，gradient clipping，callback アーキテクチャを適用．

主な追加コンポーネント:
- `Stage2ModelAdapter` — HeadlessNetwork + LegalMovesHead を `(policy, value)` 出力に変換
- `Stage2TrainingLoop` — `_compute_policy_loss` をオーバーライドし BCEWithLogitsLoss を直接使用
- `Stage2StreamingAdapter` — 2-tuple → 3-tuple データ変換
- `Stage2F1Callback` — サンプル平均 F1 スコア計算

**perf-engineer 評価**: H2D 転送のオーバーラップ改善．ボトルネックが I/O ではないため効果なし．

**結果**: 効果なし（3 it/s のまま）

### 3. F1 Callback GPU 同期排除

**コミット**: `f05386c perf(learning): eliminate per-batch GPU sync in Stage2F1Callback`

`Stage2F1Callback.on_batch_end()` で毎バッチ `.item()` を 2 回呼んでいた GPU 同期を排除．GPU テンソル上で F1/loss を蓄積し，エポック終了時のみ `.item()` で CPU に転送する方式に変更．

**perf-engineer 評価**: `.item()` のオーバーヘッドは ~20-100μs/バッチ．batch 時間 ~330ms に対して <0.1% であり，6-7x の差の主因ではない．GPU パイプラインの衛生改善として有効だが速度差の原因ではない．

**結果**: 効果なし（3 it/s のまま）

### 4. 小ファイル結合(10 ファイル)

**コミット**: `4e172f0 perf(learning): concatenate small Stage 2 files to reduce I/O stalls`

`StreamingStage2Dataset.__iter__` で 10 ファイル(~100K 行 × 10 = ~1M 行)をバッファリングし `ColumnarBatch.concatenate()` で結合してからバッチ生成．バッチ/ファイルロード比率を ~97 → ~970 に改善(Stage 3 と同等)．

**perf-engineer 評価**: I/O パターンがボトルネックではなかったことを示唆．

**結果**: 効果なし（3 it/s のまま）

## 排除された仮説

### I/O ボトルネック(workers=0 テストで完全否定)

workers=0 と workers=8 で速度が同一(3 it/s)であることから，**DataLoader/I/O は一切ボトルネックではない**ことが確定．上記 4 つの改善が全て効かなかった理由はこれである．

### バッチサイズの不一致(ユーザー確認で否定)

perf-engineer が最も可能性が高いと評価した仮説だが，ユーザーから Stage 2・Stage 3 ともに `batch_size=1024` で同一と確認され否定された．

### workers が反映されていない

コードフロー精査済み．正しく伝搬されている(ただしそもそも workers は速度に影響しない)．

### GPU 同期ボトルネック

`.item()` 修正済み．perf-engineer 分析: ~20-100μs/バッチは <0.1% で主因ではない．

### Rust I/O バックエンド差異

perf-engineer が Rust ソースコード(`rust/maou_rust/src/maou_io.rs`)を確認．両者は同一の内部関数 `maou_io_core::arrow_io::load_feather()` を呼んでおり差異なし．

### ファイルサイズの違い

10 ファイル結合でバッチ/ファイルロード比率を Stage 3 と同等にしたが改善なし．I/O パターンは無関係．

### BCEWithLogitsLoss backward が主因

perf-engineer の分析により否定．勾配テンソルの形状が Stage 2・Stage 3 とも [1024, 1496] で同一のため，backward の Linear GEMM コストと backbone backward コストは同一．さらに Stage 3 は masked_fill + log_softmax + normalize_policy_targets が追加されており，**Stage 3 の方がむしろ計算量が多い．** BCEWithLogitsLoss の `σ(x)-y` と KLDivLoss+log_softmax の backward はどちらも element-wise 操作で GPU 上では trivial であり，6-7x の差は理論的にあり得ない．

## コンピュート側の差異分析

workers=0 テストとバッチサイズ確認により，**速度差の原因は 100% コンピュート側**にある．以下が Stage 2 と Stage 3 の計算パスの差異:

### モデル Forward Pass

| 項目 | Stage 2 | Stage 3 |
|------|---------|---------|
| Backbone | `HeadlessNetwork.forward_features()` | 同左(共通) |
| Policy Head | `LegalMovesHead`: `Linear(input_dim, 1496)` | `PolicyHead`: `Linear(input_dim, 1496)` |
| Value Head | なし(dummy zeros 生成) | `ValueHead`: `Linear(input_dim, 1)` |

Forward pass は**ほぼ同一**．Stage 3 は ValueHead が追加されているが，1 次元出力なので計算量は無視できる．

### Loss 計算

**訂正**: Stage 3 は `GCELoss` ではなく `KLDivLoss` を使用している．`LossOptimizerFactory.create_loss_functions()` (`setup.py:761-785`)で `gce_parameter` は無視されている．

| 項目 | Stage 2 | Stage 3 |
|------|---------|---------|
| Policy Loss | `BCEWithLogitsLoss(logits[1024, 1496], targets[1024, 1496])` | `KLDivLoss(log_softmax(masked_logits), normalized_targets)` |
| 計算ステップ | 1 fused kernel | masked_fill + log_softmax + normalize_policy_targets + KLDivLoss (5+ kernels) |
| 勾配形状 | [1024, 1496] | [1024, 1496] |
| 勾配計算 | `σ(x) - y` (element-wise) | `-targets/N` (element-wise) + log_softmax backward |
| Value Loss | `MSELoss(dummy, dummy)` — 実質ゼロコスト | `MSELoss(value, target)` — 1024 要素 |
| Legal Move Mask | なし | あり(masked_fill + 全ゼロ行検出) |

**perf-engineer 分析**: 勾配テンソルの形状が同一([1024, 1496])のため，backward の Linear GEMM コストと backbone backward コストは同一．Stage 3 の方がむしろ計算量が多い(masked_fill + log_softmax + normalize が追加)．**BCEWithLogitsLoss backward が 6-7x 遅いことは理論的にあり得ない．**

### Stage2F1Callback の計算

`on_batch_end` で毎バッチ以下を実行(GPU 同期は排除済みだが計算自体は残る):
- `sigmoid(outputs[1024, 1496])` → `> 0.5` → bool 変換
- TP/FP/FN 計算: AND/NOT + float cast + sum → 6 カーネル
- Precision/Recall/F1 計算: 4 カーネル
- `both_empty` 判定 + `torch.where`: 3 カーネル
- **合計 ~15 CUDA カーネル/バッチ**

Stage 3 の `LoggingCallback` にはこの計算がない．ただし perf-engineer 分析では合計 ~0.2ms/batch で，333ms の差を説明できない．

## 残る仮説・次の調査候補

### 理論分析の限界

perf-engineer の分析により，**理論的に特定可能な差異(Loss 関数，F1 Callback，Forward Pass)はいずれも 6-7x の速度差を説明できない**ことが判明．Stage 3 の方がむしろ計算量が多い．

原因特定には**経験的切り分け(アブレーションテスト)**が必要．

### 最重要: 経験的切り分けテスト (perf-engineer 推奨)

以下の 3 テストのうち，**テスト B が最も情報量が多い**:

#### テスト B: Training step 全体をスキップ(最推奨)

データイテレーションのみ計測し，compute vs data を完全に切り分ける:

```python
# run_epoch の training ループを一時的に変更:
for batch_idx, context in self._iterate_with_transfer(dataloader):
    pass  # 何もしない — データイテレーションのみ計測
```

- → 20 it/s になれば training step (forward+loss+backward) がボトルネック
- → 3 it/s のままならデータイテレーション自体がボトルネック(workers=0 テストと矛盾するため，さらに深い調査が必要)

#### テスト A: Loss を dummy に置換

```python
# Stage2TrainingLoop._compute_policy_loss を一時的に変更:
def _compute_policy_loss(self, context):
    return context.outputs_policy.mean()  # Dummy loss
```

- → 20 it/s になれば Loss がボトルネック(理論と矛盾するが事実を優先)
- → 3 it/s のままなら Loss は無関係

#### テスト C: Stage2F1Callback を無効化

Stage 2 のコールバックリストから `Stage2F1Callback` を除外して実行:

- → 速度が変われば callback がボトルネック
- → 速度が変わらなければ callback は無関係

### 高優先度

1. **GPU 利用率の直接測定** — 学習中の GPU utilization を確認
   ```bash
   nvidia-smi dmon -s u -d 1
   ```
   - GPU Util < 30%: GPU がアイドル状態
   - GPU Util > 80%: GPU 計算が飽和

2. **`nsys profile` での GPU タイムライン分析** — カーネル実行，backward pass の内訳を可視化

## perf-engineer による調査アプローチの振り返り

> 全ての改善が「Stage 2 は I/O バウンドで，ファイル I/O を改善すれば速くなる」という仮説に基づいていた．しかし:
>
> 1. Stage 2 の forward pass 計算量は Stage 3 以下 → forward がボトルネックではない
> 2. 4 つの I/O 改善が全て効かない → I/O ボトルネックでもない
> 3. workers=0 でも速度が同じ → DataLoader は完全に無関係
> 4. バッチサイズは同一 → 設定の問題ではない
>
> **理論的に特定可能な差異はいずれも 6-7x を説明できない．次の調査はアブレーションテスト(経験的切り分け)で原因を絞り込むべきである．**

## 適用済みコミット一覧

Stage 2 速度改善に関連する `update-model` ブランチ上のコミット:

| コミット | 種別 | 内容 | 速度効果 |
|----------|------|------|----------|
| `6b7191e` | fix | numpy-backed tensor の clone 対応(マルチプロセス) | 前提修正 |
| `a09e73e` | fix | streaming モードでの OOM 回避 | 前提修正 |
| `7a2ba7b` | fix | worker ログレベルを DEBUG に変更 | ログ改善 |
| `e05759f` | refactor | `_cap_total_workers` 削除 | 制約除去 |
| `b5dde92` | perf | Stage 2 マルチワーカー DataLoader 化 | なし |
| `3142063` | feat | Stage 2 TrainingLoop 統合(CUDA stream overlap) | なし |
| `f05386c` | perf | F1 Callback GPU 同期排除 | なし |
| `4e172f0` | perf | 小ファイル結合(10 ファイル × 100K 行) | なし |

## 関連ファイル

- `src/maou/app/learning/streaming_dataset.py` — StreamingStage2Dataset, Stage2StreamingAdapter
- `src/maou/app/learning/training_loop.py` — TrainingLoop, Stage2TrainingLoop
- `src/maou/app/learning/callbacks.py` — Stage2F1Callback
- `src/maou/app/learning/multi_stage_training.py` — Stage2ModelAdapter, run_stage2_with_training_loop()
- `src/maou/interface/learn.py` — _run_stage2_streaming(), learn_multi_stage()
- `src/maou/domain/data/columnar_batch.py` — ColumnarBatch.concatenate()
- `src/maou/domain/loss/loss_fn.py` — LegalMovesLoss(BCEWithLogitsLoss), GCELoss(未使用)
- `src/maou/app/learning/setup.py` — LossOptimizerFactory(Stage 3 は KLDivLoss を使用)
- `src/maou/app/learning/network.py` — LegalMovesHead, PolicyHead, HeadlessNetwork
- `rust/maou_rust/src/maou_io.rs` — load_feather_file(), load_preprocessing_feather()
