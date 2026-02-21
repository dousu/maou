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

## コンピュート側の差異分析

workers=0 テストとバッチサイズ確認により，**速度差の原因は 100% コンピュート側**にある．以下が Stage 2 と Stage 3 の計算パスの差異:

### モデル Forward Pass

| 項目 | Stage 2 | Stage 3 |
|------|---------|---------|
| Backbone | `HeadlessNetwork.forward_features()` | 同左(共通) |
| Policy Head | `LegalMovesHead`: `Linear(input_dim, 1496)` | `PolicyHead`: `Linear(input_dim, 1496)` |
| Value Head | なし(dummy zeros 生成) | `ValueHead`: `Linear(input_dim, 1)` |

Forward pass は**ほぼ同一**．Stage 3 は ValueHead が追加されているが，1 次元出力なので計算量は無視できる．

### Loss 計算(最大の差異)

| 項目 | Stage 2 | Stage 3 |
|------|---------|---------|
| Policy Loss | `BCEWithLogitsLoss(logits[1024, 1496], targets[1024, 1496])` | `GCELoss(logits[1024, 1496], targets[1024])` |
| 損失計算方式 | **multi-label**: 各 1496 要素が独立した binary classification | **single-label**: softmax → one-hot → cross-entropy |
| 勾配の要素数 | **1,532,416** 個(1024 × 1496，全要素に独立した勾配) | softmax の Jacobian は sparse で効率的 |
| Value Loss | `MSELoss(dummy, dummy)` — 実質ゼロコスト | `MSELoss(value, target)` — 1024 要素 |
| Legal Move Mask | なし | あり(masked_fill + 全ゼロ行検出) |

**BCEWithLogitsLoss の backward pass が重い可能性**:
- 1,532,416 個の要素それぞれに sigmoid + log の勾配が伝播
- `reduction='mean'` により全要素の平均 → backward で勾配が全要素に均等分配
- GCELoss は softmax ベースで勾配計算が効率的(PyTorch の内部最適化)

### Stage2F1Callback の計算

`on_batch_end` で毎バッチ以下を実行(GPU 同期は排除済みだが計算自体は残る):
- `sigmoid(outputs[1024, 1496])` → `> 0.5` → bool 変換
- TP/FP/FN 計算: AND/NOT + float cast + sum → 6 カーネル
- Precision/Recall/F1 計算: 4 カーネル
- `both_empty` 判定 + `torch.where`: 3 カーネル
- **合計 ~15 CUDA カーネル/バッチ**

Stage 3 の `LoggingCallback` にはこの計算がない．

## 残る仮説・次の調査候補

### 最重要: BCEWithLogitsLoss backward の計算量

1496 次元 multi-label の `BCEWithLogitsLoss` backward が Stage 3 の `GCELoss` backward より 6-7 倍遅いことは理論的にあり得るか？ 以下の確認が必要:

**確認方法**:
```python
# 単体ベンチマーク: BCEWithLogitsLoss vs GCELoss の forward+backward 時間
import torch, time

logits = torch.randn(1024, 1496, device='cuda', requires_grad=True)
targets_bce = torch.randint(0, 2, (1024, 1496), device='cuda').float()
targets_gce = torch.randint(0, 1496, (1024,), device='cuda')

# Stage 2: BCEWithLogitsLoss
loss_fn_bce = torch.nn.BCEWithLogitsLoss()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    loss = loss_fn_bce(logits, targets_bce)
    loss.backward()
    logits.grad = None
torch.cuda.synchronize()
bce_time = time.perf_counter() - start

# Stage 3: GCELoss (softmax + one_hot + custom)
# ... 同様のベンチマーク
```

### 高優先度

1. **GPU 利用率の直接測定** — 学習中の GPU utilization を確認
   ```bash
   nvidia-smi dmon -s u -d 1
   ```
   - GPU Util < 30%: GPU がアイドル状態(データ供給以外の原因)
   - GPU Util > 80%: GPU 計算が飽和(BCEWithLogitsLoss の backward が重い)

2. **`nsys profile` での GPU タイムライン分析** — カーネル実行，backward pass の内訳を可視化

3. **Stage2F1Callback の計算コスト** — 毎バッチ ~15 CUDA カーネルの実行時間を計測．`on_batch_end` 内の計算を N バッチごとに間引くことで効果を確認

### 中優先度

4. **`torch.compile` の効果** — Stage 2 で `torch.compile()` を有効にし，BCEWithLogitsLoss + F1 計算のカーネル融合による高速化を確認
5. **mixed precision の確認** — `autocast` 下での BCEWithLogitsLoss の精度と速度のバランス

## perf-engineer による調査アプローチの振り返り

> 全ての改善が「Stage 2 は I/O バウンドで，ファイル I/O を改善すれば速くなる」という仮説に基づいていた．しかし:
>
> 1. Stage 2 の forward pass 計算量は Stage 3 以下 → forward がボトルネックではない
> 2. 4 つの I/O 改善が全て効かない → I/O ボトルネックでもない
> 3. workers=0 でも速度が同じ → DataLoader は完全に無関係
> 4. バッチサイズは同一 → 設定の問題ではない
>
> **次の調査は loss 関数の backward pass と F1 Callback の計算コストに焦点を当てるべきである．**

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
- `src/maou/domain/loss/loss_fn.py` — LegalMovesLoss(BCEWithLogitsLoss), GCELoss
- `src/maou/app/learning/network.py` — LegalMovesHead, PolicyHead, HeadlessNetwork
- `rust/maou_rust/src/maou_io.rs` — load_feather_file(), load_preprocessing_feather()
