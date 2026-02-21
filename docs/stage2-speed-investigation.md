# Stage 2 速度改善 調査報告

## 問題の概要

Stage 2(合法手学習，~40M レコード)の学習スループットが Stage 3(棋譜学習)の約 1/7 にとどまる．

| 項目 | Stage 2 | Stage 3 |
|------|---------|---------|
| スループット | ~3 it/s | ~20 it/s |
| データ規模 | ~40M レコード | ~37M レコード |
| ファイル数 | 404 | 37 |
| 行/ファイル | ~100K | ~1M |

同一設定(`--dataloader-workers 8 --pin-memory --epochs 1`)で 6-7 倍の差が発生する．

## 実行環境

- **GPU**: NVIDIA A100
- **DataLoader workers**: 8
- **pin_memory**: True
- **persistent_workers**: True
- **multiprocessing context**: spawn
- **PyTorch**: DataLoader with IterableDataset, `batch_size=None`

## 実施した改善と結果

### 1. マルチワーカー DataLoader 化

**コミット**: `b5dde92 perf(learning): enable multi-worker DataLoader for Stage 2 streaming`

Stage 2 を `workers=0`(メインプロセスのみ)から `workers=N` のマルチワーカー DataLoader に移行．ワーカーファイル分割，spawn コンテキスト対応，persistent_workers を追加．

**結果**: 効果なし（3 it/s のまま）

### 2. TrainingLoop 統合(CUDA stream overlap)

**コミット**: `3142063 feat(learning): integrate Stage 2 with TrainingLoop for CUDA stream overlap`

Stage 2 を `SingleStageTrainingLoop` から `TrainingLoop`(Stage 3 と同一)に移行．`_iterate_cuda_overlap()` による非同期 H2D 転送，tqdm，gradient clipping，callback アーキテクチャを適用．

主な追加コンポーネント:
- `Stage2ModelAdapter` — HeadlessNetwork + LegalMovesHead を `(policy, value)` 出力に変換
- `Stage2TrainingLoop` — `_compute_policy_loss` をオーバーライドし BCEWithLogitsLoss を直接使用
- `Stage2StreamingAdapter` — 2-tuple → 3-tuple データ変換
- `Stage2F1Callback` — サンプル平均 F1 スコア計算

**結果**: 効果なし（3 it/s のまま）

### 3. F1 Callback GPU 同期排除

**コミット**: `f05386c perf(learning): eliminate per-batch GPU sync in Stage2F1Callback`

`Stage2F1Callback.on_batch_end()` で毎バッチ `.item()` を 2 回呼んでいた GPU 同期を排除．GPU テンソル上で F1/loss を蓄積し，エポック終了時のみ `.item()` で CPU に転送する方式に変更．

**perf-engineer 分析**: `.item()` のオーバーヘッドは ~20-100μs/バッチ．batch 時間 ~330ms に対して <0.1% であり，6-7x の差の主因ではない．ただし GPU パイプラインの衛生改善として有効．

**結果**: 効果なし（3 it/s のまま）

### 4. 小ファイル結合(10 ファイル)

**コミット**: `4e172f0 perf(learning): concatenate small Stage 2 files to reduce I/O stalls`

`StreamingStage2Dataset.__iter__` で 10 ファイル(~100K 行 × 10 = ~1M 行)をバッファリングし `ColumnarBatch.concatenate()` で結合してからバッチ生成．バッチ/ファイルロード比率を ~97 → ~970 に改善(Stage 3 と同等)．

**結果**: 効果なし（3 it/s のまま）

## 排除された仮説

### workers が反映されていない

コードフローを精査し，CLI の `--dataloader-workers` が Stage 2 DataLoader に正しく伝搬されていることを確認済み:

```
learn_multi_stage(dataloader_workers=8)
  → _run_stage2_streaming(dataloader_workers=8)
    → DataLoaderFactory.create_streaming_dataloaders(dataloader_workers=8)
      → _clamp_workers(8, n_files=404, ...)
        → DataLoader(num_workers=8)
```

### GPU 同期ボトルネック

`.item()` による GPU 同期は修正したが速度改善なし．バッチ処理時間(~330ms)に対して同期コスト(~20-100μs)は無視できるレベルであり，根本原因ではない．

### Rust I/O バックエンド差異

perf-engineer がソースコード(`rust/maou_rust/src/maou_io.rs`)を確認:

- Stage 2: `load_feather_file()` → `maou_io_core::arrow_io::load_feather()`
- Stage 3: `load_preprocessing_feather()` → `maou_io_core::arrow_io::load_feather()`

**同一の内部関数を呼んでおり，差異は存在しない．**

### ファイルサイズの違い

Stage 2(100K 行/ファイル)と Stage 3(1M 行/ファイル)のファイルサイズ差を，10 ファイル結合で解消したが改善なし．I/O パターンが原因ではなかったか，結合のアプローチが不十分な可能性がある．

## 残る仮説・次の調査候補

### 高優先度

1. **`workers=0` での速度比較** — I/O ボトルネックの切り分け．`workers=0`(メインプロセス)で速度が同じなら I/O ではなく compute が律速
2. **`nsys profile` での GPU タイムライン分析** — 実際のカーネル実行，H2D 転送，GPU idle 時間を可視化．ボトルネックの所在を特定
3. **バッチサイズの実効値確認** — `batch_size=None`(DataLoader 側)で `_yield_stage2_batches` が生成する実効バッチサイズが適切か確認

### 中優先度

4. **モデル構造の違い** — Stage 2 は `HeadlessNetwork + LegalMovesHead`，Stage 3 は `HeadlessNetwork + PolicyHead + ValueHead`．forward pass の計算量差を確認
5. **データ変換コスト** — Stage 2 の `legal_moves_label` は `uint8→float32` 変換(4 倍膨張，5.8MB/バッチ)．Stage 3 は `float16` の `.clone()`(2.9MB)．IPC 転送サイズの差が影響する可能性
6. **`Stage2StreamingAdapter` のオーバーヘッド** — `dummy_value` 生成と tuple 再構成のコスト(理論上は ~11μs/バッチで無視できるが要確認)

### 低優先度

7. **`prefetch_factor` の調整** — 2 → 3-4 に増加．PyTorch #97432(pinned memory leak)のリスクあり
8. **ワーカー内スレッド先読み** — `StreamingFileSource` で現ファイル yield 中に次ファイルをバックグラウンドロード

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
- `src/maou/interface/learn.py` — _run_stage2_streaming()
- `src/maou/domain/data/columnar_batch.py` — ColumnarBatch.concatenate()
- `rust/maou_rust/src/maou_io.rs` — load_feather_file(), load_preprocessing_feather()
