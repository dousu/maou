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

4 つの改善を実施したが，**すべて効果なし**(3 it/s のまま)．

### 1. マルチワーカー DataLoader 化

**コミット**: `b5dde92 perf(learning): enable multi-worker DataLoader for Stage 2 streaming`

Stage 2 を `workers=0`(メインプロセスのみ)から `workers=N` のマルチワーカー DataLoader に移行．ワーカーファイル分割，spawn コンテキスト対応，persistent_workers を追加．

**perf-engineer 評価**: I/O 並列化の改善．ただしボトルネックが I/O でなければ効果なし．

**結果**: 効果なし（3 it/s のまま）

### 2. TrainingLoop 統合(CUDA stream overlap)

**コミット**: `3142063 feat(learning): integrate Stage 2 with TrainingLoop for CUDA stream overlap`

Stage 2 を `SingleStageTrainingLoop` から `TrainingLoop`(Stage 3 と同一)に移行．`_iterate_cuda_overlap()` による非同期 H2D 転送，tqdm，gradient clipping，callback アーキテクチャを適用．

主な追加コンポーネント:
- `Stage2ModelAdapter` — HeadlessNetwork + LegalMovesHead を `(policy, value)` 出力に変換
- `Stage2TrainingLoop` — `_compute_policy_loss` をオーバーライドし BCEWithLogitsLoss を直接使用
- `Stage2StreamingAdapter` — 2-tuple → 3-tuple データ変換
- `Stage2F1Callback` — サンプル平均 F1 スコア計算

**perf-engineer 評価**: H2D 転送のオーバーラップ改善．I/O がボトルネックでなければ効果なし．

**結果**: 効果なし（3 it/s のまま）

### 3. F1 Callback GPU 同期排除

**コミット**: `f05386c perf(learning): eliminate per-batch GPU sync in Stage2F1Callback`

`Stage2F1Callback.on_batch_end()` で毎バッチ `.item()` を 2 回呼んでいた GPU 同期を排除．GPU テンソル上で F1/loss を蓄積し，エポック終了時のみ `.item()` で CPU に転送する方式に変更．

**perf-engineer 評価**: `.item()` のオーバーヘッドは ~20-100μs/バッチ．batch 時間 ~330ms に対して <0.1% であり，6-7x の差の主因ではない．ただし GPU パイプラインに不必要な同期ポイントを入れるのは原則として悪いため，衛生改善として有効．

**結果**: 効果なし（3 it/s のまま）

### 4. 小ファイル結合(10 ファイル)

**コミット**: `4e172f0 perf(learning): concatenate small Stage 2 files to reduce I/O stalls`

`StreamingStage2Dataset.__iter__` で 10 ファイル(~100K 行 × 10 = ~1M 行)をバッファリングし `ColumnarBatch.concatenate()` で結合してからバッチ生成．バッチ/ファイルロード比率を ~97 → ~970 に改善(Stage 3 と同等)．

**perf-engineer 評価**: 期待効果は高い(~50-100%)と評価していたが，効果なし．I/O パターンがボトルネックではなかったことを示唆．

**結果**: 効果なし（3 it/s のまま）

## perf-engineer 総合分析: なぜ 4 つの改善がすべて効果なしだったか

### 核心: Stage 2 の計算は Stage 3 より軽い

perf-engineer のコード分析により，Stage 2 の計算パスは Stage 3 よりも **明確に軽量** であることが確認された:

| 項目 | Stage 2 | Stage 3 |
|------|---------|---------|
| Policy Loss | `BCEWithLogitsLoss(logits, targets)` — 1 カーネル | `masked_fill → log_softmax → normalize → KLDivLoss` — 5+ カーネル |
| Legal Move Mask | なし (None) | あり (mask 処理 + 全ゼロ行検出) |
| Value Loss | `MSELoss(outputs, dummy_zeros)` — 実質ゼロコスト | `MSELoss(outputs, result_value)` |
| コールバック | LoggingCallback + Stage2F1Callback | LoggingCallback |

**同じバッチサイズなら Stage 2 の方が速いはず**であり，3 it/s vs 20 it/s は**矛盾**している．

### 改善が効かなかった根本理由

4 つの改善はすべて I/O またはデータ転送に関するものだった:

1. マルチワーカー化 → I/O 並列化
2. TrainingLoop 統合 → H2D 転送オーバーラップ
3. F1 Callback GPU 同期 → GPU パイプライン改善
4. ファイル結合 → ファイルロード頻度削減

**すべてが効かないということは，ボトルネックは I/O でも GPU 同期でもない．** しかし計算量は Stage 3 以下なので，計算がボトルネックとも考えにくい．

**残る可能性は設定/計測の問題である．**

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

`.item()` による GPU 同期は修正したが速度改善なし．perf-engineer 分析: バッチ処理時間(~330ms)に対して同期コスト(~20-100μs)は <0.1% であり，6-7x の差を説明できない．

### Rust I/O バックエンド差異

perf-engineer が Rust ソースコード(`rust/maou_rust/src/maou_io.rs`)を確認:

```rust
// Stage 2: load_feather_file
fn load_feather_file(py: Python, file_path: String) -> PyResult<PyObject> {
    let batch = py.allow_threads(|| maou_io_core::arrow_io::load_feather(&file_path));
    ...
}

// Stage 3: load_preprocessing_feather
fn load_preprocessing_feather(py: Python, file_path: String) -> PyResult<PyObject> {
    let batch = py.allow_threads(|| maou_io_core::arrow_io::load_feather(&file_path));
    ...
}
```

**両者は同一の内部関数 `maou_io_core::arrow_io::load_feather()` を呼んでおり，差異は存在しない．** Stage 2 専用 Rust ローダーの開発は不要．

### ファイルサイズの違い(I/O パターン)

10 ファイル結合でバッチ/ファイルロード比率を Stage 3 と同等にしたが改善なし．perf-engineer 分析: **I/O がそもそもボトルネックではなかった**ことを示唆．

## 残る仮説・次の調査候補

### 最重要 (perf-engineer 推奨)

#### 仮説 A: バッチサイズの不一致

perf-engineer が**最も可能性が高い**と評価する仮説．

Stage 2 のバッチサイズは `stage2_batch_size or batch_size` で決定される(`src/maou/interface/learn.py:1151`)．`--stage2-batch-size` CLI オプションが設定されている場合，Stage 2 だけ小さいバッチサイズで実行されている可能性がある．

- バッチサイズが小さい → GPU 利用率が低い → it/s が低い
- 例: Stage 2 が `batch_size=1024`，Stage 3 が `batch_size=4096` の場合:
  - Stage 2: ~39,500 batches × 3 it/s = ~13,000 秒
  - Stage 3: ~9,000 batches × 20 it/s = ~450 秒
  - **これで速度差が完全に説明できる**

**確認方法**:
```bash
# 実行コマンドの --batch-size と --stage2-batch-size の値を確認
# または学習開始時のログから effective batch size を確認
```

#### 仮説 B: tqdm の "it" の定義が異なる

Stage 2 と Stage 3 で total batches が大きく異なれば，it/s の数値は直接比較できない．tqdm の progress bar に表示される total 数を確認する必要がある．

**確認方法**:
```bash
# tqdm のプログレスバーの total 数を Stage 2 と Stage 3 で比較
# Stage 2: total batches = ~40M / batch_size
# Stage 3: total batches = ~37M / batch_size
```

### 高優先度

1. **`workers=0` での速度比較** — `workers=0`(メインプロセス)で速度が同じなら I/O は完全に無関係(確認)
   ```bash
   uv run maou learn --stage 2 --dataloader-workers 0 --epochs 1
   ```

2. **GPU 利用率の直接測定** — 学習中の GPU utilization を確認
   ```bash
   nvidia-smi dmon -s u -d 1
   ```
   - GPU Util < 30%: データ供給不足またはバッチサイズ不足
   - GPU Util > 80%: GPU 計算がボトルネック

3. **`nsys profile` での GPU タイムライン分析** — カーネル実行，H2D 転送，GPU idle 時間を可視化

### 中優先度

4. **データ変換コスト** — Stage 2 の `legal_moves_label` は `uint8→float32` 変換(4 倍膨張，5.8MB/バッチ)．Stage 3 は `float16` の `.clone()`(2.9MB)．ただし変換はワーカープロセス内で実行されるため GPU を直接阻害しない

5. **IPC 転送サイズ** — Stage 2 の `.float()` 変換により IPC Queue 経由の転送サイズが Stage 3 より大きい(5.8MB vs 2.9MB)．ただし perf-engineer はこれも主因ではないと評価

### 低優先度

6. **`prefetch_factor` の調整** — 2 → 3-4 に増加．perf-engineer 評価: 期待効果は低〜中(+10-20%)．PyTorch #97432(pinned memory leak)のリスクあり
7. **ワーカー内スレッド先読み** — perf-engineer 評価: 策 3(ファイル結合)がよりシンプルで推奨(実装済み)

## perf-engineer による調査アプローチの振り返り

> 全ての改善が「Stage 2 は I/O バウンドで，ファイル I/O を改善すれば速くなる」という仮説に基づいていた．しかし:
>
> 1. Stage 2 の計算は Stage 3 より軽い → 計算ボトルネックではない
> 2. 4 つの I/O 改善が全て効かない → I/O ボトルネックでもない
> 3. **残る可能性は設定/計測の問題**
>
> 最も可能性が高いのは**バッチサイズの不一致**である．

次の調査では，まず**実行時の設定値(バッチサイズ，total batches)を正確に把握**してから改善策を検討すべきである．

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
- `rust/maou_rust/src/maou_io.rs` — load_feather_file(), load_preprocessing_feather()
