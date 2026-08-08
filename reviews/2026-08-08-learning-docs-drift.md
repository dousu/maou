---
status: pending
applied_in: ""
date: 2026-08-08
target:
  - docs/performance.md
  - README.md
  - docs/architecture.md
  - docs/loss-functions.md
  - docs/rust-backend.md
  - docs/commands/utility_benchmark_training.md
  - docs/adr-003-training-performance-optimization-attempts.md
  - docs/stage3-hang-investigation.md
  - docs/torch-compile-state-dict.md
  - docs/commands/learn_model.md
risk: low
reversibility: trivial (prose-only, per-section revert)
---

# `src/maou/app/learning` を説明する doc の drift 修正 (14 件)

## Trigger

`/audit-and-fix src/maou/app/learning high` の step 4 (doc accuracy)．
`src/maou/app/learning/` の 17 モジュールと，そこで定義されるクラス・
関数名を `docs/` / `README.md` / `CLAUDE.md` / `AGENTS.md` に対して
突き合わせ，コード側で真偽が判定できる主張のみを検証した．

WRONG 9 件 / STALE 5 件．以下の主張はモデルの推測ではなく，全て
`src/...:line` で裏を取っている (主要なものは監査時に再確認済み)．

**最も重いのは #1** — 削除済みモジュールの性能数値が「有効な既定機能」
として残っており，読者は存在しない機能を前提にチューニングすることに
なる．

## Proposed change

### 1. WRONG `docs/performance.md:35-43` — 削除済み機能の性能数値

`gpu_prefetcher.py` は存在せず (`ls` で不在，`grep -rn "gpu_prefetch\|GPUPrefetcher" src/` は
`setup.py:591` のコメント「DataPrefetcher 除去後は…」1 件のみ)，
`TrainingLoop.__init__` (`training_loop.py:60-83`) に
`enable_gpu_prefetch` / `gpu_prefetch_buffer_size` 引数は無い．
引用されているスループット数値は削除済みコードに帰属している．

**Before**
```markdown
### GPU Prefetching (Auto-Enabled)
Automatic GPU prefetching overlaps data loading with computation. **Enabled by default** on CUDA devices.

**Performance**: -93.6% data loading time, +53.2% training throughput (2,202 → 3,374 samples/sec)

```python
# Default: enable_gpu_prefetch=True, gpu_prefetch_buffer_size=3
# To disable: enable_gpu_prefetch=False (not recommended)
```
```

**After**
```markdown
### GPU Prefetching (削除済み)

GPU プリフェッチ (`DataPrefetcher` / `gpu_prefetcher.py`) は削除された．
H2D 転送は DataLoader の `--pin-memory` + `--prefetch-factor` と，
`TrainingLoop._iterate_cuda_overlap` の CUDA ストリーム
オーバーラップに一本化されている．
```

### 2. WRONG `README.md:185-206` — 2026-08-04 に削除された CLI フラグ

`--tensorboard-histogram-frequency` / `--tensorboard-histogram-module` は
`src/maou/infra/console/learn_model.py` に登録されていない
(`grep -rn "tensorboard-histogram" src/maou/infra/console/` はゼロ件)．
`docs/commands/learn_model.md` の変更履歴は既に削除 (2026-08-04) を
記録している．

**Before**: 「## TensorBoardヒストグラムの制御」節全体
(`--tensorboard-histogram-frequency` の説明と，2 つのフラグを使った
`uv run maou learn-model` の例を含む)

**After**: 節を削除する．TensorBoard 記述を残す場合は
「TensorBoard にはスカラー指標のみを書き出す．ヒストグラム出力の
`--tensorboard-histogram-*` は 2026-08-04 に削除された．」の 1 文に置換．

### 3. WRONG `docs/commands/utility_benchmark_training.md:31` — 実行時に必ず失敗する選択肢

**Before**
```markdown
| `--stage12-lr-scheduler CHOICE` | optional | Learning rate scheduler for Stage 1/2 benchmarks. Choices: `warmup_cosine_decay`, `cosine_annealing`, `step`. Overrides `--lr-scheduler` for Stage 1/2. |
```

`utility.py:487-495` の `click.Choice` は確かに 3 値を宣言するが，値は
`normalize_lr_scheduler_name` (`utility_interface.py:416`) に渡り，その
別名表 (`interface/learn.py:66-82`) は `warmupcosinedecay` /
`warmup+cosinedecay` / `cosineannealinglr` しか持たない．
`cosine_annealing` は `cosineannealing` に正規化されて表に無く，`step`
も無いため，どちらも `interface/learn.py:117` の
`ValueError("Unsupported learning rate scheduler...")` に到達する．

**After**
```markdown
| `--stage12-lr-scheduler CHOICE` | optional | Stage 1/2 ベンチマークの LR スケジューラ．実際に構築できるのは `warmup_cosine_decay` のみ．CLI の `Choice` には `cosine_annealing` / `step` も残っているが，`normalize_lr_scheduler_name` が受理しないため実行時に `ValueError` になる (別名表: `interface/learn.py:66-82`)．未指定なら `--lr-scheduler` を継承． |
```

> 注: これは doc の誤りであると同時に **コード側の不具合**でもある
> (CLI が常に失敗する選択肢を提示している)．修正先は
> `src/maou/infra/console/utility.py` で本 audit の path 外のため，
> `audits/coverage.md` の out-of-scope backlog に登録した．

### 4. WRONG `docs/performance.md:61,69` — 存在しないフラグ

`--input-path` は `generate-stage2-data` 専用 (`utility.py:1598`)．
`benchmark-dataloader` (`utility.py:68`) と `benchmark-training`
(`utility.py:461`) はどちらも `--stage3-data-path` を取る．
`docs/commands/utility_benchmark_dataloader.md:88` は既に正しい．

**Before** (両行): `  --input-path /path/to/processed \`
**After** (両行): `  --stage3-data-path /path/to/processed \`

### 5. WRONG `docs/performance.md:9-10` — 実際に構築される net と違う

`ModelFactory.create_shogi_backbone` / `create_shogi_model` は
`layers=(2, 2, 2, 2)`, `strides=(1, 2, 2, 2)`,
`out_channels=(64, 128, 256, 512)` を渡す
(`setup.py:721-723`, `:761-763`; `network.py:53-60` の既定も同一)．

**Before**
```markdown
- Layers: [2, 2, 2, 1] - Wide and shallow
- Bottleneck widths: [24, 48, 96, 144]
```
**After**
```markdown
- Layers: [2, 2, 2, 2]，strides [1, 2, 2, 2]
- Stage 出力チャンネル: [64, 128, 256, 512] (`BottleneckBlock`)
```
隣接する `~40% fewer parameters than ResNet-50` (line 11) は上の古い
構成に帰属した数値なので，同時に削除する (再測定するまで復活させない)．

### 6. STALE `docs/architecture.md:142` — tree が追い越した列挙

`array_type` は 4 メンバの `Literal` である
(`infra/file_system/file_data_source.py:83-85`, `:203`, `:228`;
`app/learning/polars_datasource.py:33-35`)．`PolarsDataFrameSource.__getitem__`
は `stage1` / `stage2` の分岐を実際に持つ．

**Before**
```markdown
Available types: `"hcpe"` (game records), `"preprocessing"` (training features)
```
**After**
```markdown
Available types: `"hcpe"` (game records), `"preprocessing"` (Stage 3 training features), `"stage1"` (reachable squares), `"stage2"` (legal moves)．
正準の定義は `src/maou/infra/file_system/file_data_source.py` の
`array_type` Literal であり，増えた場合はそちらが真．
```
末尾の一文は，同じ列挙が次に増えたときこの節が黙って古くならない
ようにするためのもの (「今のメンバを並べ直す」だけでは同じ時計を
巻き戻すだけになる)．

### 7. WRONG `docs/loss-functions.md:204` — layer の帰属が import と合わない

`src/maou/interface/learn.py` に損失クラスへの参照は 1 件も無い
(`grep -c "Loss\|loss_fn"` → 0)．損失の構築は全て app 層:
`ReachableSquaresLoss` (`stage_component_factory.py:116`, `:158`),
`LegalMovesLoss` (`:234`, `:310`),
Stage 3 は `LossOptimizerFactory.create_loss_functions` (`setup.py:781-802`)．

**Before**
```markdown
損失関数の選択とパラメータ設定はinterface層(`src/maou/interface/learn.py`)で行い，
```
**After**
```markdown
損失関数の選択とパラメータ設定はapp層で行う (Stage 1/2 は
`src/maou/app/learning/stage_component_factory.py`，Stage 3 は
`LossOptimizerFactory.create_loss_functions` in
`src/maou/app/learning/setup.py`)．CLIオプションはinfra層
(`src/maou/infra/console/learn_model.py`)で定義し，interface層
(`src/maou/interface/learn.py`)は値の検証・正規化のみを担う．
```

### 8. WRONG `docs/loss-functions.md:94` — フラグ名が反転している

`--streaming` は存在しない．ストリーミングは既定で，`--no-streaming`
で無効化する (`learn_model.py:558-562`, help = "Disable streaming mode
for file input")．

**Before**
```markdown
**注意**: ストリーミングモード(`--streaming`)では検証分割は非対応(警告ログを出力)．
```
**After**
```markdown
**注意**: ストリーミングモード(既定．`--no-streaming` で無効化)では検証分割は非対応(警告ログを出力)．
```

### 9. WRONG `README.md:208-218` — 削除済みの mmap 経路

`grep -rn "mmap_mode" src/` と `grep -rn "preprocessing_mmap_mode" src/`
はどちらもゼロ件．現在は
`FileDataSource.CacheMode = Literal["file", "memory"]`
(`file_data_source.py:73`, 既定 `"file"`) で，`mmap` は deprecated な
CLI 別名として `file` に変換されるのみ (`utility.py:107-115`)．

**Before**: 「前処理済みの`.npy`ファイルはデフォルトでコピーオンライト
(`mmap_mode="c"`)としてメモリマップされます。」から
「`preprocessing_mmap_mode`引数を追加しており…」までの段落

**After**
```markdown
前処理済みデータは Arrow IPC (`.feather`) が既定で，読み込み方式は
`--input-cache-mode {file,memory}` で選びます (`mmap` は deprecated で，
内部的に `file` に変換されます)。`KifDataset` は `torch.from_numpy()` で
ゼロコピー変換するため、read-only 配列を渡すと `ValueError` になります
(`src/maou/app/learning/dataset.py:186-198`)。
```

### 10. STALE `docs/rust-backend.md:732-736` — target が 4 要素になり得る

`KifDataset.__getitem__` は `moveWinRate` 列があれば **4 要素**
`(move_label, result_value, legal_move_mask, move_win_rate)` を返す
(`dataset.py:101-125`)．既定の `--policy-target-mode` は `win-rate`
(`learn_model.py:564-572`) なので，現行データでは 4 要素が通常．

**Before**
```python
for features, targets in dataloader:
    board, pieces = features
    move_label, result_value, legal_move_mask = targets
```
**After**
```python
for features, targets in dataloader:
    board, pieces = features
    # moveWinRate 列があれば 4 要素 (旧データでは 3 要素)
    move_label, result_value, legal_move_mask, *rest = targets
    move_win_rate = rest[0] if rest else None
```

### 11. STALE `docs/adr-003-training-performance-optimization-attempts.md:435`

**Before**: `- **GPU Prefetching Implementation**: `src/maou/app/learning/gpu_prefetcher.py``
**After**: `- **GPU Prefetching Implementation**: `src/maou/app/learning/gpu_prefetcher.py` (削除済み — 本 ADR 記述時点の実装)`

日付つき ADR なので当時の記述は残し，注記のみ足す．

### 12. STALE `docs/stage3-hang-investigation.md:7,12` — 削除済みコンポーネント

`DataPrefetcher` は存在しない (痕跡は `setup.py:591` のコメントのみ)．
`StreamingKifDataset` は現存 (`streaming_dataset.py:168`)．
調査記録なので本文は保存し，冒頭に注記を追加する:

**After** (ファイル冒頭に追加)
```markdown
> 注: 本調査時点の `DataPrefetcher` はその後削除された
> (現在は DataLoader の pin_memory / prefetch_factor と
> `TrainingLoop._iterate_cuda_overlap` のみ)．
```

### 13. STALE `docs/torch-compile-state-dict.md:21` — version スタンプ

節が挙げる API は全て現存し配置も正しい (`ModelIO._strip_orig_mod_prefix`
`model_io.py:23` 他)．古いのはバージョン表記だけ (`pyproject.toml` は 0.82.2)．

**Before**: `## 現在の対応状況 (v0.5.5)`
**After**: `## 現在の対応状況`

バージョンを打ち直すと同じ時計を巻き戻すだけなので，記載自体をやめる．

### 14. STALE `docs/commands/learn_model.md:13,135,139,160,185` — 行アンカーの drift

`【F:src/maou/app/learning/dl.py†L94-L209】` が指す先は現在
モジュールレベルの `should_stop_early()` / `monitored_value()` であり，
`Learning` のセットアップ経路ではない．散文の主張自体は正しく，
`TrainingSetup` は `setup.py:1083`，`LossOptimizerFactory` は
`setup.py:777` にある — ずれたのはアンカーのみ．

**After**: `TrainingSetup.setup_training_components` の言及は
`【F:src/maou/app/learning/setup.py†L1083-L1190】` に張り替え，
残る 4 箇所の `dl.py†L94-L209` アンカーは削除する．

## Motivation

これらは全て「コード側に真偽がある主張」であり，read 時に検証しない
限り黙って古くなる種類の drift である．特に #1 と #2 は，読者が
**存在しない機能**を前提に設定を書くことになるため実害が大きい．
#3 は doc が「動く」と書いている CLI 値が実行時に必ず失敗する．

## Alternatives considered

1. **#1/#2/#9 の節を残し「廃止」と注記するだけ．** ADR (#11) と
   調査記録 (#12) は日付つき記録なので注記が正しいが，
   `docs/performance.md` と `README.md` は「現在の使い方」を示す文書
   なので，廃止機能の手順が載っていること自体が誤り．削除を採る．
2. **#5 の性能数値を再測定して更新．** 本 audit の範囲を超える
   (GPU 実機が要る)．誤った数値を残すより削除が安全で，必要になったら
   測って足せる．
3. **#6 で現在の 4 メンバを並べるだけ．** 次に `stage3` 等が増えた
   時点でまた古くなる．正準の定義位置を指す一文を添えて，列挙の
   時計を止める．

## What this enables

- `docs/performance.md` と `README.md` が，実在するフラグと実在する
  データ経路だけを説明するようになる．
- #6 と #13 は「今の値を書き直す」のではなく「値が住んでいる場所を
  指す」形にしたので，同じ列挙・同じバージョンで再発しない．
- #3 の記述が，コード側の不具合 (backlog 登録済み) を修正するまでの
  間，ユーザーを誤った選択肢から守る．

## What this constrains

- #11/#12 の注記により，ADR と調査記録は「当時の記述 + 現在との差分」
  の二層構造になる．以後この 2 ファイルを読むときは注記を先に読む
  必要がある．
- #14 の `【F:…†L…】` アンカーは行番号を含む以上，構造的に drift する．
  今回は張り替えるだけで，仕組みは変えていない (アンカー形式そのものの
  見直しは別提案)．

## Rollback plan

全て prose のみの変更で，コードにも生成物にも依存が無い．節単位で
`git revert` するか，該当ファイルを直前の版に戻せば足りる．
