---
status: applied
date: 2026-08-14
targets:
  - docs/commands/pre_process.md
  - docs/commands/utility_benchmark_dataloader.md
  - docs/commands/utility_benchmark_training.md
  - docs/rust-backend.md
---

# bundling ノブの削除と Stage アダプタ統合に伴う doc drift

## 背景

`/audit-backlog` 2026-08-14 の run で，過去の run でユーザが回答済み
だった設計判断のうち**未実装だった 2 件**を実装した．

- **O5(c) → bundling ノブごと削除**
  ([記録](../audits/2026-08-14-backlog-cache-knob-removal.md) の Q2)．
  `--input-enable-bundling` / `--input-bundle-size-gb` が `pre-process` /
  `benchmark-dataloader` / `benchmark-training` の 3 コマンドから消え，
  `ObjectStorageDataSource` / `PageManager` / `DataSourceSpliter` の
  `enable_bundling` / `bundle_size_gb` 引数と，読み手のいなかった
  `bundle_cache` / `bundle_id` も消えた．
- **Deferred 3 → Stage1/Stage2 アダプタ 6 クラスを 3 つに統合**
  (同記録の Q4)．`Stage1ModelAdapter` / `Stage2ModelAdapter` →
  `StageModelAdapter`，`Stage1DatasetAdapter` / `Stage2DatasetAdapter` →
  `StageDatasetAdapter`，`Stage1StreamingAdapter` /
  `Stage2StreamingAdapter` → `StageStreamingAdapter`．**旧 6 名は別名
  として残す**ので import は壊れない．

以下の doc はいずれも「削除されたオプション」「統合前のクラス名」を
現存するものとして説明している．訂正後の本文は現行コードから一意に
決まる (オプションは存在しない / 生成されるクラスは 1 つ) ため，
CLAUDE.md § "Standing approval — drift corrections only" の P2 に該当
する．

## 変更内容

### 1. `docs/commands/pre_process.md` (GCS 行)

**Before**

> Supports worker counts and optional local caching.
> `--input-enable-bundling` / `--input-bundle-size-gb` are accepted but
> currently have **no effect**: the download path writes each `.feather`
> object individually and never reads either value.

**After**

> Supports worker counts and optional local caching. The download path
> writes each `.feather` object individually; there is no bundling
> option.

### 2. `docs/commands/utility_benchmark_dataloader.md` (cloud 行)

**Before**

> Requires the corresponding optional extras. `--input-enable-bundling` /
> `--input-bundle-size-gb` are accepted but currently have **no effect**
> (see `pre_process.md`).

**After**

> Requires the corresponding optional extras. There is no bundling
> option; each `.feather` object is downloaded individually (see
> `pre_process.md`).

### 3. `docs/commands/utility_benchmark_training.md`

cloud 行は 2 と同型の訂正．加えて Execution flow の 1 番:

**Before**

> Stage 1 creates `Stage1ModelAdapter` + `ReachableSquaresLoss`
> (map-style only); Stage 2 creates `Stage2ModelAdapter` +
> `LegalMovesLoss` (map-style or streaming via `Stage2StreamingAdapter`)

**After**

> Stage 1 creates `StageModelAdapter` + `ReachableSquaresLoss`
> (map-style only); Stage 2 creates `StageModelAdapter` +
> `LegalMovesLoss` (map-style or streaming via `StageStreamingAdapter`)

### 4. `docs/rust-backend.md` (S3DataSource の例)

**Before**

```python
    max_workers=16,
    # enable_bundling / bundle_size_gb は受理されるが現状は効果がない
    # (.feather は常に個別保存される)．docs/commands/pre_process.md 参照．
    enable_bundling=True,
    bundle_size_gb=1.5,
)
```

**After**

```python
    max_workers=16,
)
```

## 触れなかったもの

- **`docs/stage2-speed-investigation.md`** は特定時点の調査報告であり，
  当時のクラス名を書いていること自体は誤りではない．訂正後の本文が
  一意に決まらない (歴史的記述をどう扱うかは書き手の判断) ため P2 に
  該当せず，据え置いた．
- **`.claude/skills/gh-pr/SKILL.md` / `pr-preparation-checks/SKILL.md` /
  `feature-branch-setup/SKILL.md`** の "array bundling" は**架空の PR /
  ブランチ名の例**であり，CLI の記述ではないので据え置いた．実運用の
  コマンド例を持っていた `cloud-integration-tests` /
  `benchmark-execution` / `data-pipeline-validator` の 3 skill は
  更新済み (これらは `.claude/` 配下なので P1，本提案の対象外)．
