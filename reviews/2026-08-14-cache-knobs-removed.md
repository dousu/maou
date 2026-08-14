---
status: applied
date: 2026-08-14
targets:
  - README.md
  - docs/commands/utility_benchmark_dataloader.md
  - docs/commands/utility_benchmark_training.md
  - docs/rust-backend.md
---

# `--input-cache-mode` と `--input-local-cache` の廃止に伴う doc drift

## 背景

`/audit-backlog` 2026-08-14 の run で，2026-08-14 にユーザが回答した
2 つの設計判断を実装した ([記録](../audits/2026-08-14-backlog-design-decisions.md)
の Q1 / Q4)．

- **Q1 → `cache_mode` ノブごと削除** (backlog 行 D5 + O5(d))．
  `--input-cache-mode` が `benchmark-dataloader` / `benchmark-training`
  から消え，`FileDataSource` / `FileManager` /
  `FileDataSourceSpliter` / `interface.learn.learn()` /
  `learn_multi_stage()` / `Learning.LearningOption` からも
  該当パラメータが消えた．
- **Q4 → local-cache は dir に一本化** (backlog 行 O5(a))．
  bool flag `--input-local-cache` が 3 コマンドから消え，
  BigQuery のキャッシュ有効化は `--input-local-cache-dir` の有無だけで
  決まるようになった．

これらの doc は削除されたオプションを現存するものとして説明している．

## P2 判定

**drift correction である．** 訂正後の本文は現行コードから一意に決まる —
「存在しないオプションの行を消す」以外の書き方が無い．新しい指針も
節の再構成も含まない．したがって CLAUDE.md § "Standing approval —
drift corrections only" の恒久承認が適用され，この run 内で適用する．

なお**コード変更そのもの (P6) は判断帯**であり，ユーザの受理を待つ．
この doc 修正はコード変更と同じ PR に乗るので，コードが受理されなければ
doc も一緒に落ちる (両者が乖離することはない)．

## 変更内容

### 1. `README.md` § "Preprocessingデータの読み込み方式"

**Before** (`:192-198`):

```markdown
前処理済みデータは Arrow IPC (`.feather`) が既定で、読み込み方式は
`--input-cache-mode {file,memory}` で選びます (`mmap` は deprecated で、
内部的に `file` に変換されます)。`KifDataset` は `torch.from_numpy()` で
ゼロコピー変換するため、read-only 配列を渡すと `ValueError` になります
(`src/maou/app/learning/dataset.py:186-198`)。
```

**After**:

```markdown
前処理済みデータは Arrow IPC (`.feather`) が既定です。入力ファイルは
初期化時に全てメモリへ読み込まれ、ファイル 1 つにつき 1 配列を保持します
(データセットを常駐させたくない場合は streaming 経路を使ってください)。
`KifDataset` は `torch.from_numpy()` で
ゼロコピー変換するため、read-only 配列を渡すと `ValueError` になります
(`src/maou/app/learning/dataset.py:186-198`)。
```

### 2. `docs/commands/utility_benchmark_dataloader.md`

- `:22` の BigQuery cache knobs 行から `` `--input-local-cache`, `` を削除．
- `:23` の `` | `--input-cache-mode {file,memory,mmap}` | ... | `` 行を
  **行ごと削除**．

### 3. `docs/commands/utility_benchmark_training.md`

- `:35` の BigQuery cache knobs 行から `` `--input-local-cache`, `` を削除．
- `:36` の `` | `--input-cache-mode {file,memory,mmap}` | ... | `` 行を
  **行ごと削除**．

### 4. `docs/rust-backend.md` のコード例

**Before** (`:675-679`):

```python
datasource = FileDataSource(
    file_paths=[Path("data.feather")],
    array_type="hcpe",
    cache_mode="file",
)
```

**After**:

```python
datasource = FileDataSource(
    file_paths=[Path("data.feather")],
    array_type="hcpe",
)
```

## 触らない箇所

- `docs/commands/learn_model.md:189-191` — 2026-02-22 の変更履歴で
  「learn-model から `--input-cache-mode` を削除した」と書いてある．
  **過去の事実として今も正しい**ので変更しない．
- `docs/commands/pre_process.md:23`,`:108` および
  `docs/commands/utility_benchmark_*.md` の
  `--input-local-cache-dir` の記述 — dir は**残る**側なので変更しない．
- `--input-enable-bundling` / `--input-bundle-size-gb` の
  「受理するが効果なし」の記述 — このノブの去就は backlog 行 O5(c) の
  未回答の設計判断であり，この run では触らない．

## 追記: `docs/commands/pre_process.md`

`scripts/check-cli-docs.sh` は `pre_process.py` の変更時に
`pre_process.md` の同時ステージングを要求する．この doc は bool flag を
一度も載せていなかったので**誤りは無い**が，「何がキャッシュを
有効にするのか」は書かれていなかった．コードから一意に決まる 1 文を
BigQuery 行に足す (P2 のまま)．

**Before** (`:22`):

```
| BigQuery | `--input-dataset-id` + `--input-table-name` | Streams HCPE rows with configurable batch size, cache limits, clustering, and partition hints. Requires the `gcp` optional extra.
```

**After**:

```
| BigQuery | `--input-dataset-id` + `--input-table-name` | Streams HCPE rows with configurable batch size, cache limits, clustering, and partition hints. Local caching is enabled by supplying `--input-local-cache-dir`; there is no separate on/off flag. Requires the `gcp` optional extra.
```
