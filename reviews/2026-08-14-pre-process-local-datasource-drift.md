---
status: applied
applied_in: PENDING
date: 2026-08-14
target: [docs/commands/pre_process.md]
risk: low
reversibility: trivial
---

# `docs/commands/pre_process.md` の「local `FileDataSource`」がコードと食い違う

## Trigger

`/audit-backlog` (2026-08-14, `92hcyo`)．backlog 行 D14(b) の消化
(`FileDataSource` から `preprocess.DataSource` の継承を外す) で
`docs/` を洗ったところ，pre-process のローカル入力の記述が
**この変更より前から**コードと食い違っていたことが判った．

## 検証結果 (HEAD `e978ea4` + 本 run の変更)

`docs/commands/pre_process.md:58` — § "Execution flow" 1.:

> The CLI checks provider exclusivity, instantiates a `DataSource`
> (local `FileDataSource`, `BigQueryDataSource`, `GCSDataSource`,
> or `S3DataSource`), …

ローカル入力に対して CLI が構築するのは **`StreamingHcpeDataSource`**
である:

```
src/maou/infra/console/pre_process.py:533  StreamingHcpeDataSource,
src/maou/infra/console/pre_process.py:536  datasource = StreamingHcpeDataSource(
```

`FileDataSource` は `console/pre_process.py` から一度も構築されない．
本 run で `FileDataSource` は `preprocess.DataSource` の継承自体を
失ったので (backlog 行 D14(b))，この記述は「古い」だけでなく
**型としても成り立たない**記述になった．

## P2 判定

**drift correction である．** 訂正後の本文は現行コードから一意に決まる —
ローカル経路が構築するクラスは `StreamingHcpeDataSource` ただ 1 つで，
書き方の選択肢が無い．よって CLAUDE.md § MUST rules の standing approval
が適用され，本 run で適用・コミットする．

## 変更内容

### `docs/commands/pre_process.md:57-60`

**Before**

```markdown
1. **Datasource resolution** – The CLI checks provider exclusivity, instantiates
   a `DataSource` (local `FileDataSource`, `BigQueryDataSource`, `GCSDataSource`,
   or `S3DataSource`), and pins `array_type="hcpe"` so only HCPE tensors enter
   the workflow.【F:src/maou/infra/console/pre_process.py†L66-L360】
```

**After**

```markdown
1. **Datasource resolution** – The CLI checks provider exclusivity, instantiates
   a `DataSource` (local `StreamingHcpeDataSource`, `BigQueryDataSource`,
   `GCSDataSource`, or `S3DataSource`), and pins `array_type="hcpe"` so only HCPE
   tensors enter the workflow. The local source streams one file at a time rather
   than loading them all up front, so peak memory is one file rather than the whole
   input; `FileDataSource` is the *learning* path's source and is not a
   `preprocess.DataSource`.【F:src/maou/infra/console/pre_process.py†L66-L360】
```

## 影響

- CLI のオプション表には触れない (挙動の記述ではなく実装クラス名の訂正)．
- `docs/rust-backend.md` の `FileDataSource.iter_batches_df()` の例は
  **drift ではない** — 当該メソッドは具象メソッドとして残っており，
  ABC 経由でない直接呼び出しは今までどおり動く．
