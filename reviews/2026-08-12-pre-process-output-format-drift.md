---
status: applied
applied_in: 4e335ee
date: 2026-08-12
target: [docs/commands/pre_process.md]
risk: low
reversibility: trivial
---

# `docs/commands/pre_process.md` が pre-process の出力を `.npy` と説明している

## Trigger

`audits/coverage.md` § Out-of-scope backlog の O11 行
([2026-08-10 infra/file_system](../audits/2026-08-10-src-maou-infra-file-system.md))．
`/audit-backlog` (2026-08-12) で HEAD に対して再検証し，6 箇所すべてが
現行コードと食い違っていることを確認した．

`docs/adr-004-arrow-ipc-migration.md` の Arrow IPC 移行から漏れた記述で，
2026-08-10 の承認済み提案 (`1c6a442`) は `infra/file_system` を説明する
主張だけを対象にしていたため，pre-process 自身の出力形式は残っていた．

## 現行コードが書くもの (訂正文の一意性の根拠)

| 経路 | 実装 | 実際の出力 |
|---|---|---|
| ローカル出力 | `app/pre_process/hcpe_transform.py:574` | `transformed_chunk{NNNN}.feather` (`save_preprocessing_df`) |
| GCS/S3 アップロード | `infra/object_storage/feature_store.py:144` | `batch{ID}_{N}dfs_{SIZE}MB.feather` (Arrow IPC bytes) |
| GCS/S3 ダウンロード | `infra/object_storage/data_source.py:159-161` | `.feather` 以外の suffix は `ValueError` で拒否 |

`.npy` を書く経路も読む経路も pre-process には存在しない．したがって
訂正後の本文は現行コードから一意に決まる (P2 = drift correction)．

## Before / After

### `:6` (Overview)

```diff
-  by converting raw `.hcpe` inputs into preprocessed `.npy` shards and optional
-  BigQuery/GCS/S3 uploads.
+  by converting raw `.hcpe` inputs into preprocessed `.feather` shards (Arrow IPC)
+  and optional BigQuery/GCS/S3 uploads.
```

### `:22` (Input selection / GCS)

```diff
-Downloads `.npy` shards tagged `array_type="hcpe"`.
+Downloads `.feather` shards (Arrow IPC) tagged `array_type="hcpe"`; any other suffix is rejected.
```

### `:32` (Feature-store outputs / Local only)

```diff
-Writes `.npy` shards tagged `array_type="preprocessing"` to disk.
+Writes `transformed_chunk{NNNN}.feather` shards (Arrow IPC) tagged `array_type="preprocessing"` to disk.
```

### `:34` (Feature-store outputs / GCS)

```diff
-Uploads `.npy` shards as `array_type="preprocessing"`
+Uploads `batch{ID}_{N}dfs_{SIZE}MB.feather` shards (Arrow IPC) as `array_type="preprocessing"`
```

### `:66` (Execution flow 3)

```diff
-   temporary directory, and emits final `.npy` shards. When a feature store is
+   temporary directory, and emits final `.feather` shards. When a feature store is
```

### `:90` (Outputs and usage)

```diff
-- Local runs write `.npy` shards derived from HCPE inputs into `--output-dir`
+- Local runs write `.feather` shards derived from HCPE inputs into `--output-dir`
```

## 承認

CLAUDE.md § "Standing approval — drift corrections only" が与える恒久承認の
範囲内 (訂正後の本文が現行コードから一意に決まる drift correction)．
`/audit-backlog` の P2 として適用済み (`4e335ee`, PR #456)．
