---
status: pending
applied_in:
date: 2026-08-12
target: [docs/rust-backend.md]
risk: low
reversibility: trivial
---

# `docs/rust-backend.md` § Performance Comparison の表が現行コードと合わない

## Trigger

`audits/coverage.md` § Out-of-scope backlog の NEW-3 行
([2026-08-12 backlog auto-band-and-n1](../audits/2026-08-12-backlog-auto-band-and-n1.md))．
`/audit-backlog` (2026-08-12) で HEAD に対して再検証した．

## 現状 (`docs/rust-backend.md:724-728`)

```
| File Format | iter_batches() | iter_batches_df() | Notes |
|-------------|----------------|-------------------|-------|
| `.feather` | ❌ Not supported | ✅ Zero-copy load | Most efficient |
| `.npy` | ✅ mmap/memory | ✅ Auto-convert | Conversion overhead |
| Cloud (cached) | ✅ numpy arrays | ✅ Auto-convert | Same as .npy |
```

## 検証結果

| 表の主張 | 現行コード | 判定 |
|---|---|---|
| `.feather` で `iter_batches()` は「Not supported」 | `tests/maou/infra/file_system/test_file_data_source.py:156` `test_file_data_source_iter_batches` が `.feather` の `FileDataSource` で 2 バッチ 8 行を通す | **誤り** |
| `.npy` が読める | `file_data_source.py:306` / `object_storage/data_source.py:168` / `bigquery/bq_data_source.py:326` がいずれも `Only .feather files are supported. Got: {suffix}` で拒否 | **存在しない経路** |
| Cloud (cached) は `.npy` と同じ | 同上 (キャッシュも `.feather`) | **存在しない経路** |

## なぜ P2 (drift correction) ではないか

`.feather` 行の訂正 (❌ → ✅) は現行コードから一意に決まる．しかし
`.npy` / `Cloud (cached)` の 2 行は「削除する」「legacy として注記を残す」
のどちらも現行コードと矛盾しない — **表の構成そのものが著者判断**になる．
`.npy` は `CLAUDE.md` が "Legacy Support: Numpy .npy format still supported"
と書いており (どの経路を指すかは別途要確認)，黙って消すと今度は
`CLAUDE.md` との整合が問題になる．したがって承認を待つ．

## Before / After (案 A — 対応形式だけを残す)

### `:722-730`

```diff
 ### Performance Comparison

-| File Format | iter_batches() | iter_batches_df() | Notes |
-|-------------|----------------|-------------------|-------|
-| `.feather` | ❌ Not supported | ✅ Zero-copy load | Most efficient |
-| `.npy` | ✅ mmap/memory | ✅ Auto-convert | Conversion overhead |
-| Cloud (cached) | ✅ numpy arrays | ✅ Auto-convert | Same as .npy |
-
-**Recommendation:** Use `.feather` files for new data pipelines to take advantage of direct DataFrame loading.
+データソースが受け付けるのは `.feather` (Arrow IPC) だけである．
+`.npy` は `FileDataSource` / object storage / BigQuery キャッシュの
+いずれでも `Only .feather files are supported` で拒否される．
+
+| File Format | iter_batches() | iter_batches_df() | Notes |
+|-------------|----------------|-------------------|-------|
+| `.feather` | ✅ structured array へ変換 | ✅ Zero-copy load | 唯一の対応形式 |
+
+`iter_batches()` は `ColumnarBatch` を structured array に変換して返すため，
+DataFrame のまま扱える場面では `iter_batches_df()` の方が変換 1 回分安い．
```

### 案 B — legacy 行を注記付きで残す

`.npy` 行を残し `Notes` を
`Legacy: データソース経由では読めない (変換ユーティリティのみ)` に
差し替える．`CLAUDE.md` の "Legacy Support" と表記を揃えられる一方，
「読めない形式が対応表に載っている」状態は残る．

## 判断してほしい点

- 案 A (削除) か案 B (legacy 注記) か．
- 案 B を採るなら，`.npy` が実際に生きている経路 (変換ユーティリティ
  など) を先に確定する必要がある — この run では未確認．
