---
status: applied
applied_in: 466adac
date: 2026-08-12
target: [docs/commands/hcpe_convert.md]
risk: low
reversibility: trivial
---

# `docs/commands/hcpe_convert.md` が hcpe-convert の出力を `.npy` と説明している

## Trigger

`/audit-backlog` (2026-08-12, `cz2r2u`) で `docs/rust-backend.md` の
Performance Comparison を「`.feather` のみ」に直した (`cf3e4ca`) 直後の
step 4d チェック — **source/doc の修正が別の prose を無効化していないか**
の走査で見つけた．`docs/commands/` を `.npy` で grep して 2 行が残っていた．

`docs/adr-004-arrow-ipc-migration.md` の Arrow IPC 移行から漏れた記述で，
**2026-08-12 に同じ欠陥で `docs/commands/pre_process.md` を直している**
(`reviews/2026-08-12-pre-process-output-format-drift.md`, `4e335ee`)．
そのときは `pre_process` だけを対象にしていたため，`hcpe_convert` が残った．

## 現行コードが書くもの (訂正文の一意性の根拠)

| 経路 | 実装 | 実際の出力 |
|---|---|---|
| ローカル出力 | `app/converter/hcpe_converter.py:90` | 「`maou._rust.maou_convert` で各ファイルを **`.feather`** に一括変換」 |
| ファイル名 | 同 `:196-198` | `output_dir / file.with_suffix(".feather").name` |
| GCS/S3 アップロード | 同 `:189` | 「個別 **`.feather`** ファイルをチャンクにまとめ，feature_store にアップロードする」 |

`.npy` を書く経路は hcpe-convert に存在しない．同ファイルの他の箇所
(`:5`, `:19`) は既に `.feather` と書いており，表の 2 行だけが取り残されて
いた．したがって訂正後の本文は現行コードから一意に決まる
(**P2 = drift correction**)．CLAUDE.md の standing approval が及ぶので，
この run 内で適用した．

## Before / After

### `:45` (`--output-dir`)

```diff
-| `--output-dir PATH` | ✅ | Destination directory for `.npy` shards. …
+| `--output-dir PATH` | ✅ | Destination directory for `.feather` shards (Arrow IPC). …
```

### `:57` (`--output-gcs`)

```diff
-| … | Uploads `.npy` shards to GCS with configurable worker counts, …
+| … | Uploads `.feather` shards (Arrow IPC) to GCS with configurable worker counts, …
```

## 対象外

CLAUDE.md の "Legacy Support: Numpy .npy format still supported" と
`docs/architecture.md` の `.npy` の例は**この提案に含めない**．訂正後の
本文が一意に決まらないため，別提案
(`2026-08-12-npy-legacy-support-status.md`) で承認を待つ．
