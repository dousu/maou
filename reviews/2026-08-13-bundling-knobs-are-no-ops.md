---
status: applied
applied_in: a1ce41c
date: 2026-08-13
target:
  - docs/commands/pre_process.md
  - docs/commands/utility_benchmark_training.md
  - docs/commands/utility_benchmark_dataloader.md
risk: low
reversibility: trivial
---

# docs 3 本が `--input-enable-bundling` / `--input-bundle-size-gb` を「使える機能」として説明している

## Trigger

`audits/coverage.md` § Out-of-scope backlog の **O5(c)** 行
([2026-08-10 infra/file_system](../audits/2026-08-10-src-maou-infra-file-system.md))．
`/audit-backlog` (2026-08-13) で HEAD に対して再検証した．

O5 行そのものは「ノブの意味が層をまたいで分裂している」という**設計判断
を要する** finding (P6 + G4) で，本提案はその中の **(c) の doc drift 部分
だけ**を切り出したものである．ノブを消すか残すかの判断には踏み込まない —
「現状は受け取るが効果がない」という事実だけを docs に書く．

## 現行コードが何をするか (訂正文の一意性の根拠)

| 位置 | 実装 | 事実 |
|---|---|---|
| `src/maou/infra/object_storage/data_source.py:192-200` | `__download_all_to_local(*, enable_bundling, bundle_size_gb)` の docstring | *"Note: enable_bundling and bundle_size_gb are ignored for .feather files (kept for API compatibility but not used)"* と明言 |
| `src/maou/infra/object_storage/data_source.py:288-320` | ダウンロード完了ハンドラ | `# Save .feather files directly (no bundling)` の下で，チャンク内の各オブジェクトを `feather_path.write_bytes(byte_data)` で**無条件に個別保存**する．`enable_bundling` を読む分岐は存在しない |
| grep `enable_bundling` (同ファイル) | `:45,:57,:102,:138,:194,:394,:410,:430` | 引数として層をまたいで受け渡されるだけで，値を**判定に使う場所が一つも無い** |

つまり両オプションは CLI から受理され，コンストラクタまで運ばれ，そこで
捨てられる．データ入力は現在すべて `.feather` (Arrow IPC) なので，bundling
が効く経路は残っていない (`.npy` 時代の遺物)．

したがって訂正後の本文は現行コードから一意に決まる:
**「受け取るが現状は効果がない」以外の書き方が無い**．P2 = drift correction．

## Before / After

### `docs/commands/pre_process.md:23` (Input selection / GCS)

```diff
-Downloads `.feather` shards (Arrow IPC) tagged `array_type="hcpe"`; any other suffix is rejected. Supports worker counts, bundling (`--input-enable-bundling`, `--input-bundle-size-gb`), and optional local caching.【F:src/maou/infra/console/pre_process.py†L200-L360】
+Downloads `.feather` shards (Arrow IPC) tagged `array_type="hcpe"`; any other suffix is rejected. Supports worker counts and optional local caching. `--input-enable-bundling` / `--input-bundle-size-gb` are accepted but currently have **no effect**: the download path writes each `.feather` object individually and never reads either value.【F:src/maou/infra/console/pre_process.py†L200-L360】【F:src/maou/infra/object_storage/data_source.py†L192-L320】
```

### `docs/commands/utility_benchmark_training.md:34` (Input sources)

```diff
-Downloads tensors via `GCSDataSource`/`S3DataSource` splitters. Supports worker counts, bundling (`--input-enable-bundling`, `--input-bundle-size-gb`), and optional sampling ratios; requires the respective optional extras.【F:src/maou/infra/console/utility.py†L869-L951】
+Downloads tensors via `GCSDataSource`/`S3DataSource` splitters. Supports worker counts and optional sampling ratios; requires the respective optional extras. `--input-enable-bundling` / `--input-bundle-size-gb` are accepted but currently have **no effect** (see `pre_process.md`).【F:src/maou/infra/console/utility.py†L869-L951】【F:src/maou/infra/object_storage/data_source.py†L192-L320】
```

### `docs/commands/utility_benchmark_dataloader.md:21` (Input sources)

```diff
-Downloads shards via `GCSDataSource` or `S3DataSource` splitters. Supports worker counts (`--input-max-workers`), bundling (`--input-enable-bundling`, `--input-bundle-size-gb`), and optional sampling ratios. Requires the corresponding optional extras.【F:src/maou/infra/console/utility.py†L306-L374】
+Downloads shards via `GCSDataSource` or `S3DataSource` splitters. Supports worker counts (`--input-max-workers`) and optional sampling ratios. Requires the corresponding optional extras. `--input-enable-bundling` / `--input-bundle-size-gb` are accepted but currently have **no effect** (see `pre_process.md`).【F:src/maou/infra/console/utility.py†L306-L374】【F:src/maou/infra/object_storage/data_source.py†L192-L320】
```

## この提案がやらないこと

- ノブそのものの削除．CLI オプションの撤去は P6 (契約の破壊) であり，
  O5 行が名指しする「bool flag と dir のどちらがキャッシュを有効にするのか」
  という層をまたいだ整合の決めと一体で扱うべきもの．O5 行は backlog に
  残す．
- 既定値の不一致 (`data_source.py:45` が `True`，`:102`/`:394` が `False`)
  の是正．値が読まれない以上いま観測可能な差は無いが，ノブを生かす向きに
  決めた瞬間に効いてくるので，これも O5 の判断と一体．

## 承認

CLAUDE.md § "Standing approval — drift corrections only" が与える恒久承認の
範囲内 — 訂正後の本文が現行コードから一意に決まる drift correction であり，
新しい指針・節の再構成・規則の追加を含まない．
`/audit-backlog` の P2 として適用済み (`a1ce41c`)．
