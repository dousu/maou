---
status: applied
applied_in: ed8dcab
date: 2026-08-12
target: [CLAUDE.md, docs/architecture.md]
risk: medium
reversibility: trivial
---

# `.npy` の「Legacy Support」が何を指しているのかコードから読み取れない

## Trigger

`/audit-backlog` (2026-08-12, `cz2r2u`)．`docs/rust-backend.md` の
Performance Comparison から `.npy` 行を削除する承認 (案 A) を受けて適用した
あと，**同じ主張が他の durable doc に残っていないか**を走査して見つけた．

`docs/rust-backend.md` が「受け付けるのは `.feather` だけ」と書く一方で
CLAUDE.md が「`.npy` はまだサポートされている」と書いている状態は，
どちらが正しいのか読者が判断できない．

## 検証結果 (HEAD)

`src/maou/` 全体で `.npy` を読み書きするコードは **1 ファイルだけ**:

| 場所 | 内容 | CLI から到達するか |
|---|---|---|
| `infra/utility/benchmark_polars_io.py:35-55` | `save_hcpe_array` / `load_hcpe_array` / `save_preprocessing_array` / `load_preprocessing_array` | **しない** (`infra/console/` と `interface/` からの参照はゼロ) |

これは `.feather` と `.npy` の I/O 性能を比較するためのベンチマーク台であり，
データパイプラインの入出力経路ではない．実際のデータソースは 3 経路とも
`Only .feather files are supported. Got: {suffix}` で `.npy` を拒否する
(`file_data_source.py:306` / `object_storage/data_source.py:168` /
`bigquery/bq_data_source.py:326`)．

### `docs/architecture.md:133,140-141` は**そもそも動かない**

```python
from maou.domain.data.io import save_hcpe_array, load_hcpe_array
save_hcpe_array(array, "output.hcpe.npy", validate=True)
```

`maou.domain.data.io` というモジュールは存在しない
(`ModuleNotFoundError` を実測で確認)．`domain/data/array_io.py` は
例外クラス 2 つだけを持つ 62 行のファイルで，`save_hcpe_array` は無い．
実体は上記のベンチマーク台にあり，`validate` 引数も持たない．
つまりこの例は **import 行・関数の所在・引数・ファイル形式のすべてが誤り**．

## なぜ P2 (drift correction) ではないか

「`.npy` の現状」という**事実**は一意に決まるが，**それをどう書くか**が
一意に決まらない．少なくとも次の 3 通りがあり，どれも現行コードと矛盾しない．

1. **削除**: CLAUDE.md から Legacy Support 行を消し，`architecture.md` の
   例を `.feather` の API に差し替える．「`.npy` はもう無い」という立場．
2. **限定して残す**: 「Legacy Support: `.npy` はベンチマーク
   (`infra/utility/benchmark_polars_io.py`) でのみ使用．データソースは
   `.feather` のみ受け付ける」と範囲を明示する．
3. **移設**: `.npy` の話を `docs/performance.md` 側 (ベンチマークの文脈) に
   移し，アーキテクチャの説明からは外す．

1 と 2 は**将来 `.npy` 入力を再開する余地を残すかどうか**という方針の違いで，
3 は doc の構成の判断である．いずれも「現行コードから一意に決まる訂正」では
ないので，CLAUDE.md の standing approval は及ばない．

## 提案 (案 2 を推す)

`.npy` を読み書きするコードが実在する以上「もう無い」(案 1) は言い過ぎで，
一方で現状の書き方は「データ形式として選べる」と読めてしまう．**範囲を
明示する案 2** が事実に最も近い．

### CLAUDE.md `:19`

```diff
-- **Legacy Support**: Numpy .npy format still supported
+- **Legacy Support**: `.npy` は `infra/utility/benchmark_polars_io.py`
+  (性能比較用) でのみ読み書きする．データソース (file / object storage /
+  BigQuery キャッシュ) はいずれも `.feather` 以外を拒否する
```

### `docs/architecture.md` `:133,140-141`

```diff
-from maou.domain.data.io import save_hcpe_array, load_hcpe_array
+from maou.domain.data.rust_io import save_hcpe_df, load_hcpe_df

 # Standardized data types
 hcpe_dtype = get_hcpe_dtype()
 preprocessing_dtype = get_preprocessing_dtype()

 # High-performance I/O
-save_hcpe_array(array, "output.hcpe.npy", validate=True)
-loaded_array = load_hcpe_array("input.hcpe.npy", validate=True)
+save_hcpe_df(df, "output.feather")
+loaded_df = load_hcpe_df("input.feather")
```

## 判断してほしい点

- 案 1 (削除) / 案 2 (範囲を明示・推奨) / 案 3 (performance.md へ移設) のどれか．
- 案 2 を採る場合，`architecture.md` の差し替え先 API は
  `domain/data/rust_io` の `save_hcpe_df` / `load_hcpe_df` でよいか
  (structured array ではなく DataFrame を扱う API になる — 例の主題が
  「配列の I/O」から「DataFrame の I/O」に変わる)．

## 決定 (2026-08-12)

ユーザ判断で **案 1 (削除)** を採用した．理由は「`.npy` はもう使う予定が
ない」．提案側は案 2 (範囲を明示) を推していたが，`.npy` を将来の入力形式
として残す意思が無い以上，範囲を書き残すと「選べる形式」と読まれ続ける —
という判断で案 1 が採られた．

適用内容:

- `CLAUDE.md:19` の `- **Legacy Support**: Numpy .npy format still supported`
  を削除．§ Data Pipeline は Arrow IPC / Polars / Rust I/O の 3 行になる．
- `docs/architecture.md` の § Centralized Schema Management の例を，存在しない
  `maou.domain.data.io` から `maou.domain.data.array_io` の
  `save_hcpe_df` / `load_hcpe_df` に差し替え，`.npy` を `.feather` にした．
  例の主題が「配列の I/O」から「DataFrame の I/O」に変わるため，structured
  array は DataFrame から変換して得る旨を 1 段落補った．

### 残る不整合 (この提案の対象外)

`infra/utility/benchmark_polars_io.py` は `.npy` の save/load を持ち続ける．
これは `.feather` と `.npy` の I/O 性能を比較するベンチマーク台で，CLI から
到達しない．**「もう使う予定がない」形式との比較を維持する意味があるか**は
コード側の判断なので，`audits/coverage.md` の N-2 行に残した．
