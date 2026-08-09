---
status: pending
applied_in:
date: 2026-08-09
target: [docs/commands/pre_process.md]
risk: low
reversibility: trivial
---

# `--search-value-path` のディレクトリ対応を docs へ反映する

## 背景

`--search-value-path` はこのプロジェクトのパス指定オプションの規約
(ファイルでもディレクトリでも受ける) から外れており，ディレクトリを渡すと
polars の `IsADirectoryError` で落ちていた．しかも遅延ロードだったため，
HCPE 変換と DuckDB 集約が全部終わってから落ちる．パスの取り違え 1 つで
数時間の実行が丸ごと無駄になっていた．

コード側は本 PR で修正済み:

- ディレクトリを受け付け，配下の `**/*.feather` を union する
- 読み込み時に `id` / `searchWinRate` の 2 列へ射影する (診断列の構成差で
  union が失敗しないため．`elapsedMs` は 0.82.0 で追加された列で，
  新旧の出力が同じディレクトリに並ぶことは普通に起こる)
- 全ファイルのスキーマを**読み込みより先に**検査し，feather として読めない・
  必要な列が無い・型が使えない場合は `PreProcess.__init__` の時点で
  `ValueError` を投げる (変換を始める前に落ちる)

`docs/commands/pre_process.md` の `--search-value-path` 行は現状「出力
feather」としか書いておらず，ディレクトリ対応にも早期失敗にも触れていない．

## 提案する変更

`docs/commands/pre_process.md` の `--search-value-path` 行を次へ差し替える．

```markdown
| `--search-value-path PATH` | optional | [`maou utility search-values`](utility_search_values.md) の出力．**単一の `.feather` でも，それらを含むディレクトリでもよい**(ディレクトリなら配下の `**/*.feather` を union する)．該当する局面の `resultValue` を対局結果由来の値から**探索値**へ差し替える．対局結果は 1 対局の約 110 局面すべてで同じ値になるため「どの対局か」を思い出す近道が成立するが，探索値は局面ごとに異なるのでその近道が効かなくなる．入力に無い局面は対局結果由来のまま残るので**部分適用できる**．`--position-count-threshold` とは**独立**で，出現回数によらずカバーされた局面すべてに適用される(ただし `--drop-below-threshold` で除外された行には届かない)．読み込みでは `id` / `searchWinRate` の 2 列だけを使うので，診断列の構成が違う世代の出力同士でも union できる．feather として読めない・この 2 列が無い・型が使えないファイルが 1 つでもあれば，**変換を始める前に** `ValueError` で落ちる(差し替えは集約の後に走るので，遅延させると数時間後に失敗が判明する)．差し替えた行数は結果 JSON の `search_values_applied` に出る．背景は [docs/design/training-quality/](../design/training-quality/index.md) §3.3． |
```

## 判断の余地があった点

- **`--position-count-threshold` との独立性を書くかどうか**: 元の記述は
  threshold との関係に触れておらず，コードとの矛盾は無かった．ただし
  「fallback」という語がファイルのカバレッジ基準を指すのか
  `--position-count-threshold` を指すのか，実際に読み違えが起きたので明記する．
- **union の重複解決**: 同じ id が複数ファイルにあるとパス順で後勝ちになる．
  行数不変は `apply_search_values` が保証しており，運用上は resume の重なり
  程度でしか起きないので，表には書かず docstring 側に留める．

## リスク

ドキュメントのみ．コードの挙動は本提案では変えない．
