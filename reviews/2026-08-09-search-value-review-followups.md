---
status: pending
applied_in:
date: 2026-08-09
target: [docs/commands/pre_process.md, docs/commands/utility_search_values.md]
risk: low
reversibility: trivial
---

# コードレビュー指摘 (PR #452) の docs 反映

## 背景

[reviews/2026-08-09-search-value-path-directory-support.md](2026-08-09-search-value-path-directory-support.md)
で入れたディレクトリ対応に対し，`/code-review` が 3 件を指摘した．
いずれも実コードで裏を取り，事実と確認した上で修正済み．
そのうち 2 件が文書化済みの挙動を変えるので，ここで提案する．

| 指摘 | 実装した対応 | docs 影響 |
|---|---|---|
| F1: 重複排除と警告がチャンクごとに反復する | union 直後に 1 回だけ解決 (`maintain_order=True`) | なし (内部) |
| F2: 「変換を始める前に落ちる」が文字通りでない | 検査を `pre_process()` 冒頭へ引き上げ | あり |
| F3: `**/*.feather` のみで取りこぼしが無言 | `.arrow` も拾う + 読んだファイルをログ + 生成側で拡張子を強制 | あり |

### F2 の詳細

検証した実行順序 (`src/maou/infra/console/pre_process.py`):

- 376-486 行: データソース構築．GCS/S3 は全ダウンロード + メモリロード
- 481 行: `resize_input_files` が入力 HCPE コーパス全体を work_dir へ書き直す
- 598 行: `preprocess.transform` → `PreProcess.__init__` → 検証

つまり打ち間違えても全ダウンロードか全リサイズ 1 回分は払っていた．
`validate_search_value_source()` (フッタのみ読む) を切り出し，
`interface.preprocess.validate_search_value_path()` 経由で
`pre_process()` の最初の実処理として呼ぶようにした．
`PreProcess.__init__` のロードは他の呼び出し元向けの防御として残す．

### F3 の詳細

`search-values --output-path` に拡張子の検査が無いため `values.bin` が書けてしまい，
`pre-process --search-value-path` のディレクトリ指定はそれを無言で飛ばす
(0 件のときだけエラー)．数日分の GPU 時間がエラーも警告も無く落ちる経路なので，
拾う拡張子を `.feather` / `.arrow` に広げ，読んだファイル名を INFO でログし，
生成側の `--output-path` にも同じ拡張子を強制した．

`_write` の中間ファイル (`*.feather.tmp`) はどちらの glob にも一致しない
(テストで固定済み)．HCPE 側の列挙は `.feather` のみのまま変えていない．

## 提案する変更 1: `docs/commands/pre_process.md`

`--search-value-path` 行のうち 2 点を差し替える．

- 「配下の `**/*.feather` を union する」→
  「配下の `**/*.feather` と `**/*.arrow` を union する」
- 「**変換を始める前に** `ValueError` で落ちる」→
  「**入力のダウンロードやリサイズより前に** `ValueError` で落ちる」

加えて重複解決を 1 文追記する:
「同じ id が複数ファイルにある場合はパス順で後勝ちになる (union 時に 1 回だけ解決する)．」

## 提案する変更 2: `docs/commands/utility_search_values.md`

130 行の `--output-path` 行を差し替える．

```markdown
| `--output-path PATH` | required | 出力する feather (`id` / `searchWinRate` / `playouts` / `stop`)．拡張子は `.feather` か `.arrow` でなければ**エラー**になる．`pre-process --search-value-path` にディレクトリを渡すとこの拡張子で配下を拾うため，別の名前で書くとそのファイルだけ無言で飛ばされて部分適用になる． |
```

## 判断の余地があった点

- **F3 (c) は破壊的か**: 現在 `.bin` 等で書いている運用があれば壊れる．ただし
  ヘルプは元から "Feather file to write" と書いており，未検証だっただけで
  サポートされた挙動ではなかった．文書化された意図に検証を合わせる変更として
  `fix:` 寄りに扱い，glob 拡張 (`feat:`) と合わせて **minor** を提案する
  (`0.84.0` → `0.85.0`)．CLAUDE.md の「breaking change → major」を字義どおり
  適用すると `1.0.0` になるが，CLI の入力検証を厳しくしただけで
  リリース成熟度の宣言に相当する major を切るのは釣り合わないと判断した．
  **ここは異論があれば従う．**
- **`maintain_order=True` のコスト**: union 1 回分の並べ直しのみ．join は左の
  順序で出るので前処理出力には影響しないが，順序が実行ごとに変わると
  「パス順で後勝ち」を結果から確かめられなくなるので付けた．
- **F1 の残コスト**: 毎チャンクの `apply_search_values` 内 `unique()` は残る．
  join 自体がチャンクごとに O(V) のハッシュテーブルを作るので支払い済みの
  処理に対する定数倍であり，API に `values_are_unique` のような引数を足して
  まで消す価値は無いと判断した．警告の反復は本対応で 0 回になる．

## リスク

docs は実装済みの挙動へ追随するのみ．
