---
status: applied          # pending | approved | applied | rejected
applied_in: d8676e90
date: 2026-09-23
target: [docs/commands/utility_search_values.md]
risk: low
reversibility: trivial
---

# `search-values` に `--max-ply` と来歴記録を足したのでドキュメントを追随させる

## Trigger

Arm 1 (探索値を ply 60-99 に投与する対照) の設計で 2 つ分かった．

1. **帯の上限が切れない．** `--min-ply` (既定 60) はあるが上限フィルタが
   CLI・interface・app のどこにも無く，`ply 60-99` を対象にできない．
2. **シャードに teacher が残らない．** 出力の 6 列
   (`id` / `searchWinRate` / `playouts` / `stop` / `elapsedMs` / `warmupMs`) にも
   併設ファイルにもモデルの識別子が無い．このため**既に蓄積済みの ply>=120 の
   11M 局面 (240 時間ぶん) が「どの teacher で探索したか分からない」という理由で
   Arm 1 に再利用できない**ことが判明した (user 確認済み)．

Arm 1 は実測 45,800 局面/時 から **約 266 時間 / 12〜13 セッション**を見込む．
同じことを繰り返さないよう，蓄積を始める前に両方を実装した (PR 同梱，版数 0.104.0)．

## 変更 1 (`### 選定はラベルと独立` の本文, 28 行目)

現行:

```
絞り込みには**手数 (`--min-ply`) と重複しか使わない**．
```

変更後:

```
絞り込みには**手数 (`--min-ply` / `--max-ply`) と重複しか使わない**．
帯は `[--min-ply, --max-ply)` で，`scripts/soften_result_value.py` と同じ
意味論にそろえてある (下限は含み，上限は含まない)．`--max-ply` を省けば上限なし．
```

## 変更 2 (`## CLI options` の表, `--min-ply` の行の直後に 1 行追加)

```
| `--max-ply INT` | optional | この手数**未満**の局面のみ対象にする．帯は `[--min-ply, --max-ply)`．省略すると上限なし (従来動作)．`--min-ply` 以下の値はエラーにする． |
```

## 変更 3 (`## Output` に来歴の節を追加)

`## Output` の末尾へ:

```
### `provenance.json` (来歴)

出力ディレクトリには，シャードとは別に `provenance.json` が置かれる．
**シャードには探索したモデルが一切残らない**ので，どの teacher が出した値かは
ここだけが知っている．実行のたびに 1 レコードが追記される:

| フィールド | 意味 |
|---|---|
| `started` | 実行開始時刻 (UTC) |
| `model_name` | `--model-path` のファイル名．**VM ごとにパスは変わるので，記録として有効なのはこちら** |
| `model_sha256` | モデルの sha256 (mock 評価器なら `null`) |
| `model_path` | 実行時に見えていたパス (参考情報) |
| `min_ply` / `max_ply` | その実行の帯 |
| `max_playouts` / `time_ms` | 探索予算 |

`--resume` のとき，**既存の来歴と違う sha256 のモデルを渡すとエラーで止まる**．
1 本の蓄積を十数セッションに分けて継ぎ足す運用では，途中で別のモデルを渡しても
気付けず教師が混ざったデータが黙ってできあがるため (実際に過去の蓄積分が
この理由で再利用できなくなった)．teacher を変えるときは `--output-path` を
新しいディレクトリにする．

`.feather` で終わらないので，`--search-value-path` の読み手は拾わない．
```

## 変更 4 (`## Usage` に Arm 1 の実行例を追加)

`# GPU で 100 万局面ぶんを探索する` の例の後へ:

```
# 帯を切って貯める (Arm 1: ply 60-99)．Colab の 1 セッションぶんに区切る
maou utility search-values \
  --input-path hcpe --output-path search_values/ply60_99 \
  --model-path teacher.onnx --min-ply 60 --max-ply 100 \
  --max-positions 900000 --playouts 800 --batch-size 64 --threads 1 \
  --cuda --tensorrt --resume
```

## 影響

- 既定動作は変わらない (`--max-ply` 省略時は従来どおり上限なし)
- 既存の出力ディレクトリに `provenance.json` は無いので，初回の `--resume` は
  来歴が空として通り，そのときのモデルが以後の基準になる
