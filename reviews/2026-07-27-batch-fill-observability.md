---
status: pending
applied_in:
date: 2026-07-27
target:
  - docs/commands/search.md
  - docs/design/usi-engine/verification.md
risk: low
reversibility: trivial
---

# 提案: バッチ充填率と衝突数を `maou search` の Stats に出し，「session 変動」の断定を撤回する

## Trigger

単発 30 秒探索の `nps` が同一コマンド・同一ビルドで 1 時間のうちに
**10,095 → 4,595 (2.2 倍低下)** した．これを **Colab の session 変動**と
断定して verification.md §1 に書いたが，**その断定は根拠不足だった** (user 指摘)．

会計側の疑いは薄い:

- 空回りは **0.008%** (303,171 playouts に対し 24 件) — 水増しではない
- `elapsed_ms` は 30,031 / 30,065 でほぼ同一，`warmup_ms` は別掲
- 同一ビルドなので `playouts == eval_items` の意味も同一

しかし **nps の解釈に穴がある**: TensorRT は固定 shape のため
`pad_to = batch_size` で**毎バッチを `--batch-size` へ padding** する．
つまり**バッチ充填率が下がると，GPU の仕事量が同じでも `playouts` / `nps` が
下がる**．そして充填率を示す `avg_batch` / `eval_batches` / `collisions` は
`SearchStats` に存在するのに **`maou search` の Stats 行に出していなかった**．
⇒ 「GPU が遅くなった」と「充填率が落ちた」を**区別できない状態で断定していた**．

## コード変更 (既に実装済み — 本レビューはその docs 反映が対象)

`maou 0.60.2`: `maou search` の Stats 行へ `eval_batches` / `avg_batch` /
`collisions` / `nodes_used` を追加 (いずれも `SearchStats` に既存で，PyO3 でも
公開済み．表示だけの追加で探索挙動は不変)．

```
Stats: playouts=2000 terminal_backprops=0 nps=69631 eval_batches=250 avg_batch=8.0
       collisions=0 nodes_used=2001 elapsed_ms=28 ... stop=playout_limit
```

## ドキュメント変更内容 (本レビューの承認対象)

### (a) `docs/commands/search.md` — Stats 例と読み方

例の行を新フィールド込みへ差し替え，**`avg_batch ÷ --batch-size` が充填率**で
あること，**`nps` より先に充填率を読むこと** (TensorRT の padding のため充填率
低下がそのまま nps 低下に見える)，**`--threads` を上げたときの `collisions` が
充填率低下の主因になりやすい** (衝突は収集途中のバッチを即時評価に回す) を
追記する．

### (b) `docs/design/usi-engine/verification.md` §1 — 断定の撤回

「**Colab の session 変動である**」→ 「**原因は未特定**」へ書き換え，
会計の水増しではない根拠 (空回り 0.008% / `elapsed_ms` 同一) と，
**GPU 側か充填側かは当時の出力では区別できなかった**ことを明記する．
併せて次回の計測規則を追加: **`avg_batch ÷ --batch-size` と `collisions` を
必ず併記し，充填率が一定なら GPU 側・落ちていれば充填側と判定する**．
「絶対値を session をまたいで比較しない / 比は同一 session の天井で出す」は
どちらの原因でも成り立つので維持する．

## 代替案と棄却理由

- **§1 の記述をそのまま残す (session 変動と書き切る)**: 棄却．区別できていない
  ことを「変動」と名付けると，**次に同じ現象を見た人が充填率を疑わない**．
  compass の TRIPWIRE「性能数値を報告する前に実測か概算か明示する」と同じ趣旨で，
  原因未特定は未特定と書く．
- **`--threads 1` での再測定を待って書く**: 棄却しないが後続．計測器が無いまま
  再測定しても同じ「区別できない 2 点」が増えるだけなので，**先に計測器を入れる**
  順序にした．
- **`eval_items` も出す**: 棄却．`avg_batch × eval_batches` で足り，行が長くなる．
  (`playouts == eval_items` は実装上の恒等式で，回帰テストで pin してある．)

## リスクと理由

- **risk: low** — 表示フィールドの追加のみ．探索挙動・既定値は不変．
- **reversibility: trivial** — Stats 行の 4 フィールドと docs の追記を戻すだけ．

## ロールバック

`src/maou/app/search/run.py` の追加フィールド，`search.md` の例と読み方，
§1 の書き換えを元に戻す．
