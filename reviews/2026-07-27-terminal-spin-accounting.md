---
status: pending
applied_in:
date: 2026-07-27
target:
  - docs/commands/selfplay.md
  - docs/design/usi-engine/verification.md
risk: low
reversibility: trivial
---

# 提案: 空回り (terminal spin) の分離計上を docs へ反映する

## Trigger

GPU 追認 run で `throughput: 668,449 playouts/秒` = 物理上限の約 61 倍が観測され，
`playouts` が探索量の指標として使えないことが数値で確定した．原因は
`maou_search` が **終端到達だけで折り返した backprop (空回り) を `playouts` に
合算していた**こと (`search.rs` の `Selection::Backpropped` → `complete_playouts(1)`)．

コード側で会計を分離した (本レビューはその**ドキュメント反映**が対象):

- `SearchStats.playouts` = 葉評価を伴う実 playout のみ (`eval_items` と一致)
- `SearchStats.terminal_backprops` = 空回り (新設)
- **予算 (`max_playouts`) は両者の合計で消費** — 停止点は従来と同じ
- `carried_visits` の算出も両者の合計を引くよう修正 (空回り分が引き継ぎとして
  二重計上されていた)

計測値の裏取り: mock 6 手の A/B テスト対局でも消費予算の **約 25%** が空回り
(`test_selfplay_ab_per_side_budget`)．深さ上限を超えた降下は `mark_terminal` を
しない = 証明で畳めないため，空回りは原理的に残る
(`test_terminal_spin_is_separated_from_playouts` で pin)．

## ドキュメント変更内容 (本レビューの承認対象)

### (a) `docs/commands/selfplay.md` § Output — JSONL のフィールド追加

per-game レコードの列挙へ `terminal_backprops` を追加し，`playouts` の意味を
明記する:

> `"playouts": N` は**葉評価を伴った実探索量**で，終端に当たって折り返しただけの
> backprop は `"terminal_backprops": N` に分離する (予算は両者の合計で消費)．

### (b) `docs/commands/selfplay.md` § Output — サマリ行の追加

stdout サマリの説明へ 1 項目追加:

> `terminal spin:` 空回りの総数と，消費予算に占める割合．証明済み終端・千日手・
> 最大手数が近い終盤ではここが大半を占め，公称予算に対する実探索量が大きく
> 下がる．`throughput:` の分子は**実 playout のみ**なので，この行と併せて読む．

### (c) `docs/design/usi-engine/verification.md` §4.2 — tripwire の更新

本セッションで追記した実測段落 (668,449 playouts/秒) に，会計修正後の読み方を
1 文追加する:

> (この水増しは `maou_search` 0.23.0 / `maou_usi` 0.15.0 で解消した．
> `throughput:` は実 playout のみを分子に取るようになり，空回りは
> `terminal spin:` 行に分離される．**修正前の run の playouts と直接比較しない**
> — 同じ探索でも報告値が 1-2 桁小さくなる．)

## 代替案と棄却理由

- **`playouts` の意味を変えず，別名フィールドだけ足す**: 棄却．`throughput:` /
  `nps` / `carried_visits` の分母・分子が水増しのままになり，「水増しに気づいた
  人だけが正しく読める」状態が続く．実際 §4.2 の tripwire は「気づけ」という
  運用回避策であり，会計を直せば不要になる性質のもの．
- **空回りを予算から外す (ユーザ提案)**: **本レビューには含めない**．深さ上限
  超過は `mark_terminal` しないため証明で畳めず，予算から外すと葉収集ループが
  永久に回る (playout 制では停止条件が消える)．空回り自体の上限を別途設けた
  うえで棋力 A/B が要る変更なので，別レビュー・別 PR とする．
- **docs を更新しない**: 棄却．`selfplay.md` § Output は JSONL の全フィールドを
  列挙する契約になっており，追加フィールドを載せないと実体と乖離する．

## リスクと理由

- **risk: low** — ドキュメントの記述追加のみ．コード変更は本レビューの対象外
  (既にテストで pin 済み)．
- **reversibility: trivial** — 追記した行を消すだけ．

## ロールバック

`docs/commands/selfplay.md` の追記 2 箇所と，`verification.md` §4.2 の追記
1 文を削除する．
