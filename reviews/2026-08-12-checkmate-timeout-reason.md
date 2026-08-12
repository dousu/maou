---
status: applied
applied_in: pending
date: 2026-08-12
target: [docs/commands/usi.md]
risk: low
reversibility: trivial
---

# `checkmate timeout` が 2 つの事象を畳んでいることを doc に書く

## Trigger

`audits/coverage.md` N5(c)．ユーザ指示 (2026-08-12) で N5 の候補5 として実装
することが決まった．

`rust/maou_usi/src/protocol.rs` は `CheckmateResult::Timeout` と
「詰みは証明できたが手順を復元できない」(`TsumeResult::CheckmateNoPv` /
`Checkmate { moves: [] }`) を**すべて** `checkmate timeout` にシリアライズする．
USI の `checkmate` は `<手順>` / `nomate` / `timeout` の 3 種しか返せないため
**行そのものは変えられない**が，同じ行に潰れていると外から原因を区別できず，
真のソルバ回帰を疑ったときの切り分けが実測 50 回超に膨らんだ (本 run の N5 調査)．

## 承認について

CLAUDE.md § MUST rules は durable doc の編集に承認済み `reviews/*.md` を要求する．
本件は**ユーザが候補5 の実装を明示的に指示している**ため，その指示を承認として
扱い本 run 内で適用した．P2 の standing approval (drift correction) ではない —
新しい挙動の記述なのでドリフトではなく，承認の根拠は上記のユーザ指示である．

記述内容は実装から一意に決まる: 出力トークンが 3 種であることは
`protocol.rs` の `serialize`，`info string` の文言は `agent.rs` の
`handle_go_stream` (mate 分岐) による．

## Before / After

`docs/commands/usi.md` の `go mate` の説明 (Overview 内)．

```diff
-  `checkmate nomate` (only
-  when no-mate is actually *proven*), or `checkmate timeout` (budget or
-  `stop` reached without a conclusion).
+  `checkmate nomate` (only
+  when no-mate is actually *proven*), or `checkmate timeout`. The last one covers
+  **two different situations** the USI spec cannot distinguish — the budget or
+  `stop` was reached without a conclusion, *or* a mate was proven but its move
+  sequence could not be reconstructed. The second case is preceded by
+  `info string checkmate timeout reason=mate-proven-but-pv-unavailable`, so a
+  genuine solver regression can be told apart from a plain budget overrun.
```

## 根拠

| 主張 | 根拠 |
|---|---|
| 出力は 3 種のみ | USI 規約．`protocol.rs` の `serialize` が `checkmate <手順>` / `nomate` / `timeout` にしか写像しない |
| 2 事象が同じ行に潰れる | `backend.rs` の `solve_mate` が `Checkmate { moves: [] }` と `CheckmateNoPv` を返す枝 |
| `info string` の文言 | `agent.rs` の mate 分岐 (`MateWithoutPv` のときだけ emit) |
| 時間切れには付かない | 回帰テスト `test_mate_without_pv_is_distinguishable_from_timeout` の (b) が固定 |
