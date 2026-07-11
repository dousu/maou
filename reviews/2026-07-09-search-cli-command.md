---
status: applied
applied_in: e291ba4
title: maou search CLI コマンドの追加 (docs/commands/search.md 新設の追認)
target: docs/commands/search.md
---

# maou search CLI コマンドの追加 (docs/commands/search.md 新設の追認)

## Trigger

worklog/2026-07-09-113225.md — 探索コード完成トラック④で新 CLI コマンド
`maou search` を追加した．public API 面 (`maou._rust.maou_search` /
`maou.interface.search` / `maou.app.search.run.SearchRunner`) と
durable-doc target (docs/commands/) に触れるため起票する．

## Proposed change (適用済み — 追認を求める)

- **docs/commands/search.md を新設** (Overview / CLI options 表 / wheel
  feature 要件 / 出力仕様 / 実装参照 — evaluate.md と同形式)．
  適用済み commit: **e291ba4** (Release latest-gpu 参照の追記は d3b024a)
- 根拠: CLAUDE.md Documentation ルール「MUST create a new
  docs/commands/<command-name>.md when adding a new CLI command」を
  standing mandate として先行適用した (レビュー承認を待つと CLI 追加の
  MUST 義務と衝突するため，事後追認方式とした)

## What this enables

- `maou search` の CLI 仕様が evaluate.md と同じ形式で参照可能になる．
  特に評価値の意味論 (Eval = 600 × logit を **探索後勝率**に適用 —
  evaluate の生 logit とスケール互換・値は別物) を仕様として固定する
- GPU wheel (Release latest-gpu) の feature 要件と入手方法が
  コマンド doc から辿れる

## What this constrains

- `search_board.py` のオプション変更時は docs/commands/search.md の
  同期義務が発生する (CLAUDE.md Documentation ルール)
- CLI 出力の `Eval:` / `WinRate:` ラベルは evaluate と共通のパース面として
  維持する

## Rollback plan

docs/commands/search.md を削除し，app.py の LAZY_COMMANDS から
"search" エントリを外す (コード側は e291ba4 の revert)．
