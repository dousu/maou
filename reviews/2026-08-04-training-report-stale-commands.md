---
title: 過去の調査レポートに「当時のコマンドであり現在は動作しない」注記を入れる
date: 2026-08-04
status: applied
applied_in: ccc6d29
target:
  - docs/TRAINING_INVESTIGATION_REPORT.md
risk: low
reversibility: easy
---

# 提案: `TRAINING_INVESTIGATION_REPORT.md` に時点注記を追加する

## 背景

`reviews/2026-08-04-value-target-and-dead-options.md` で `--gce-parameter` を
削除した結果，`docs/TRAINING_INVESTIGATION_REPORT.md` の実行例 3 箇所
(L55 / L182 / L205) がそのままではエラーになる状態になった．

## 変更内容

**本文は書き換えない．** 冒頭に注記ブロックを 1 つ追加するだけ:

- 本レポートが 2025-11-01 時点の記録であること
- コマンド例は当時の CLI をそのまま残しており現在は動作しないこと
- `--gce-parameter` は 2026-08-04 に削除されたこと (根拠 review へのリンク)
- 現行のオプションは `docs/commands/learn_model.md` を参照すること

## 選択の理由

コマンドを現行 CLI に書き換える案 (A) も検討したが却下した．
調査レポートは「その時点で何を実行したか」の記録であり，コマンドを
書き換えると**再現性の記録が壊れる**．今後も CLI が変わるたびに追随が必要になり，
そのたびに当時の実行内容が失われていく．

注記方式なら，レポートは記録として不変のまま，読者は現行 CLI へ誘導される．

## リスク

- **低**: 記述の追加のみ．挙動に影響しない．逆行は容易
- 同種の時点依存ドキュメントが他にもある場合，同じ扱いを繰り返す必要がある
  (今回の grep では本ファイルのみが該当)
