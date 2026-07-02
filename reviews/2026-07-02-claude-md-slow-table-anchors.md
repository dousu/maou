---
title: "CLAUDE.md [SLOW] table の canonical anchor を 3.4.x 実態へ更新"
status: applied
applied_in: 44d8cd3
date: 2026-07-02
branch: feat/tsume-solver
target: CLAUDE.md
risk: low  # 表の数値更新のみ
---

## Trigger

[2026-07-02-ghi-rediagnosis-verify-memo-fix.md](2026-07-02-ghi-rediagnosis-verify-memo-fix.md)
から分離 (CLAUDE.md は設計ドキュメントと別承認とするため)．子 φ 予算の default 変更
(3.4.4, 08c3a6f) で canonical node anchor が変わり，かつ現行の表の値は 3.4.x の実 assert
とも不一致 (first-mate 時代の値のまま)．

## Proposed change (承認後に適用)

CLAUDE.md「重いテスト (Rust dfpn)」の表を更新:

| テスト名 | 現行記述 | 提案 |
|---|---|---|
| `test_29te` | mid 1te/3te/29te canonical (9,288 nodes / mate-29 / STRICT Some(29)) | mid 1te/3te/29te canonical (**396,516 nodes** (find_shortest 総数) / mate-29 / STRICT Some(29)) |
| `test_39te_measure` | 39te canonical (4,272,957 nodes / Some(55) / sound) | 39te canonical (**17,545,528 nodes / mate-39 / STRICT Some(39) / canonical PV**) |

数値の根拠: 3.4.4 (08c3a6f) の canonical anchor (tests.rs の assert と一致)．
`Some(55)` は既に誤り (3.4.0 以降 39 手到達済) のため合わせて訂正する．
