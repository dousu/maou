---
title: "GHI 再診断: 偽証明は誤診 (実体は verify memo 汚染の偽 Unknown)．設計ドキュメント反映 (loop-ghi §7.5 訂正ほか)"
status: applied
applied_in: 22e1c03
date: 2026-07-02
branch: feat/tsume-solver
target: docs/design/tsume-solver/ (loop-ghi.md, threshold-control.md, move-ordering-and-pv.md)
risk: low  # docs 訂正のみ; コード変更 (0088291=3.4.3, 08c3a6f=3.4.4, 0a764f9=3.4.6) は別 commit で既済
---

> **適用メモ (2026-07-02)**: user 指示「詰将棋ソルバー設計ドキュメントに今回の変更を反映して
> ください」を承認とみなし，設計ドキュメント 3 ファイルへ適用した (22e1c03)．適用時に scope を
> 拡張: loop-ghi §7.5 訂正に加え，threshold-control §3.1 (子 φ 予算 1+ε ε=1/8, 3.4.4) と
> move-ordering-and-pv §9-b (dep memo ゲート + 2-tier verify, 3.4.3/3.4.6 + 旧 3.2.0 状況記述の
> 更新) を反映．**CLAUDE.md [SLOW] table の anchor 更新は本 review から分離** →
> [2026-07-02-claude-md-slow-table-anchors.md](2026-07-02-claude-md-slow-table-anchors.md) (pending)．

## Trigger

user 指示「GHI 偽証明の根本対策を調査，FRONTK の健全性を確認して高速オプションを
デフォルト化」(2026-07-02)．調査の結果，**前セッション (worklog/2026-07-02-073805.md,
reviews/2026-07-02-ghi-false-proof-verify-authoritative.md → loop-ghi §7.5 @ b6031c8)
の診断が誤りだった**ことが確定した．

## 再診断の要旨 (commit 0088291 = maou_shogi 3.4.3)

- 局所化されていた「偽証明」局面 `9/9/4+N1pS1/9/4+R4/5+B3/6R1P/3S2k2/9 w` は
  **全 hand で真の詰み** (全 4 防御 3h2i/3h2h/3h4h/3h4i の子局面を default 構成で
  個別に solve し STRICT Some を確認)．「玉が 3h2i で脱出可能」は誤認だった．
- 真因は **verify_proof の memo 汚染**: 証明候補が path 上の祖先へ戻る探査枝
  (王 3h↔2i・飛 3g↔2g の 4-ply 循環近傍) で当該ノードを verify すると，防御 3h2i の
  子が祖先一致で千日手拒否 → None (その経路では正当) → これを**経路非依存の memo に
  書き込んでいた**．以後，本物の証明線が同ノードを要求するたび memo=None が返り
  root まで連鎖 → STRICT None → (44aeb41 の authoritative 化により) **偽 Unknown**．
  order-dependent なのは「どの文脈で先に verify するか」が探索順で変わるため．
- fix: KH の rep-close gate と同型の dep 伝播 (`dep_out` = 結果が依存した最浅 path
  index)．依存が自 subtree 内で閉じた結果のみ memo．祖先依存 None と budget 枯渇
  None は memo しない．**探索側は完全不変** (fast repro の探索 2,181,791 nodes 一致の
  まま STRICT None → Some(51) へ反転)．
- 帰結: **探索側に偽証明 (false proof) は存在しなかった**．「TT proven の cross-branch
  再利用による proof-tree 循環で非詰みを proven 化」という §7.5 の機構記述は撤回が必要．
  verify authoritative (44aeb41) は defense-in-depth として妥当なので維持．

## 閾値デフォルト変更 (commit 08c3a6f = maou_shogi 3.4.4)

誤診の撤回により REFUTED「閾値ゆるめ = GHI 偽証明露出」の前提が消滅．sweep
(K∈{1,2,4,8} × DIV∈{4,8,16}) の結果，子 φ 予算を `2nd_phi + 1` →
`2nd_phi + 2nd_phi/8 + 1` に変更:

| anchor | 旧 (3.4.3) | 新 (3.4.4) |
|---|---|---|
| 29te find_shortest 総 nodes | 531,296 | **396,516** (−25%) |
| 39te find_shortest 総 nodes | 24,773,536 | **17,545,528** (−29%) |
| 39te wall (4 runs avg) | 110.7s | **95.4s** (−13.8%) |
| 答え / PV | 29・39・31 / canonical | **不変** |
| 161 tests / STRICT | 161-0 / Some | **不変** |

## Proposed doc changes (承認後に適用)

1. **docs/design/tsume-solver/loop-ghi.md §7.5**: 「転置による偽証明 (proof-tree 循環)」の
   機構記述を撤回し，再診断の内容へ書き換える:
   - 事象の実体 = verify_proof の path 依存 None の memo 汚染 (偽 Unknown; 健全性は
     侵されていなかった)．
   - fix = dep 伝播による memo ゲート (3.4.3)．verify authoritative は維持 (多層防御)．
   - 「探索側 GHI-safe 化 (proof-path 循環追跡) が research-level 課題」の記述は削除
     (前提が消滅)．
2. **CLAUDE.md [SLOW] table**: `test_29te` / `test_39te_measure` の canonical 記述を更新:
   - test_29te: 396,516 nodes (find_shortest 総数) / mate-29 / STRICT Some(29)
   - test_39te_measure: 17,545,528 nodes / Some(39) / canonical PV
   (現行の「9,288 nodes」「4,272,957 nodes / Some(55)」は first-mate 時代の値で
   3.4.x の実 assert とも不一致だった)．

## 却下しない代替案

- §7.5 を丸ごと削除する案 → 却下: 誤診の経緯と再診断は再発防止の記録価値が高い．
  訂正追記の形で残す．
