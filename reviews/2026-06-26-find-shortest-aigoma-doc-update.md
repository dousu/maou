---
status: applied
applied_in: c5de0fb
title: find_shortest 余詰探索・無駄合いフィルタ再配線・len-bound 修正を tsume-solver doc へ反映
target: docs/design/tsume-solver/move-ordering-and-pv.md, docs/design/tsume-solver/aigoma-optimization.md
risk: low (doc only; 旧記述が現実装と乖離している箇所の更新)
---

## Trigger

worklog/2026-06-26-161404.md のセッションで find_shortest を honor 実装 (余詰探索),
無駄合いフィルタを再配線, len-bound 偽 proof を修正 (3.2.0, commit 03b5849)．
既存 doc の以下 2 セクションが旧状態を記述しており現実装と乖離:

- `move-ordering-and-pv.md` §9-b.3: find_shortest を「PV 上で未証明王手を追加証明する
  `complete_or_proofs` 相当・最短保証なし・CheckmateNoPv」と記述 = orphaned だった旧設計．
- `aigoma-optimization.md` §8.2: 無駄合い filter を「opt-in (muda_filter 既定 off)・寄与は小さい
  ため既定無効」と記述 = filter が適用されていなかった旧状態 (この主張は誤りで，無駄合いが
  最短手数に算入されていた)．

## Problem

doc が現実装と矛盾し，find_shortest/無駄合いの正しい仕様 (最短手数のための必須機構) と
今セッションで判明した重要事項 (詰将棋の oracle はユーザ; KH は無駄合いを数え不可; 無駄合いの
正確な定義; 39te len-43 完全性バグ) が記録されていない．

## Proposed change

1. `move-ordering-and-pv.md` §9-b.3 を「find_shortest (余詰探索) と最短手数」へ全面改稿:
   len-aware df-pn の余詰ループ (len=d-2 反復・dual-range TT・loop guard), len 予算強制の
   3 点 (MateLen::sub clamp / look-ahead budget gate / len 予算切れ cutoff), oracle=ユーザ
   (KH 不可), 現状 (29te=29 confirm / 39te 45 で len-43 false-disproof 残) + divergence-probe．
2. `aigoma-optimization.md` §8.2 を「無駄合いの除外 (最短手数のため既定 on)」へ改稿:
   futile/chain 分類と futile drop skip, 無駄合いの正確な定義 (支えなし**だけでは不十分**=
   同手数以下で詰む必須・駒ずらし保護), 3.2.0 再配線 (PNS→mid 移行で未適用だった), 旧「寄与は
   小さい」が誤りだった旨．

実際の置換テキストはこの review 承認後に model が適用する (step 5 on-approval-applies flow)．

## Alternatives considered

- doc 更新せず worklog のみ: 却下 (find_shortest/無駄合いは durable な仕様であり doc が
  source of truth; 旧記述が誤情報として残る)．
