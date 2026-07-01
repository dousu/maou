---
title: "GHI 偽証明 (proof-tree 循環) と STRICT verify 権威化を loop-ghi §7.5 に追記"
status: applied
applied_in: b6031c8
date: 2026-07-02
branch: feat/tsume-solver
target: docs/design/tsume-solver/loop-ghi.md
risk: low  # 既存 §7 (GHI 対策) への追記のみ; コード変更 (44aeb41) は別 commit で既済
---

## Trigger
worklog/2026-07-02-073805.md にて，閾値ゆるめ (FRONTK/FRONTDIV) が露出させた **GHI 偽証明
(false proof)** を局所化・根治した (commit 44aeb41, maou_shogi 3.4.2: STRICT verify を
authoritative 化)．user が「今回のバグ修正に関して詰将棋ソルバードキュメントに記載する内容が
あるか確認して必要なら追記」を指示 (2026-07-02)．

既存 loop-ghi.md は §7.1 (現探索スタック上の循環=`path_depths`)・§7.2 (horizon false NoMate)・
§7.3 (visit-history dominance)・§7.4 (len-aware) を扱うが，**転置 proven エントリの cross-branch
再利用による proof-tree 循環 = false PROOF** と，その健全化 (verify 権威化) を**記述していない**．
= durable-doc target への追記が必要．

## Proposed change (適用済)
`docs/design/tsume-solver/loop-ghi.md` に **§7.5「転置による偽証明 (proof-tree 循環) と STRICT
verify の権威化」** を追記:
- proof-tree 循環による false proof の機構 (proven X を分岐 B2 で TT 再利用 → 証明内へ戻る循環)．
  §7.1 の `path_depths` は探索スタック上の循環のみ検出し，これは捕捉できない．localize 例 (AND
  局面で玉が 3h2i 脱出可なのに proven)．
- TT look_up proven 分岐 (`pn==0`) が unknown 分岐の rep 再チェックを欠く穴と，cross-branch
  clean-proof では `is_possible_repetition` が立たず targeted fix が不発なこと．探索側 GHI-safe 化
  (proof-path 循環追跡) は research-level 課題として残す旨．
- 採用した健全化: `verify_proof` (GHI-correct) を `solve_impl` の最終権威にし，STRICT Some のみ
  Checkmate・None は Unknown (soundness > completeness)．default 挙動不変．

## Rationale
GHI soundness の設計節 (§7) に，新規に判明した false-PROOF ケースと現行の健全化方針を残さないと，
将来「なぜ verify が authoritative なのか」「探索側 GHI-safe 化が未了なのはなぜか」が失われる．
コード変更は 44aeb41 で既済ゆえ本 review は docs 追記のみ (low risk)．
