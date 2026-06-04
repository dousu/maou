---
title: "mid / mid_v2 の配線を全削除して廃止・全テストを mid_v3 へ移行・pns を deprecated 化"
date: 2026-06-05
status: approved
crate: maou_shogi
version: TBD (breaking, major 候補)
---

# mid / mid_v2 廃止 + pns deprecated 化 + 全テスト mid_v3 移行

## Trigger
- ユーザ指示（`/checkpoint-context` 引数, worklog/2026-06-05-010632.md）:
  「次のセッションでは mid と mid_v2 の配線をすべて削除して廃止とし，すべてのテストを mid_v3 に置き換える方針とします．pns も deprecated にする方針とします．」
- 背景: Phase 36 で **mid_v3 が default で GOAL 達成**（29te 18,539 nodes / mate-29 / STRICT Some(29) < KH 19,270）．
  mid / mid_v2 はもはや主戦場でなく，正しさ修正（[[project_dfpn_domoff_none_mate1_bugs]]）で mid_v2 SLOW テストが退行中（削除予定で許容）．

## 提案
1. **mid / mid_v2 経路を全削除**: `solve_via_v2` / `solve_via_v2_find_shortest` / `solve_via_v1`(mid) 等の探索本体と，それらだけが使う TT/param/診断フィールドを除去．
2. **全テストを mid_v3 ベースへ移行**: mid/mid_v2 を呼ぶテストを `solve_via_v3` ベースへ書き換える or 削除する．
   - 特に退行中の `test_tsume_39te_ply24_mate15_regression`(+ _with_both/_dag_correction/_delayed_move_list/_dominance/_handset) は **mid_v3 で同局面が解けることを確認**してから移行（解ければ退行解消・移行完了，解けなければ別途調査）．
3. **pns を deprecated 化**．

## ⚠ critical: pns の依存切り分け（壊さないための前提）
`pns.rs` は探索本体（PNS）と movegen ヘルパが同居している．**mid_v3 は以下のヘルパに依存**するため一律削除は不可:
- `generate_check_moves` / `generate_check_moves_cached`（OR ノード手生成 + look-ahead 候補）
- `generate_defense_moves_inner`（AND ノード手生成 + STRICT VERIFY）
- `is_legal_quick`（逆王手 drop 合法手検証, Phase 36 で配線）
- `has_mate_in_1_with`（※ mate_move_in_1ply 経由; 削除可否は要確認）
- `board.rs::mate_move_in_1ply` / `is_checkmate_after_bb`（Phase 36 で `&mut self` / `Option<bool>` 化, mid_v3 必須）

→ **探索本体(PNS solve)は廃止対象，movegen ヘルパは存続 or 別モジュールへ移設**．deprecate は段階的に（`#[deprecated]` 属性 → 参照ゼロ確認 → 削除）．

## 影響
- 公開 API surface の縮小（`solve_via_v2` 等の除去）= breaking change → **major bump 候補**（要判断）．
- `CLAUDE.md` の記述更新が必要な可能性（重いテスト一覧の mid_v2 系, docs/design/tsume-solver/）．本 review は方針合意のためのもので，実装は次セッション．
- compass invariant「mid/mid_v2 は触らない」→「mid/mid_v2 は削除済」へ更新．

## 検証方針
- 移行後も非 ignored suite green + production soundness guard（`test_tsume_39te_ply2_no_false_nomate`, `..._soundness_depth25`）pass．
- mid_v3 で 29te(18,539/Some 29) + 逆王手例題(mate-7) + 39te ply24 系を健全に解けること．
