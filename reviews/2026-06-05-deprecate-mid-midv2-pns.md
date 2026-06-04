---
title: "mid / mid_v2 の配線を全削除して廃止・全テストを mid_v3 へ移行・pns を deprecated 化"
date: 2026-06-05
status: approved
crate: maou_shogi
version: 2.0.0 (breaking; Phase 1 = mid_v2 廃止のみ実施済, 2026-06-05)
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

---

## 実施結果 — Phase 1: mid_v2 廃止のみ (2026-06-05, uncommitted, v2.0.0)

ユーザ判断で **段階的** に実施（v1 mid と production→mid_v3 配線は次フェーズへ）．

### 実施したこと
1. **mid_v2 探索本体を solver.rs から全削除**: `mid_v2` / `solve_v2` / `solve_v2_with_budget` / `solve_v2_with_pv` / `solve_via_v2` / `solve_v2_find_shortest` / `eliminate_double_count_mid_v2` / `extend_threshold_for_mid_v2(_mode)` / 死コード `refine_mate_distance(_inner)`（`mid_v2` を呼ぶ未使用 PV-walk）．
2. **mid_v2.rs → local_expansion.rs に rename**（`git mv`, 履歴保持）+ v2探索専用 `search_impl` / `MiniMidContext` + その 3 テストを strip．**共有 LE 型（`MidSearchResult` / `MidLocalExpansion` / `REPETITION_NONE` / `build_delayed_chain(_chuai)`）は mid_v3 が依存するため存続**．mod.rs / mid_v3.rs / solver.rs の `super::mid_v2::` → `super::local_expansion::` に更新．
3. **テスト surgery**: 159 の `solve_via_v2` caller + 直接 `solve_v2*` caller の研究/診断/sweep/benchmark テスト計 **166 関数を削除**．基本 correctness **12 関数を `solve_via_v3` へ移行**（`test_tsume_1te(_gote)` / `test_tsume_5`(17手 PV) / `test_tsume_9te` / `test_no_checkmate(_gote)` / `test_uchifuzume_promoted_rook_fails` / `test_tsume_3_ryu_2a_not_checkmate` / `test_timeout`）→ **mid_v3 で 17/17 pass**．tests.rs 18,228 → ~2,300 行．
4. **v2 専用 診断フィールド/メソッド除去**: `mid_v2_visit_counts` / `mid_v2_first_ply_count` / `get_visit_counts` / `parent_meta` / `diag_or_nodes` / `diag_or_recursions` / `diag_and_nodes` / `diag_and_recursions` / `diag_trace_count` / `look_up_pn_dn_md_bounded` / `is_shallow_remaining` / `mid_via_pns_boundary`（compiler が never-read/never-used 確認済）．

### 計画からの逸脱（理由つき）
- **mid_v2.rs 丸ごと削除は不可**: mid_v3（GOAL 達成路）が LE データ構造の上に構築されている → **rename + strip** へ変更（ユーザ承認）．Explore 報告の「mid_v2.rs entire file deletable」は誤り．
- **proof_hand.rs は削除せず**: `hand_gte_forward_chain` が **tt.rs（production TT antichain 比較, 7 箇所）で使用** → 削除不可．Phase-28 dead 関数のみ残置（別 experiment, 本タスク対象外）．
- **39te ply24 soundness guard 3 件は migrate せず削除**（ユーザ判断）: → 下記 gap 参照．

### ⚠ mid_v3 39te ply24 gap（要 follow-up, 別 review 記録）
移行候補だった 3 guard は **mid_v3 で release 実行すると全 FAIL**:
- `test_tsume_39te_ply24_mate15_regression`: got **Mate(27)**, expected Mate(15)（`find_shortest(false)` を mid_v3 が無視 → 非最短 mate）．
- `test_tsume_39te_ply24_mate15_soundness_depth25`: depth-25 PV mismatch．
- `test_tsume_39te_ply2_no_false_nomate`: **10M nodes で NoCheckmate**（budget 切れを Unknown でなく NoCheckmate 報告 = robustness/soundness 疑い）．
これらは **mid_v2(HEAD) でも既に退行中**（Phase 36 正しさ修正の余波）で両エンジン未 pass．ユーザ判断で削除し，gap は follow-up review に記録．

### 検証
- 非 ignored lib suite: **170 passed / 0 failed / 11 ignored**（`--test-threads=1`, release）．
- 基本 correctness 12 件 + `test_mid_v3(_counter_check_example)`: mid_v3 で全 pass（tsume_5 の 17手 PV 一致）．
- build green（warning 11→ lib-only 22; 残りは entry/tt の `disproven_len` feature・proof_hand Phase-28・mid_v3 `phi/delta`・`prev_hits`・`PostCaptureSummary` = pre-existing/shared, 別 follow-up）．
- version: maou_shogi **1.14.0 → 2.0.0**（pub `solve_via_v2`/`solve_v2*`/`get_visit_counts` 除去 = breaking）．

### 次フェーズ（未実施）
- **pns deprecated 化**（探索本体 vs movegen ヘルパの切り分けは上記 critical 節の通り）．
- **v1 `mid()` 削除 + production `solve()`/`solve_tsume*` を mid_v3 へ配線**（現状 production は依然 v1 mid．Rust 単体テストは v1 を 1 件も叩いていない点に注意）．
- **mid_v3 39te ply24 gap 調査**（find_shortest 無視・budget 切れ NoCheckmate 報告）．
- 残 dead-code（entry/tt `disproven_len`, proof_hand Phase-28, mid_v3 `phi/delta`, tests.rs `print_result`）の整理．
