---
title: 詰将棋ソルバー設計ドキュメントの実装忠実化 (dead-flag 参照の除去・stale 記述の更新)
status: applied
approved_by: user (/goal directive, 2026-07-02)
applied_in: 8b853cd
---

# 詰将棋ソルバー設計ドキュメントの実装忠実化

## 承認の経緯

user が 2026-07-02 の /goal 指示で本作業を明示的に指示した:

> 詰将棋ソルバー設計ドキュメントが現状の実装に忠実な設計資料となっているかを
> 評価して修正してください。

(同指示は調査コード・dead code・タスク文脈依存コメントの整理も含み，コード側は
refactor(dfpn)! @ d662e70 で適用済．本 review はその docs 側の適用記録である．)

## 評価結果 (要旨)

docs/design/tsume-solver/ の記述 15 項目をコードと突合した結果，機構レベルの記述は
全て正確だった (閾値式・TT 構造・GC 条件・API デフォルト・verify 2-tier 等)．
一方で以下の系統的な不忠実があった:

1. **dead flag を制御点として記述**: `Params::scope_disproof / repetition /
   use_visit_history_dominance / path_dominance / decouple_edge_cost /
   obvious_final_max_depth / use_delayed_move_list / dml / muda_filter /
   capture_dedup / cross_dedup / use_dag_correction` は探索から一切読まれない
   化石フィールドだった (d662e70 で削除)．実際の機構は常時有効 (または不存在)．
2. **版数スタンプ**: index.md 自身が「版数は記さない」と定める一方，複数 doc に
   3.x.y 参照が残存．
3. **stale な perf/状態記述**: aigoma §8.4 の「39te 24.8M nodes/113s・collapse は
   今後の課題」(現況: 17.5M/70s 帯・collapse は計測済み perf 中立で不採用)，
   39te-oracle.md の解決済み PENDING 節，「非 ignored 90/0」等．
4. **セッション文脈の残存**: 「案A」ラベル・「再診断による訂正」等の campaign
   物語体 (docs は現行設計を時制フリーに記述すべき)．
5. **軽微な式の不正確**: root 閾値成長は ⌈⌉ でなく切り捨て (`(x*1.7) as u64 + 1`)．

## 適用した修正

| ファイル | 修正 |
|---|---|
| index.md | 実装ファイル表: 版数除去・`solver.rs` の内容を現行へ・存在しない `dfpn/board.rs` を除去 |
| search-architecture.md | 1.7× 成長式を切り捨て表記へ・§2.6 版数除去 |
| threshold-control.md | 1+ε の版数/日付スタンプ除去 (数値は保持)・TCA 擬似コードの increment 位置を実装通り「入口のみ」へ・§3.4 切り捨て表記 |
| proof-disproof-numbers.md | 存在しない `use_dag_correction` opt-in の note を削除 |
| loop-ghi.md | §7.2/§7.3: dead flag 参照 → 「常時有効」記述へ (存在しない repetition 精緻化・path_dominance note を削除)・§7.5 を時制フリーに再構成 (誤診訂正の物語 → 現行設計 + 設計根拠) |
| move-ordering-and-pv.md | decouple モード記述削除・版数除去・「案A」→ 無駄合い len credit・canonical anchor は tests.rs assert 参照へ |
| initial-heuristics.md | §5.2 を folded 設計の記述へ書換 (不存在の切替オプション記述を除去)・§5.3 の `obvious_final_max_depth` 参照を実装 (深さ制限なし) へ |
| aigoma-optimization.md | §8.1/§8.2 dead flag → 常時有効・cross-deduction は「機構は現行コードに無い」と明記・§8.4 版数/案A 除去 + perf 記述を現況 (反証 pass ~90% 支配・collapse 計測済み中立) へ |
| 39te-oracle.md | 解決済み FIXED/(旧)PENDING 節を「解決済の過去乖離」へ集約 (oracle データの CONFIRMED サブ局面表は全て保持)・更新履歴に 2026-07-02 追記 |

## 不変条件

- oracle データ (SFEN・手数・PV・CONFIRMED マーク) は一切変更しない．
- 計測数値 (nodes −29% 等) は保持し，版数アンカーのみ除去．
- legacy/ 配下は無変更 (アーカイブ)．
