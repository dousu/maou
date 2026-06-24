---
status: applied
applied_in: 1636885
date: 2026-06-24
target:
  - docs/design/tsume-solver/index.md
  - docs/design/tsume-solver/search-architecture.md
  - docs/design/tsume-solver/threshold-control.md
  - docs/design/tsume-solver/proof-disproof-numbers.md
  - docs/design/tsume-solver/initial-heuristics.md
  - docs/design/tsume-solver/transposition-table.md
  - docs/design/tsume-solver/loop-ghi.md
  - docs/design/tsume-solver/move-ordering-and-pv.md
  - docs/design/tsume-solver/aigoma-optimization.md
  - docs/design/tsume-solver/optimization-proposals.md
  - docs/design/tsume-solver/references.md
  - docs/design/tsume-solver/benchmarks.md
  - docs/design/tsume-solver/pn-dn-distribution.md
risk: medium
reversibility: trivial
---

## Trigger

ユーザ指示 (2026-06-24): 「詰将棋ソルバー設計ドキュメントの maou 独自部分が旧 pns + 旧 mid の
ものになっているので，有用なものは新 mid に取り込んだ上で新 mid ベースのドキュメントに
書き換えてください．」

worklog 2026-06-24-144811 §Unresolved で「docs/design/tsume-solver/index.md の実装ファイル表が
依然 pre-v3.0.0」「user が後続の最適化で修正と明言 → 別タスク」と延期されていた本件を着手する．

## Context — ドリフトの実態

現行実装は **maou_shogi 3.1.9 の統一 `mid` 一本** (threshold-iterating df-pn)．
一方 `docs/design/tsume-solver/` の全 13 ファイルは **旧二エンジン期 (v0.x〜v0.55)** の記述:

- **二エンジン構成**: Phase 1 Best-First PNS (arena/`PnsNode`/`pns_main`) + Phase 2 IDS-dfpn +
  Frontier Variant (PNS→局所 MID) + solve() レベル warmup．→ **全廃**．現行は `solve()`→`solve_impl()`→
  `search_impl()` の単一再帰 (solver.rs:1312, search/mod.rs:786)．
- **Dual TT** (ProvenTT + WorkingTT) + FrontierTT + LeafDisproofTT + refutable disproof entry
  (ProvenEntry flags bit 7)．→ **全廃**．現行は単一 64-byte len-aware TT (tt/entry.rs:53-65)．
- **WPN / CD-WPN / VPN / SNDA** (証明数の二重計数補正)．→ **コードから消滅**．現行の二重計数除去は
  δ-sum + `sum_mask` (max-vs-sum) + DAG (`parent_board_key`) + cross-hand dominance．
- **Killer move / 捨て駒ブースト / TT Best Move 動的手順改善 / Deep df-pn / DFPN-E 形式コスト**．
  → **消滅**．現行 move-ordering は `move_brief_eval` + `king_supports` + δ tie-break．
- **9000 行近い実験ログ** (`benchmarks.md` 6429 + `pn-dn-distribution.md` 2303 +
  `optimization-proposals.md` の v0.2x 実験節) は dead-architecture のチューニング履歴．
  memory-architecture では rust/ algorithmic tuning は worklog/compass/git 管轄であり docs 非対象．

### 現行 mid に生き残る概念 (有用 → 新 mid 版に取り込む)

df-pn 基礎 (OR/AND・pn/dn) / 持ち駒優越 (hand_gte, hand_gte_forward_chain) / 1+ε (threshold_epsilon) /
TCA inc_flag (Kishimoto) / df-pn+ 風初期化 (init_pn_dn_or/and + edge_cost) / インライン 1 手詰
(constructive mate1ply + CheckCache) / 中合い遅延展開 (DelayedMoveList + chain 対称性) /
GHI・ループ対策 (path-dependent disproof + repetition taint + scope_disproof) /
dominance 枝刈り (visit_history dominance) / PV 復元 (verify_proof STRICT + build_pv) /
EliminateDoubleCount + cross-hand + look_up_parent (DAG/持ち駒越境) / 並列不採用方針 (binding VETO)．

## User decisions (2026-06-24, AskUserQuestion)

1. **実験ログ = 当初「削除」→ 追補で「保全」に変更**: 当初 `benchmarks.md` / `pn-dn-distribution.md`
   削除で承認したが，user 追補 (2026-06-24「消したドキュメントで有用かもしれない方法論は今後の改善で
   試してみるのでドキュメント貸しておいてください」) により方針変更．**削除せず
   `docs/design/tsume-solver/legacy/` へ verbatim 退避** (git mv) し，banner で「旧二エンジン期記録・
   今後の mid 改善で再検討しうる方法論を含む (新 mid 未適用)」を明示する．`optimization-proposals.md`
   の experiment 節も legacy へ退避し，main は不採用方針 (並列 VETO 等) の新 mid 版に絞る．
2. **学術スタイル維持 (新 mid 版)**: 出典論文引用・図表・機構ごとの設計根拠という網羅的スタイルを保ちつつ
   内容を現行 mid に rebase する．
3. **KH 中立記述**: KomoringHeights は references の一参考実装に留め，maou の機構を独立に記述する
   (KH 依存の印象を避ける)．

## Plan — ファイルごとの処置

| ファイル | 処置 | 概要 |
|---|---|---|
| `benchmarks.md` (6429) | **legacy/ へ退避** | dead-arch 実験ログ．`legacy/benchmarks.md` へ git mv + banner (方法論保全) |
| `pn-dn-distribution.md` (2303) | **legacy/ へ退避** | 対数正規化 campaign ログ (v0.35-0.51)．`legacy/` へ git mv + banner |
| `optimization-proposals.md` (547) | **legacy 退避 + 新規 main** | 原本を `legacy/optimization-proposals.md` へ退避．main は不採用方針 (並列 VETO 等) の新 mid 版を新規作成 |
| `legacy/README.md` (新規) | **新規** | legacy/ の位置づけ banner: 旧二エンジン期記録・将来 mid 改善で再検討しうる方法論・新 mid 未適用・節番号は旧 docs 基準 |
| `index.md` | **全面改稿** | 統一 mid 概要 + 3.1.9 実装ファイル表 + 実装手法表 (version pin 廃止・出典のみ) |
| `search-architecture.md` | **全面改稿** | df-pn 基礎 + 統一 MID 再帰 + threshold-driven IDS (root loop)．PNS/Frontier/warmup 除去 |
| `threshold-control.md` | **全面改稿** | 1+ε + TCA inc_flag + PN_UNIT scaling + threshold 成長．depth-adaptive 系 (旧 mid) 除去 |
| `proof-disproof-numbers.md` | **全面改稿** | pn/dn 集約と二重計数除去 (δ-sum / sum_mask max-vs-sum / DAG / cross-hand)．WPN/SNDA は prior-art 参照に降格 |
| `initial-heuristics.md` | **全面改稿** | df-pn+ 風初期化 (init_pn_dn) + edge_cost + decouple + インライン 1 手詰．Deep df-pn 除去 |
| `transposition-table.md` (1240) | **全面改稿** | 単一 len-aware TT (64B entry/board-key/hand-dominance/look_up_parent cross-hand/amount GC)．Dual TT/FrontierTT/refutable/Zobrist clustering 除去 |
| `loop-ghi.md` | **全面改稿** | GHI + path-dependent disproof + repetition taint + scope_disproof + len-aware NM (proven_len/disproven_len) + dominance |
| `move-ordering-and-pv.md` | **全面改稿** | move_brief_eval/king_supports/δ tie-break + PV (verify_proof STRICT + build_pv)．Killer/捨て駒/3-phase 除去 |
| `aigoma-optimization.md` (431) | **全面改稿** | 中合い = DelayedMoveList (chain 対称性) + futile filter + capture cross-deduction．refutable disproof 除去 |
| `references.md` | **改稿** | 論文・既存ソルバー・JP リソースは保持．変更履歴表は git/compass 参照に置換 (or 主要 milestone のみ) |

## 新ドキュメントの内容方針 (accuracy 原則)

- **always-on core vs opt-in lever を区別する**: 現行 Params は 60+ flag を持つが，その大半は
  A/B 実験で棄却された opt-in lever (doc コメント自身が「default OFF / 悪化する」と記録)．
  これらは worklog/compass 管轄であり design doc には書かない．docs には **constructor の
  shipping default で active な機構のみ** を「as-built 設計」として記述する．
  - 実測 default (solver.rs:563-609): `dml=true`, `scope_disproof=true`,
    `use_delayed_move_list=true`, `use_visit_history_dominance=true` / 一方
    `use_dag_correction=false`, `use_tca=false`, `root_ids_enable=false`,
    `decouple_edge_cost=false`, `minimal_proof_hand=false`．
  - compass 確定 lever (「EliminateDoubleCount」「TCA inc_flag DEC」「root持続IDS」「dominance(DOM)」)
    は **always-on の core 機構** を指し，同名 Params flag (use_dag_correction/use_tca/root_ids_enable)
    は core の上に乗る追加 opt-in (default off)．docs では core を記述し，flag 名は持ち出さない．
  - **execute フェーズで search/mod.rs + search/expansion.rs の探索 core を読み，always-on の
    δ-sum・inc_flag・dominance・root threshold loop を実コードで裏取りしてから記述する**
    (本 review 時点の機構リストは Explore + constructor read に基づく; 最終文面はコード oracle で確定)．
- **version pin を廃止**: 旧 docs の `v0.24.xx 導入` 列は campaign 状態であり compass 管轄．
  新 docs は出典論文の引用のみ残し，maou 内部の版数は記さない．
- **KH 中立**: 「KomoringHeights v0.4.0」等の機構帰属は references へ集約し，本文は maou の機構として記述．
- **JP 記法**: 読点 `，`・句点 `．`・半角括弧 (CLAUDE.md 規則)．mermaid/ASCII 図は現行構成に更新．
- **index.md 実装ファイル表は 3.1.9 構成に同期**: mod.rs/api.rs/solver.rs(core+Params+Diagnostics)/
  search/{mod,expansion,pv}/movegen/{mod,mate1ply,delayed_move_list,check_cache}/tt/{mod,entry}/
  heuristics.rs/proof_hand.rs/leaf 値型 (compass §環境リファレンス 3.1.9 と整合)．

## Risks / reversibility

- **risk: medium** — 大規模改稿 (10 改稿 + 2 削除 + 1 trim)．コード非変更・docs 限定のため
  挙動への影響なし．誤記リスクは「always-on/opt-in の取り違え」が主 → execute 時にコード oracle で裏取り．
- **reversibility: trivial** — docs-only．git revert で完全復元可能．削除ファイルも git 履歴に存続．
- **version bump 不要**: src/rust 非変更 (CLAUDE.md versioning 規則の対象外)．dirty-tree gate も非該当．
- **CLAUDE.md [SLOW] テーブル**: tests.rs パス・test 名は不変のため影響なし (reviews 追加不要)．
- **docs/commands/**: CLI 非変更のため影響なし．

## 承認後の手順 (on approval)

1. execute フェーズで search core (search/mod.rs, expansion.rs) を読み always-on 機構を裏取り．
2. 上表のとおり 2 削除 + 1 trim + 10 改稿を適用．
3. pre-commit (markdown lint 等) を通し commit (`--no-verify` 禁止)．
4. 本 review を `status: applied` + `applied_in: <sha>` に更新し即 commit (audit trail)．
