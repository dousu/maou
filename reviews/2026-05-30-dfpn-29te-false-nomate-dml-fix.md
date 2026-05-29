---
status: approved
applied_in: working-tree (uncommitted; v1.5.0 bump pending user commit)
date: 2026-05-30
target: [rust/maou_shogi/src/dfpn/mid_v2.rs, rust/maou_shogi/src/dfpn/solver.rs, rust/maou_shogi/src/dfpn/tests.rs, rust/maou_shogi/Cargo.toml]
risk: medium
reversibility: easy (param-gated; revert = set param_kh_dml default false)
---

# dfpn 29te false-NoMate 発見 + KH parity DML 修正 (Phase 26, v1.5.0)

> Phase 25 holistic re-design の継続．「sum_mask/DML の KH 整合性」を実装する過程で，
> **canonical test `test_tsume_6_29te` が HEAD で RED (false NoMate)** という既存
> soundness バグを発見し，KH parity DML (`kh_dml`) でこれを修正した．
> ユーザ判断 (AskUserQuestion): 「kh_dml を default 化 + 根本調査」．

## 最重要発見: production path は 29te で false NoMate を返していた

`test_tsume_6_29te` (canonical, `solve_via_v2`, depth=31) は HEAD `b0fe183` で
**`NoCheckmate { nodes_searched: 811241 }`** を返す (= 詰みを「詰まない」と誤判定)．
clean HEAD を `git stash` で確認済 → 私の変更ではなく**既存バグ**．

- 旧セッションが「baseline 190,646 / PV29 intact」と記録したのは **depth=33** の
  visit-diag 診断パス (`test_mid_v2_tsume_29te_visit_per_ply` は `with_timeout(33,…)`)．
  canonical test は **depth=31**．両者で結果が割れていた (production = 偽 NoMate)．
- 「93 lib pass」は正しいが，29te は `#[ignore]` のため 93 に含まれず，RED が見逃されていた．

## 修正: KH parity DML (`param_kh_dml`, default true)

`build_delayed_chain` を KH `delayed_move_list.hpp` 忠実版に拡張:
- 旧: AND ノードの同 to_sq drops のみ defer．
- 新 (`kh_dml=true`): 上記に加え **非駒打ち 成/不成ペア** (歩/角/飛 + 香 rank2/8) を
  **OR/AND 両方**で (from,to) 同一として chain 化．`IsDelayable`/`IsSame` を移植
  (`initial_estimation.hpp:227` IsSumDeltaNode は narrow すぎるため未実装)．
- `MidLocalExpansion::new_with_fh_dml(…, kh_dml, us_is_black)` を追加，
  solver 呼び出し側で `self.param_kh_dml` + `board.turn==Black` を渡す．

効果: 29te を **正しい Mate(29)** で解く (depth=31)．strict PV assertion pass．

## A/B 実測 (29te, 50M/120s)

| config | depth=31 (canonical) | depth=33 (旧 baseline) |
|---|---|---|
| dml_off (旧 default) | **NoMate! 811,241** ❌ | Mate(29) 190,646 init / 571,938 full |
| dml_on (新 default) | Mate(29) 218,195 init / 654,585 full ✓ | Mate(29) 260,604 / 781,812 |

- **co-tune 仮説 (DML + deferred penalty /8 floor) は棄却**: pen8 で 654K→812K 退行，
  depth=33 では PV が Mate31 に破綻．→ 最適は kh_dml=ON + penalty OFF (denom=0)．
- kh_dml は depth=33 で dml_off より ~37% 遅い (260K vs 190K init)．robustness と
  efficiency のトレードオフ．

## 根本調査: depth horizon = mate-length-bound の混同 (真因)

depth cliff 実測 (`test_tsume_29te_depth_cliff`, init solve):

| depth | margin | dml_off | dml_on |
|---|---|---|---|
| 29 | 0 | NoMate! | NoMate! |
| 30 | 1 | Mate(29) 176K | Mate(29) 213K |
| 31 | 2 | **NoMate! 811K** | Mate(29) 218K |
| 32 | 3 | Mate(31) 253K | Mate(29) 254K |
| 33 | 4 | Mate(29) 190K | Mate(29) 260K |
| 34 | 5 | Mate(29) 391K | Mate(31) 280K |

**depth 感度は cliff ではなく chaotic**．両 config が特定 depth で false NoMate / 非最小
PV を出す．dml_on は margin 1–4 で安定 Mate(29) だが depth-independent ではない．

**真因**: maou は GHI depth-horizon と mate-length-bound を `remaining = depth - ply` で
**同一視**している．`look_up_pn_dn_impl` の `remaining==0` で **hard disproof `(INF,0)`**
を返す (solver.rs:2149-2154)．tight horizon (margin 2) では探索の tentative 深掘りが
horizon に当たり偽反証が pn/dn 集約を汚染して root を偽反証する．margin が変わると
探索順が変わり結果も chaotic に変わる．

→ KH は両者を分離: `komoring_heights.cpp:375` で `GetDepth() >= kDepthMax` (巨大固定値) を
**Repetition** で返し (hard disproof ではない)，実際の長さ制約は `MateLen len` (= md_budget
相当) が独立に担う．maou もこの分離 (内部 horizon を self.depth より十分大きく取り，長さ
制約は md_budget に委譲) を実装すれば depth-independent な soundness が得られ，dml_off の
efficiency (190K) も保てる可能性が高い．

## 次の実装方向 (提案, 未着手)

1. **[本丸] depth horizon / length bound の分離** — `self.depth` (length bound) とは別に
   内部 GHI horizon `depth_horizon = self.depth + Δ` (or 固定大) を導入し，`remaining` を
   horizon 基準にする．`remaining==0` (horizon) は repetition 扱い，length 超過は md_budget
   が担当．これで canonical depth=31 でも dml_off が解け，kh_dml workaround が不要になる見込み．
   risk: 高 (termination/soundness 影響大)．要 93 + no_false_nomate 回帰．
2. **[暫定 bankable] kh_dml=true を default 維持** — canonical test green．本 review で適用済．

## 検証 (kh_dml=true default)

- 93 dfpn lib tests: pass (single-thread; 並列は release で OOM/SIGTERM するため -j1 推奨)．
- `test_tsume_6_29te`: **Mate(29) strict PV pass** (false NoMate 修正)．
- `test_tsume_39te_ply2_no_false_nomate`: pass (Unknown, 偽反証なし)．
- `test_tsume_5` (17手): pass (93 内)．
- `test_tsume_39te_ply24_mate15_soundness_depth25`: pass (偽反証なし)．

## Rollback

`param_kh_dml` default を false に戻すだけ (1 行)．DML 実装自体は param-gated で
旧挙動 (kh_dml=false) は byte-identical に保存．version 1.5.0→1.4.0 戻し．

## What this constrains

depth horizon / length bound 分離 (提案 1) を実装するまで，29te は depth に対し
chaotic で，canonical test の depth=31 は kh_dml に依存して green を保つ．efficiency↑
施策は依然 PV 長 (=29) を壊しうる (Phase 25 鉄則は継続)．
