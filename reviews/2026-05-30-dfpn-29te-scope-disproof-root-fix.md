---
status: applied
applied_in: pending-commit
date: 2026-05-30
target: [rust/maou_shogi/src/dfpn/solver.rs, rust/maou_shogi/src/dfpn/tests.rs, rust/maou_shogi/Cargo.toml]
risk: medium
reversibility: easy (param-gated; revert = set param_scope_disproof default false)
---

# dfpn 29te false-NoMate **根治** (scope_disproof, Phase 26b, v1.6.0)

> [[2026-05-29-dfpn-19k-holistic-redesign]] / 2026-05-30-dfpn-29te-false-nomate-dml-fix の継続．
> ユーザ指示「偽反証してしまうバグを根本原因から修正してから次セッションで効率改善」．
> v1.5.0 (kh_dml) は workaround だった偽反証を，v1.6.0 で **真因から修正**する．

## 真因 (確定)

mid_v2 の集約 disproof store (solver.rs:7508) が `REMAINING_INFINITE` (= ProvenTT
**confirmed/絶対** disproof; `tt.rs:211 is_proven_entry`) で格納していた．depth-limit
偽反証 (`look_up_pn_dn_impl` の `remaining==0 → (INF,0)`) が伝播して生じた集約 disproof も
confirmed 化され，TT を**恒久的に汚染**．より浅い ply (remaining 大) の transposition lookup が
この絶対 disproof を読み，真の mate 経路を塞いで root を false NoMate にしていた．

KH は length-disproof を `disproven_len` で scope 化し，GHI horizon (`kDepthMax`) は
Repetition で返す (confirmed にしない)．maou はこの分離を欠いていた．

## 修正 (scope_disproof, default true)

solver.rs:7508 の集約 disproof を `param_scope_disproof` 時 `REMAINING_INFINITE` →
**`remaining` scope** で store する．これにより WorkingTT に入り (`is_proven_entry` は
dn==0 を remaining<INF なら confirmed 扱いしない)，lookup は `tt.rs:1267
e.remaining() >= remaining` で scope される — より深い ply の transposition では
disproof が無効化され**再探索**される．KH `disproven_len` scope 相当．

新 default = **kh_dml=true + scope_disproof=true** (on+scope)．

## A/B (29te depth cliff, init solve, 50M/120s, find_shortest=false)

| depth | margin | dml_off | dml_off+scope | on+scope (新default) |
|---|---|---|---|---|
| 29 | 0 | NoMate! | NoMate! | NoMate! (※margin0 は仕様: 詰み局面が horizon と一致し terminal 検出前に偽反証．margin≥1 必須) |
| 30 | 1 | Mate(29) | Mate(29) | Mate(31) |
| **31** | **2** | **NoMate! 811K** | **Mate(29) 268K** | **Mate(29) 162K** (最速) |
| 32 | 3 | Mate(31) | Mate(31) | Mate(33) |
| 33 | 4 | Mate(29) | Mate(31) | Mate(29) |
| 34 | 5 | Mate(29) | Mate(29) | Mate(31) |

→ **margin≥1 で false NoMate 消滅**．非最小 init (Mate31/33) は find_shortest=true (default)
で 29 に refine される (canonical test_tsume_6_29te が実証)．

## 失敗した試行 (killing signal 付き, 保存)

- **horizon-inflation (param_horizon_margin)**: `self.depth` (GHI horizon) を margin 分
  inflate し length を md_budget に委譲する案 (提案1 の素直な実装)．→ **全 depth で Unknown
  (50M budget 枯渇)**．horizon は有用な pruning を担っており，inflate すると md_budget では
  抑えられない探索爆発が起きる．**棄却・revert 済**．scope_disproof が正しい lever だった．

## 検証 (on+scope = 新 default, 全 pass)

- 93 dfpn lib (single-thread): pass．
- `test_tsume_6_29te` (depth=31, find_shortest=true): **Mate(29) strict PV, 1.90s**．
- `test_tsume_39te_ply2_no_false_nomate`: pass (Unknown, 偽反証なし)．
- `test_tsume_39te_ply24_mate15_soundness_depth25`: pass (偽反証なし)．
- depth_cliff: margin≥1 で NoMate 消滅．

## 残課題 (次セッション = 効率)

- 54× breadth gap (KH 5,269 SearchImpl vs maou ~162K init) は残存．偽反証が消えたので
  次セッションは純粋に **KH 同等ノード数** を目指す効率改善に集中できる (ユーザ計画通り)．
- margin0 (depth==mate_len) の NoMate は terminal 検出の別問題．実用では margin≥1 を前提とする．
- scope_disproof で genuine 絶対 disproof も再探索されうる (効率コスト)．現状 29te/93/no_false_nomate
  では問題なし．horizon-only 由来に限定する精密化は将来の最適化候補．

## Rollback

`param_scope_disproof` default を false に戻すだけ (1 行)．version 1.6.0→1.5.0．
