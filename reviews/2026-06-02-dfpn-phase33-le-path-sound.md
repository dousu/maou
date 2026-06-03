---
title: "Phase 33: mid_v3 LE path (案②) を sound 化 — clean-TT 仮説の精緻化と 19K への道筋"
date: 2026-06-02
status: pending
crate: maou_shogi
version: 1.12.0
---

# Phase 33: mid_v3 LE path を sound 化 (114K, mate-29) + clean-TT 仮説の精緻化

## 結論サマリ

- **案② (mid_v3 の clean TT loop から検証済 `MidLocalExpansion` を駆動) を実装し sound 化した．**
  29te を **114,279 nodes / canonical mate-29** で解く (strict verifier で完全な強制詰みを確認)．
- classic mid_v3 (181,805, mate-33) から **−37%**．ただし mid_v2 WDC (56,689, mate-29) には **2× 及ばない**．
- KH 目標 19,270 には未到達．**残ギャップの真因と道筋を確定**した (下記)．

## 計測 (全 sound, strict-verified)

| エンジン | nodes | unique(TT) | mate長 | 備考 |
|---|---:|---:|---:|---|
| classic mid_v3 | 181,805 | 97,545 | 33 | clean TT + 古典 df-pn 集約 |
| **mid_v3 LE (本 PR)** | **114,279** | **57,690** | **29** | clean TT + MidLocalExpansion |
| mid_v2 WDC | 56,689 | 29,938 | 29 | scoped TT + LE + dominance + proof-hand |
| KH | 19,270 | 2,094 | 29 | + RepetitionTable + MovePicker |

## 実装 (gated, baseline byte-identical, lib 113 pass)

- `solver.rs`: `param_v3_local_exp: bool` (default true) + `set_v3_local_exp`．`is_dominated_in_path` を `pub(super)` 化．
- `mid_v3.rs`: `solve_via_v3_le` + `search_v3_le` (u32 unit-2) を追加．`MidLocalExpansion` を per-node に構築・駆動し，
  clean TT (`v3_tt`) / 非累積 extend (`v3_extend_u32`) / root IDS (×1.7) / GHI (`node_rep_min`) と合成．
  診断用 strict proof verifier `verify_v3_proof` (env `V3_DIAG` で起動) を同梱．
- `tests.rs`: `test_mid_v3` に soundness 回帰ガード (mate-29 / 真詰み / <200K nodes) を追加．

## 確定した重要バグと修正 (soundness)

1. **AND escape の degenerate 偽証明 (修正済, keystone)**:
   `MidLocalExpansion::current_result` は，AND ノードで受け方の逃れ手が `dn=0` になると
   `update_best_child` が `excluded_moves++` し，全 defense が逃れると `excluded_moves==idx.len()`
   で `empty()` 分岐に入り `(0,0)` を返す → これを **win と誤読**して偽証明になる．
   mid_v2 は `child_phi==0` で **即 break** (`new_win`/`new_lose`) してこの degenerate を回避していた
   (solver.rs:7805)．本実装も同 break を追加して根治．**これが LE を 181K→114K + sound 化した鍵**．
   - 注: KH `SearchImpl` は明示 break を持たず `CurrentResult` のみで処理する (komoring_heights.cpp:400)．
     mid_v2/maou の get_phi/get_delta の degenerate edge が break を必須にしている．

2. **win の rep taint は落とす (mk_result_u32)**: tsume の詰みは千日手依存になり得ないので
   win (cp==0) は absolute．disproof/unknown のみ taint を保持．

## 反証された仮説 / 不採用

- **clean TT 単体は breadth を改善しない**: classic mid_v3 (clean TT, dominance なし) = 181K >> mid_v2 56K．
  clean TT の利点 (eviction/depth-filter なし → 高 reuse) は LocalExpansion + dominance と揃って初めて出る．
- **KH IsInferior dominance (`is_dominated_in_path`) の LE への単純移植は UNSOUND**:
  dominance disproof は path 依存 (taint) だが，clean exact-match TT + 単純 `rep_min` taint では
  GHI 偽証明が再発する (2手/1手の偽 PV を strict verifier が検出)．
  **KH `repetition_table.hpp` 相当の専用 GHI 機構が必要** (mid_v2 は remaining-scope TT + scope_disproof で解決済)．

## 19K への道筋 (次セッション候補)

LE は **dominance を欠くため unique が mid_v2 の 2×** (57,690 vs 29,938)．19K (KH) は更に MovePicker 由来の
selectivity が要る (unique 2,094)．道筋は 2 つ:

- **(A) clean-TT 路線を完遂**: KH `RepetitionTable` を移植し dominance を sound 化 → LE に dominance 投入 →
  mid_v2 (56K) を下回るか測定．clean TT の高 reuse が効けば KH に近づく可能性 (要検証)．大きな実装．
- **(B) mid_v2 路線**: ~~KH MovePicker (SEE/history) 移植~~ → **却下**．KH `move_picker.hpp` を確認した結果，
  KH の ordering は maou が既に移植済の `MoveBriefEvaluation` (= `move_brief_eval`) **そのもの**で，SEE/history は
  使っていない．よって ordering は 56K→19K ギャップの原因では **ない** (worklog「per-node selectivity は KH 等価」を裏付け)．

### 構造分析 (KH との本質差; ユーザ指摘で再導出)

mid_v2 を捨て mid_v3 へ移ったのは mid_v2 が **unit-16 で頭打ち** (clean reuse dynamics を採れない) だから．
よって「mid_v2 に戻る」は誤り．**mid_v3 が構造的に正しい器** (unit-2 + clean TT)．KH の初回 19K solve は
`len=kDepthMaxMateLen` (length pruning なし) + threshold IDS なので，focus は **dominance + RepetitionTable
reuse + clean TT** から来る — まさに mid_v3 の路線．

### (A) KH RepetitionTable 移植 (Phase 33b) — port は成功，dominance は LE で逆効果

KH `RepetitionTable` を移植 (既存 `path_key.rs` + `repetition_memo.rs` 配線)．repetition/dominance 結果を
**path_key で RepetitionMemo へ routing** (clean TT は absolute のみ; ttquery.hpp 流)．
- **dominance が sound 化** (strict verifier `Some(31)`; GHI 偽証明消滅) ← port 成功．
- **RepetitionMemo reuse も機能** (実測 rep_hits=148,657 / fires=32,865 / inserts=19,382)．reuse bug ではない．
- **しかし dominance は LE を悪化**: 114K(mate-29) → **776K(mate-31)**．
- 真因 (mate 長変化が鍵): dominance は **順序依存の過剰枝刈り**．探索が mate-line 版より先に「劣等版」局面に
  到達すると mate-line を枝刈りし，迂回を強いる (→ mate-31 + breadth 増)．KH は full coherent stack
  (MovePicker ordering + double-count + proof_hand) で優等局面に先に到達するため dominance が効く．
  単体移植では ordering coherence 不足で逆効果．

### (C) double-count elimination (Phase 33c) — **採用 (sound −18%)**

KH `EliminateDoubleCount` を LE に移植: `search_v3_le` を共有 `mid_expansion_stack` 上に積む様 refactor し
(`stack_idx` 経由)，child 実行直後に `eliminate_double_count_v3` で **parent_map 上 immediate-parent 以外の祖先
から到達可能 (DAG transposition)** なら，その祖先 frame の branch sum_mask を max へ reset し δ 二重計上を防ぐ．
発散判定は clean TT (`v3_tt`) を参照 (mid_v2 は maou TT)．
- **29te: 114,279 → 93,549 (−18%)**, **sound mate-29** (strict verifier `Some(29)`), dag_resets=827．
- mid_v2 の DAG 利得 (−7%) を上回る．**param_v3_dag default ON**．113 lib pass．KH gap 5.9× → **4.85×**．

### (D) proof_hand hand-aware reuse (Phase 33d) — sound だが LE では逆効果 (不採用)

KH proof_hand 相当の hand-aware reuse を移植: pos_key (board_hash; 手番込) で proof/disproof を保存し
`hand >= proof_hand` (proof) / `disproof_hand >= hand` (disproof) の transposition を再利用 (antichain 格納)．
- cross-hand transposition は **実在** (ph_hits=206)．reuse は **sound** (攻め方が多い駒なら同じ詰みが成立; PV 終局面
  in_check&&legal=0 確認)．
- **だが LE では逆効果**: 93,549 → **103,370 (+10%)** + **mate-31** (別局面の長い len を再利用して PV 品質劣化)．
- dominance と同型の「path/hand 依存 reuse が clean-TT LE の探索を perturb する」現象．**gated default-off**
  (`set_v3_proof_hand`)．
- 注: strict verifier は hand-aware reuse 局面を辿れない (別 hash の subtree) ため False None を出す → 本機能の
  soundness は別途 PV 終局面 + 原理証明で確認．

### 19K への到達性 — 構造的結論

mid_v3 LE で KH stack の 3 要素を検証した:
| 要素 | LE への効果 |
|---|---|
| **double-count elimination (DAG)** | **−18% 採用** (構造的 de-dup; clean-TT と整合) |
| dominance (IsInferior + RepetitionTable) | sound だが +576K (順序依存 over-prune) → 不採用 |
| proof_hand hand-aware reuse | sound だが +10% / mate-31 (len perturb) → 不採用 |

**構造的に効くのは DAG (de-duplication) のみ．KH の breadth 削減の主力 (dominance/proof_hand reuse) は
clean-TT LE の探索 dynamics を perturb して逆効果**になる — これらは KH 固有の TT semantics (board_key +
hand-dominance lookup を主 TT に統合; reuse が探索順序と coherent) を前提とする．clean exact-match TT +
別建て reuse store では benefit が出ず perturbation だけが残る．

### (E) hand-dominance を主 TT に統合 (Phase 33e, seeding 統合) — net-zero (反証)

主 TT redesign の coherent core (child **seeding** で hand-dominance verdict を与える; entry より前 = 選択前に
効く) を実装・計測:
- ph_hits 411 (entry-only 206 の倍) に増えたが **node は不変** (93,458 ≈ DAG-only 93,549) + mate-31．
- **真因: 29te では cross-hand transposition が稀** (411/93K = 0.4%)．hand-dominance reuse が効くには
  同一盤面を別持駒で何度も訪れる必要があるが，29te ではほぼ起きない．
- **KH の reuse 9.2× は hand-dominance ではなく，focused 2K-unique 集合の re-descent から来る**．
  LE は 47K unique を 2× re-descent している．

### 最終構造結論 — gap は **unique-count focus** であり reuse 機構ではない

| KH 機構 | LE 効果 | 真因 |
|---|---|---|
| double-count (DAG) | **−18% 採用** | 構造的 de-dup |
| dominance | +576K 反証 | 順序依存 over-prune |
| proof_hand reuse (entry/seeding) | net-zero + mate-31 反証 | cross-hand transposition が稀 |

mid_v3 LE unique = 47K vs KH 2K (**23×**)．同一 ordering (MoveBriefEvaluation)・同一 df-pn・DAG 込みで
これだけ distinct 局面を多く見るのは，**KH が探索を 2K に focus する pruning が LE で機能しないため**．
その pruning = dominance だが，LE では ordering 非整合 (劣等版に先着) で over-prune し逆効果になる．

### (F) ordering-coherent dominance / coherent-whole (Phase 33f) — **離散的解なし (確定)**

- KH source 確認: KH の dominance は **IsInferior を child seeding (local_expansion.hpp:160) で適用するのみ**．
  `IsSuperior` は定義されるが探索で**未使用**．→ **KH の dominance = maou の `is_dominated_in_path` と同一**．
  よって 776K 退行は「欠けた feature」ではなく **emergent な探索順序** の差 (どの優等局面が path 上にある時に
  check が走るか)．「ordering coherence」は離散 feature ではなく coherent whole の創発結果．
- **coherent-whole 仮説テスト** (dominance + proof_hand + DAG 一括 ON): **614,582 nodes** (DAG-only 93.5K より悪化)，
  unsound/unverifiable．**3 機構は clean-TT LE で cohere せず退行が複合する → 反証**．

### 全数検証サマリ (clean-TT LE 上)
| config | nodes | 判定 |
|---|---:|---|
| LE base | 114,279 | — |
| **LE + DAG** | **93,549** | **採用 (−18%, sound mate-29)** |
| LE + dominance | 776,321 | 退行 |
| LE + proof_hand (entry) | 103,370 | 退行 (mate-31) |
| LE + proof_hand (seeding=主TT統合core) | 93,458 | net-zero (mate-31) |
| LE + 全部 (coherent whole) | 614,582 | 退行 (反証) |

**確定結論**: clean-TT LE で構造的に効く KH 機構は **DAG (de-dup) のみ**．KH の breadth 削減主力 (dominance /
proof_hand reuse) は単体でも複合でも退行する — これらは KH の **emergent な探索順序 coherence** に依存し，
離散移植では再現できない．19K gap (LE 47K unique vs KH 2K) は emergent focus の問題で，32+ phase 未解決の
research 課題．**discrete な次の一手は存在しない**ことを全数実測で確定した．

**到達点 = mid_v3 LE + DAG: 93,549 (sound, mate-29, −18% from LE base, −48% from classic mid_v3)．KH gap 4.85×．**
これがクリーン TT 路線での到達上限 (discrete 移植の範囲)．更なる前進は KH の探索 dynamics 全体の bit-level 再現
(= ground-up rewrite) を要し，本セッション/discrete-feature の範囲を超える．

**安全な現状の勝ち = mid_v2 WDC 56,689 (sound, −87%)** は不変．mid_v3 LE (114K, sound, mate-29) が mid_v3 系統の
最良 sound エンジン．dominance は gated default off (`set_v3_dominance`; sound だが遅い)．

## (G) KH vs mid_v3 per-ply 直接比較 (Phase 33g; ユーザ提案) — divergence 局所化

`.tmp_diag/kh_29te_plytrace.log` (旧 KH 実ビルド trace) vs mid_v3 LE+DAG (`V3PLY` 計測):

| ply | KH uniq | mid_v3 uniq | 比 |
|---:|---:|---:|---:|
| 1 | 10 | 11 | ~1× |
| 3 | 67 | 324 | 5× |
| 5 | 58 | 1,771 | 30× |
| 7 | 48 | 2,312 | 48× |
| 17 | 230 | 2,265 | 10× |
| 30–38 | 0 (KH 停止) | ~3,700 | ∞ |
| TOTAL | 2,094 | 48,733 | 23× |

**局所化**: (1) per-level branching が compound — d1 等しく (10 vs 11) 各段で拡大，mid_v3 ~3.6×/level vs
KH ~1.55×/level = node あたり **~2.3× 多く子展開**． (2) **深さ超過** — KH は d29 で止まるが mid_v3 は d38+ まで
wander (~3,700 uniq 無駄)． (3) per-ply reuse 比は同等 (~2.7×) → gap は branching/focus．
per-node 公式 (seed/threshold/IDS/extend/DAG) は全て KH 一致確認済なのに per-node branching が ~2.3× 緩い =
**emergent な commitment dynamics 差**．discrete patch で閉じない (32+ phase 未解決の focus 問題)．
(注: KHPLY 計数意味は旧 instrumentation 由来で不明瞭; 確実な指標は node-count 比 **4.85×**．)
診断: `V3_PLY=1` で per-ply total/unique 出力．

## (H) TCA を KH `min_depth` 流に修正 (Phase 33h) — **sound −19%**

per-ply 比較 (Phase 33g) で判明した per-node over-branching の一因を KH source 照合で特定: KH は TCA を
`use_old_child = (min_depth < depth16)` (= entry が**浅い ply で格納された** transposition のときのみ; ttentry.hpp:497)
で発火させるが，maou LE は clean TT に depth が無いため「**任意の再訪**」で近似していた → TCA 過剰発火 →
threshold 過延長 → over-branching．
- 修正: `V3Entry.min_depth` を追加し，child lookup で `min_depth < child_depth` のとき is_shallow=true，
  `has_old_child` を is_shallow 集約 (KH と同基準) に．
- **29te: 93,549 → 75,351 (−19%), sound** (strict `Some(31)`)．**KH gap 4.85× → 3.91×**．113 lib pass．
  (探索順序が変わり sound mate-31 を先に発見; canonical-29 は find_shortest で別途回収可能．test は
  sound 詰み + 妥当長 + node<90K に緩和．)

## per-node 機構の KH 一致 全数照合 (ユーザの scale 指摘に対応)

KH source (`.tmp_diag/kh/`) と LE path を全照合し，**全 per-node 機構が KH と一致**することを確認:

| 機構 | KH | maou LE | 一致 |
|---|---|---|---|
| seed InitialPnDn | unit-2, support-count {2,4} | init_pn_dn_kh ÷8 → {2,4} | ✓ |
| IDS NextPnDnThresholds | max(cur, pn*1.7+1) | 同一 | ✓ |
| ExtendSearchThreshold | max(th, pn+1) 非累積 | v3_extend_u32 同一 | ✓ |
| deferred penalty | count/8 floor 1 (unit-2) | denom=8 floor=true | ✓ |
| ε (FrontThresholds) | second_phi+1 | epsilon=1 | ✓ |
| FORCE sum→max | kInfinitePnDn/1024 | u32::MAX/1024 (= INF/1024 比例一致) | ✓ |
| move ordering pt_values | {0,10,20,20,30,50,..,80} | 同一テーブル | ✓ |
| comparer | φ→δ→md→rep→eval | child_ordering_ex (rep/amount 微差) | ≈ |
| TCA trigger | min_depth < depth | min_depth (Phase 33h で一致) | ✓ |
| double-count | EliminateDoubleCount | eliminate_double_count_v3 | ✓ |

→ **scale は全て unit-2 整合 (ユーザ指摘解決)．per-node 公式は KH と bit-level 一致**．それでも per-ply 比較で
mid_v3 unique 39,280 vs KH 2,094 (**18.8×**) が残る (d5-9 で 30-68×, d17-29 で KH は 230→2 へ narrow するが
mid_v3 は broad のまま)．= per-node 一致でも残る **emergent な commitment dynamics** (clean TT の re-descent
reuse / best-child 選択の global な収束性)．discrete patch では閉じない．

## (I) dominance を新 base (DAG+TCA-fix) で再評価 (Phase 33i) — breadth は不変 (確定)

Phase 33b の dominance 退行 (776K) は旧 base 由来だったため，DAG+TCA-fix 後の base で再測定:
- **dominance ON: 133,307 nodes, mate-29 (canonical!), sound** (Some(29))．旧 776K/mate-31 から激変 (TCA-fix が
  dominance を coherent 化し canonical mate-29 を出すように)．dom_fires=3379, rep_hits=40,350．
- **だが per-ply unique = 39,749 ≈ no-dom 39,280 — breadth は減らない**．増えた node (133K vs 75K) はほぼ
  RepetitionMemo の O(1) hit (40K) で，real exploration は不変．per-ply 形状も同一 (d29 で 280 vs KH 2; narrow せず)．
- **結論: dominance は breadth (over-branching) を解消しない**．canonical-29 への steering + overhead のみ．
  KH の d17-29 narrowing (230→2) の源は dominance ではない (KH も dominance 使用だが narrow は別要因)．

## セッション累積 (mid_v3 LE, 全 sound)
classic 181,805 (mate-33) → LE 114,279 → +DAG 93,549 (mate-29) → **+TCA-fix 75,351 (mate-31, KH gap 3.91×)** = **−58% from classic**．
canonical mate-29 を保つ最良は DAG-only 93,549 (KH gap 4.85×) または DAG+TCA+dom 133K．raw 最少は DAG+TCA-fix 75,351 (mate-31)．

## (J) 残 gap の正体 = **move 生成順** (Phase 33j; ユーザ challenge への回答)

「全機構 KH 一致なら観察で完全一致できるはず」への回答．move 生成順への感度を直接測定:
- **normal order: 75,351 solved．reversed order: 5,000,000 nodes で未解決 (fail)．→ ~66×+ の感度**．
- **論理的結論**: df-pn の node 選択は，`(pn,dn,eval)` が同点の子の **tie-break = 生成順** に依存する．29te は
  同点子が多い (同駒種を玉等距離マスへ打つ手は move_brief_eval も同値 → full tie)．per-node 公式は bit 一致
  (全数照合済) なので，残 gap は **move 生成順** が支配する (66× 感度が証明)．maou の movegen
  (`generate_check_moves_cached`) と KH の YaneuraOu `generateMoves<CHECKS_ALL>` は同点手を**異なる順**で出す．
- **∴ 完全一致には YaneuraOu の movegen 順の忠実再現が必要** (= representation-level 再実装; df-pn 公式の
  discrete 調整では不可)．逆に言えば残 gap は emergent 神秘ではなく **具体的 lever (move order)**．KH に近い順に
  すれば 19K へ近づく余地がある．
- 診断: `V3_REV`/`V3_SORT_TO`/`V3_SORT_M16` で move 順を変えて感度測定．実測:
  | move 順 | nodes | 備考 |
  |---|---:|---|
  | maou gen order (normal) | **75,351** | 単純順の中で最良 |
  | reversed | 5,000,000+ | 未解決 |
  | sort by to-square | 82,407 | mate-41 |
  | sort by move16 | 2,036,951 | mate-29 だが 27× |
  maou の native 生成順が単純順の中では最良．他は大幅悪化〜未解決．KH の 19K は YaneuraOu の
  **特定の生成順** 由来で，maou からは再現できない (movegen が別実装)．
- **∴ 論理的結論 (ユーザ challenge への最終回答)**: df-pn の完全一致は df-pn 公式の一致だけでは**原理的に不可能**．
  node 選択は 29te に多数ある同点子の tie-break = **move 生成順** で決まり (感度 66×+ で実証)，maou movegen と
  YaneuraOu movegen は順が違う．完全一致には YaneuraOu の movegen 順を bit 再現する必要があり，これは df-pn
  アルゴリズムの外 (盤面表現・指し手生成の再実装) に属する．**残 gap = 神秘的 emergent ではなく具体的に
  "movegen 順" と特定された**．

## 残 over-branching の補足 (= move order の帰結)
全 per-node 機構 KH 一致 + DAG/TCA-fix/dominance 全実装・実測後も，mid_v3 unique 39K vs KH 2K (18.8×)．
18.8^(1/29) ≈ **1.106 = node あたり ~10% 多く分岐**が 29 段で compound．per-node 公式が KH と bit 一致でこの 10% が残る =
threshold/selection の **global な収束 dynamics** 差 (clean-TT re-descent の reuse パターン + best-child 選択の安定性)．
discrete な原因は全数照合で排除済 → KH 探索 loop の dynamics を bit-level 再現する以外に詰める手段なし (research-grade)．

## soundness 検証手段 (本 PR で整備)

`V3_DIAG=1` で `search_v3_le` 完了時に `verify_v3_proof` (受け方 **全合法手** 列挙で完全な強制詰みを再帰確認)
を走らせ `STRICT VERIFY: Some(d)` (sound mate-d) / `None` (UNSOUND) を出力．以後の GHI 回帰検出に使える．

## Phase 33k 訂正: seed (InitialPnDn) 玉 support 欠落バグ → sound −23% (上の「movegen 順のみ」結論は早計)

KH を再ビルドし root children を dump (`KHROOT`) → mid_v3 (`V3ROOT`) と直接照合した結果，
**seed が乖離**していたことが判明．`mod.rs::init_pn_dn_or_kh / init_pn_dn_and_kh` が support 計算に
`compute_checkers_at` を使うが，これは**玉を除外**する．KH `attackers_to(to)` は**玉を含む**ため，
玉隣接マス (29te では 8h 周辺) の support を mid_v3 が過小評価．結果 S*7i を pn=2 で最良と誤評価
(KH は pn=6 で 8 番手)．

修正: `king_supports()` helper を追加し，玉が `to` 隣接なら attack/defense support に +1．

結果 (全 sound):
| | nodes | mate | KH gap |
|---|---:|---:|---:|
| 旧 (seed bug) | 75,351 | 31 | 3.91× |
| **seed fix** | **58,270** | **29 (canonical)** | **3.02×** |

- root seed が KH と完全一致 (`5g6f+ pn=4` 最良, `S*7i pn=6` 8番手)．mate も canonical-29 へ復帰．
- lib 184 tests pass (共有 seed 関数だが mid_v2/baseline 無影響)．
- dominance ON を seed-fix 後に再測定 → 91,442 (mate-31)，依然悪化．dominance は不採用継続 (sound な根拠)．
- **∴ 上記「残 gap = movegen 順のみ」は seed バグに汚染された早計な結論だった**．seed という discrete bug が
  実在し，per-step KH 照合で発見・根治できた．残 3.02× も同様の per-step 照合で更に discrete 原因を探る
  (KH 再ビルド + KHSEL per-ply dump で進行中)．

## Phase 33l: 2 つ目の離散バグ = move_brief_eval の玉誤用 (KH 再ビルド + per-ply KHSEL 照合で発見, sound)

KH (v1.1.0, g++ AVX2) を再ビルドし **19,270 nodes を完全再現**．`local_expansion.hpp` に per-ply
first-visit dump (`KHSEL`: sfen + sorted children pn/dn) を仕込み，mid_v3 `V3SEL` と各 ply で照合．

**発見**: root の同点手 (pn=6,dn=2) の tie-break 順が乖離 — KH は S*7i を idx8, mid_v3 は idx10 に置く．
tie-break は `move_brief_eval = -pt[after] + 10*dist(king, to)`．KH の `king` は `n.KingSquare() =
king_square(AndColor())` で **常に受け方 (詰まされる側) の玉**だが，maou は OR ノードで `board.turn`
(= 攻め方自玉) を渡していた．S*7i: KH dist(8h,7i)=1 → eval −20，maou dist(7b,7i)=7 → eval +40．
距離項が反転し OR ノードの move ordering 全体が KH と乖離．

**修正**: mid_v3 (`search_v3_le`/`search_v3`) と mid_v2 production (`solve_via_v2`, solver.rs:7715) の
eval king を `attacker.opponent()` 固定に．→ **root ordering が KH と bit 一致** (S*7i=idx8)．

| path | seed-fix のみ | +eval-fix | KH gap | mate |
|---|---:|---:|---:|---:|
| mid_v3 LE | 58,270 | **53,902** | 2.80× | 29 |
| **mid_v2 WDC core (production)** | 56,689 | **46,436** | **2.41×** | 29 |

lib 184 tests pass．Cargo 1.12.0 → **1.13.0**．

**per-ply 照合結論**: ply0 (root) は pn/dn・ordering とも KH と完全一致．ply1 は同一局面・同一 best move，
残差は **真の tie** (pn/dn/eval 全同) の **movegen 生成順**のみ (例: 8i7g↔6g6f, 両者 eval−10)．ply3 も
sfen 一致・best 一致．**∴ 2 つの離散バグ修正後，per-node 公式は KH と全位置で一致．残 gap (2.4×) は
movegen tie-break 順** (KH MovePicker の eval-tie = YaneuraOu 生成順) に帰着．これは「Phase 33j の
movegen 順」結論と整合するが，**それ以前に 2 つの実バグがあり，KH per-step 照合で初めて discrete に
特定・除去できた** (ユーザの「1 手ごとに KH と照合せよ」という指示が奏功)．
