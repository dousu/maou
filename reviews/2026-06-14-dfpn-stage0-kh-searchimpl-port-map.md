---
title: "Stage 0 — KH SearchImpl 移植マップ + iter6 真乖離の局所化 (39te wholesale port)"
status: approved
date: 2026-06-14
branch: feat/tsume-solver
base-sha: ef49730  # v2.9.0
campaign: "39te を KH 相当時間で解く (builds 3.73× / per-build 2.7× = ~10.5×)"
supersedes-context: docs/plans/steady-burning-lantern.md §3,§6
---

## 0. この doc の位置づけ

ユーザが「wholesale KH port を start」を選択 (2026-06-14)．本 doc は **Stage 0
の成果物**: KH `SearchImpl` stack を maou `search_v3_le` stack へ line-map し，
「何が既に移植済か / 何が genuine 残務か」を確定する．**重要な含意があるため
status: pending とし，ユーザ確認の上で次段の方針を決める．**

## 1. 🎯 Headline finding (port の前提を更新する)

**maou の `search_v3_le` は既に KH `SearchImpl` の忠実な line-by-line 移植である．**
制御フロー・comparer・move 順 tie-break・sum_mask・DML・seed・GHI/repetition・
劣等局面 pruning が **全て移植済かつ default で active**．これは前 session が
「verbatim 400 行 copy の soundness risk を避け，検証済 LE path を再利用する」と
決めた (steady-burning-lantern §6.4) 帰結であり，矛盾ではない．

→ **「Stage 0 = SearchImpl skeleton を書く」は字義どおりやると現挙動を再生産する**
(search_v3_le がそれ)．port の genuine 残務は *再翻訳ではなく*，§3 に挙げる
少数の未移植要素 + co-adapted equilibrium の同時再導出である．

## 2. KH ↔ maou 対応表 (本 session で逐行確認)

| KH (`.tmp_diag/.../user-engine/`) | maou (`rust/maou_shogi/src/dfpn/`) | 状態 |
|---|---|---|
| `komoring_heights.cpp` SearchImpl/SearchImplForRoot | `mid_v3.rs` search_v3_le / search_v3_le_frame | ✅ 忠実 (TCA inc_flag, ExtendSearchThreshold, threshold loop, Pop) |
| `NextPnDnThresholds` (×1.7+1) | mid_v3.rs:819-822 | ✅ 一致 |
| `local_expansion.hpp` FrontPnDnThresholds (min(thphi, 2ndphi+1), thdelta−sumd) | front_pn_dn_thresholds | ✅ 一致 |
| `search_result.hpp:258` SearchResultComparer (φ→δ→Len→amount→`mp_.value`) | `local_expansion.rs` child_ordering_ex (φ→[δ]→Len→[amount]→move_eval) | ✅ 忠実 (criterion 4 と δ/amount は下記) |
| comparer 最終 tie-break `mp_[i].value` (MoveBriefEvaluation) | move_eval (KHPAR で knight-pref 一致) | ✅ ≈一致 (残: §3.2) |
| `LocalExpansion` ctor (mp_ 順 build → idx_ を comparer で sort) | MidLocalExpansion build (build_v3_le_expansion) | ✅ 同形 |
| `FrontSumMask` → Emplace の sum_mask | param_v3_maskkeep / sum_mask | ⚠ 移植済だが **個別有効化で 39te 退行** (§4) |
| `delayed_move_list_` (DML) | dml_prev/dml_next, build_delayed_chain | ✅ 移植済 (Phase 12 sound) |
| `InitialPnDn` seed | seed_pn/seed_dn (KH 一致, v1.13.0) | ✅ 一致 |
| `RepetitionTable` | v3_rep_memo (path_key keyed) | ✅ 配線済 default-on (Phase 29 「未配線」は解消済) |
| `IsRepetitionOrInferiorAfter` (rep + **IsInferior** 劣等局面) | v3_path / is_dominated_in_path (param_v3_dominance) | ✅ 配線済 **default true** |
| `EliminateDoubleCount` (DAG) | eliminate_double_count_v3 (param_v3_dag) | ✅ 移植済 |

結論: **KH SearchImpl stack の主要 12 機構は全て maou に存在し，大半は default で
動いている．** port は機構の不在問題ではない．

## 3. Genuine 残務 (未移植 / 部分移植)

### 3.1 comparer criterion 4 (disproven `repetition_start` 順序)
KH は dn==0 の子を `repetition_start` で並べる (search_result.hpp:279-291)．maou は
**未採用** (child_ordering_ex コメント: 採用すると一部テストが非終了化＝探索発散)．
深い木 (39te) は disproven/repetition 子が多く，この順序欠落が selection を乱す
可能性がある．非終了化の原因 = maou の repetition_start tracking が KH 比で不完全
(GHI fidelity)．→ **GHI 完全化とセットでないと安全に入れられない**．

### 3.2 exact MoveBriefEvaluation / mp_ 生成順の残差
KHPAR で knight-pref は一致したが，v1.13.0 memo に「残=movegen tie 順」．move_eval が
**完全同値でタイ**になったとき，両者とも `std::sort` (unstable) に落ちる＝順序不定．
ここは原理的に co-adapted/arbitrary で，YOORDER (to_sq tie-break) は退行確定 (§4)．
**「exact YaneuraOu generate<CHECKS>/<EVASIONS> 順」が真に必要か**は §5 の iter6
trace で判定する (現データは move_eval ほぼ一致を示唆)．

### 3.3 sum_delta / sum_mask の depth-robustness (★本丸候補)
§6.3 (steady-burning-lantern) の核心: coherent bundle は **29te で KH 一致
(10,308 ≈ KH 9,296) だが 39te 非転移**．機構 (RECALC=incremental δ更新,
MASKKEEP=sum_mask継承) は KH と同形だが個別有効化で退行 (§4)．= maou の他部分が
co-adapt しており，KH 機構を單独移植しても KH equilibrium に乗らない．

## 4. 🔴 既に反証済 (再実行禁止, compass invariant)
RECALC 単独 29te+42%/39te+29%; MASKKEEP +64%/+10%; **full coherent bundle
29te 10,308(=KH 1.11× sound) だが 39te 23.37M 非転移**; YOORDER (KH movegen 順)
29te+145%; unbounded cross-hand +2.8%; re-sort 法 同結果; eps/PN_UNIT/TCA/
proof_hand/caching/full-drop/compound格子36/per-node ALU/check_cache/TT-flat 全棄却．

## 5. 🎯 本 session の fresh oracle data (v2.9.0, committed build で再取得)

root-iter oracle (`V3ROOTI` vs `KHROOT`, 同 39te) を **committed ef49730** で再取得
(前データは uncommitted v2.8.18)．`grep '^V3ROOTI iter=' / '^KHROOT iter='`:

| iter | th | maou pn/dn | KH pn/dn | 判定 |
|---|---|---|---|---|
| 2 | (4,45) | 4/16 | 4/16 | ✅ byte一致 |
| 3 | (7,45) | 8/16 | 8/16 | ✅ |
| 4 | (14,45) | 10/46 | 10/46 | ✅ |
| 5 | (18,79) | 15/81 | 15/81 | ✅ |
| 6 | (26,138) | 26/**110** | 26/**107** | ⚠ **dn 初乖離** |
| 7 | (45,_) | **30**/188 | **27**/186 | pn 乖離 |
| 8 | | 52 | 46 | |
| 11 | | **131** | **91** | gap 44% (compounding) |

### ★ 真の first divergence = iter6 の 2 手 (前 session の「iters 2-6 完全一致」を更新)
同一 th=(26,138) で root children を突合 — 8 手中 6 手は byte 一致，**2 手のみ乖離**:

| move | maou pn/dn | KH pn/dn |
|---|---|---|
| 7b7a | **28**/15 | **26**/15 |
| 7b8a | 26/**18** | **28**/15 |

→ root dn 110 vs 107 はこの 7b8a (dn 18 vs 15) 由来．**閾値は同一なのに 7b7a/7b8a
subtree の評価値が違う** = threshold allocation でなく subtree exploration の乖離．

### iter7 の帰結: KH は集中, maou は分散 (bare config)
KH は **7b6b** (最終 PV 手) に予算集中し pn=27 へ証明．maou の 7b6b は pn=37 へ劣化し，
N\*4d(30)/N\*6d に分散．以降 root best が P\*5c→N\*4d→N\*6d→7b6b と **振動** (= 無駄)．

### ★★ iter6 乖離の根 = 中合い代表駒 (本 session trace で確定, V3THX/KHTHX)
7b8a (=8b の +R が discovered check; 玉 5b へ rank-b の利き) の中合い 6b で:
- **maou (bare)**: `L*6b` (香) を中合い代表に生成 (node_movegen.rs:405 前方利き系=香系).
- **KH**: `N*6b` (桂) を生成．→ 別駒=別 subtree=別 pn/dn → 7b8a 値 26/18 vs 28/15．
これは既知の「中合い代表駒」乖離 (memory: project_dfpn_caching_refuted_interposition_lever)．

### ★★★ 決定的: production config (KHPAR) は iter6 を **解消** し root を KH に追従させる
同 oracle を **production env (KHPAR+CHUAI+ROOTKEEP+XHAND)** で再取得:
| iter | th | maou-prod | KH | maou-bare |
|---|---|---|---|---|
| 6 | (26,138) | 26/**107** ✅ | 26/107 | 26/110 |
| 7 | (45,182)✅ | 32/182 | 27/186 | 30/188 |
| 11 | | **94**/1546 | **91**/1361 | 131 |
→ **KHPAR の knight-pref が iter6 中合いを N\*6b に揃え dn=107 が KH 一致**．iter7 th も
(45,182) で KH と完全一致．iter11 で root pn 94 vs 91 (≈3%, bare 131 から激減)．
**= production の root-selection 層は既に KH を追従している．**

### 🔴 真の gap 再局所化: root selection ではなく **subtree visit-count**
production が root pn/dn を KH に追従するのに full-solve は依然 **builds 3.73×**
(14.48M vs 3.88M KHEMP)．⇒ **同じ root pn/dn に到達するのに maou は ~3.7× の builds を
使う**．gap は comparer/threshold/中合い選択 (= 既に一致) でなく，**各 subtree を
KH より多く visit する (breadth/revisit) inefficiency**．これは §6.3 の
「sum_delta/sum_mask depth 劣化」「visit-count equilibrium」と一致するが，本 session で
**root-selection は除外され subtree-visit に絞り込まれた** のが新しい．

## 6. 推奨する次の concrete step (port の照準を再定義)

本 session の trace で **Step 6.1 (iter6 乖離の root 特定) は完了** し，結論として
**port の照準が変わった**:
- ❌ root-selection (comparer / threshold / 中合い代表) を KH 化する: **不要 — production は既に KH 追従** (iter11 94 vs 91)．iter6 の唯一の乖離 (中合い L\*6b) は KHPAR で解消済．
- ✅ **真の target = subtree visit-count inefficiency**: 同じ root pn/dn に到達するのに
  maou は KH の ~3.7× builds を使う．= 各 subtree の breadth/revisit を KH 級に減らす．

### Step 6.1 (済) — iter6 乖離 = 中合い代表駒 (§5 ★★)．KHPAR で解消．surgical bug ではなく既知 co-adapt．

### Step 6.2 (次) — subtree visit-count の定量と分解 (最優先)
KH と maou-prod で **同一 subtree (例: 7b6b 後 or 7b8a 後) の (unique positions, total visits)**
を計測し，gap が ①breadth (unique 数; maou が KH の訪れない局面へ wander = guidance/selectivity)
か ②revisit (同一 unique を複数回; TT/GHI 再利用不足 = 親再入) かを分離する．
- 既知の手掛り (要再測): Phase 28 で「90K unique vs KH 2094 + 1.8× revisit」(別局面)．
  breadth が支配項なら guidance 設計，revisit 支配なら TT/GHI/parent-reentry．
- 計測: maou `V3_DIAG` の nodes/tt + per-subtree counter (要 instrumentation 追加),
  KH `KHEMP` builds + node 内訳．**単位を揃える** (KHEMP builds = maou v3_le builds)．
- plan §1 の既知観測「maou は 1 子/visit で親再入, KH は 4 子/visit 消化」= revisit 説の傍証．

### Step 6.3 — 分解結果で port 機構を選ぶ (param_v3_v4 gate)
- revisit 支配なら: 親再入を減らす機構 (frame 寿命 / TT proven 保持 / GHI 完全化) を
  param_v3_v4 下で実装．⚠ frame 持続 (inner) は KH 機構でない (compass) ので
  「再入を安くする」方向 (TT reuse 強化 / EliminateDoubleCount 連動) に限る．
- breadth 支配なら: selectivity 機構 (dominance 拡張 / sum_mask depth / multi_pv) を実装．
- gate = default 29te 18,539 死守 + STRICT Some(d) + 同一 subtree visits が KH 方向へ減るか．
  **co-adapt 脱出判定** = 29te が悪化しても 39te builds が KH 方向 (<14.48M) へ減れば採用．

## 7. 環境 (再現用)
- bin: `/tmp/cargo-target/release/deps/maou_shogi-*` (src 編集で hash 変動; `ls -t ... | grep -v '\.d$' | head -1`).
  rebuild: `CARGO_TARGET_DIR=/tmp/cargo-target cargo test --release -p maou_shogi --no-run`.
- 39te: `9/1+R+N1kP2S/6pn1/9/9/5+B3/1R2S4/3p5/9 b NPb4g2sn4l14p 1`
- maou oracle: `V3ROOTI=1 V3_ROOTKEEP=1 V3_BUDGET=2e6 <bin> dfpn::tests::test_mid_v3_39te_measure --ignored --exact --nocapture --test-threads=1 2>&1 | grep '^V3ROOTI'`
- KH: `printf 'usi\nsetoption name Threads value 1\nsetoption name PostSearchLevel value None\nisready\nusinewgame\nposition sfen <39te>\ngo mate 100000\n' | KHROOT=1 timeout 35 .tmp_diag/KomoringHeights/source/KomoringHeights-by-gcc 2> /tmp/kh_root.err`
- baseline 確認 (committed ef49730): default 29te=**18,539** / mate-29 / canonical PV (再現済).
- KH clean solve: `(printf '...\nsetoption name USI_Hash value 4096\n...\ngo mate 600000\n'; sleep 75) | KHEMP=1 timeout 85 ./KomoringHeights-by-gcc` — **stdin を sleep で開いたままにする** (EOF=即 quit→"checkmate timeout" の罠)．option 名は `USI_Hash` (not Hash)．

## 8. Step 6.2/6.3 実測結果 (本 session, committed ef49730)

### 8.1 decomposition: gap は **breadth** (selectivity)．revisit ではない
| | builds | unique | revisit (builds/unique) | wall | mate |
|---|---|---|---|---|---|
| **maou-prod** (KHPAR+CHUAI+RK+XHAND) | 14,478,765 | **9,977,876** (tt) | **1.451×** | 106.7s | Some(55) sound |
| **KH** (clean, USI_Hash 4096) | ~3.88M (KHEMP) | **≤ 3.88M** | ≥ 1 | 10.9s | mate-33 (ub:55), 7.45M info-nodes |
- **maou の unique (9.98M) > KH の総 builds (3.88M)** ⇒ **breadth ratio ≥ 2.57×** (rigorous bound)．
  = maou は KH が *存在すら数えない* distinct 局面を 2.6×+ 探索する．**gap の本体は breadth**．
- **maou revisit=1.451× は低い** ⇒ re-exploration (親再入/TT 再利用不足) は maou の問題でない．
  → revisit-reduction (frame 寿命 / TT reuse) は **lever でない**．selectivity が lever．
- (KH unique の厳密値は hashfull=31‰ × USI_Hash 4096MB が sampling かつ entry size 不明で
  ±大．bound ≤3.88M で十分結論が出る．KH の PV は maou と同形 start: 7b6b 5b4c 8b9c 4c3d…)

### 8.2 🔴 Step 6.3 の含意更新: guidance signal は **既に KH 忠実** — breadth は seed/comparer 起因でない
breadth=selectivity 不足だが，**selectivity を決める guidance signal は mid_v3 で既に KH 移植済**:
- seed = `init_pn_dn_or_kh`/`init_pn_dn_and_kh` = KH `InitialPnDn` 忠実 (support-count 難易度;
  **edge_cost を pn に折込まない＝Phase 25 の decouple は mid_v3 では設計上達成済**)．
- move ordering = `move_brief_eval` = KH `MoveBriefEvaluation`; comparer = `child_ordering_ex` = KH．
- ❌ **Phase 25-31 の guidance/seed lever は全て mid_v2 専用で mid_v3 未配線** (edge_cost/decouple/
  kh_scale/rich_seed)．かつ mid_v2 で実測反証済 (rich_seed +15-16%, δ-tie Mate-31, 「seed quality
  は lever でない」)．→ **guidance signal の局所値を更に KH 化する余地はない (既に一致)**．
- ⇒ breadth の残差は **per-position guidance でなく，sum_delta/sum_mask aggregation の depth 挙動**
  (= §3.3/§6.3 の co-adapted core)．RECALC/MASKKEEP (KH aggregation 機構の単独移植) は退行 (§4)．

### 8.3 結論: port の残務は唯一 **depth-robust aggregation の同時再導出** (gamble)
本 session で **root-selection・guidance signal・seed・comparer・revisit を全て gap 要因から除外**
した．残るのは sum_delta/sum_mask の depth 劣化 1 点のみ．これは個別移植で退行する co-adapted
core であり，KH 級に減らすには「aggregation + それに co-adapt する全部位」を同時 KH 化する
wholesale gamble しかない (29te coherent bundle は 10,308 で実証可能, 39te 深部へ非転移)．
**= compass の構造的結論を最深部で再確認．次段は §8.3 の core を param_v3_v4 下で attack するか，
gamble の妥当性を再判断するか．**

## 9. mid_v4 深さ gate 実験 (2026-06-14, v2.9.1) — 🔴 threshold/aggregation 系は全敗

ユーザ指示で mid_v4 (param_v3_v4) を実際に改善すべく，breadth-is-deep に基づく深さ gate レバーを
3 種実装 (`V3_DEEPRECALC`/`V3_DEEPMASK`/`V3_DEEPEPS`, mid_v3.rs)・full 39te で実測した．

### 9.1 per-depth breadth 局所化 (V3_PLY, prod base)
`V3PLY` 計装で確定: **unique 局面の 88% が d≥11，peak は d=23-27 = 奇数=AND 節 (defender fan-out)．
深部 revisit は 1.2-1.4× (低)** = maou は深部の distinct 局面を ~1 回ずつ訪れる (再探索でなく breadth)．
proof DAG は ~8M distinct (KH ≤3.88M の ~2×)．maou 出力 mate-55 vs KH mate-33．

### 9.2 実験結果 (prod base = 14,478,765 / mate-55 / sound)
| mid_v4 lever | 39te builds | vs base | mate | 解釈 |
|---|---|---|---|---|
| **V3_DEEPRECALC=11** | 18,705,827 | **+29%** | 57 | accurate sum_delta → 再選択増 → breadth 拡大 |
| **V3_DEEPMASK=11** | 14,478,765 | no-op | 55 | v3_dag_masks 未populate (DAG off) で未発火 |
| **V3_DEEPEPS=11:2** | 19,897,874 | **+37%** | 57 | AND-commit (dn 予算↑) → breadth 拡大 |
| **V3_DEEPEPS=11:3** | 22,029,920 | **+52%** | 55 | 同上 (より強く悪化) |

gate: default (無印) 29te=**18,539** 維持確認 (全レバー env-gate)．全 lever STRICT sound (SOLVE)．

### 9.3 結論: threshold 配分は深部で局所最適 — breadth は param でなく構造
- sum_delta を accurate 化 (RECALC) も ε で AND-commit 強化も **両方 breadth を増やす** =
  maou native の threshold 配分 (stale sum_delta + ε=1) は **pn 軸・dn 軸双方で深部局所最適**．
- ⇒ 深部 breadth は **tunable parameter ではない**．§8.3 の「aggregation 同時再導出」も
  個別 param では到達不能を再確認 (perturbation は必ず悪化)．
- **残る非-threshold lever** (未着手, より大きな構造変更):
  1. **proof-width 削減**: 深 AND 節の defender 手数を減らす — 中合い代表駒の削減 (node_movegen,
     soundness risk) / DML futile-defer 強化 / dominance 拡張．breadth=AND fan-out の直撃だが movegen
     変更は unsound risk．param_v3_v4 gate 必須．
  2. **mate-length 短縮**: maou 55 vs KH 33 = maou は長い証明木を建てる．OR 節で短手詰へ誘導する
     bias (find_shortest 不在; mid_v3 未対応)．proof depth↓ = tree↓ だが mechanism 大．
- 反証済 knob (DEEPRECALC/DEEPMASK/DEEPEPS) は comment 付きで残置 (repo 慣習; 再実行禁止)．

## 10. wholesale gamble 着手 — deep-mechanism 監査 (2026-06-14, ユーザ選択)

ユーザが「wholesale gamble」を選択．gamble の現実的形は「matched root と 2× deep DAG の間の
**deep-divergent mechanism** を特定・KH 化」．最有力候補 = `EliminateDoubleCount` (DAG δ 補正;
深部は transposition 多 → 補正不全なら δ 過大 → over-explore → deep breadth) を監査:

### 10.1 maou eliminate_double_count_v3 vs KH FindKnownAncestor
- maou: `parent_map` (child→first-parent, or_insert で更新せず) を遡り祖先 stack で sum_mask reset．
  threshold `3*unit`・OR reset dn / AND reset pn は KH 一致．
- KH: `FindKnownAncestor` が **現 TT best-move 鎖**を上方向に辿り branch root を特定．
- 疑い: maou の stale first-parent は深部で branch root を取り逃す → δ 過大 → breadth．

### 10.2 🔴 反証: maou の DAG 補正は **既に高効率** (V3_NODAG=1 診断)
prod + `V3_NODAG=1` (DAG 補正 OFF) → **26,690,597 builds (+84%)** / mate-55 / sound．
⇒ maou の parent_map 版 DAG 補正は **削ると +84% 悪化＝極めて有効に効いている**．
= 「stale first-parent で無効」仮説は反証．**DAG は deep gap でない** (KH FindKnownAncestor 移植は lever でない)．

### 10.3 gamble の含意: deep gap は単一機構でなく emergent equilibrium
今 session で監査した deep 機構は **全て maou で既に有効/局所最適**:
- DAG δ 補正: 有効 (OFF=+84%)．  - threshold 配分: 局所最適 (両軸 perturbation 悪化, §9)．
- guidance/seed/comparer: KH 忠実 (§8)．  - revisit: 低 1.3× (§8.1)．
⇒ **deep breadth (2× DAG) を説明する単一の「欠落/乖離機構」は存在しない**．gap は
co-adapted system 全体が別 equilibrium にある emergent な結果．
⇒ gamble の「find & fix deep mechanism」形は**消尽**．残るは §6.4 が soundness risk で回避した
**from-scratch 忠実 re-impl (KH SearchImpl 一括 verbatim)** のみ — 数週間, 到達保証なし, soundness risk 大．
**評価**: 11 refuted lever + 本 session の 5 機構監査が全て「maou は既に効いている」を示す．
parity への incremental/mechanism 経路は構造的に閉じた．次段判断 = from-scratch gamble 続行 vs equilibrium 受容．

## 11. faithful verbatim 移植 着手 (2026-06-14, ユーザ指示 = 忠実最優先・近道禁止・single-thread)

ユーザがエンジニアリング近道 (LE path 再利用・env knob) を禁じ，**KH を byte 単位で verbatim 再現**せよと指示．
mid_v3 LE path の非忠実な近道 (確定済): ① **resort 後 stale sum_delta** (KH は RecalcDelta; local_expansion.rs:410-415，
忠実化 V3_RECALC は退行) ② **MateLen len を一切 thread しない** (KH は全 Emplace で len-1) ③ **len 非依存 TT**．
→ 「忠実 bundle は 29te=10,308≈KH だが 39te=23.37M で 6× 乖離」= bundle は 29te-relevant のみ忠実で深さ特有の
非忠実 (②③) が残る．**近道なしの verbatim 再現でのみ偶然一致でなく真の忠実性を検証できる**．

### 移植ロードマップ (verbatim, single-thread, 依存順) — v2.10.0
| # | KH source | maou module | 状態 |
|---|---|---|---|
| 1 | `mate_len.hpp` | `dfpn/mate_len.rs` | ✅ done, 4/4 test (len_plus_1, sub の∞飽和, 順序, 定数) |
| 2 | `search_result.hpp` | `dfpn/search_result.rs` | ✅ done, 7/7 test (u64 PnDn, Make*, TCA, Comparer φ→δ→Len→rep→amount) |
| 4a | `ttentry.hpp` (Entry, 606L のうち core) | `dfpn/ttentry.rs` | ✅ done, 6/6 test (**len-aware LookUpExact: len≥proven→詰/len≤disproven→不詰**, **cross-hand LookUpSuperior/Inferior**, null slot, min_depth/old-child) |
| 4b | `regular_table.hpp` + `ttquery.hpp` + `repetition_table.hpp` | `dfpn/tt_v4.rs` | ✅ done, 5/5 test (cluster probe 循環走査, **set_result→look_up roundtrip**, **cross-hand superior**, repetition (rep_table.Contains len-gate), 別 board 非衝突)．noise/atomic/lock 排除 (single-thread)．生ポインタ→cluster index 化．GC/Save/Load は探索意味論外で後回し |
| 3a | `local_expansion.hpp` core (δ/閾値/選択) | `dfpn/kh_local_expansion.rs` | ✅ done, 5/5 test．**GetPn/Dn/Phi/Delta (deferred penalty 込), FrontPnDnThresholds (second_phi+1), RecalcDelta, ResortFront/Back/ExcludedBack (lower_bound+rotate), UpdateBestChild**．🎯 **resort 後 RecalcDelta を忠実実装** (maou 旧近道 local_expansion.rs:410-415 が落としていた箇所)．synthetic ctor `from_parts` でテスト |
| 3b | `local_expansion.hpp` 構築 + GetWin/LoseResult | kh_local_expansion.rs (current_result) + mid_v4.rs (build_v4_expansion) | ✅ done．MovePicker(movegen)/TT LookUp(seed/rep)/current_result(win/lose/rep) 配線．proof-hand は現持駒で代用 (sound)．DML/1手詰先読み/HandSet 極小化は効率 TODO |
| 5 | `komoring_heights.cpp` SearchImpl | `dfpn/mid_v4.rs` | ✅ done．**search_impl_v4 (len threading 込) + solve_via_v4 (IDS root) + V3_V4ENG gate**．tt は local `&mut` で再帰へ (field 不要)． |

## 🎉 12. 忠実移植エンジン **動作確認** (2026-06-14, v2.10.0) — 29te を mate-29 で sound 解

`V3_V4ENG=1` で全モジュール結線エンジンが起動し，**1te/3te/29te を正しい mate 手数で解いた**:
| 例題 | root pn | mate_len | nodes |
|---|---|---|---|
| 1te | 0 | **1** ✓ | 3 |
| 3te | 0 | **3** ✓ | 26 |
| **29te** | 0 | **29** ✓ | 81,496 |

= **MateLen threading + faithful LocalExpansion (RecalcDelta-after-resort) + len-aware/cross-hand TT** の
verbatim 移植が end-to-end で動き **健全** (mate_len=29 = 既知正解 = 独立な soundness 信号)．
node 81,496 (native 18,539 / KH 9,296 比 大) は **意図的に省いた最適化** (DML deferral / 1手詰先読み) のため．

**default (無印) 29te=18,539 / canonical PV 不変** (V3_V4ENG gate)．**26 faithful unit test green**
(mate_len 4 / search_result 6 / ttentry 6 / tt_v4 5 / kh_local_expansion 5)．crate 2.10.0 uncommitted．

### 残 TODO (効率/完成度; soundness ではない)
- DML deferral・1手詰先読み (CheckObviousFinalOrNode) = node 数を native/KH 方向へ削減．
- PV 抽出 + STRICT replay の v4 版 (現状は root mate_len で健全性評価; Checkmate は空 PV 返却).
- proof/disproof HandSet 極小化 (cross-hand reuse 効率).
- 39te 実走 + KH との node 比較 (本来の目的; 最適化投入後)．

**module 4 完了 = maou 旧 v3_tt との決定的差 (深さ特有非忠実 §11 ③) を埋めた**: 各局面に
proven_len/disproven_len を持ち探索 len で proven/disproven/unknown を判定 (depth-coupled) + 持駒包含で
cross-hand 合成 + len-aware repetition．計 **21 faithful unit test green** (mate_len 4 / search_result 6 /
ttentry 6 / tt_v4 5)．default 29te=18,539 不変 (param_v3_local_exp 別経路)．crate 2.10.0 uncommitted．
次 = module 3 LocalExpansion (TT 上に構築; MovePicker/DML/HandSet 依存)．

**設計原則**: maou primitive (Board/movegen/hash/Hand=[u8;7]) は再利用，**探索構造 (LocalExpansion/SearchResult/TT/len)
は KH と byte 一致**．mid_v3 default (param_v3_local_exp 別経路) は gate で死守 (29te=18,539 不変)．
反証済 deep-gate knob (§9-10) は別件で残置．**並列化は除外** (KH single-thread が対象; maou は single-thread 最適化が目標)．
