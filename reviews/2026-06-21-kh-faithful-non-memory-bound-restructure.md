---
title: "mid_v4 を KH 忠実な non-memory-bound 構造へシフト — per-node HashMap 撤廃 + TT cache-cluster 化"
status: applied
applied_in: a202cd6  # v2.48.0 (R1 flat path stack + R2 TT 64B cache-line)
date: 2026-06-21
branch: feat/tsume-solver
base-sha: 0355910  # v2.47.0
campaign: "39te を KH 相当の実行時間で安定して解く (真因 = mid_v4 の memory 帯域律速)"
trigger: worklog/2026-06-21-165925.md
---

## 1. 背景 / 動機 (実測で確定した真因)

39te 探索時間 gap の真因は **mid_v4 の memory 帯域律速** (本日 2026-06-21 同一 host window min-of-3 で確定):
- **maou は host 負荷で 2.3× 振れる (16→37s)，KH は安定 14s**．一様負荷なら比は不変のはず → maou だけ膨れる ＝
  maou は DRAM/メモリ帯域律速，KH は cache/計算効率で安定．**静かな host では maou は既に KH の 1.16× (16.3/14)**．
- **回帰は無い**: 16.3s を出した v2.39.0 (87e7664) 自体が今日 36.8s; v2.42.0 36.2s; 現 OFF 36.0s ＝ 全て同等．
  「baseline が無駄に増えた」のは host メモリ競合であって code 変更ではない．
- gap はアルゴリズム (count は KH 級, mate1ply も KH 忠実化で sound) でなく **メモリ帯域**．effect 機構が
  net 償却されなかった (review §10) のも同根: effect は query 計算高速化で，メモリ帯域 gap には効かない．

V4PROF (本日, 一瞬空いた 16.0s run) の memory-touching phase: cl_other(seed/query/**pathcheck**) 24.3% /
mate1ply 23.3% / **tt_lookup 12.9%** (20.2M lookup × 102ns DRAM)．

## 2. メモリ律速の出所 (コードで特定)

- **① per-node HashMap (散在 DRAM)**: `mid_v4.rs:1017-1077`
  - `self.v3_path.insert(board.hash, depth)` / `.remove` — 千日手検出 (board.hash キー)．
  - `self.v4_dom_path.entry(position_key(board))` / `.get_mut` / `.remove` — DAG (EliminateDoubleCount)．
  - 毎ノード insert/remove/get ＝ HashMap bucket への散在 DRAM アクセス (cl_other pathcheck の主因)．
- **② 576MB TT (DRAM-bound)**: `tt_v4.rs` `entries: Vec<Entry>`, 8M × 72B = 576MB (tt_v4.rs:346 コメントで
  「L3 を遥かに超え look_up は memory-bound」明記)．Entry 72B は cache-line (64B) 跨ぎ．prefetch (v2.40) で
  緩和済だが本質はレイアウト．

KH/YaneuraOu は (a) 千日手を **path_key ベースの flat RepetitionMemo** で持ち (HashMap でない), (b) DAG/parent を
TT 内に持ち, (c) TT は **cluster (1 cache-line 内 probe, cache-line 整列)**．これが KH の host 負荷非依存性の源．

## 3. 提案 (段階導入; 各段で探索不変 + 同一 window 計測)

- **Stage R1: per-node HashMap 撤廃**
  - `v3_path` (千日手) → KH 流 flat `RepetitionMemo` (path_key ベース)．`dfpn/repetition_memo.rs` /
    `dfpn/path_key.rs` は Phase 29 で Stage 1a/1b 実装済・**未配線** ([[project_dfpn_phase29_kh_port]])．これを配線．
  - `v4_dom_path` (DAG) → TT/parent ベース，または flat 構造へ．
  - ⚠ 探索不変 (千日手判定・DAG 結果が bit 一致) を canonical 18,539 + 差分検証で死守．
- **Stage R2: TT cache-cluster 化**
  - Entry 72B (cache-line 跨ぎ) → cluster (1 cache-line 内に複数 entry, cache-line 整列, probe は cluster 内)．
  - pn/dn=K_INFINITE_PN_DN sentinel + parent_* (DAG) 必須ゆえ単純 u32 化は不可 (既知) → レイアウト/cluster で攻める．
  - look_up の DRAM miss と帯域敏感性を下げる．

## 4. 受け入れ基準

- **探索不変死守**: canonical mid_v3 **18,539** / node (3.11M/9,288/17,720) / 199 + ignored 8 pass / mate-len soundness．
- **計測**: 同一 host window で KH と interleave min-of-N (cross-session 禁止; 静かな host を引くまで複数 round)．
  目標 = host 負荷敏感性を圧縮し (16-37s の振れを縮小)，KH 14s に安定接近 (現 1.16×@quiet を維持しつつ contention 耐性)．
- docs/architecture.md (TT/memory 構造) / docs/rust-backend.md を更新 (完了後; モデルは直接編集しない)．

## 5. リスク / 留意

- 大規模・soundness-critical (千日手/DAG/TT は探索の核)．段階導入 + 差分検証必須．
- TT cluster 化は探索結果 (collision/replacement 方針) を変え得る → 探索不変を厳格検証．
- wheel 可搬性 binding (HW 命令 runtime gate のみ)．
- effect コード (v2.43-47, feature `effect_table` gated) は **保持** (ユーザ指示; 本リストラとは独立)．

## 6. 承認後のアクション

着手は承認後．完了時に docs を更新し status: applied + SHA を記入する (docs 編集はユーザ; モデルは直接編集しない)．

## 7. 適用結果 (2026-06-21, a202cd6 / v2.48.0)

**実装完了・全 search-invariant** (canonical mid_v3 18,539 / 39te 3,105,196 nodes /
6,384,324 do_moves / mate-65 byte 一致; 203 suite + 14 新 unit pass):
- **R1**: `dfpn/path_stack.rs` (`PathStack` drop-in + `DomPathStack`)．`v3_path`/`v4_dom_path`
  FxHashMap を flat 連続配列 + 逆順線形走査へ置換 (KH `ContainsInPath` 同型)．
- **R2**: `ttentry::Entry` 72B→**64B = 1 cache line** (`#[repr(C, align(64))]`; proven/disproven_len
  u16, min_depth 15bit + rep flag 1bit pack)．TT 576MB→512MB．

**🎯 但し memory 構造は wall lever でないと実測確定 (重要; 当初動機を部分的に反証)**:
- 同一 window min-of-N: quiet host = base 21.2s / **new 21.3s** / KH 14.1s．
  **DRAM contention 下** (自作 streaming stressor ×3 を core 1-3 pin) = base 45.9s / **new 45.9s**．
- 理由: search-invariant = memory **アクセス回数**不変 → layout (bytes/access, scatter) 変更では
  latency 律速の wall は縮まらない (contention 下で new==base が決定的証拠)．
- **R2 は小 micro-win** (V4PROF: tt_lookup 100→86ns/op, -14%, ~-2% wall)．**R1 は neutral**．
- **真の gap = per-node compute** (maou node 少 3.11M<KH 3.88M だが 1.5×/node 遅):
  mate1ply 23.6% (1207ns) / cl_other(seed/query) 26.6%．
- → 旧 compass「真因 = memory 帯域律速 (lever)」を実験棄却．次 lever (ユーザ選択) =
  **mate1ply の KH 忠実移植** (`Mate::mate_1ply` king-centric constructive; movegen 回避が ~6× 定数差)．
