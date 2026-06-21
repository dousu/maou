---
title: "mid_v4 を KH 忠実な non-memory-bound 構造へシフト — per-node HashMap 撤廃 + TT cache-cluster 化"
status: pending
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
