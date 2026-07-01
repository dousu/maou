---
status: applied
applied_in: bc1aa8b
target: docs/design/tsume-solver/aigoma-optimization.md
date: 2026-07-01
---

# 合駒最適化ドキュメントに 案A (無駄合い-free len budget) を反映

## 背景

commit 03f635a (maou_shogi 3.4.0) で `find_shortest` の探索側に **案A (透過中合い drop を
len 予算から credit)** を導入し，39te root の 43 手過大評価を根治した (真の最短 39 手を報告)．
これは §8.2 の post-pass (報告手数の補正) とは別の，探索本体 (len 予算 + mate_len 集計) の変更である．
user から「設計ドキュメントにも今回の探索アルゴリズムの変更を反映せよ」と明示要求 (2026-07-01)．

## 提案

`docs/design/tsume-solver/aigoma-optimization.md` に **§8.4 無駄合い-free len budget —
find_shortest 探索側** を追加:

- 問題: find_shortest の len 予算が無駄合い込み raw ply を数え，短い詰みを偽 disproof (旧「完全性
  バグ」誤診の実体).
- 修正: `transparent_interposition_squares`(movegen) / `child_len`(search/mod) / `current_result`
  AND-proven mate_len sub(2)(expansion) / `len<DEPTH_MAX` gate で first-mate 不変.
- 探索側 credit と post-search post-pass の役割分担 (過去の proven_len discount 棄却との差異).
- perf 影響 (未解決; 刈りは unsound, collapse は今後).

## 承認

user 明示要求 (2026-07-01) → approved．commit 時に applied 化 (applied_in に SHA 記録)．

## 適用

- `aigoma-optimization.md` §8.4 追記 (この doc commit で適用)．
