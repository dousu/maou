---
title: cshogi 遺構リファクタ完了に伴う docs の stale 記述一掃
date: 2026-07-15
status: applied
applied_in: 6de97dc
target:
  - docs/architecture.md
  - docs/rust-backend.md
  - docs/git-workflow.md
  - docs/visualization/shogi-conventions.md
  - docs/commands/evaluate.md
  - docs/commands/build_game_graph.md
  - docs/testing-guide.md
  - docs/design/position-search/index.md
---

# 提案: 現在の挙動を cshogi 語彙で説明している docs 記述の是正

## 背景

cshogi 依存除去 (Parser 層削除 / Rust エンジン移行 / feature.py 削除 /
LUT 一本化) が完了したが，docs には削除済み API
(`_cshogi_piece_to_piece_id` / `_reorder_piece_planes_cshogi_to_pieceid` /
`csa_parser.py` / `feature.py`) や「cshogi が現行実装である」前提の
記述が残っていた．VETO「cshogi 依存 (コード/定数/語彙) は複雑でも完全
削除」(user, 2026-07-14) に基づき一掃する．

user は 2026-07-15 に「このセッションではすべてのドキュメント編集を
approve とします」と包括承認済み．

## 適用内容

- architecture.md: §「cshogi Library Encapsulation」を
  §「Shogi Engine (Rust) Encapsulation」に全面書き換え．
  raw ID↔PieceId 変換表 (RAW_PIECE_TO_PIECEID) を単一の真実として記載．
  一括 API (maou_search/maou_convert/maou_io) の app 層直接利用を
  設計として明文化．「Replacing cshogi」節は削除
- rust-backend.md: 削除済み `_reorder_piece_planes_cshogi_to_pieceid()`
  参照を RAW_PIECE_TO_PIECEID に置換．104ch 特徴量に legacy 注記
- git-workflow.md: コミット例文の cshogi 語彙を中立化
- shogi-conventions.md: 「cshogi Integration」→ Rust エンジン記述，
  stale な行番号参照 (shogi.py:425) を削除
- commands/evaluate.md: SFEN 検証の主体を cshogi → Rust エンジンに修正
- commands/build_game_graph.md: 「cshogi move16」→「cshogi 互換 move16」
  (データ形式名としての「cshogi 互換」は維持)
- testing-guide.md: 削除した tests/benchmarks/ (死蔵 README のみ，
  対象 API は Phase 5a で削除済み) のセクションを削除
- design/position-search/index.md: feature.py 削除と
  Board.get_normalized_* 委譲を反映

## 保持したもの (設計経緯・provenance として正当)

- design/maou-shogi-concept.md, maou-shogi-crate.md: cshogi 置換の
  設計動機・互換性方針そのものを説明する文書
- 「cshogi 互換」というデータ形式名 (move int / move16 / HCP / raw ID)．
  既存データ互換性の invariant を指す用語として維持
- commands/hcpe_convert.md: 「cshogi は dev 依存の parity oracle のみ」
  という現状を正確に記述済み
- pyproject.toml の cshogi dev 依存 (golden 再生成・交差検証 oracle)
