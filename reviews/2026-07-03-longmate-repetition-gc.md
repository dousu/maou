---
title: 長手数詰将棋対応 (千日手テーブル GC / depth 上限 2047) を設計docへ反映
status: applied
approved_by: user (2026-07-03 指示: 正式機能化 + 設計doc 反映)
applied_in: cd18976 (code) + 8b51216 (docs)
---

# 長手数詰将棋対応を設計ドキュメントへ反映

## 承認の経緯

user が 2026-07-03 に指示:
> 正式機能化してください．詰将棋ソルバー設計ドキュメントにも反映してください

コード側の対応: `feat(dfpn)! 長手数詰将棋対応` (cd18976)．本 review はその設計doc 反映．

## 背景 (実証)

depth 上限 (旧 47) を撤廃し，千日手テーブル (`RepetitionTable`) を無制限 `FxHashMap` から
KomoringHeights 流の **固定サイズ + generation GC** へ置換した結果，maou は
**ミクロコスモス (1525 手詰) を探索可能**になった:

- find_shortest=False: STRICT-verified 詰みを 151s / 26.5M nodes で取得．
- find_shortest 短縮: 無駄合い-free で mate_len を 1549→**1525** (canonical) へ収束させ，
  STRICT VERIFY Some(1525) を確認．
- 旧実装は千日手テーブルが青天井膨張して OOM していたが，generation GC 導入で len=1523
  反証を 224M nodes 走らせても OOM しなくなった (最短性の formal 証明は時間律速で継続)．

健全性: 160 lib + 2 canonical anchor (29te 396,516 / 39te 17,545,528) 全通過・node 数不変
(rep GC は Microcosmos 規模でのみ発火, 通常問題は挙動不変)．

## 適用した doc 変更

| ファイル | 変更 |
|---|---|
| index.md | 設計目標 2 に「超長手数 (1525 手) 対応・depth 上限 2047・千日手テーブル GC」を追記．公開 API 節の depth 上限を 47→2047 へ |
| transposition-table.md | **§6.7 新設**「千日手テーブルの generation GC」(固定サイズ open-addressing + generation GC・健全性=miss は再探索・REPSIZE)．§6.6→§6.8 に繰上げ．§6.5 に final エントリ GC 保護を追記 |
| loop-ghi.md §7.1 | `RepetitionTable` が固定サイズ + generation GC で bound される旨と，eviction が再探索で健全な旨を追記 |
| rust-backend.md | Python API 例・パラメータガイドの depth 上限を 47→2047 (長手数対応) へ |

## 不変条件

- 実装記述・アルゴリズム説明・出典は正確性を維持 (KomoringHeights 帰属を明記)．
- 千日手処理の健全性 (GHI) の説明は不変 — GC は「memo の eviction = 再探索」であり，
  千日手の正しさを担う on-path 検出 (`path_depths`) には触れない．
