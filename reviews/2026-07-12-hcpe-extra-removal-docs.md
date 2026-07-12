---
title: hcpe extra 削除に伴う docs 同期
date: 2026-07-12
status: applied
applied_in: PENDING
target:
  - docs/commands/hcpe_convert.md
---

# 提案: hcpe extra 削除を docs に反映

## 背景

production の cshogi 依存が完全に無くなったため，後方互換で残していた
`hcpe` optional extra (cshogi) を pyproject.toml から削除した (Phase 5c)．
cshogi は dev dependency group に parity oracle として残る．

## 提案内容

docs/commands/hcpe_convert.md の Requirements: 「legacy hcpe extra は fixture
再生成のため残置」の記述を「cshogi は runtime/optional 依存ではなく，dev
group の parity oracle としてのみ残る」に更新する．

## 根拠

- pyproject.toml から `[project.optional-dependencies] hcpe` を削除 (Phase 5c)
- dev group の cshogi は存置 (golden 再生成 + 交差検証)
