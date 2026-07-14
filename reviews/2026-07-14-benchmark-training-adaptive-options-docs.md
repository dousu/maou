---
title: benchmark-training への adaptive batch オプション追加に伴う docs 同期
date: 2026-07-14
status: applied
applied_in: PENDING
target:
  - docs/commands/utility_benchmark_training.md
---

# 提案: benchmark-training の新オプションを docs に反映

## 背景

learn-model に存在する 7 オプション (--gradient-accumulation-steps と
adaptive batch 系 6 種) が benchmark-training に欠けており，CLI 同期テスト
(test_cli_option_compatibility) が失敗していた．オプションを CLI→interface→
app (TrainingLoop) に配線して追加した．

## 提案内容

docs/commands/utility_benchmark_training.md の「Training & benchmarking
knobs」表に --gradient-accumulation-steps と adaptive batch 系オプションの
行を追加する (adaptive は Stage 3 のみ適用，learn-model と同じ構築規則)．

## 根拠

- CLI 同期は benchmark-training-sync の検証対象 (learn-model と同一
  パラメータでベンチマークできることが要件)
- test_cli_option_compatibility が green になることを確認済み
