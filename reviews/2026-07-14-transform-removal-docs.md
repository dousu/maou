---
title: Transform 経路削除に伴う docs 同期
date: 2026-07-14
status: applied
applied_in: 26a14b4
target:
  - docs/commands/utility_benchmark_training.md
  - docs/rust-backend.md
---

# 提案: Transform/cache_transforms 削除を docs に反映

## 背景

cshogi 時代の HCPE→特徴量オンザフライ変換 (Transform クラス) は
production 参照ゼロ (setup.py で transform=None 固定) の死蔵コードだった．
前処理は Rust 一括 API に移行済みのため，Transform クラスと
cache_transforms プラミング (CLI --cache-transforms 含む) を全廃した．

## 提案内容

- utility_benchmark_training.md: --cache-transforms 行を削除
- rust-backend.md: KifDataset 使用例から transform/cache_transforms
  引数を削除

## 根拠

- Transform() の production インスタンス化ゼロ (監査で確認)
- cache_transforms は transform=None 固定のため常に no-op だった
