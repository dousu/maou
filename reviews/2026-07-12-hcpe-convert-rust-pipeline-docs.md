---
title: HCPE 変換の Rust パイプライン移行に伴う docs 同期
date: 2026-07-12
status: applied
applied_in: PENDING
target:
  - docs/commands/hcpe_convert.md
  - docs/rust-backend.md
---

# 提案: hcpe-convert の Rust 一括変換移行を docs に反映

## 背景

HCPE 変換を `maou._rust.maou_convert` の一括パイプラインへ移行した
(c8adeb6, Phase 4)．per-move PyO3 往復 + ProcessPoolExecutor を廃し，
ファイル直読み + rayon 並列 + Arrow 直出力を Rust 側で完結する．承認済みの
挙動変更 (複数局 CSA 全変換 / cp932 .kif 対応) も有効化された．

## 提案内容

### docs/commands/hcpe_convert.md

- Overview / Requirements: 変換エンジンが `maou._rust.maou_convert` に
  なったこと，複数局 CSA 全変換・cp932 fallback を明記
- `--process-max-workers`: プロセスワーカー → rayon スレッドへのマップに更新
- Execution flow: ProcessPool 記述を Rust 一括変換 (rayon) に置換

### docs/rust-backend.md

- crate 一覧 / モジュールツリーに `maou_convert` (棋譜→HCPE 一括変換) を追加
- PyO3 サブモジュール `maou._rust.maou_convert` を追記

## 根拠

- 実装: c8adeb6 (converter 切替) / daa2877 (maou_convert crate)
- 挙動: test_hcpe_golden.py / test_rust_convert_parity.py で bit-exact 実証
