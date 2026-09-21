---
status: pending          # pending | approved | applied | rejected
applied_in:
date: 2026-09-22
target: [docs/commands/learn_model.md]
risk: low
reversibility: trivial
---

# `learn-model` の worker メモリに関する案内を実態に合わせる

## Trigger

2026-09-21 の Colab G4 (176GB) で Stage 3 の学習 DataLoader worker 5 本が各 ≈34 GiB
(合計 ≈169 GiB) に達して memory cgroup の OOM になった (`DataLoader worker ... killed
by signal: Killed`)．worker のメモリは **担当 1 ファイルの展開後サイズ** (前処理済 1M 行 ≈
12 GB: `moveLabel` / `moveWinRate` が各 1496×f32) で決まり，バッチサイズはほぼ無関係．

`docs/commands/learn_model.md` の `--stage3-batch-size` の行は「worker が OOM-kill されたら
これを下げよ」と案内しているが，バッチサイズを下げても worker のピーク (ファイル丸ごとの
展開 + 変換) は変わらない．同 PR で `src/maou/app/learning/setup.py` の見積を
「圧縮サイズ × 4 × 2」から「最大ファイルの展開後サイズ (スキーマ × 行数) × 2.0 × 1.15」に
変え，`--dataloader-workers` は利用可能メモリの 0.8 をこの値で割った本数に自動で
clamp されるようになった (ログ `Memory-based worker limit: N (...)`)．

## 変更 1 (`--stage3-batch-size` の行)

現行:

```
Lower this to reduce Stage 3 streaming DataLoader worker memory when workers are OOM-killed (`DataLoader worker ... killed by signal: Killed`).
```

変更後:

```
Batch size does not change streaming DataLoader worker memory (each worker decompresses one whole preprocessing file, ≈12 GB per 1M rows); if workers are OOM-killed (`DataLoader worker ... killed by signal: Killed`), lower `--dataloader-workers` or write smaller preprocessing chunk files.
```

## 変更 2 (`--dataloader-workers` の行)

現行:

```
Worker processes for PyTorch DataLoaders. Negative values raise `ValueError`.
```

変更後:

```
Worker processes for PyTorch DataLoaders. Negative values raise `ValueError`. Stage 3 streaming clamps this to what fits in memory: each worker holds one whole decompressed preprocessing file (≈12 GB per 1M rows, estimated from the file schema and row count), and at most 80% of available RAM is given to workers — see the `Memory-based worker limit` log line.
```

## 影響

案内の訂正のみ．CLI の動作は同 PR の `src/` 変更 (見積の是正) で既に変わっている．
