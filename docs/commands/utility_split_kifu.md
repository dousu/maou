# `maou utility split-kifu`

## Overview

棋譜ファイルを**対局単位**で学習用と検証用に分割する．

分割の単位はファイルである．floodgate の棋譜は 1 ファイル 1 対局なので，
これがそのまま対局単位の分割になる (複数対局を含むファイルは分割されず
まとめて片側に入る)．

## なぜ前段で分けるのか

`maou pre-process` は局面を Zobrist hash で**全コーパス横断に集約**するため，
集約後に対局の同一性は復元できない．`maou learn-model --test-ratio` による
分割は前処理出力のチャンクファイル単位であり，チャンクは hash 順に並ぶので
**同一対局の局面が train と val の両方に入る**．1 手違いのほぼ同一局面が
両側に散るため検証損失が楽観的になり，early stopping も機能しにくい．

したがって対局単位の分割は棋譜 (または HCPE) の段階でしか行えない．

背景と測定結果は [docs/design/training-quality/](../design/training-quality/index.md)
を参照．

## CLI options

| Flag | Default | Description |
| --- | --- | --- |
| `--input-path PATH` | required | 分割対象の棋譜ディレクトリ (またはファイル)．再帰的に走査する．【F:src/maou/infra/console/utility.py】 |
| `--train-dir PATH` | required | 学習側の出力ディレクトリ． |
| `--val-dir PATH` | required | 検証側の出力ディレクトリ．`--train-dir` と同じパスは拒否される． |
| `--val-ratio FLOAT` | `0.1` | 検証側に回すファイルの割合．`0 < ratio < 1`．端数で 0 件になる場合も最低 1 件は割り当てる． |
| `--seed INT` | `42` | シャッフルの乱数シード．**同じ入力と同じ seed からは常に同じ分割**が得られる (入力の列挙順には依存しない)． |
| `--ext TEXT` | (全ファイル) | 収集する拡張子 (例 `.csa`)． |
| `--mode [copy\|symlink\|hardlink]` | `copy` | `copy` は実体複製．`symlink` / `hardlink` は追加のディスクを消費しない (`hardlink` は同一ファイルシステムのみ)． |
| `--dry-run` | `False` | 件数だけ報告してファイルを作らない． |

## Output

入力ディレクトリからの**相対パスを出力先でも保つ**ので，
floodgate の `YYYY/MM/DD` 構造がそのまま残る．

標準出力に JSON を 1 行返す:

```json
{"input_path": "...", "train_dir": "...", "val_dir": "...",
 "total_files": 878, "seed": 42, "mode": "hardlink", "dry_run": false,
 "train_files": 790, "val_files": 88}
```

## Example

```bash
# 1. 棋譜を取得
maou utility fetch-floodgate --start-date 2020-01-01 --end-date 2025-12-31 \
  --output-dir floodgate --strategy archive --archive-cache-dir floodgate-archives

# 2. 対局単位で分割 (ディスクを節約するため hardlink)
maou utility split-kifu --input-path floodgate \
  --train-dir floodgate-train --val-dir floodgate-val \
  --val-ratio 0.05 --ext .csa --mode hardlink

# 3. それぞれ HCPE 化して前処理
for split in train val; do
  maou hcpe-convert --input-path floodgate-$split --input-format csa \
    --output-dir hcpe/$split
  maou pre-process --input-path hcpe/$split --output-dir preprocess/$split \
    --position-count-threshold 3 --win-rate-fallback neutral
done

# 4. 学習 (検証は別ディレクトリから読む)
maou learn-model --stage 3 \
  --stage3-data-path preprocess/train \
  --stage3-validation-data-path preprocess/val \
  --early-stopping-patience 5 --epoch 100 ...
```

**注意**: 前処理は train / val を別々に走らせる．同じ集約に混ぜると
分割の意味が失われる．

## Validation and guardrails

- `--train-dir` と `--val-dir` が同一パスの場合はエラー．
- 入力にファイルが 1 件も無い場合はエラー (拡張子指定の誤りを早期に検出する)．
- ファイルが 2 件未満の場合はエラー (分割不能)．
- 出力先に同名ファイルが既にある場合は上書きする．
