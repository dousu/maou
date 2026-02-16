# `maou utility generate-stage2-data`

## Overview

Stage2データ生成コマンドは，HCPEデータから合法手予測用の学習データを生成する．盤面ハッシュによる重複排除（Phase 1）を行い，各ユニーク局面に対して合法手ラベルを生成（Phase 2）する．

### 3段階学習パイプライン

| Stage | 学習内容 | パターン数 |
|-------|----------|------------|
| Stage 1 | 駒の移動先（到達可能マス） | ~1,105 |
| **Stage 2** | 合法手の制約 | HCPEデータ依存 |
| Stage 3 | 最適な指し手（従来の学習） | 既存実装 |

Stage2はStage1で学習した駒の移動可能範囲の上に，味方駒や王手回避などの制約を加えた**合法手（legal moves）** を学習する段階である．

### 出力フォーマット

Arrow IPC（.feather）形式，LZ4圧縮．

## CLI オプション

| フラグ | 必須 | 説明 |
|--------|------|------|
| `--input-path PATH` | Yes | HCPEフェザーファイルを含む入力ディレクトリパス． |
| `--output-dir PATH` | Yes | 出力ファイルのディレクトリパス． |
| `--output-gcs` | No | Google Cloud Storageへの出力を有効にするフラグ． |
| `--output-bucket-name TEXT` | No | 出力先のGCSバケット名． |
| `--output-prefix TEXT` | No | 出力先のGCSプレフィックスパス． |
| `--output-data-name TEXT` | No | GCS上でデータを識別する名前（デフォルト: `stage2`）． |
| `--chunk-size INT` | No | 1チャンクあたりの局面数（デフォルト: `100000`）． |
| `--intermediate-cache-dir PATH` | No | 中間データキャッシュ用ディレクトリ（デフォルト: 一時ディレクトリ）． |

## 実行フロー

1. **Phase 1: ユニーク局面の収集** - 入力HCPEデータから盤面ハッシュによる重複排除を行い，ユニークな局面を収集する
2. **Phase 2: 合法手ラベルの生成** - 各ユニーク局面に対して合法手ラベルを生成する
3. **データ保存** - Arrow IPC（.feather）形式，LZ4圧縮でチャンク分割して出力

【F: `src/maou/infra/console/utility.py` L1105-1248】

## 使用例

### 基本的な使用

```bash
# Stage2データを生成
maou utility generate-stage2-data --input-path ./converted_hcpe/ --output-dir ./stage2_data/
```

### GCSアップロード付き

```bash
# GCSにもアップロードする場合
maou utility generate-stage2-data \
  --input-path ./converted_hcpe/ \
  --output-dir ./stage2_data/ \
  --output-gcs \
  --output-bucket-name my-bucket \
  --output-prefix stage2/latest
```

### 出力例

```
✓ Input: 500000 positions
✓ Unique: 123456 positions
✓ Output: 2 files
  - ./stage2_data/stage2_chunk_0.feather
  - ./stage2_data/stage2_chunk_1.feather

Stage 2 data generation complete!
```

## 関連コマンド

- [`maou utility generate-stage1-data`](./generate-stage1-data.md) - Stage1データ生成（駒の到達可能マス）
- [`maou learn-model`](./learn_model.md) - モデル学習（Stage3）
