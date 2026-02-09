# `maou utility generate-stage1-data`

## Overview

Stage1データ生成コマンドは，将棋AIの基礎学習用データセットを生成する．Stage1では**駒の到達可能マス（reachable squares）** を学習するための最小限の盤面パターンを作成する．

### 3段階学習パイプライン

| Stage | 学習内容 | パターン数 |
|-------|----------|------------|
| **Stage 1** | 駒の移動先（到達可能マス） | ~1,105 |
| Stage 2 | 合法手の制約 | 未実装 |
| Stage 3 | 最適な指し手（従来の学習） | 既存実装 |

Stage1は最も基礎的な段階で，単一の駒が盤上のどのマスに移動できるかを学習する．

## 生成されるデータ

### パターン構成

1. **盤上パターン（1,098件）**
   - 各駒種（歩，香，桂，銀，金，角，飛，王 + 成駒）を盤上の全合法位置に配置
   - 駒種ごとの配置制約を考慮（例：歩は1段目に配置不可）

2. **持ち駒パターン（7件）**
   - 先手の持ち駒として各駒種（歩，香，桂，銀，金，角，飛）を1枚保持
   - 盤上は空の状態

### 出力フォーマット

Arrow IPC（.feather）形式，LZ4圧縮．

| カラム | 型 | 説明 |
|--------|-----|------|
| `id` | UInt64 | 一意のレコードID |
| `boardIdPositions` | List[List[UInt8]] | 9×9の駒配置（駒IDの2次元配列） |
| `piecesInHand` | List[UInt8] | 持ち駒配列（14要素：先手7種 + 後手7種） |
| `reachableSquares` | List[List[UInt8]] | 9×9の到達可能マス（バイナリマップ） |

## CLI オプション

| フラグ | 必須 | 説明 |
|--------|------|------|
| `--output-dir PATH` | Yes | 出力ディレクトリパス．存在しない場合は自動作成される． |

## 使用例

### 基本的な使用

```bash
# Stage1データを生成
maou utility generate-stage1-data --output-dir ./data/stage1/
```

### 出力例

```
✓ Generated 1105 patterns
✓ Saved to: ./data/stage1/stage1_data.feather

Stage 1 data generation complete!
```

## 生成されたデータの可視化

生成したデータは`maou visualize`コマンドで確認できる：

```bash
# 可視化サーバーを起動
maou visualize --input-path ./data/stage1/stage1_data.feather --array-type stage1
```

ブラウザで http://localhost:7860 にアクセスすると：
- 盤面上の駒位置
- 到達可能マス（ハイライト表示）
- レコードナビゲーション

が確認できる．

## 実行フロー

1. **パターン生成** - `Stage1DataGenerator`が盤上パターンと持ち駒パターンを列挙
2. **合法手計算** - 各パターンに対して到達可能マスを計算（cshogi非依存の独自実装）
3. **データ保存** - Polars DataFrameとしてArrow IPC形式で出力

## 関連コマンド

- [`maou visualize`](./visualize.md) - データの可視化
- [`maou learn-model`](./learn_model.md) - モデル学習（Stage3）
