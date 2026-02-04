# Mock Data Expansion Design

## Overview

`maou visualize --use-mock-data` で HCPE 以外のデータ型（Stage1, Stage2, Preprocessing）のモックデータも生成できるようにする．

## Goals

- UI/レイアウトの確認: 各データ型のUIが正しく表示されるかテスト
- 視覚的なデモ: 実際のデータに近い見た目で機能のデモンストレーション

## Architecture

```
DataRetriever._create_mock_record()  (既存)
    │
    ├─ _create_hcpe_mock()          (既存ロジックを分離)
    ├─ _create_stage1_mock()        (新規: Stage1Generatorを参照)
    ├─ _create_stage2_mock()        (新規: row_numberベース生成)
    └─ _create_preprocessing_mock() (新規: row_numberベース生成)
```

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/maou/app/visualization/data_retrieval.py` | `_create_mock_record()` を分割，各データ型専用メソッド追加 |
| `src/maou/infra/visualization/search_index.py` | `build_mock()` をデータ型別に対応 |

## Data Type Specifications

### Stage1 (到達可能マス)

**データ構造:**
```python
{
    "id": "mock_stage1_{row_number}",
    "boardIdPositions": [[...], ...],  # 9x9 盤面
    "piecesInHand": [...],             # 14要素
    "reachableSquares": [[...], ...]   # 9x9 binary (0/1)
}
```

**生成方針:**
- `Stage1DataGenerator` の `enumerate_board_patterns()` から代表的なパターンを取得
- `row_number % パターン数` でパターンを選択
- 各パターンには既に `reachableSquares` が計算済み

**バリエーション:**
- 序盤: 歩・香・桂など移動範囲が狭い駒のパターン
- 中盤: 角・飛車など移動範囲が広い駒のパターン
- 終盤: 馬・龍など成駒のパターン

### Stage2 (合法手リスト)

**データ構造:**
```python
{
    "id": "mock_stage2_{row_number}",
    "boardIdPositions": [[...], ...],  # 9x9 盤面
    "piecesInHand": [...],             # 14要素
    "legalMovesLabel": [...]           # 2187要素 binary (0/1)
}
```

**生成方針:**
- 盤面は `row_number` に応じて駒の配置を変化（中盤風: 駒が適度に交換された状態）
- 合法手は `row_number` に応じて数が変化:
  - `row_number % 5 == 0`: 少なめ（10〜20手）
  - `row_number % 5 == 1-3`: 中程度（30〜50手）
  - `row_number % 5 == 4`: 多め（60〜80手）
- 合法手のインデックスは連続ではなく，実際の将棋に近い分布

### Preprocessing (着手確率)

**データ構造:**
```python
{
    "id": "mock_preprocessing_{row_number}",
    "boardIdPositions": [[...], ...],  # 9x9 盤面
    "piecesInHand": [...],             # 14要素
    "moveLabel": [...],                # 2187要素 float (確率分布, 合計=1.0)
    "resultValue": float               # 勝敗値 (-1.0 〜 1.0)
}
```

**生成方針:**
- 盤面は `row_number` に応じて終盤風（駒が少なめ）
- 確率分布は `row_number` に応じてパターンが変化:
  - `row_number % 3 == 0`: 集中型（1手に80%以上）
  - `row_number % 3 == 1`: 分散型（上位3手に均等）
  - `row_number % 3 == 2`: 混合型（1手が50%，残りが分散）
- `resultValue` は `row_number` に応じて -1.0 〜 1.0 で変化

## SearchIndex Changes

| データ型 | インデックスの特徴 |
|---------|-----------------|
| **HCPE** | 現状維持（eval -3000〜3000） |
| **Stage1** | eval不要，IDインデックスのみ |
| **Stage2** | eval不要，IDインデックスのみ |
| **Preprocessing** | resultValue でソート可能に（-100〜100 にスケール） |

## New Methods (data_retrieval.py)

- `_create_hcpe_mock(row_number)` - 既存ロジックを分離
- `_create_stage1_mock(row_number)` - Stage1Generator参照
- `_create_stage2_mock(row_number)` - 中盤風盤面 + 合法手生成
- `_create_preprocessing_mock(row_number)` - 終盤風盤面 + 確率分布生成
- `_get_stage1_patterns()` - Stage1パターンのキャッシュ
- `_create_midgame_board(row_number)` - 中盤風盤面生成
- `_create_endgame_board(row_number)` - 終盤風盤面生成

## Testing

各データ型で以下のコマンドが動作することを確認:

```bash
maou visualize --use-mock-data --array-type hcpe
maou visualize --use-mock-data --array-type stage1
maou visualize --use-mock-data --array-type stage2
maou visualize --use-mock-data --array-type preprocessing
```
