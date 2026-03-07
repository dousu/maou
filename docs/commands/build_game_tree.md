# `maou build-game-tree`

## Overview

- preprocessデータ(局面単位・集約済み `.feather`)からBFSでゲームツリーを構築し，
  `nodes.feather` + `edges.feather` として出力する．
  【F:src/maou/infra/console/build_game_tree.py†L1-L129】
- 初期局面(平手)からBFSで探索を行い，各局面の `moveLabel` から
  `min_probability` 以上の指し手をエッジとして展開する．
  【F:src/maou/app/game_tree/builder.py†L1-L189】
- 出力データは Arrow IPC 形式(LZ4圧縮)で保存される．
  【F:src/maou/interface/game_tree_io.py†L1-L111】

## CLI options

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--input-path PATH` | Yes | — | preprocessデータのディレクトリまたはファイルパス．再帰的に `.feather` ファイルを収集する．【F:src/maou/infra/console/build_game_tree.py†L45-L50】 |
| `--output-dir PATH` | Yes | — | ツリーデータ(`nodes.feather`, `edges.feather`)の出力先ディレクトリ．存在しない場合は自動作成される．【F:src/maou/infra/console/build_game_tree.py†L51-L56】 |
| `--max-depth INT` | No | `30` | BFSの最大探索深さ．初期局面からの手数上限．【F:src/maou/infra/console/build_game_tree.py†L57-L63】 |
| `--min-probability FLOAT` | No | `0.001` | 指し手の最小確率閾値．この値未満の指し手はツリーに含まれない．表示時のフィルタリング(Epic 2)より小さい値を設定すべき．【F:src/maou/infra/console/build_game_tree.py†L64-L70】 |

## Example invocation

```bash
# 基本的な使用方法
maou build-game-tree \
  --input-path ./data/preprocess/ \
  --output-dir ./data/game-tree/

# パラメータを指定
maou build-game-tree \
  --input-path ./data/preprocess/ \
  --output-dir ./data/game-tree/ \
  --max-depth 20 \
  --min-probability 0.005
```

## Output format

### `nodes.feather`

| Column | Type | Description |
| --- | --- | --- |
| `position_hash` | UInt64 | Zobrist hash(局面の一意識別子) |
| `result_value` | Float32 | 局面の勝率(手番側視点，0.0〜1.0) |
| `best_move_win_rate` | Float32 | 最善手の勝率 |
| `num_branches` | UInt16 | 分岐数(`min_probability`以上の指し手数) |
| `depth` | UInt16 | 初期局面からの最短距離 |

### `edges.feather`

| Column | Type | Description |
| --- | --- | --- |
| `parent_hash` | UInt64 | 親局面のZobrist hash |
| `child_hash` | UInt64 | 子局面のZobrist hash |
| `move16` | UInt16 | cshogi move16形式の指し手 |
| `move_label` | UInt16 | moveLabelのインデックス(0〜1495) |
| `probability` | Float32 | moveLabel値(親局面からの相対出現確率) |
| `win_rate` | Float32 | moveWinRate値(この手の勝率) |

## Implementation references

| Component | File |
| --- | --- |
| CLI command | `src/maou/infra/console/build_game_tree.py` |
| BFS builder | `src/maou/app/game_tree/builder.py` |
| Data I/O | `src/maou/interface/game_tree_io.py` |
| Data models | `src/maou/domain/game_tree/model.py` |
| Polars schemas | `src/maou/domain/game_tree/schema.py` |
