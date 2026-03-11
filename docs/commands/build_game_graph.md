# `maou build-game-graph`

## Overview

- preprocessデータ(局面単位・集約済み `.feather`)からBFSでゲームグラフ(有向グラフ)を構築し，
  `nodes.feather` + `edges.feather` として出力する．
- 初期局面(平手)からBFSで探索を行い，各局面の `moveLabel` から
  `min_probability` 以上の指し手をエッジとして展開する．
- 同一局面への合流(transposition)や局面循環(千日手等)により，構築されるグラフは閉路を含む有向グラフとなる．
  各ノードの `depth` はBFS最短距離を記録する．
- 出力データは Arrow IPC 形式(LZ4圧縮，Rustバックエンド使用)で保存される．

## CLI options

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--input-path PATH` | Yes | — | preprocessデータのディレクトリまたはファイルパス．再帰的に `.feather` ファイルを収集する． |
| `--output-dir PATH` | Yes | — | グラフデータ(`nodes.feather`, `edges.feather`)の出力先ディレクトリ．存在しない場合は自動作成される． |
| `--max-depth INT` | No | `30` | BFSの最大探索深さ．初期局面からの手数上限． |
| `--min-probability FLOAT` | No | `0.001` | 指し手の最小確率閾値(0.0〜1.0)．この値未満の指し手はグラフに含まれない．表示時のフィルタリング(Epic 2)より小さい値を設定すべき． |
| `--initial-hash INT` | No | 平手初期局面 | 開始局面のZobrist hash(preprocessデータのID)．指定した局面からBFSを開始する．`--initial-sfen` と併用必須． |
| `--initial-sfen TEXT` | No | — | 開始局面のSFEN文字列．`--initial-hash` 指定時に必須．BFSで正しい盤面を復元するために使用する． |
| `--max-cache-files INT` | No | `1` | List型カラムのLRUキャッシュファイル数．1ファイルあたり約11.5GBのメモリを使用する(100万行 × 1496要素 × 4bytes × 2列)．メモリに余裕がある場合は2〜3に増やすことでキャッシュヒット率が向上する． |

## Example invocation

```bash
# 基本的な使用方法
maou build-game-graph \
  --input-path ./data/preprocess/ \
  --output-dir ./data/game-graph/

# パラメータを指定
maou build-game-graph \
  --input-path ./data/preprocess/ \
  --output-dir ./data/game-graph/ \
  --max-depth 20 \
  --min-probability 0.005

# 特定の局面からグラフを構築(hash + SFEN の両方を指定)
maou build-game-graph \
  --input-path ./data/preprocess/ \
  --output-dir ./data/game-graph/ \
  --initial-hash 12345678901234567 \
  --initial-sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2"
```

## Output format

### `nodes.feather`

| Column | Type | Description |
| --- | --- | --- |
| `position_hash` | UInt64 | Zobrist hash(局面の一意識別子) |
| `result_value` | Float32 | 局面の勝率(手番側視点，0.0〜1.0) |
| `best_move_win_rate` | Float32 | 最善手の勝率 |
| `num_branches` | UInt16 | 分岐数．`is_depth_cutoff=false` のノードでは実際にエッジが生成された指し手数．`is_depth_cutoff=true` のノードでは `min_probability` 以上の候補手数(未展開) |
| `depth` | UInt16 | 初期局面からの最短距離 |
| `is_depth_cutoff` | Boolean | `max_depth` に到達して展開を打ち切ったノードの場合 `true` |

### `edges.feather`

| Column | Type | Description |
| --- | --- | --- |
| `parent_hash` | UInt64 | 親局面のZobrist hash |
| `child_hash` | UInt64 | 子局面のZobrist hash |
| `move16` | UInt16 | cshogi move16形式の指し手 |
| `move_label` | UInt16 | moveLabelのインデックス(0〜1495) |
| `probability` | Float32 | moveLabel値(親局面からの相対出現確率) |
| `win_rate` | Float32 | moveWinRate値(この手の勝率) |
| `is_leaf` | Boolean | child_hashがpreprocessデータに存在しない場合True |

## Implementation references

| Component | File |
| --- | --- |
| CLI command | [`src/maou/infra/console/build_game_graph.py`](../../src/maou/infra/console/build_game_graph.py) |
| BFS builder | [`src/maou/app/game_graph/builder.py`](../../src/maou/app/game_graph/builder.py) |
| Data I/O | [`src/maou/interface/game_graph_io.py`](../../src/maou/interface/game_graph_io.py) |
| Data models | [`src/maou/domain/game_graph/model.py`](../../src/maou/domain/game_graph/model.py) |
| Polars schemas | [`src/maou/domain/game_graph/schema.py`](../../src/maou/domain/game_graph/schema.py) |
| Lazy loading | [`src/maou/interface/lazy_list_columns.py`](../../src/maou/interface/lazy_list_columns.py) |
| Process info | [`src/maou/app/process_info.py`](../../src/maou/app/process_info.py) |
