# `maou visualize-game-tree`

## Overview

- 構築済みのゲームツリーデータ(`nodes.feather` + `edges.feather`)を
  Cytoscape.jsベースのインタラクティブUIで可視化する．
- ツリー表示(ノードの色=勝率，サイズ=確率)と詳細パネル(盤面表示，指し手一覧，分岐分析)を提供する．
- ノードクリックで詳細表示，ダブルクリックでサブツリー展開が可能．

## CLI options

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--tree-path PATH` | Yes | — | ツリーデータディレクトリ(`nodes.feather` + `edges.feather`)のパス． |
| `--port INT` | No | Gradio自動選択 | サーバーポート．未指定時はGradioの自動選択に委任する． |
| `--server-name TEXT` | No | `127.0.0.1` | サーバーバインドアドレス． |
| `--share` | No | `False` | Gradio公開リンクを生成する． |

## Example invocation

```bash
# 基本的な使用方法
maou visualize-game-tree --tree-path ./data/game-tree/

# ポートを指定
maou visualize-game-tree --tree-path ./data/game-tree/ --port 7861

# 公開リンクを生成
maou visualize-game-tree --tree-path ./data/game-tree/ --share

# 全ネットワークからアクセス可能にする
maou visualize-game-tree --tree-path ./data/game-tree/ --server-name 0.0.0.0
```

## UI機能

### ツリービュー(左パネル)

- **Cytoscape.js** + **dagre** レイアウトによる階層的ツリー表示
- ノードの**色**: 勝率(`result_value`)に応じたグラデーション
  - 青(先手有利，>55%) / グレー(互角) / 赤(後手有利，<45%)
- ノードの**サイズ**: 親エッジの確率(`probability`)に比例
- エッジの**太さ**: 確率に比例
- **シングルクリック**: 詳細パネルを更新
- **ダブルクリック**: 選択ノードを新しいルートとしてサブツリーを展開

### 詳細パネル(右パネル)

- **盤面表示**: SVGによる将棋盤の描画(親からの指し手を矢印表示)
- **局面統計**: 勝率，最善手勝率，深さ，分岐数
- **指し手一覧**: 確率降順の指し手テーブル(指し手，確率，勝率)
- **分岐分析**: 上位10手の確率分布バーチャート

### コントロール

- **表示深さ**: サブツリーの表示階層数(1-10，デフォルト3)
- **最小確率**: 表示するエッジの最小確率閾値(0.001-0.3，デフォルト0.01)
- **ルートに戻る**: ツリーの根に戻る

## Implementation

| Item | Path |
| --- | --- |
| CLI command | `src/maou/infra/console/visualize_game_tree.py` |
| Gradio server | `src/maou/infra/visualization/game_tree_server.py` |
| Interface adapter | `src/maou/interface/game_tree_visualization.py` |
| Query logic | `src/maou/app/game_tree/query.py` |
| Frontend JS | `src/maou/infra/visualization/static/game_tree.js` |
| Frontend CSS | `src/maou/infra/visualization/static/game_tree.css` |
