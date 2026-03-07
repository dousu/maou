# エピック設計: 棋譜ツリー機能 (Game Tree Feature)

## 概要

preprocessデータ(局面単位・全ゲーム横断で集約済み)から統計的な棋譜ツリーを構築し，
インタラクティブに可視化する機能を追加する．

### 目的

- 大量の棋譜データから「どの局面でどの手がどれだけ指されたか」をツリー構造で俯瞰する
- 各分岐の出現頻度・勝率を視覚的に把握し，定跡研究や学習データの傾向分析に活用する

### 入力データ

preprocessed `.feather` ファイル(Arrow IPC，LZ4圧縮)．各行は一意の局面を表す:

| フィールド | 型 | 説明 |
|-----------|------|------|
| `id` | UInt64 | Zobrist hash(局面の一意識別子) |
| `boardIdPositions` | List(List(UInt8)) | 9×9盤面(Fortran order) |
| `piecesInHand` | List(UInt8) | 持ち駒(14要素: 先手7 + 後手7) |
| `moveLabel` | List(Float32) | 指し手確率分布(1,496要素) |
| `moveWinRate` | List(Float32) | 指し手ごとの勝率(1,496要素) |
| `bestMoveWinRate` | Float32 | 最善手の勝率 |
| `resultValue` | Float32 | 局面の勝率(0.0〜1.0，手番側視点) |

### 出力データ(ツリー)

**nodes.feather:**

| Column | Type | Description |
|--------|------|-------------|
| `position_hash` | UInt64 | Zobrist hash |
| `visit_count` | UInt32 | この局面の出現回数(moveLabelの合計から算出) |
| `result_value` | Float32 | 局面の勝率(手番側視点) |
| `depth` | UInt16 | 初期局面からの最短距離 |

**edges.feather:**

| Column | Type | Description |
|--------|------|-------------|
| `parent_hash` | UInt64 | 親局面のZobrist hash |
| `child_hash` | UInt64 | 子局面のZobrist hash |
| `move16` | UInt16 | cshogi move16形式の指し手 |
| `move_label` | UInt16 | moveLabelのインデックス(0〜1495) |
| `probability` | Float32 | moveLabel値(出現確率) |
| `win_rate` | Float32 | moveWinRate値(この手の勝率) |

### 処理パイプライン

```
preprocess .feather (局面単位・集約済み)
    │
    ▼
[1] 全局面を id(zobrist hash) → row のルックアップテーブルに読み込み
    │
    ▼
[2] 初期局面(平手)の zobrist hash からBFS開始
    │
    ▼
[3] 各局面の moveLabel から非ゼロの指し手を取得
    │  ├── move_label index → cshogi の指し手に変換
    │  ├── cshogi.Board で指し手を適用 → 子局面の zobrist hash を取得
    │  └── 子局面がルックアップテーブルにあれば → エッジ追加・子をキューへ
    │
    ▼
[4] max_depth or min_count で打ち切り
    │
    ▼
[5] nodes.feather + edges.feather 出力
```

---

## アーキテクチャ概要

Clean Architecture の依存フロー(`infra → interface → app → domain`)に従う．

```
infra/console/build_game_tree.py          ← CLI コマンド定義
infra/visualization/game_tree_server.py   ← Gradio UI (Cytoscape.js)
infra/visualization/static/game_tree.js   ← フロントエンド (Cytoscape.js)
infra/visualization/static/game_tree.css  ← ツリー表示用スタイル
    │
    ▼
interface/game_tree_io.py                 ← ツリーデータの読み書き
interface/game_tree_visualization.py      ← UI ↔ App 層アダプタ
    │
    ▼
app/game_tree/builder.py                  ← ツリー構築ロジック (BFS)
app/game_tree/query.py                    ← ツリー検索・フィルタリング
    │
    ▼
domain/game_tree/model.py                 ← GameTreeNode, GameTreeEdge
domain/game_tree/schema.py                ← Polars スキーマ定義
```

---

## Epic 1: ツリーデータ構築基盤

### 目標

preprocessデータからゲームツリーを構築し，`.feather` ファイルとして保存する
`maou build-game-tree` CLIコマンドを実装する．

### Story 1.1: Domain — ツリーデータモデル定義

**ファイル:** `src/maou/domain/game_tree/model.py`

ツリーのノードとエッジを表すデータクラスを定義する．

```python
@dataclass(frozen=True)
class GameTreeNode:
    """ゲームツリーのノード(一意の局面)."""

    position_hash: int    # Zobrist hash
    visit_count: int      # 出現回数
    result_value: float   # 局面勝率(手番側視点)
    depth: int            # 初期局面からの最短距離

@dataclass(frozen=True)
class GameTreeEdge:
    """ゲームツリーのエッジ(局面間の遷移)."""

    parent_hash: int      # 親局面 Zobrist hash
    child_hash: int       # 子局面 Zobrist hash
    move16: int           # cshogi move16
    move_label: int       # moveLabel index (0-1495)
    probability: float    # 出現確率
    win_rate: float       # この手の勝率
```

**ファイル:** `src/maou/domain/game_tree/schema.py`

Polarsスキーマ定義(既存の`domain/data/schema.py`パターンに準拠)．

**完了条件:**
- データクラスに型ヒントとdocstring
- Polarsスキーマが`nodes.feather`/`edges.feather`のカラムと一致
- ユニットテスト

### Story 1.2: App — ツリー構築ロジック (BFS)

**ファイル:** `src/maou/app/game_tree/builder.py`

preprocessデータのルックアップテーブルを構築し，初期局面からBFSでツリーを展開する．

```python
class GameTreeBuilder:
    """preprocessデータからゲームツリーを構築する．"""

    def build(
        self,
        preprocess_df: pl.DataFrame,
        max_depth: int = 30,
        min_probability: float = 0.01,
    ) -> tuple[list[GameTreeNode], list[GameTreeEdge]]:
        """BFSでツリーを構築する．"""
```

**処理の流れ:**
1. `preprocess_df`の`id`カラム(Zobrist hash)をキーにした辞書を構築
2. cshogiの`Board()`で初期局面を作成し，`board.zobrist_hash()`で初期ハッシュを取得
3. BFSキューに初期局面を追加
4. 各局面について:
   - `moveLabel`から`min_probability`以上の指し手インデックスを取得
   - `make_usi_move_from_label()`で各インデックスをcshogiの指し手に変換
   - `board.push(move)` → `board.zobrist_hash()` で子局面のハッシュを取得 → `board.pop()`
   - 子局面がルックアップテーブルに存在すればエッジを追加，未訪問ならキューに追加
5. `max_depth`に達したら展開を停止

**重要な設計判断:**
- `moveLabel`のインデックスからcshogiの指し手への変換には`domain/move/label.py`の`make_usi_move_from_label()`を使用
- 同一局面への複数経路(transposition)は最短距離の`depth`を採用
- メモリ効率のため，ルックアップテーブルは必要最小限のカラムのみ保持

**完了条件:**
- BFS構築が正しく動作する(初期局面→depth 1→depth 2...と展開)
- `min_probability`フィルタリングが機能する
- transposition(合流)を正しく処理する
- プログレスバー用のコールバック対応
- ユニットテスト(モック局面データで検証)

### Story 1.3: Interface — ツリーデータI/O

**ファイル:** `src/maou/interface/game_tree_io.py`

ツリーデータの`.feather`ファイルへの読み書きを担当する．

```python
class GameTreeIO:
    """ゲームツリーデータのI/O."""

    def save(
        self,
        nodes: list[GameTreeNode],
        edges: list[GameTreeEdge],
        output_dir: Path,
    ) -> None:
        """nodes.feather, edges.feather を出力する．"""

    def load(self, tree_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
        """nodes.feather, edges.feather を読み込む．"""
```

**完了条件:**
- Arrow IPC (LZ4圧縮)で正しく保存・読み込みできる
- スキーマバリデーション
- ユニットテスト

### Story 1.4: Infra — CLIコマンド `maou build-game-tree`

**ファイル:** `src/maou/infra/console/build_game_tree.py`

```
maou build-game-tree \
  --input-path ./data/preprocess/ \
  --output-dir ./data/game-tree/ \
  --max-depth 30 \
  --min-probability 0.01
```

| オプション | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--input-path` | PATH | (必須) | preprocessデータのディレクトリまたはファイル |
| `--output-dir` | PATH | (必須) | ツリーデータの出力先 |
| `--max-depth` | INT | 30 | 最大探索深さ |
| `--min-probability` | FLOAT | 0.01 | 指し手の最小確率閾値 |

**既存パターンに準拠:**
- `app.py`の`LAZY_COMMANDS`に登録
- Click デコレータでオプション定義
- `FileSystem.collect_files()`で`.feather`ファイル収集

**完了条件:**
- CLIが正しく動作する
- プログレスバー表示(処理済み局面数/全局面数)
- エラーハンドリング(入力パスが存在しない，初期局面が見つからないなど)
- `docs/commands/build-game-tree.md` を作成
- `pyproject.toml` のバージョンバンプ

### Story 1.5: テストとドキュメント

**テスト:**
- `tests/maou/domain/game_tree/test_model.py`
- `tests/maou/app/game_tree/test_builder.py`
- `tests/maou/interface/test_game_tree_io.py`
- `tests/maou/infra/console/test_build_game_tree.py`(CLIスモークテスト)

**ドキュメント:**
- `docs/commands/build-game-tree.md`(CLIリファレンス)

---

## Epic 2: ゲームツリー可視化 UI

### 目標

構築済みのツリーデータをGradio上でインタラクティブに表示する．
Cytoscape.jsによるツリーグラフと将棋盤の連動表示を実現する．

### UI全体設計

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚡ Maou 棋譜ツリービューア                                       │
│  [📂 データ読込] [Depth: ▼3] [Min Count: ▼10] [🔄 更新]          │
└─────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────┬──────────────────────────────┐
│   TREE VIEW (scale=3)           │   DETAIL PANEL (scale=2)     │
│                                  │                              │
│   ┌─────────┐                   │   ┌────────────────────────┐ │
│   │ 初期局面 │                   │   │   🎴 盤面表示          │ │
│   │ N=85000 │                   │   │   [SVG 450×540px]      │ │
│   └────┬────┘                   │   │                        │ │
│    ┌───┴───┐                    │   └────────────────────────┘ │
│   ┌┴┐    ┌┴┐                   │                              │
│   │76│   │26│                   │   ┌────────────────────────┐ │
│   │歩│   │歩│                   │   │ 📊 局面統計            │ │
│   └┬─┘   └┬─┘                  │   │ 出現回数: 85,000       │ │
│  ┌─┴─┐  ┌─┴─┐                  │   │ 勝率: 52.3%           │ │
│  │34歩│ │84歩│                  │   │ 深さ: 0               │ │
│  └───┘  └───┘                   │   └────────────────────────┘ │
│                                  │                              │
│  ノードの色 = 勝率              │   ┌────────────────────────┐ │
│  🔵 先手有利 ← → 後手有利 🔴   │   │ 📋 指し手一覧          │ │
│                                  │   │ ▲7六歩  45.2%  52.1% │ │
│  ノードのサイズ = 出現回数      │   │ ▲2六歩  32.1%  51.8% │ │
│                                  │   │ ▲5六歩   8.3%  50.2% │ │
│                                  │   │ ...                    │ │
│                                  │   └────────────────────────┘ │
│                                  │                              │
│                                  │   ┌────────────────────────┐ │
│                                  │   │ 📈 分岐分析            │ │
│                                  │   │ [確率分布グラフ]       │ │
│                                  │   └────────────────────────┘ │
└──────────────────────────────────┴──────────────────────────────┘
```

### Story 2.1: Domain — ツリー検索・フィルタリング

**ファイル:** `src/maou/app/game_tree/query.py`

ツリーデータに対する検索・フィルタリング機能を提供する．

```python
class GameTreeQuery:
    """ツリーデータの検索・フィルタリング．"""

    def __init__(
        self, nodes_df: pl.DataFrame, edges_df: pl.DataFrame
    ) -> None: ...

    def get_subtree(
        self,
        root_hash: int,
        max_depth: int = 3,
        min_count: int = 10,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """指定ノードを起点としたサブツリーを取得する．

        UIに送信するデータ量を制限するため，
        表示深さとカウント閾値でフィルタリングする．
        """

    def get_node_detail(self, position_hash: int) -> dict[str, Any]:
        """ノードの詳細情報を取得する．"""

    def get_children(self, position_hash: int) -> pl.DataFrame:
        """指定ノードの子エッジ一覧を取得する．"""

    def get_path_to_root(self, position_hash: int) -> list[int]:
        """指定ノードから根までのパスを取得する．"""
```

**完了条件:**
- サブツリー取得が`max_depth`と`min_count`で正しくフィルタリングされる
- 子ノード一覧が確率降順でソートされる
- ルートまでのパス取得が動作する
- ユニットテスト

### Story 2.2: Interface — 可視化アダプタ

**ファイル:** `src/maou/interface/game_tree_visualization.py`

App層の`GameTreeQuery`とInfra層のGradio UIを接続するアダプタ．

```python
class GameTreeVisualizationInterface:
    """ゲームツリー可視化のインターフェース層．"""

    def get_cytoscape_elements(
        self,
        root_hash: int,
        display_depth: int,
        min_count: int,
    ) -> dict:
        """Cytoscape.js用のノード・エッジデータを生成する．

        Returns:
            {"nodes": [...], "edges": [...]} 形式のCytoscape elements
        """

    def get_board_svg(self, position_hash: int) -> str:
        """指定局面のSVGを生成する．"""

    def get_move_table(
        self, position_hash: int
    ) -> list[list[str]]:
        """指定局面の指し手一覧テーブルを生成する．

        Returns:
            [["▲7六歩", "45.2%", "52.1%", "38,400"], ...]
        """
```

**Cytoscapeノードデータの形式:**
```python
{
    "data": {
        "id": str(position_hash),
        "label": "▲7六歩",           # 親からの指し手(USI → 日本語表記)
        "visit_count": 85000,
        "result_value": 0.523,        # 勝率
        "depth": 1,
        "probability": 0.452,         # 親からの出現確率
    }
}
```

**Cytoscapeエッジデータの形式:**
```python
{
    "data": {
        "source": str(parent_hash),
        "target": str(child_hash),
        "label": "45.2%",            # 確率表示
        "probability": 0.452,
    }
}
```

**完了条件:**
- Cytoscape elements形式が正しく生成される
- 盤面SVGが既存の`SVGBoardRenderer`を使って生成される
- 指し手一覧テーブルがUSI→日本語表記に変換される
- ユニットテスト

### Story 2.3: Infra — Cytoscape.jsフロントエンド

**ファイル:** `src/maou/infra/visualization/static/game_tree.js`
**ファイル:** `src/maou/infra/visualization/static/game_tree.css`

Cytoscape.jsを使ったインタラクティブなツリー表示コンポーネントを実装する．

**技術選定: Cytoscape.js**

Gradioの`gr.HTML`内に埋め込み，CDNから読み込む．

| 選択肢 | 評価 | 理由 |
|--------|------|------|
| **Cytoscape.js** | ✅ 採用 | 宣言的スタイル，dagre自動レイアウト，クリックイベント対応 |
| D3.js | △ | 低レベルすぎる．同等の実装コストが高い |
| Plotly (gr.Plot) | ✗ | クリックイベントコールバック非対応 |

**機能:**

1. **ツリーレイアウト:**
   - `cytoscape-dagre`プラグインによるトップダウン階層レイアウト
   - ノード間の適切な間隔調整

2. **ノードの視覚エンコーディング:**
   - **サイズ:** `visit_count`に比例(対数スケール)
   - **色:** `result_value`(勝率)のグラデーション
     - 先手有利(> 0.55): 青系 `#2196F3`
     - 互角(0.45〜0.55): グレー `#9E9E9E`
     - 後手有利(< 0.45): 赤系 `#F44336`
   - **ラベル:** 親からの指し手(日本語表記)

3. **エッジの視覚エンコーディング:**
   - **太さ:** `probability`に比例
   - **ラベル:** 確率表示(例: "45.2%")
   - **透明度:** 確率が低いほど薄く

4. **インタラクション:**
   - **ノードクリック:** 選択ノードをハイライト，詳細パネルを更新
   - **ノードダブルクリック:** 選択ノードを新しいルートとしてサブツリーを展開
   - **ズーム/パン:** マウスホイール・ドラッグ
   - **ツールチップ:** ホバーで統計サマリー表示

5. **Python連携:**
   - ノードクリック時にhidden `gr.Textbox`を更新 → Python callbackをトリガー
   - Python側で盤面SVG・統計情報を生成 → UIを更新

```
gr.HTML (Cytoscape.js)   ──click──→  gr.Textbox (hidden)
                                          │
                                     Python callback
                                          │
                         ┌────────────────┤
                         ▼                ▼
               gr.HTML (盤面SVG)    gr.JSON (統計)
```

**完了条件:**
- Cytoscape.jsが正しくレンダリングされる
- dagre自動レイアウトが階層的に表示される
- ノードの色・サイズが統計値を反映する
- クリックイベントがPython callbackに伝達される
- ズーム・パンが動作する
- 既存テーマ(`theme.css`)との整合性

### Story 2.4: Infra — Gradioサーバー統合

**ファイル:** `src/maou/infra/visualization/game_tree_server.py`

ゲームツリー専用のGradioサーバーを実装する．
既存の`GradioVisualizationServer`とは**独立した新しいサーバー**として実装する
(既存サーバーの複雑さを考慮し，結合を避ける)．

**レイアウト構成:**

```python
with gr.Blocks(title="Maou Game Tree Viewer") as demo:
    # ヘッダー
    gr.Markdown("# ⚡ Maou 棋譜ツリービューア")

    # コントロールバー
    with gr.Row():
        tree_path_input = gr.Textbox(label="ツリーデータパス")
        load_btn = gr.Button("📂 読込", variant="primary")
        depth_slider = gr.Slider(1, 10, value=3, step=1, label="表示深さ")
        min_count_slider = gr.Slider(1, 1000, value=10, label="最小出現回数")
        refresh_btn = gr.Button("🔄 更新")

    # メインコンテンツ
    with gr.Row():
        # 左: ツリービュー
        with gr.Column(scale=3):
            tree_html = gr.HTML(label="ツリー表示")

        # 右: 詳細パネル
        with gr.Column(scale=2):
            board_html = gr.HTML(label="盤面")
            stats_json = gr.JSON(label="局面統計")
            move_table = gr.Dataframe(
                headers=["指し手", "確率", "勝率", "出現回数"],
                label="指し手一覧",
            )
            analytics_plot = gr.Plot(label="分岐分析")

    # Hidden state
    selected_node = gr.Textbox(visible=False)
    tree_data_state = gr.State()
```

**イベントフロー:**

| トリガー | 処理 | 更新対象 |
|----------|------|----------|
| 読込ボタン | ツリーデータ読込 + 初期表示 | tree_html, board_html, stats_json, move_table |
| depth/min_count変更 | サブツリー再計算 | tree_html |
| ノードクリック(JS→hidden textbox) | 盤面・統計・指し手の更新 | board_html, stats_json, move_table, analytics_plot |
| ノードダブルクリック(JS→hidden textbox) | 新ルートでサブツリー再展開 | tree_html, board_html, stats_json, move_table |

**完了条件:**
- Gradioサーバーが正しく起動する
- ツリーデータの読み込みと表示が動作する
- ノードクリック→盤面更新の連動が動作する
- depth/min_countスライダーでツリーが再描画される

### Story 2.5: 詳細パネル — 盤面表示と指し手統計

**右側パネルの詳細設計:**

#### 盤面表示
- 既存の`SVGBoardRenderer`を再利用
- 選択ノードの局面を表示
- 親からの指し手を矢印(`MoveArrow`)で表示

#### 局面統計
```json
{
    "局面ハッシュ": "0x1234567890ABCDEF",
    "出現回数": "85,000",
    "勝率": "52.3%",
    "深さ": 0,
    "分岐数": 30,
    "主要手数": 3
}
```

#### 指し手一覧テーブル
| 指し手 | 確率 | 勝率 | 出現回数 |
|--------|------|------|----------|
| ▲7六歩 | 45.2% | 52.1% | 38,400 |
| ▲2六歩 | 32.1% | 51.8% | 27,300 |
| ▲5六歩 | 8.3% | 50.2% | 7,100 |
| ... | ... | ... | ... |

- 確率降順でソート
- クリックでその指し手の子ノードを選択(ツリービューと連動)

#### 分岐分析チャート
- Plotly棒グラフ: 上位10手の確率分布
- x軸: 指し手，y軸: 確率，色: 勝率

**完了条件:**
- 盤面SVGが正しく表示される
- 指し手一覧テーブルが確率降順でソートされる
- 分岐分析チャートが描画される
- テーブル行クリック→ツリーノード選択の連動

### Story 2.6: CLIコマンド `maou visualize-game-tree`

**ファイル:** `src/maou/infra/console/visualize_game_tree.py`

```
maou visualize-game-tree \
  --tree-path ./data/game-tree/ \
  --port 7861 \
  --share
```

| オプション | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--tree-path` | PATH | (必須) | ツリーデータディレクトリ(nodes.feather + edges.feather) |
| `--port` | INT | 7861 | サーバーポート |
| `--server-name` | TEXT | "127.0.0.1" | サーバーバインドアドレス |
| `--share` | FLAG | False | Gradio公開リンク生成 |

**LAZY_COMMANDSへの登録:**
- `required_packages`に`gradio`，`matplotlib`，`playwright`を指定
- 既存の`visualize`コマンドとは独立

**完了条件:**
- CLIが正しく動作する
- ポート自動選択(衝突時の7861-7959のフォールバック)
- `docs/commands/visualize-game-tree.md` を作成

### Story 2.7: テストとドキュメント

**テスト:**
- `tests/maou/app/game_tree/test_query.py`
- `tests/maou/interface/test_game_tree_visualization.py`
- `tests/maou/infra/visualization/test_game_tree_server.py`(起動スモークテスト)

**ドキュメント:**
- `docs/commands/visualize-game-tree.md`(CLIリファレンス)

---

## Epic 3: UI拡張機能

### 目標

ツリービューアのUX品質を向上させる高度な機能群．
Epic 2完了後に着手する．

### Story 3.1: パンくずリストナビゲーション

選択ノードの「初期局面からのパス」をパンくずリストとして表示する．

```
初期局面 > ▲7六歩 > △3四歩 > ▲2六歩
```

- 各パンくず要素をクリックでそのノードに移動
- Gradio `gr.HTML` で実装(クリック→hidden textbox→Python callback)

### Story 3.2: ツリー比較モード

2つの異なるツリーデータ(例: 異なる棋力帯のデータ)を並べて比較する．

```
┌──────────────────┬──────────────────┐
│  Tree A (高段)    │  Tree B (低段)    │
│  ▲7六歩 45.2%    │  ▲7六歩 38.1%    │
│  ▲2六歩 32.1%    │  ▲2六歩 41.5%    │
└──────────────────┴──────────────────┘
```

- 同一局面の統計値の差分をハイライト
- 確率差が大きい分岐を強調表示

### Story 3.3: 定跡データベース連携

既知の定跡名をツリーノードにアノテーションする．

```
初期局面 > ▲7六歩 > △3四歩 > ▲6六歩  ← [四間飛車]
```

- 定跡名のマッピングファイル(JSON/YAML)を読み込み
- ツリーノードに定跡名バッジを表示
- 将来拡張: 定跡のカテゴリ別フィルタリング

### Story 3.4: キーボードショートカット

既存の`visualize`コマンドのパターンに準拠:

| キー | アクション |
|------|------------|
| `↑` / `K` | 親ノードに移動 |
| `↓` / `J` | 最頻子ノードに移動 |
| `←` / `H` | 前の兄弟ノードに移動 |
| `→` / `L` | 次の兄弟ノードに移動 |
| `+` / `-` | 表示深さの増減 |
| `R` | ルートノードに戻る |
| `?` | ヘルプモーダル |

### Story 3.5: エクスポート機能

ツリーの特定サブツリーを各種形式でエクスポートする．

| 形式 | 用途 |
|------|------|
| PNG/SVG | ツリー画像(レポート・論文用) |
| CSV | 統計データ(スプレッドシート分析) |
| SFEN列挙 | 選択ノードまでのSFEN手順(エンジン検証用) |

---

## 実装順序と依存関係

```
Epic 1: ツリーデータ構築基盤
  Story 1.1 ──→ Story 1.2 ──→ Story 1.3 ──→ Story 1.4 ──→ Story 1.5
  (domain)      (app)         (interface)   (infra/CLI)   (test/docs)

        ↓ (ツリーデータが生成可能になった時点で着手可能)

Epic 2: ゲームツリー可視化 UI
  Story 2.1 ──→ Story 2.2 ──→ Story 2.3 ──┐
  (app/query)   (interface)   (JS frontend) ├→ Story 2.4 ──→ Story 2.5 ──→ Story 2.6 ──→ Story 2.7
                                             │  (Gradio)      (detail)     (CLI)         (test/docs)
                                             │
                              Story 2.3は2.1・2.2と並行開発可能

        ↓ (Epic 2完了後)

Epic 3: UI拡張機能
  Story 3.1〜3.5 は独立して着手可能
```

## バージョニング

| エピック完了時 | バージョン | 変更種別 |
|---------------|-----------|----------|
| Epic 1 | 0.11.0 | feat: 新コマンド追加(minor) |
| Epic 2 | 0.12.0 | feat: 新コマンド追加(minor) |
| Epic 3 (各Story) | 0.12.x | feat: 機能追加(minor or patch) |

現在のバージョン: **0.10.18**
