# Maou将棋データ可視化ツール - API仕様

このドキュメントでは，可視化ツールの各レイヤーにおけるAPI仕様を説明します．

## 目次

1. [CLI API](#cli-api)
2. [Python API](#python-api)
   - [Domain Layer](#domain-layer)
   - [App Layer](#app-layer)
   - [Interface Layer](#interface-layer)
   - [Infra Layer](#infra-layer)
3. [Rust API](#rust-api)

---

## CLI API

### `maou visualize`

将棋データ可視化サーバーを起動します．

#### シグネチャ

```bash
poetry run maou visualize [OPTIONS]
```

#### オプション

| オプション | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|-----|----------|------|
| `--input-path` | Path | ❌ | - | 入力ファイルまたはディレクトリパス（複数指定可） |
| `--array-type` | Choice | ✅ | - | データ型（hcpe, preprocessing, stage1, stage2） |
| `--port` | Int | ❌ | 7860 | Gradioサーバーポート |
| `--share` | Flag | ❌ | False | Gradio公開リンク作成 |
| `--server-name` | String | ❌ | 127.0.0.1 | サーバーバインドアドレス |
| `--model-path` | Path | ❌ | - | ONNXモデルパス（評価表示用） |
| `--debug-mode` | Flag | ❌ | False | 詳細ログ有効化 |

#### 使用例

```bash
# ディレクトリからHCPEデータを可視化
poetry run maou visualize --input-path ./data/hcpe --array-type hcpe

# 特定ファイルでpreprocessingデータを可視化
poetry run maou visualize --input-path data1.feather --input-path data2.feather --array-type preprocessing

# カスタムポートと公開リンク
poetry run maou visualize --input-path ./data --array-type hcpe --port 8080 --share
```

---

## Python API

### Domain Layer

#### `SVGBoardRenderer`

将棋盤のSVG描画を行うレンダラークラス．

**ファイル**: `src/maou/domain/visualization/board_renderer.py`

##### `render()`

```python
def render(
    self,
    position: BoardPosition,
    highlight_squares: Optional[List[int]] = None,
) -> str
```

**引数**:
- `position`: `BoardPosition` - 描画する盤面状態
- `highlight_squares`: `Optional[List[int]]` - ハイライトするマス（0-80のインデックス）

**戻り値**: `str` - 完全なSVG文字列（HTML埋め込み可能）

**使用例**:

```python
from maou.domain.visualization.board_renderer import (
    BoardPosition,
    SVGBoardRenderer,
)

# ボード状態を作成
board = [[0 for _ in range(9)] for _ in range(9)]
board[0][4] = 16 + 8  # 後手王
board[8][4] = 8  # 先手王

hand = [0] * 14
position = BoardPosition(
    board_id_positions=board,
    pieces_in_hand=hand,
)

# SVG描画
renderer = SVGBoardRenderer()
svg = renderer.render(position, highlight_squares=[40])  # 中央をハイライト
```

#### `BoardPosition`

不変な将棋盤の状態表現．

```python
@dataclass(frozen=True)
class BoardPosition:
    board_id_positions: List[List[int]]  # 9×9の駒配置
    pieces_in_hand: List[int]            # 14要素の持ち駒
```

**バリデーション**:
- `board_id_positions`: 9行×9列の2次元リスト
- `pieces_in_hand`: 14要素のリスト（先手7種 + 後手7種）

---

### App Layer

#### `DataRetriever`

データ検索オーケストレータ．

**ファイル**: `src/maou/app/visualization/data_retrieval.py`

##### `get_by_id()`

```python
def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]
```

**引数**:
- `record_id`: `str` - 検索するレコードID

**戻り値**: `Optional[Dict[str, Any]]` - レコードデータ，見つからない場合は`None`

**返却データ構造**:

```python
{
    "id": str,
    "eval": int,
    "moves": int,
    "boardIdPositions": List[List[int]],  # 9×9
    "piecesInHand": List[int],           # 14要素
}
```

##### `get_by_eval_range()`

```python
def get_by_eval_range(
    self,
    min_eval: Optional[int],
    max_eval: Optional[int],
    offset: int,
    limit: int,
) -> List[Dict[str, Any]]
```

**引数**:
- `min_eval`: `Optional[int]` - 最小評価値（Noneで-∞）
- `max_eval`: `Optional[int]` - 最大評価値（Noneで+∞）
- `offset`: `int` - スキップする件数
- `limit`: `int` - 取得する最大件数

**戻り値**: `List[Dict[str, Any]]` - レコードデータのリスト

**使用例**:

```python
from pathlib import Path
from maou.app.visualization.data_retrieval import DataRetriever
from maou.infra.visualization.search_index import SearchIndex

# SearchIndexを構築
search_index = SearchIndex.build(
    file_paths=[Path("data.feather")],
    array_type="hcpe",
    num_mock_records=1000,
)

# DataRetrieverを初期化
retriever = DataRetriever(
    search_index=search_index,
    file_paths=[Path("data.feather")],
    array_type="hcpe",
)

# ID検索
record = retriever.get_by_id("mock_id_42")
if record:
    print(f"Found: {record['id']}, eval={record['eval']}")

# 評価値範囲検索
records = retriever.get_by_eval_range(
    min_eval=-100,
    max_eval=100,
    offset=0,
    limit=20,
)
print(f"Found {len(records)} records")
```

#### `BoardDisplayService`

ボード表示サービス．

**ファイル**: `src/maou/app/visualization/board_display.py`

##### `render_from_record()`

```python
def render_from_record(
    self,
    record: Dict[str, Any],
    highlight_squares: Optional[List[int]] = None,
) -> str
```

**引数**:
- `record`: `Dict[str, Any]` - レコードデータ
- `highlight_squares`: `Optional[List[int]]` - ハイライトするマス

**戻り値**: `str` - SVG文字列

**使用例**:

```python
from maou.app.visualization.board_display import BoardDisplayService
from maou.domain.visualization.board_renderer import SVGBoardRenderer

# サービス初期化
service = BoardDisplayService(renderer=SVGBoardRenderer())

# レコードからボード描画
record = {
    "id": "test_id",
    "boardIdPositions": [[0]*9 for _ in range(9)],
    "piecesInHand": [0]*14,
}

svg = service.render_from_record(record)
```

---

### Interface Layer

#### `VisualizationInterface`

可視化インターフェースアダプター．

**ファイル**: `src/maou/interface/visualization.py`

##### `search_by_id()`

```python
def search_by_id(
    self, record_id: str
) -> Tuple[str, Dict[str, Any]]
```

**引数**:
- `record_id`: `str` - 検索するレコードID

**戻り値**: `Tuple[str, Dict[str, Any]]`
- `[0]`: ボードSVG文字列
- `[1]`: レコード詳細辞書

**使用例**:

```python
from pathlib import Path
from maou.interface.visualization import VisualizationInterface
from maou.infra.visualization.search_index import SearchIndex

# SearchIndex構築
search_index = SearchIndex.build(
    file_paths=[Path("data.feather")],
    array_type="hcpe",
)

# Interface初期化
viz_interface = VisualizationInterface(
    search_index=search_index,
    file_paths=[Path("data.feather")],
    array_type="hcpe",
)

# ID検索
board_svg, record_details = viz_interface.search_by_id("mock_id_42")
print(f"Record: {record_details}")
# board_svgをGradioで表示
```

##### `search_by_eval_range()`

```python
def search_by_eval_range(
    self,
    min_eval: int,
    max_eval: int,
    page: int,
    page_size: int,
) -> Tuple[List[List[Any]], str, str, Dict[str, Any]]
```

**引数**:
- `min_eval`: `int` - 最小評価値
- `max_eval`: `int` - 最大評価値
- `page`: `int` - ページ番号（1始まり）
- `page_size`: `int` - ページサイズ

**戻り値**: `Tuple[List[List[Any]], str, str, Dict[str, Any]]`
- `[0]`: テーブルデータ（2次元リスト）
- `[1]`: ページ情報文字列
- `[2]`: 最初のレコードのボードSVG
- `[3]`: 最初のレコードの詳細

**テーブルデータ構造**:

```python
[
    [index, id, eval, moves],  # 各行
    [1, "mock_id_0", 100, 50],
    [2, "mock_id_10", 150, 55],
    ...
]
```

##### `get_dataset_stats()`

```python
def get_dataset_stats(self) -> Dict[str, Any]
```

**戻り値**: `Dict[str, Any]` - データセット統計情報

```python
{
    "total_records": int,
    "array_type": str,
    "num_files": int,
}
```

---

### Infra Layer

#### `SearchIndex`

検索インデックスクラス（Pythonラッパー）．

**ファイル**: `src/maou/infra/visualization/search_index.py`

##### `build()` (クラスメソッド)

```python
@classmethod
def build(
    cls,
    file_paths: List[Path],
    array_type: str,
    num_mock_records: int = 1000,
) -> "SearchIndex"
```

**引数**:
- `file_paths`: `List[Path]` - データファイルパスリスト
- `array_type`: `str` - データ型（"hcpe", "preprocessing", "stage1", "stage2"）
- `num_mock_records`: `int` - モックレコード数（テスト用）

**戻り値**: `SearchIndex` - 構築済みインスタンス

##### `search_by_id()`

```python
def search_by_id(
    self, record_id: str
) -> Optional[Tuple[int, int]]
```

**引数**:
- `record_id`: `str` - 検索するレコードID

**戻り値**: `Optional[Tuple[int, int]]`
- `(file_index, row_number)` または `None`

##### `search_by_eval_range()`

```python
def search_by_eval_range(
    self,
    min_eval: Optional[int] = None,
    max_eval: Optional[int] = None,
    offset: int = 0,
    limit: int = 20,
) -> List[Tuple[int, int]]
```

**引数**:
- `min_eval`: `Optional[int]` - 最小評価値
- `max_eval`: `Optional[int]` - 最大評価値
- `offset`: `int` - スキップする件数
- `limit`: `int` - 取得する最大件数

**戻り値**: `List[Tuple[int, int]]` - `(file_index, row_number)`のリスト

##### `total_records()`

```python
def total_records(self) -> int
```

**戻り値**: `int` - 総レコード数

---

## Rust API

### `SearchIndex` (Rust)

Rustベースの高性能検索インデックス（PyO3バインディング）．

**ファイル**: `rust/maou_index/src/lib.rs`

#### `new()`

```python
def __init__(
    file_paths: list[str],
    array_type: str,
) -> SearchIndex
```

**引数**:
- `file_paths`: `list[str]` - データファイルパスリスト
- `array_type`: `str` - データ型（"hcpe", "preprocessing", "stage1", "stage2"）

**戻り値**: `SearchIndex` - 新しいインスタンス

#### `build_mock()`

```python
def build_mock(self, num_records: int) -> None
```

**引数**:
- `num_records`: `int` - 生成するモックレコード数

**説明**: テスト用モックデータでインデックスを構築

#### `search_by_id()`

```python
def search_by_id(self, id: str) -> Optional[tuple[int, int]]
```

**引数**:
- `id`: `str` - 検索するレコードID

**戻り値**: `Optional[tuple[int, int]]`
- `(file_index, row_number)` または `None`

**計算量**: O(1)（Hash map）

#### `search_by_eval_range()`

```python
def search_by_eval_range(
    self,
    min_eval: Optional[int] = None,
    max_eval: Optional[int] = None,
    offset: int = 0,
    limit: int = 20,
) -> list[tuple[int, int]]
```

**引数**:
- `min_eval`: `Optional[int]` - 最小評価値（Noneで-∞）
- `max_eval`: `Optional[int]` - 最大評価値（Noneで+∞）
- `offset`: `int` - スキップする件数
- `limit`: `int` - 取得する最大件数

**戻り値**: `list[tuple[int, int]]` - `(file_index, row_number)`のリスト

**計算量**: O(log n + k)（B-tree範囲検索 + k件取得）

#### `count_eval_range()`

```python
def count_eval_range(
    self,
    min_eval: Optional[int] = None,
    max_eval: Optional[int] = None,
) -> int
```

**引数**:
- `min_eval`: `Optional[int]` - 最小評価値
- `max_eval`: `Optional[int]` - 最大評価値

**戻り値**: `int` - レコード数

**計算量**: O(log n + m)（m = マッチするレコード数）

#### `total_records()`

```python
def total_records(self) -> int
```

**戻り値**: `int` - 総レコード数

**計算量**: O(1)

---

## データ型仕様

### `array_type`

サポートされるデータ型：

| 値 | 説明 | 主要フィールド |
|----|------|--------------|
| `"hcpe"` | 棋譜データ | `id`, `eval`, `bestMove16`, `gameResult` |
| `"preprocessing"` | 学習用前処理データ | `id`, `boardIdPositions`, `piecesInHand`, `moveLabel`, `resultValue` |
| `"stage1"` | 到達可能マス予測 | `id`, `boardIdPositions`, `piecesInHand`, `reachableSquares` |
| `"stage2"` | 合法手予測 | `id`, `boardIdPositions`, `piecesInHand`, `legalMovesLabel` |

### ボード座標系

- **マスインデックス**: 0-80（行優先，0=左上，80=右下）
- **行**: 0-8（上から下）
- **列**: 0-8（左から右）
- **変換式**: `index = row * 9 + col`

### 駒ID

- **先手駒**: 0-14（`PieceId.EMPTY=0`, `PieceId.FU=1`, ..., `PieceId.RYU=14`）
- **後手駒**: 16-30（先手駒ID + 16）

---

## エラー処理

### CLI

- **ClickException**: ユーザー入力エラー（不正なパス，パラメータ）
- **LazyGroup**: 依存関係不足（Gradio未インストール）

### Python

- **ValueError**: 不正なデータ形式，バリデーションエラー
- **FileNotFoundError**: ファイルが見つからない
- **IndexError**: 範囲外アクセス

### Rust

- **IndexError**: インデックス操作エラー（PyExceptionに変換）

---

## 性能特性

| 操作 | 計算量 | 実測（推定） | 備考 |
|------|--------|-----------|------|
| インデックス構築（100万件） | O(n) | ~5秒 | 起動時のみ |
| ID検索 | O(1) | < 1ms | Hash map |
| 評価値範囲検索 | O(log n + k) | ~5ms（k=20） | B-tree |
| レコード読み込み | O(1) | ~10ms | mmap |
| SVG生成 | O(1) | < 1ms | 固定サイズ |

---

## まとめ

本APIは，Clean Architectureに基づいた設計により，各レイヤーの責務が明確に分離されています．型ヒントとドキュメント文字列により，型安全性と可読性を確保しています．
