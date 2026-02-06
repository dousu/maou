# Stage2データ生成機能 設計ドキュメント

## 概要

HCPEファイルからStage2（合法手学習）データを生成する機能を追加する．
Stage2は実践の局面を利用した合法手予測ヘッドの学習に使用するデータである．

## 要件

- **入力**: HCPEディレクトリ（featherファイル）
- **出力**: Stage2スキーマのfeatherファイル（LZ4圧縮）
- **重複排除**: HCPハッシュで同一局面を排除
- **GCS対応**: オプションでGCSへ自動アップロード

## ファイル構成

| ファイル | 内容 |
|---------|------|
| `src/maou/app/utility/stage2_data_generation.py` | UseCase（新規） |
| `src/maou/interface/utility_interface.py` | インターフェース関数追加 |
| `src/maou/infra/console/utility.py` | CLIコマンド追加 |
| `tests/maou/app/utility/test_stage2_data_generation.py` | テスト（新規） |

## 処理フロー

PreProcessの大規模ファイル処理パターンを参考に，2フェーズ処理を採用する．

```
Phase 1: HCP収集・重複排除
├── HCPEファイルをバッチ読み込み
├── HCPハッシュ値を計算
└── DuckDBにupsert（重複排除）

Phase 2: 合法手ラベル生成・出力
├── DuckDBからユニークHCPをチャンク読み出し
├── 各HCPから合法手ラベルを生成（make_move_label使用）
└── チャンク単位でfeather出力（+ GCSアップロード）
```

### パフォーマンス上の利点

- 重複排除がPhase 1で完了するため，合法手生成は必要最小限
- メモリに収まらない大規模データでもDuckDBで処理可能
- チャンク出力でメモリ使用量を制御

## クラス設計

```python
# src/maou/app/utility/stage2_data_generation.py

@dataclass
class Stage2DataGenerationConfig:
    """Stage2データ生成の設定."""
    input_dir: Path              # HCPE featherファイルのディレクトリ
    output_dir: Path             # 出力先ディレクトリ
    output_data_name: str = "stage2"  # 出力ファイル名ベース
    chunk_size: int = 100_000    # チャンクサイズ（局面数）
    gcs_bucket: Optional[str] = None   # GCSバケット名
    gcs_prefix: Optional[str] = None   # GCSプレフィックス
    cache_dir: Optional[Path] = None   # 中間データキャッシュ

class Stage2DataGenerationUseCase:
    """Stage2学習データ生成ユースケース."""

    def execute(self, config: Stage2DataGenerationConfig) -> dict:
        """生成を実行し，結果を返す."""
        ...

    def _collect_unique_hcps(self, input_dir: Path, store: IntermediateDataStore) -> int:
        """Phase 1: HCPを収集し重複排除."""
        ...

    def _generate_legal_moves_labels(self, store: IntermediateDataStore, ...) -> int:
        """Phase 2: 合法手ラベル生成・出力."""
        ...
```

## CLIコマンド

```python
@click.command("generate-stage2-data")
@click.option(
    "--input-path",
    help="Input directory containing HCPE feather files.",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    help="Directory for output files.",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--output-gcs",
    type=bool,
    is_flag=True,
    help="Output features to Google Cloud Storage.",
    required=False,
)
@click.option(
    "--output-bucket-name",
    help="GCS bucket name for output.",
    type=str,
    required=False,
)
@click.option(
    "--output-prefix",
    help="GCS prefix path for output.",
    type=str,
    required=False,
)
@click.option(
    "--output-data-name",
    help="Name to identify the data in GCS for output.",
    type=str,
    required=False,
)
@click.option(
    "--chunk-size",
    help="Positions per output chunk (default: 100000).",
    type=int,
    default=100_000,
    required=False,
)
@click.option(
    "--intermediate-cache-dir",
    help="Directory for intermediate data cache (default: temporary directory).",
    type=click.Path(path_type=Path),
    required=False,
)
@handle_exception
def generate_stage2_data(...) -> None:
    """Generate Stage 2 training data for legal moves prediction."""
```

### CLIオプション整合性（pre-processと統一）

| オプション | pre-process | generate-stage2-data |
|-----------|-------------|---------------------|
| 入力パス | `--input-path` | `--input-path` |
| 出力先 | `--output-dir` | `--output-dir` |
| GCS有効化 | `--output-gcs` | `--output-gcs` |
| バケット名 | `--output-bucket-name` | `--output-bucket-name` |
| プレフィックス | `--output-prefix` | `--output-prefix` |
| データ名 | `--output-data-name` | `--output-data-name` |
| 中間キャッシュ | `--intermediate-cache-dir` | `--intermediate-cache-dir` |

### 実行例

```bash
# ローカルのみ
maou utility generate-stage2-data \
    --input-path ./converted_hcpe/ \
    --output-dir ./stage2_data/

# GCSアップロード付き
maou utility generate-stage2-data \
    --input-path ./converted_hcpe/ \
    --output-dir ./stage2_data/ \
    --output-gcs \
    --output-bucket-name maou-training \
    --output-prefix data/stage2/ \
    --output-data-name stage2
```

## 出力データ構造

### Stage2スキーマ（既存の`get_stage2_polars_schema`を使用）

| フィールド | 型 | 説明 |
|-----------|---|------|
| `id` | UInt64 | 局面ハッシュ値（重複排除キー） |
| `boardIdPositions` | List[List[UInt8]] | 盤面（9×9） |
| `piecesInHand` | List[UInt8] | 持ち駒（14要素） |
| `legalMovesLabel` | List[UInt8] | 合法手ラベル（2187要素，binary） |

### 合法手ラベル生成ロジック

```python
board.set_hcp(hcp)
legal_labels = np.zeros(MOVE_LABELS_NUM, dtype=np.uint8)
for move in board.get_legal_moves():
    label = make_move_label(board.get_turn(), move)
    legal_labels[label] = 1
```

### 出力ファイル形式

- Arrow IPC (.feather) with LZ4 compression
- 既存の`save_stage2_df`関数を使用（Polars標準I/O）
- ファイル名パターン（PreProcessと統一）:
  - 単一ファイル: `{output_data_name}.feather`
  - チャンク分割: `{output_data_name}_chunk{0000}.feather`

## 依存関係

- `IntermediateDataStore`: 既存のDuckDBベースストア（再利用または拡張）
- `make_move_label`: 指し手ラベル生成（既存）
- `Board.get_legal_moves()`: 合法手取得（既存）
- `GCS`: クラウドアップロード（既存）
- `save_stage2_df`: Stage2データ保存（既存）

## テスト戦略

### テストファイル

```
tests/maou/app/utility/test_stage2_data_generation.py
```

### テストケース

1. **単一ファイル処理テスト**
   - 小規模HCPEファイル（数局面）から正しくStage2データが生成されること
   - 合法手ラベルが正しく設定されていること

2. **重複排除テスト**
   - 同一局面が複数回含まれる場合，1レコードに集約されること
   - ハッシュ値がIDとして正しく使用されていること

3. **チャンク出力テスト**
   - 大規模データ時に正しくチャンク分割されること
   - ファイル名パターンが`{name}_chunk{0000}.feather`であること

4. **GCSアップロードテスト**（モック使用）
   - `--output-gcs`オプション時にGCSクラスが呼び出されること

5. **スキーマ検証テスト**
   - 出力DataFrameが`get_stage2_polars_schema()`に準拠すること
