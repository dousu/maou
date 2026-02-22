# learn-modelコマンド徹底調査レポート

## 1. CLIオプション一覧

### 入力・キャッシュ関連
- `--input-file-packed` (bool, flag): Numpy ファイル展開を有効化 (default: False)
- `--input-cache-mode` (choice: file|memory|mmap, default: file): ローカル入力のキャッシュ戦略。mmap は非推奨で file に変換される
- `--no-streaming` (bool, flag): ストリーミングモード無効化 (デフォルト: enabled)

### GPU・Model・ViT関連
- `--gpu` (str): PyTorch デバイス指定 (e.g., cuda:0, cpu)
- `--model-architecture` (choice: resnet|mlp-mixer|vit, default: resnet): バックボーン
- `--vit-embed-dim` (int): ViT 埋め込み次元
- `--vit-num-layers` (int): ViT エンコーダ層数
- `--vit-num-heads` (int): ViT アテンションヘッド数
- `--vit-mlp-ratio` (float): ViT MLP hidden dim ratio
- `--vit-dropout` (float): ViT ドロップアウト率
- `--gradient-checkpointing` (flag): 勾配チェックポイント有効化
- `--compilation` (bool): torch.compile 有効化
- `--detect-anomaly` (bool, flag): anomaly detection

### 学習基本設定
- `--test-ratio` (float): テストセット比率
- `--epoch` (int): エポック数
- `--batch-size` (int): バッチサイズ (推奨: 512[8GB], 1024[16GB], 2048[24GB], 4096[40-80GB])
- `--dataloader-workers` (int): DataLoader ワーカー数
- `--pin-memory` (bool, flag): pinned memory 有効化
- `--prefetch-factor` (int): ワーカーあたりの先読みバッチ数 (default: 4)

### キャッシュ・Transform
- `--cache-transforms` (bool, flag): Dataset transform のメモリ内キャッシング
  - 実装: `transform is not None` の場合にのみ有効

### TensorBoard ログ
- `--tensorboard-histogram-frequency` (int, default: 0): パラメータヒストグラムログ間隔
- `--tensorboard-histogram-module` (str, multiple): ヒストグラムフィルタ (glob pattern)

### 損失関数・オプティマイザ
- `--gce-parameter` (float, default: 0.1): GCE損失のハイパーパラメータ
- `--policy-loss-ratio` (float, default: 1.0): Policy 損失の重み
- `--value-loss-ratio` (float, default: 1.0): Value 損失の重み
- `--learning-ratio` (float, default: 0.01): 学習率
- `--lr-scheduler` (choice, default: warmup_cosine_decay): 学習率スケジューラ
- `--momentum` (float, default: 0.9): SGD momentum
- `--optimizer` (choice: adamw|sgd, default: adamw)
- `--optimizer-beta1` (float, default: 0.9): AdamW beta1
- `--optimizer-beta2` (float, default: 0.999): AdamW beta2
- `--optimizer-eps` (float, default: 1e-8): AdamW epsilon

### チェックポイント・再開
- `--resume-from` (Path): 完全チェックポイント復帰
- `--start-epoch` (int): 開始エポック番号 (0-indexed)
- `--resume-backbone-from` (Path): バックボーンパラメータ復帰
- `--resume-policy-head-from` (Path): Policy Head 復帰
- `--resume-value-head-from` (Path): Value Head 復帰
- `--freeze-backbone` (flag): バックボーン凍結
- `--trainable-layers` (int): 訓練可能なバックボーン層グループ数

### ステージ設定
- `--stage` (choice: 1|2|3|all, default: 3): 訓練ステージ
- `--stage1-data-path` (Path): Stage 1 訓練データ
- `--stage2-data-path` (Path): Stage 2 訓練データ
- `--stage3-data-path` (Path): Stage 3 訓練データ
- `--stage1-threshold` (float, default: 0.99): Stage 1 精度閾値
- `--stage2-threshold` (float, default: 0.85): Stage 2 F1 閾値
- `--stage1-max-epochs` (int, default: 10): Stage 1 最大エポック
- `--stage2-max-epochs` (int, default: 10): Stage 2 最大エポック
- `--stage1-batch-size` (int): Stage 1 バッチサイズ (default: --batch-size に継承)
- `--stage2-batch-size` (int): Stage 2 バッチサイズ (default: --batch-size に継承)
- `--stage1-learning-rate` (float): Stage 1 学習率 (default: --learning-ratio に継承)
- `--stage2-learning-rate` (float): Stage 2 学習率 (default: --learning-ratio に継承)
- `--stage12-lr-scheduler` (choice: auto|none|..., default: auto): Stage 1/2 スケジューラ (auto = batch_size > 256 で warmup_cosine_decay)
- `--stage12-compilation` (bool, flag): Stage 1/2 torch.compile 有効化
- `--stage1-pos-weight` (float, default: 1.0): Stage 1 正例重み
- `--stage2-pos-weight` (float, default: 1.0): Stage 2 正例重み
- `--stage2-gamma-pos` (float, default: 0.0): Stage 2 ASL 正例フォーカシングパラメータ
- `--stage2-gamma-neg` (float, default: 0.0): Stage 2 ASL 負例フォーカシングパラメータ
- `--stage2-clip` (float, default: 0.0): Stage 2 ASL 負例クリッピング
- `--stage2-hidden-dim` (int): Stage 2 ヘッド隠れ層次元 (デフォルト: 単層線形)
- `--stage2-head-dropout` (float, default: 0.0): Stage 2 ヘッドドロップアウト
- `--stage2-test-ratio` (float, default: 0.0): Stage 2 検証分割比率
- `--resume-reachable-head-from` (Path): Reachable Head 復帰 (Stage 1)
- `--resume-legal-moves-head-from` (Path): Legal Moves Head 復帰 (Stage 2)

### 出力・ログ
- `--log-dir` (Path): TensorBoard ログディレクトリ
- `--model-dir` (Path): モデル出力ディレクトリ
- `--output-gcs` (bool, flag): GCS アップロード有効化
- `--gcs-bucket-name` (str): GCS バケット名
- `--gcs-base-path` (str): GCS ベースパス
- `--output-s3` (bool, flag): S3 アップロード有効化
- `--s3-bucket-name` (str): S3 バケット名
- `--s3-base-path` (str): S3 ベースパス


## 2. inputオプションと各ステージでの使用

### Stage 1 (Reachable Squares)
- `--stage1-data-path`: ファイル読み込み
- `cache_mode = "file"` (常に強制) → OOM防止のため input_cache_mode を無視
- ストリーミング: `--no-streaming` がなければ StreamingFileSource 使用
- キャッシュ関連: cache_transforms は無視 (Stage 1/2 では transforms なし)

### Stage 2 (Legal Moves)
- `--stage2-data-path`: ファイル読み込み
- `cache_mode = "file"` (常に強制)
- ストリーミング: `--no-streaming` がなければ StreamingFileSource 使用
- キャッシュ関連: cache_transforms は無視

### Stage 3 (Policy + Value)
- `--stage3-data-path`: ファイル読み込み
- `input_cache_mode`: "file" または "memory" を使用（Stage 1/2 と異なり）
- `input-file-packed`: True なら numpy bit_pack サポート
- `cache-transforms`: map-style dataset のみで有効
- ストリーミング: file_level_split で train/val 分割し StreamingFileSource 使用
  - 条件: len(files) >= 2
  - len(files) == 1 の場合: map-style にフォールバック


## 3. cache-transformsと--input-cache-modeの実際の挙動

### --input-cache-mode の効果
- **Stage 1/2**: 強制的に "file" に上書き (memory 無視)
- **Stage 3 (map-style)**:
  - "file": ファイルごとにロード (低メモリ)
  - "memory": 初期化時に全ファイル結合し一度にメモリ搭載
- **Stage 3 (streaming)**: 設定無視、ファイル単位ストリーミング

### --cache-transforms の効果
- Stage 1/2: **使用されない** (transform が None のため)
- Stage 3:
  - map-style dataset のみ有効
  - streaming dataset では無視
  - 条件: `cache_transforms AND transform is not None`
  - 実際の実装: transform が None なので実質的に **常に無効**

### Stage 3 でのキャッシュ戦略
1. **map-style (streaming=False)**:
   - FileManager が全ファイルをメモリに読み込み
   - cache_mode="memory": 全ファイルを ColumnarBatch に結合
   - cache_mode="file": ファイルごと on-demand ロード

2. **streaming (streaming=True)**:
   - StreamingFileSource がファイル単位で遅延読み込み
   - キャッシュなし、1ファイル分のみメモリ占有
   - DataLoaderFactory.create_streaming_dataloaders() で worker 分散


## 4. 各オプションの使用箇所

### CLIの learn_model() 関数
- 行 564-636: パラメータ受け取り
- 行 637-700: 入力検証・クラウドストレージ初期化
- 行 726-841: StageDataConfig 遅延初期化 + StreamingFileSource 作成

### Interface層 (maou/interface/learn.py)
- learn() 関数: Stage 3 専用 (map-style のみ)
- learn_multi_stage() 関数: Stage 1/2/3 統合
  - 行 1189-1217: Stage 1 実行 (_run_stage1 or _run_stage1_streaming)
  - 行 1232-1275: Stage 2 実行 (_run_stage2 or _run_stage2_streaming)
  - 行 1304-1365: Stage 3 実行 (learn() 呼び出し)

### App層 (maou/app/learning/)
- dl.py: Learning クラス (map-style, streaming 両対応)
  - 行 163-204: streaming コンポーネント vs map-style 判定
  - 行 325-451: _setup_streaming_components()

- setup.py: TrainingSetup.setup_training_components()
  - cache_transforms 判定 (transform is None で常に無効)

- streaming_dataset.py: StreamingKifDataset, StreamingStage1Dataset, StreamingStage2Dataset
  - worker file分割, epoch seed管理, batch yield


## 5. Streaming datasource の仕組み

### StreamingFileSource (infra/file_system/streaming_file_source.py)
- 初期化時: ファイルパス設定のみ (I/O遅延)
- row_counts アクセス時: 全ファイル行数スキャン (初回のみ)
- iter_files_columnar(): ファイル単位で PolarsDF → ColumnarBatch に変換, yield
- iter_files_columnar_subset(): worker 担当ファイルのみ読み込み

### StreamingKifDataset (app/learning/streaming_dataset.py)
- __init__: ソース設定のみ
- __iter__:
  1. worker 情報取得 (id, seed)
  2. _resolve_worker_files() で worker 担当ファイル分割
  3. ファイルごとに ColumnarBatch 読み込み
  4. _yield_kif_batches() で batch_size 単位に分割
  5. yield: ((board_tensor, pieces_tensor), (move_label, result_value, legal_move_mask))

### ファイル結合・キャッシュ戦略
- map-style FileManager._concatenate_columnar():
  - cache_mode="memory" の場合のみ実行
  - ColumnarBatch.concatenate() で全ファイル結合
  - warning: >32GB → OOM リスク

- streaming: ファイル単位ストリーミング
  - 1ファイル分のみメモリ
  - _FILES_PER_CONCAT=10 で Stage 2 小ファイル結合最適化


## 6. キー実装の追跡

### Stage 1 での cache_mode 上書き (learn_model.py:731-737)
```python
stage12_cache_mode = "file"
if input_cache_mode.lower() == "memory":
    logger.info("Stage 1/2: cache_mode='memory' is ignored for memory efficiency...")
```

### Stage 3 での datasource 遅延初期化 (learn_model.py:1321-1330)
```python
stage3_datasource = (
    None
    if streaming
    else stage3_data_config.create_datasource()
)
```
→ streaming=True で create_datasource() スキップ (OOM 防止)

### Streaming component setup (dl.py:325-451)
StreamingKifDataset + DataLoaderFactory.create_streaming_dataloaders()
