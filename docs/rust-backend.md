# Rust Backend Development

## Overview

The project uses Rust (PyO3 + maturin) for high-performance operations．
Rust ワークスペースは `rust/` 以下に配置され，単一の `cdylib` (`maou_rust`) から
3 つのサブモジュールとして Python に公開される．

```
maou._rust
├── maou._rust.maou_io      # Arrow IPC I/O
├── maou._rust.maou_index   # インデックス操作
└── maou._rust.maou_shogi   # 将棋盤面操作・合法手生成・特徴量抽出
```

## Initial Rust Setup

### Interactive Environment (Development Machine / DevContainer)

```bash
# Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build Rust extension (automatically done during uv sync)
uv sync

# Verify Rust backend
uv run python -c "from maou._rust.maou_io import hello; print(hello())"
# Expected output: "Maou I/O Rust backend initialized"
```

### Non-Interactive Environment (Google Colab / Jupyter Notebook)

Colab では Rust ツールチェインのインストールは不要．
GitHub Releases にプリビルト wheel が配置されており，`uv pip install` でインストールできる．

#### Prebuilt Wheel Installation

```python
# Colab セル
# Python バージョンに応じて cp311 または cp312 の wheel を指定
!pip install uv
!uv pip install --system "maou[cuda] @ https://github.com/{owner}/{repo}/releases/download/latest/maou-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl"
```

extras を `[]` 内に指定することで，wheel のインストールと依存パッケージの追加を 1 コマンドで実行できる:

```bash
# 複数 extras を指定する場合
uv pip install --system "maou[cuda,gcp] @ https://github.com/{owner}/{repo}/releases/download/latest/maou-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl"

# extras なし (Rust 拡張のみ)
uv pip install --system "maou @ https://github.com/{owner}/{repo}/releases/download/latest/maou-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl"
```

**Available Extras:**

| Category | Extra | Description |
|---|---|---|
| GPU / Accelerator | `cpu` | PyTorch (CPU) + ONNX Runtime |
| | `cuda` | PyTorch (CUDA) + ONNX Runtime GPU |
| | `mpu` | PyTorch + ONNX Runtime GPU (MPU 環境向け) |
| Inference | `cpu-infer` | ONNX Runtime (CPU 推論のみ) |
| | `onnx-gpu-infer` | ONNX Runtime GPU (GPU 推論のみ) |
| | `tensorrt-infer` | TensorRT + ONNX Runtime GPU |
| Cloud | `gcp` | Google Cloud Storage + BigQuery |
| | `aws` | AWS (boto3) |
| Visualization | `visualize` | Gradio + Matplotlib + Playwright |

**Notes:**
- `--system` フラグは Colab のシステム Python 環境にインストールするために必要
- wheel は main ブランチへの push 時に自動ビルドされる (`.github/workflows/build-wheel.yml`)
- wheel には Python コード，コンパイル済み Rust 拡張 (`_rust.so`)，CLI エントリポイントがすべて含まれる
- GPU / 推論 / クラウド等の依存は wheel に含まれないため extras で指定する

#### Using maou CLI in Colab

wheel のインストールにより `maou` コマンドが利用可能になる:

```python
# CLI ヘルプの表示
!maou --help

# サブコマンドの例
!maou hcpe-convert --help   # HCPE ファイルの変換
!maou pre-process --help    # データの前処理
!maou learn-model --help    # モデルの学習
!maou evaluate --help       # 局面の評価
!maou build-engine --help   # TensorRT エンジンのビルド
!maou visualize --help      # 将棋盤の可視化
!maou utility --help        # ユーティリティ (ベンチマーク等)
```

**Available Subcommands:**

| Subcommand | Description | Required Extras |
|---|---|---|
| `hcpe-convert` | HCPE ファイルを Arrow IPC 形式に変換 | — |
| `pre-process` | 学習データの前処理 | — |
| `learn-model` | モデルの学習 | `torch` |
| `evaluate` | 局面の評価値を計算 | `onnxruntime` |
| `build-engine` | ONNX → TensorRT エンジン変換 | `tensorrt` |
| `visualize` | 将棋盤の可視化 | `gradio`, `matplotlib` |
| `utility` | ベンチマーク・スクリーンショット等 | `torch` |

各サブコマンドの詳細は [docs/commands/](commands/) を参照．

## Development Workflow

```bash
# After modifying Rust code
uv run maturin develop  # Rebuild extension

# Run Rust tests
cargo test --manifest-path rust/maou_io/Cargo.toml

# Format Rust code
cargo fmt --manifest-path rust/maou_io/Cargo.toml

# Lint Rust code
cargo clippy --manifest-path rust/maou_io/Cargo.toml
```

## Memory-Constrained Build Configuration

**Problem:** Rust compilation can fail with OOM (Out of Memory) errors on systems with limited RAM (2-4GB) due to:
- Polars `dtype-full` feature compiling 18+ data types (only 6 actually used)
- High parallel codegen units (default: 256) on low-core systems
- Large dependency trees (288 packages) with complex feature interactions

**Solution:** The project is pre-configured with memory-optimized build settings that reduce peak memory usage from 3.0-3.5GB to 1.0-1.5GB (60-70% reduction).

### sccache (Build Cache)

DevContainer では [sccache](https://github.com/mozilla/sccache) によるビルドキャッシュが有効化されている．
sccache は `rustc` の呼び出しをラップし，コンパイル結果をクレート単位でキャッシュする．

**Effect:** `target/` ディレクトリが消失した場合 (DevContainer 再作成，`cargo clean` 等) でも，
依存クレート (polars, arrow, pyo3 等) の rlib がキャッシュから復元され，フルリビルドを回避できる．

**Configuration (automatically applied by `dev-init.sh`):**

| Setting | Value | Location |
|---|---|---|
| `rustc-wrapper` | `sccache` | `~/.cargo/config.toml` |
| `SCCACHE_CACHE_SIZE` | `1G` | `devcontainer.json` containerEnv |
| `SCCACHE_DIR` | `/home/vscode/.cache/sccache` | `devcontainer.json` containerEnv |
| `incremental` | `false` (dev profile) | `Cargo.toml` + `~/.cargo/config.toml` |

**Useful commands:**

```bash
# キャッシュ統計の表示
sccache --show-stats

# キャッシュのクリア
sccache --zero-stats
```

**Limitations:**
- `cdylib` (`maou_rust`) はキャッシュ対象外．最終的な `.so` の生成は常に実行される
- sccache と incremental compilation は併用不可のため，`incremental = false` に設定済み
- 通常のコード変更→ビルドサイクルでは sccache は関与しない (Cargo の fingerprint が先にスキップを判断)

### Automatic Optimizations

The following optimizations are automatically applied:

**1. Environment Variables (scripts/dev-init.sh):**
```bash
export CARGO_BUILD_JOBS=1              # Single parallel job
export RUSTFLAGS="-C codegen-units=1 -C incremental=1"  # Sequential compilation
```

**2. Build Profiles (Cargo.toml):**
- `codegen-units = 1` for all profiles (dev, release, mem-opt)
- Thin LTO (Link-Time Optimization) for smaller binaries
- Sequential compilation prioritizes memory over build speed

**3. Minimal Feature Flags:**
- **Polars:** Only 6 specific dtypes enabled (i8, i16, u8, u16, u64, date) instead of `dtype-full`
- **Arrow:** Removed `prettyprint` feature (not used in production)
- **Evidence:** Analyzed actual usage in `src/maou/domain/data/schema.py` and `rust/maou_index/src/index.rs`

**4. Optimized Linker:**
- Uses `lld` (LLVM linker) for faster, lower-memory linking
- Configured in `.cargo/config.toml`

### Expected Build Performance

| Metric | Before Optimization | After Optimization | Improvement |
|--------|--------------------|--------------------|-------------|
| Peak Memory | 3.0-3.5GB | 1.0-1.5GB | **60-70% reduction** |
| OOM Failures | Frequent | Rare (<5%) | **95%+ success rate** |
| Dev Build Time | 1-2 min | 3-5 min | 1.5-2.5x slower (acceptable) |
| Release Build Time | 2-3 min | 5-8 min | ~2x slower (acceptable) |

### Build Profiles

**Development (default):**
```bash
uv run maturin develop  # Uses [profile.dev]
# opt-level = 0, codegen-units = 1, incremental = true
```

**Production (optimized):**
```bash
uv run maturin develop --release  # Uses [profile.release]
# opt-level = 3, codegen-units = 1, lto = "thin"
```

**Balanced (memory-optimized):**
```bash
CARGO_PROFILE=mem-opt uv run maturin develop  # Uses [profile.mem-opt]
# opt-level = 2, codegen-units = 1, lto = "thin"
```

### Troubleshooting OOM Failures

If you still encounter OOM errors:

**1. Monitor memory usage:**
```bash
# Check available memory
free -h

# Monitor build process
/usr/bin/time -v uv run maturin develop 2>&1 | grep "Maximum resident"
```

**2. Reduce parallelism further:**
```bash
# Temporarily disable incremental compilation
export CARGO_INCREMENTAL=0
uv run maturin develop
```

**3. Clean build cache:**
```bash
# Clear target directory to free disk space
cargo clean

# Rebuild from scratch
uv run maturin develop
```

**4. Verify configuration:**
```bash
# Check Cargo environment
cargo --version
rustc --version

# Verify build settings
cat .cargo/config.toml
```

**5. Check system resources:**
```bash
# Ensure no swap thrashing
swapon --show

# Check for memory leaks
ps aux --sort=-%mem | head -n 10
```


### Feature Flag Minimization Rationale

The project replaces `dtype-full` with minimal dtype features based on actual usage:

**Used dtypes (6 total):**
- `dtype-i8`: Board piece types, move deltas
- `dtype-i16`: Evaluation scores (centipawns)
- `dtype-u8`: Board square representations (0-255)
- `dtype-u16`: Move labels, piece counts
- `dtype-u64`: Position hash IDs
- `dtype-date`: Timestamp fields

**Unused dtypes (12+ excluded):**
- Floating-point types (f32, f64) - not needed for integer game data
- Large integer types (i32, i64) - covered by smaller types
- Decimal types - not used in game records
- String types - handled separately
- Time types - only date is needed

**Impact:** Removing 12+ unused dtypes saves 800MB-1.2GB memory during compilation.

## Rust Project Structure

```
rust/
├── maou_rust/              # PyO3 cdylib (Python バインディング)
│   └── src/
│       ├── lib.rs          # エントリポイント，サブモジュール登録
│       ├── maou_io.rs      # Arrow I/O ラッパー
│       ├── maou_index.rs   # インデックスラッパー
│       └── maou_shogi.rs   # 将棋エンジンラッパー (PyBoard 等)
├── maou_io/                # Arrow IPC I/O クレート
│   └── src/
│       ├── lib.rs
│       ├── arrow_io.rs
│       ├── schema.rs
│       └── error.rs
├── maou_index/             # インデックスクレート
└── maou_shogi/             # 将棋エンジンクレート
    └── src/
        ├── lib.rs
        ├── board.rs        # 盤面表現 (Bitboard + Mailbox)
        ├── movegen.rs      # 合法手生成
        ├── moves.rs        # Move エンコーディング
        ├── types.rs        # PieceType, Square 等
        ├── feature.rs      # 特徴量抽出 (104×9×9)
        ├── hcp.rs          # HuffmanCodedPos
        └── dfpn.rs         # 詰将棋ソルバー (Df-Pn アルゴリズム)
```

## maou_shogi モジュール

`maou._rust.maou_shogi` は将棋エンジン機能を Python に公開する．
旧 `cshogi` ライブラリの置き換えとして設計され，API レベルの互換性を維持する．

### PyBoard クラス

| メソッド | 説明 |
|---------|------|
| `PyBoard()` | 平手初期局面で生成 |
| `set_sfen(sfen)` / `sfen()` | SFEN の設定・取得 |
| `set_turn(turn)` / `turn` | 手番の設定・取得 (0=先手, 1=後手) |
| `set_hcp(data)` / `to_hcp()` | HCP (32バイト) のデコード・エンコード |
| `legal_moves()` | 合法手リスト (`Vec<u32>`) |
| `move_from_move16(m16)` | 16-bit → 32-bit move 変換 |
| `move_from_usi(usi)` | USI 文字列 → 32-bit move 変換 |
| `push(m)` | 指し手適用 (内部 undo スタックに保存) |
| `pop()` | 直前の指し手取消 |
| `piece_planes(arr)` / `piece_planes_rotate(arr)` | 特徴量抽出 (in-place，`np.ndarray[f32]`) |
| `pieces_in_hand()` | 持ち駒 `(Vec<u8>, Vec<u8>)` |
| `piece(sq)` / `pieces()` | 駒情報取得 |
| `zobrist_hash()` | Zobrist ハッシュ値 |
| `is_ok()` | 盤面整合性検証 |

### フリー関数

| 関数 | 説明 |
|------|------|
| `move16(m)` | 32-bit → 16-bit 圧縮 |
| `move_to(m)` / `move_from(m)` | 移動先・移動元の取得 |
| `move_to_usi(m)` | USI 文字列変換 |
| `move_is_drop(m)` / `move_is_promotion(m)` | 打ち・成り判定 |
| `move_drop_hand_piece(m)` | 打ち駒の種類取得 |
| `solve_tsume(sfen, ...)` | 詰将棋ソルバー (後述) |

### 詰将棋ソルバー (Df-Pn)

`solve_tsume` は Depth-First Proof-Number (Df-Pn) アルゴリズムによる詰将棋ソルバーである．
先手番・後手番いずれの詰将棋にも対応し，デフォルトで最短手数の詰み手順を返す．

#### Python API

```python
from maou._rust.maou_shogi import solve_tsume

result = solve_tsume(
    sfen,                    # SFEN 局面文字列
    depth=31,                # 最大探索深さ (手数)
    nodes=1_048_576,         # 最大探索ノード数
    draw_ply=32767,          # 千日手判定閾値 (手数)
    timeout_secs=300,        # タイムアウト (秒)
    find_shortest=True,      # 最短手数探索の有無
)

# TsumeResult のプロパティ
result.status          # "checkmate" | "checkmate_no_pv" | "no_checkmate" | "unknown"
result.moves           # 詰み手順 (USI 形式の文字列リスト)
result.nodes_searched  # 探索ノード数
bool(result)           # status が "checkmate" or "checkmate_no_pv" のとき True
```

#### Rust API

```rust
use maou_shogi::dfpn::{solve_tsume, solve_tsume_with_timeout, DfPnSolver, TsumeResult};

// 便利関数 (デフォルトパラメータ)
let result = solve_tsume(sfen, None, None, None)?;

// タイムアウト・最短探索を細かく指定
let result = solve_tsume_with_timeout(
    sfen,
    Some(31),       // depth
    Some(5_000_000), // nodes
    None,            // draw_ply (default: 32767)
    Some(60),        // timeout_secs
    Some(true),      // find_shortest
)?;

// ソルバーを直接構築
let mut solver = DfPnSolver::with_timeout(31, 1_048_576, 32767, 300);
solver.set_find_shortest(false);  // 最短探索を無効化 (高速化)
let result = solver.solve(&mut board);

match result {
    TsumeResult::Checkmate { moves, nodes_searched }  => { /* 詰み (手順あり) */ }
    TsumeResult::CheckmateNoPv { nodes_searched }     => { /* 詰み (PV 復元不可) */ }
    TsumeResult::NoCheckmate { nodes_searched }       => { /* 不詰 */ }
    TsumeResult::Unknown { nodes_searched }            => { /* 探索打ち切り */ }
}
```

#### パラメータガイド

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `depth` | 31 | 最大探索深さ．長手数の詰将棋には引き上げが必要 |
| `nodes` | 1,048,576 (2^20) | 探索ノード数上限．メモリ使用量に比例 |
| `draw_ply` | 32,767 | 千日手判定手数．通常は変更不要 |
| `timeout_secs` | 300 | 時間制限 (秒) |
| `find_shortest` | `true` | `false` にすると最初に見つかった詰み手順を返す (高速だが最短とは限らない) |

#### 主要な最適化手法

- **持ち駒の優越関係 (Hand Dominance)**: TT キーに盤面のみのハッシュ(持ち駒を除外)を使用し，
  同一盤面・異なる持ち駒の局面を統合して探索する．攻め方の持ち駒が多いほど有利な性質を利用し，
  証明駒 (proof pieces) / 反証駒 (disproof pieces) の概念で TT ヒット率を向上させる
- **単一パス探索**: 反復深化を使わず，証明数・反証数が自然に最も有望な手順に探索を誘導する
- **最短手数探索**: `find_shortest=true` 時，初回の詰み発見後に PV 長を上限として
  `complete_or_proofs` で全 OR ノードの未証明子を追加探索し，最短手順を保証する
- **TT エントリ管理**: 各盤面ハッシュにつき最大 64 エントリの持ち駒パターンを保持

#### テスト実績

| 問題 | 手数 | 探索ノード数 | 備考 |
|------|------|-------------|------|
| 1手詰め | 1 | ~10 | 先手・後手両対応 |
| 3手詰め | 3 | ~100 | 飛車活用・金打ち等 |
| 9手詰め | 9 | ~1,000 | 中規模問題 |
| 17手詰め | 17 | ~100,000 | `find_shortest` の効果検証 |
| 29手詰め | 29 | ~50,000 | TT 保存バグの回帰テスト |

### cshogi 互換性に関する設計判断

- **Move エンコーディング**: cshogi 互換の 16-bit/32-bit ビットレイアウトを採用．
  打ち手は bit 15 フラグではなく `from_field >= 81` で識別する．
- **Piece ID**: cshogi 互換 ID を使用 (0=空, 1-14=先手, 17-30=後手)．
  Python 側の `_reorder_piece_planes_cshogi_to_pieceid()` で PieceId 順序に並び替える．
- **push/pop セマンティクス**: `maou_shogi::Board` の `do_move`/`undo_move` は
  捕獲駒の受け渡しが必要．PyO3 ラッパー側で `Vec<(u32, u8)>` の undo スタックを保持し，
  cshogi 互換の `push()`/`pop()` インターフェースを提供する．
- **HCP**: Apery 互換の 32 バイトバイナリフォーマット．既存データとの互換性を保証．
- **特徴量**: 104×9×9 (= `PIECE_TYPES_NUM * 2 + 76`) チャネル．cshogi 互換の出力順序．

## Using Polars + Rust I/O

### Basic Usage

```python
import polars as pl
from maou.domain.data.array_io import save_hcpe_df, load_hcpe_df
from maou.domain.data.schema import get_hcpe_polars_schema

# Create DataFrame
schema = get_hcpe_polars_schema()
df = pl.DataFrame(data, schema=schema)

# Save to .feather file (Rust backend)
save_hcpe_df(df, "output.feather")

# Load from .feather file
loaded_df = load_hcpe_df("output.feather")
```

### HCPE Converter (Polars version)

```python
from maou.app.converter.hcpe_converter import HCPEConverter

converter = HCPEConverter()

# Use new Polars-based method
result = HCPEConverter._process_single_file_polars(
    file=Path("game.csa"),
    input_format="csa",
    output_dir=Path("output/"),
    min_rating=None,
    min_moves=None,
    max_moves=None,
    allowed_endgame_status=None,
    exclude_moves=None,
)
# Output: .feather file instead of .npy
```

### Preprocessing Data I/O

```python
from maou.domain.data.array_io import save_preprocessing_df, load_preprocessing_df

# Save/load with Rust backend (zero-copy Arrow conversion)
save_preprocessing_df(df, "preprocessed.feather")
loaded_df = load_preprocessing_df("preprocessed.feather")
filtered_df = loaded_df.filter(pl.col("resultValue") > 0.3)
```

### Data Format Summary

| Data Type | numpy dtype | Polars Schema | File Format | Rust Backend |
|-----------|-------------|---------------|-------------|--------------|
| HCPE | `get_hcpe_dtype()` | `get_hcpe_polars_schema()` | .feather | ✅ |
| Preprocessing | `get_preprocessing_dtype()` | `get_preprocessing_polars_schema()` | .feather | ✅ |
| Intermediate | `get_intermediate_dtype()` | `get_intermediate_polars_schema()` | .duckdb | ✅ (sparse arrays) |

### Intermediate Store (DuckDB)

Preprocessing pipeline uses DuckDB for memory-efficient aggregation with Rust sparse array compression (98% reduction). Processes 10M+ positions with 1-5GB RAM.

```python
from maou.domain.data.intermediate_store import IntermediateDataStore

with IntermediateDataStore(db_path=Path("temp.duckdb")) as store:
    for batch_df in hcpe_batches:
        store.add_dataframe_batch(batch_df)
    for chunk_df in store.iter_finalize_chunks_df(chunk_size=1_000_000):
        save_preprocessing_df(chunk_df, output_path)
```

### DataSource with Polars DataFrames

```python
from maou.infra.file_system.file_data_source import FileDataSource

datasource = FileDataSource(
    file_paths=[Path("data.feather")],
    array_type="hcpe",
    cache_mode="mmap",
)

for name, df in datasource.iter_batches_df():
    filtered = df.filter(pl.col("eval") > 100)
```

### Cloud Storage DataSource with Polars

```python
from maou.infra.s3.s3_data_source import S3DataSource
from maou.infra.gcs.gcs_data_source import GCSDataSource

# S3 DataSource
s3_datasource = S3DataSource(
    bucket_name="my-bucket",
    prefix="training-data",
    data_name="hcpe-202412",
    local_cache_dir="./cache",
    array_type="hcpe",
    max_workers=16,
    enable_bundling=True,
    bundle_size_gb=1.5,
)

# Iterate as DataFrames (downloads + converts)
for name, df in s3_datasource.iter_batches_df():
    # Process cloud data as DataFrames
    aggregated = df.group_by("id").agg(pl.col("eval").mean())

# GCS DataSource (same interface)
gcs_datasource = GCSDataSource(
    bucket_name="my-gcs-bucket",
    prefix="training-data",
    data_name="preprocessing-202412",
    local_cache_dir="./cache",
    array_type="preprocessing",
)

for name, df in gcs_datasource.iter_batches_df():
    # DataFrame operations work identically
    print(df.describe())
```

### Performance Comparison

| File Format | iter_batches() | iter_batches_df() | Notes |
|-------------|----------------|-------------------|-------|
| `.feather` | ❌ Not supported | ✅ Zero-copy load | Most efficient |
| `.npy` | ✅ mmap/memory | ✅ Auto-convert | Conversion overhead |
| Cloud (cached) | ✅ numpy arrays | ✅ Auto-convert | Same as .npy |

**Recommendation:** Use `.feather` files for new data pipelines to take advantage of direct DataFrame loading.

### PyTorch Dataset with Polars DataFrames (Phase 5)

The project now supports using Polars DataFrames directly with PyTorch Dataset and DataLoader:

```python
import polars as pl
from torch.utils.data import DataLoader

from maou.app.learning.polars_datasource import PolarsDataFrameSource
from maou.app.learning.dataset import KifDataset
from maou.domain.data.rust_io import load_preprocessing_df

# Load preprocessing data as Polars DataFrame
df = load_preprocessing_df("training_data.feather")

# Create Polars-backed DataSource
datasource = PolarsDataFrameSource(
    dataframe=df,
    array_type="preprocessing",
)

# Use with existing KifDataset (no code changes needed!)
dataset = KifDataset(
    datasource=datasource,
    transform=None,  # Preprocessing data doesn't need transform
    cache_transforms=False,
)

# Create DataLoader as usual
dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# Training loop works identically
for features, targets in dataloader:
    board, pieces = features
    move_label, result_value, legal_move_mask = targets
    # ... training code ...
```

**Benefits of Polars Dataset:**

1. **Zero-Copy Efficiency**: Direct conversion from Polars → numpy → PyTorch tensors
2. **Memory Efficient**: Polars DataFrames use less memory than numpy structured arrays
3. **Modern API**: Leverage Polars for data filtering/preprocessing before training
4. **Compatible**: Works with existing Dataset and DataLoader code without changes

**Supported Data Types:**

| array_type | Schema | Dataset Class | Status |
|------------|--------|---------------|--------|
| `"preprocessing"` | Full training data | `KifDataset` | ✅ Tested |
| `"stage1"` | Reachable squares | `Stage1Dataset` | ✅ Compatible |
| `"stage2"` | Legal moves | `Stage2Dataset` | ✅ Compatible |
| `"hcpe"` | Game records | `KifDataset` (with transform) | ✅ Compatible |

## File Format Migration

**Legacy Format:**
- Extension: `.npy`
- Backend: numpy binary format
- Size: Uncompressed

**New Format:**
- Extension: `.feather`
- Backend: Arrow IPC (Rust)
- Size: LZ4 compressed (2-3x compression on typical game data)

**Note**: Both formats are currently supported．Gradual migration recommended．
