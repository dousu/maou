# Rust Backend Development

## Overview

The project uses Rust for high-performance I/O operations with Arrow IPC format．
Rust code is located in `rust/maou_io/` and integrated via PyO3 + maturin．

## Initial Rust Setup

### Interactive Environment (Development Machine / DevContainer)

```bash
# Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build Rust extension (automatically done during poetry install)
poetry install

# Verify Rust backend
poetry run python -c "from maou._rust.maou_io import hello; print(hello())"
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
| | `tpu` | PyTorch + torch-xla (TPU 環境向け) |
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
poetry run maturin develop  # Rebuild extension

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
poetry run maturin develop  # Uses [profile.dev]
# opt-level = 0, codegen-units = 1, incremental = true
```

**Production (optimized):**
```bash
poetry run maturin develop --release  # Uses [profile.release]
# opt-level = 3, codegen-units = 1, lto = "thin"
```

**Balanced (memory-optimized):**
```bash
CARGO_PROFILE=mem-opt poetry run maturin develop  # Uses [profile.mem-opt]
# opt-level = 2, codegen-units = 1, lto = "thin"
```

### Troubleshooting OOM Failures

If you still encounter OOM errors:

**1. Monitor memory usage:**
```bash
# Check available memory
free -h

# Monitor build process
/usr/bin/time -v poetry run maturin develop 2>&1 | grep "Maximum resident"
```

**2. Reduce parallelism further:**
```bash
# Temporarily disable incremental compilation
export CARGO_INCREMENTAL=0
poetry run maturin develop
```

**3. Clean build cache:**
```bash
# Clear target directory to free disk space
cargo clean

# Rebuild from scratch
poetry run maturin develop
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
rust/maou_io/
├── Cargo.toml          # Crate configuration
├── src/
│   ├── lib.rs         # PyO3 module entry point
│   ├── arrow_io.rs    # Arrow IPC I/O implementation
│   ├── schema.rs      # Arrow schema definitions
│   └── error.rs       # Error types
└── benches/           # Performance benchmarks (future)
```

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
