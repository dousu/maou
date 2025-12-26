# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Guidelines

When working on the Maou project, please follow these core principles:

1. **Clean Architecture**: Strictly maintain dependency flow: `infra → interface → app → domain`
2. **Type Safety**: Type hints are required for all code. No exceptions.
3. **Code Quality**: Follow established patterns and maintain consistency.
4. **Testing**: New features require tests. Bug fixes need regression tests.
5. **Performance**: Optimize for S3 operations with parallel processing where applicable.
6. **Documentation**: Public APIs must have docstrings.

## Project Overview

Maou (魔王) is a Shogi (Japanese chess) AI project implemented in Python following Clean Architecture principles. The name "maou" translates to "demon king" in Japanese.

### Core Components
- **Domain Layer**: Business logic and entities (network models, loss functions, parsers)
- **App Layer**: Use cases (converter, learning, pre-processing)
- **Interface Layer**: Adapters between app and infrastructure
- **Infrastructure Layer**: External systems (cloud storage, databases, logging)

### Data Pipeline Architecture (NEW: Polars + Rust)

**Modern Data Stack:**
- **Data Format**: Arrow IPC (.feather) with LZ4 compression
- **Data Processing**: Polars DataFrames for efficient operations
- **I/O Backend**: Rust (PyO3 + maturin) for high-performance file I/O
- **Legacy Support**: Existing numpy .npy format still supported

**Benefits:**
- **Performance**: 2.9-8.0x faster data loading with Polars + Rust backend
- **Compression**: LZ4 compression reduces storage by 30x on HCPE game data
- **Memory Efficiency**: Zero-copy conversions between Polars and Arrow
- **Modern API**: Polars expressions for readable data transformations
- **Type Safety**: Full type stubs generated with pyo3-stub-gen

**Performance Benchmarks (50,000 records):**

| Data Type | Metric | Numpy (.npy) | Polars + Rust (.feather) | Improvement |
|-----------|--------|--------------|--------------------------|-------------|
| **HCPE** | Load time | 0.0316s | 0.0108s | **2.92x faster** |
| | File size | 29.90 MB | 1.00 MB | **29.78x compression** |
| **Preprocessing** | Load time | 0.8754s | 0.1092s | **8.02x faster** |
| | File size | 147.68 MB | 287.95 MB | 0.51x (random data) |

**Key Findings:**
- **Read performance**: 3-8x faster loading for training workflows
- **Storage efficiency**: 30x compression on real game data (HCPE format)
- **Preprocessing data**: 8x faster loading despite larger file size
- **Write performance**: Slightly slower due to compression (acceptable tradeoff)

## Core Development Rules

### Package Management

**ONLY use Poetry. NEVER use pip directly.**

```bash
# Install dependencies
poetry install

# Environment-specific installations
poetry install -E cpu -E gcp      # CPU + GCP
poetry install -E cuda -E aws     # CUDA + AWS
poetry install -E tpu -E gcp      # TPU + GCP

# Add/remove dependencies
poetry add package-name
poetry add --group dev package-name
poetry remove package-name
```

### Code Quality Standards

#### Required Standards
- **Type hints**: Required for all functions, methods, and class attributes
- **Docstrings**: Required for all public APIs
- **Line length**: 88 characters maximum
- **Function size**: Functions must be focused and small
- **Architecture**: Follow Clean Architecture dependency rules

#### Architecture Compliance
- **Domain layer**: No dependencies on other layers
- **App layer**: Only depends on domain layer
- **Interface layer**: Adapts between app and infrastructure
- **Infrastructure layer**: Implements external system integrations

### Testing Requirements

**Framework**: Use `poetry run pytest`

```bash
poetry run pytest                           # Run all tests
poetry run pytest --cov=src/maou            # Run with coverage
TEST_GCP=true poetry run pytest             # Test GCP features
TEST_AWS=true poetry run pytest             # Test AWS features
```

#### Test Requirements
- **New features**: Must include comprehensive tests
- **Bug fixes**: Must include regression tests
- **Edge cases**: Test error scenarios and boundary conditions
- **Integration tests**: Test cloud provider integrations when applicable

## Python Tools

### Essential Commands

```bash
# Type checking (required before commits)
poetry run mypy src/

# Code formatting
poetry run ruff format src/
poetry run ruff check src/ --fix
poetry run isort src/

# Linting
poetry run flake8 src/

# Complete quality pipeline (run before commits)
poetry run ruff format src/ && poetry run ruff check src/ --fix && poetry run isort src/ && poetry run mypy src/
```

### Pre-commit Hooks
```bash
poetry run bash scripts/pre-commit.sh    # Install hooks
poetry run pre-commit run --all-files    # Run manually
```

## Rust Backend Development

### Overview

The project uses Rust for high-performance I/O operations with Arrow IPC format．
Rust code is located in `rust/maou_io/` and integrated via PyO3 + maturin．

### Initial Rust Setup

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

### Development Workflow

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

### Rust Project Structure

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

### Using Polars + Rust I/O

**Basic Usage:**

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

**HCPE Converter (Polars version):**

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

**Preprocessing Data I/O (Polars version):**

```python
import polars as pl
from maou.domain.data.array_io import (
    save_preprocessing_df,
    load_preprocessing_df,
)
from maou.domain.data.schema import get_preprocessing_polars_schema
from maou.app.pre_process.label import MOVE_LABELS_NUM

# Create preprocessing DataFrame
schema = get_preprocessing_polars_schema()

# Example data structure
data = {
    "id": [12345, 67890],
    "boardIdPositions": [
        [[i for i in range(9)] for _ in range(9)],  # 9x9 board
        [[i * 2 for i in range(9)] for _ in range(9)],
    ],
    "piecesInHand": [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # 14 pieces
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ],
    "moveLabel": [
        [0.0] * MOVE_LABELS_NUM,  # 2187 move labels
        [0.5] * MOVE_LABELS_NUM,
    ],
    "resultValue": [0.0, 0.5],
}

df = pl.DataFrame(data, schema=schema)

# Save to .feather file (Rust backend with zero-copy Arrow conversion)
save_preprocessing_df(df, "preprocessed.feather")

# Load from .feather file
loaded_df = load_preprocessing_df("preprocessed.feather")

# DataFrame operations with Polars
filtered_df = loaded_df.filter(pl.col("resultValue") > 0.3)
```

**Intermediate Data (Polars schema available):**

```python
from maou.domain.data.schema import (
    get_intermediate_polars_schema,
    create_empty_intermediate_df,
)

# Intermediate data schema (for future Polars-based aggregation)
schema = get_intermediate_polars_schema()
# Fields: id, boardIdPositions, piecesInHand, count, moveLabelCount, winCount

# Create empty DataFrame with proper schema
df = create_empty_intermediate_df(size=1000)
```

**Data Format Summary:**

| Data Type | numpy dtype | Polars Schema | File Format | Rust Backend |
|-----------|-------------|---------------|-------------|--------------|
| HCPE | `get_hcpe_dtype()` | `get_hcpe_polars_schema()` | .feather | ✅ |
| Preprocessing | `get_preprocessing_dtype()` | `get_preprocessing_polars_schema()` | .feather | ✅ |
| Intermediate | `get_intermediate_dtype()` | `get_intermediate_polars_schema()` | .duckdb | ✅ (sparse arrays) |

**Intermediate Store (DuckDB + Arrow IPC):**

The preprocessing pipeline uses DuckDB for memory-efficient aggregation of duplicate board positions:

- **Storage**: DuckDB database with Arrow-native types (zero-copy operations)
- **Compression**: Rust-accelerated sparse array compression for moveLabelCount (99% sparse，20 non-zero elements out of 1496)
- **Performance**: 2-3x faster than previous SQLite implementation，30-40% disk space reduction
- **Format**: .duckdb file (temporary，deleted after finalization)
- **Memory Efficiency**: Processes 10M+ unique positions with only 1-5GB RAM usage

**Key Features:**
```python
from maou.domain.data.intermediate_store import IntermediateDataStore

# Create intermediate store for memory-efficient preprocessing
with IntermediateDataStore(db_path=Path("temp.duckdb")) as store:
    # Add/aggregate duplicate positions (Polars DataFrame API)
    for batch_df in hcpe_batches:
        # batch_df is a Polars DataFrame with columns:
        # hash_id, count, win_count, move_label_count, board_id_positions, pieces_in_hand
        store.add_dataframe_batch(batch_df)  # UPSERT with Rust sparse compression

    # Finalize in chunks (memory-efficient for large datasets)
    for chunk_df in store.iter_finalize_chunks_df(chunk_size=1_000_000):
        # chunk_df is already a Polars DataFrame with normalized data
        save_preprocessing_df(chunk_df, output_path)

# Database automatically deleted on context exit
```

**Sparse Array Compression (Rust):**
- **Before**: 1496 int32 values = 5984 bytes per position
- **After**: ~20 indices (uint16) + 20 values (int32) = 120 bytes per position
- **Compression Ratio**: 98% reduction for typical move distributions
- **Speed**: 5-10x faster than Python implementation

**Disk Usage (10M unique positions):**
- **DuckDB Database**: 8-12 GB (vs 15-20 GB with SQLite)
- **Peak Disk**: ~15 GB with chunked output + incremental deletion
- **Final Output**: ~45 GB (.feather files with LZ4 compression)

**DataSource with Polars DataFrames (Phase 4):**

All DataSource implementations now support iterating as Polars DataFrames via `iter_batches_df()`:

```python
from maou.infra.file_system.file_data_source import FileDataSource
import polars as pl

# Create DataSource with .feather files (direct DataFrame loading)
datasource = FileDataSource(
    file_paths=[Path("data1.feather"), Path("data2.feather")],
    array_type="hcpe",
    cache_mode="mmap",
)

# Iterate as DataFrames (most efficient for .feather files)
for name, df in datasource.iter_batches_df():
    print(f"Batch: {name}, Records: {len(df)}")
    # Process DataFrame with Polars operations
    filtered = df.filter(pl.col("eval") > 100)
    high_value = df.filter(pl.col("resultValue") > 0.7)

# Also works with legacy .npy files (automatic conversion)
legacy_datasource = FileDataSource(
    file_paths=[Path("legacy_data.npy")],
    array_type="hcpe",
    cache_mode="mmap",
)

# Converts numpy arrays to DataFrames automatically
for name, df in legacy_datasource.iter_batches_df():
    # Same DataFrame interface regardless of file format
    print(df.schema)
```

**Cloud Storage DataSource with Polars:**

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

**Performance Comparison:**

| File Format | iter_batches() | iter_batches_df() | Notes |
|-------------|----------------|-------------------|-------|
| `.feather` | ❌ Not supported | ✅ Zero-copy load | Most efficient |
| `.npy` | ✅ mmap/memory | ✅ Auto-convert | Conversion overhead |
| Cloud (cached) | ✅ numpy arrays | ✅ Auto-convert | Same as .npy |

**Recommendation:** Use `.feather` files for new data pipelines to take advantage of direct DataFrame loading.

**PyTorch Dataset with Polars DataFrames (Phase 5):**

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

**Complete Training Pipeline with Polars:**

```python
import polars as pl
from pathlib import Path
from torch.utils.data import DataLoader

from maou.infra.file_system.file_data_source import FileDataSource
from maou.app.learning.polars_datasource import PolarsDataFrameSource
from maou.app.learning.dataset import KifDataset
from maou.app.learning.training_loop import TrainingLoop

# 1. Load data as DataFrames from .feather files
file_datasource = FileDataSource(
    file_paths=list(Path("data/").glob("*.feather")),
    array_type="preprocessing",
    cache_mode="memory",  # Load all into RAM
)

# 2. Collect all DataFrames into one
dataframes = []
for name, df in file_datasource.iter_batches_df():
    dataframes.append(df)

full_df = pl.concat(dataframes)

# 3. Optional: Filter/preprocess with Polars
filtered_df = full_df.filter(
    (pl.col("resultValue") > 0.3) &  # High-value positions
    (pl.col("id") % 10 != 0)  # 90% for training
)

# 4. Create training dataset
train_datasource = PolarsDataFrameSource(
    dataframe=filtered_df,
    array_type="preprocessing",
)

train_dataset = KifDataset(
    datasource=train_datasource,
    transform=None,
)

# 5. Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

# 6. Train model
training_loop = TrainingLoop(...)
training_loop.train(train_loader, ...)
```

**Supported Data Types:**

| array_type | Schema | Dataset Class | Status |
|------------|--------|---------------|--------|
| `"preprocessing"` | Full training data | `KifDataset` | ✅ Tested |
| `"stage1"` | Reachable squares | `Stage1Dataset` | ✅ Compatible |
| `"stage2"` | Legal moves | `Stage2Dataset` | ✅ Compatible |
| `"hcpe"` | Game records | `KifDataset` (with transform) | ✅ Compatible |

### File Format Migration

**Legacy Format:**
- Extension: `.npy`
- Backend: numpy binary format
- Size: Uncompressed

**New Format:**
- Extension: `.feather`
- Backend: Arrow IPC (Rust)
- Size: LZ4 compressed (2-3x compression on typical game data)

**Note**: Both formats are currently supported．Gradual migration recommended．

## Environment Setup

### Initial Setup
```bash
bash scripts/dev-init.sh                 # Initialize environment
poetry env info --path                   # Get environment path
poetry run bash scripts/pre-commit.sh    # Setup hooks
```

### Cloud Authentication

#### Google Cloud Platform (GCP)
```bash
gcloud auth application-default login
gcloud config set project "your-project-id"
```

#### Amazon Web Services (AWS)
```bash
aws configure sso --use-device-code --profile default
aws sso login --use-device-code --profile default  # Renew token
```

## CLI Commands

The project provides main CLI commands following the data pipeline:

### 1. Game Record Conversion
```bash
poetry run maou hcpe-convert \
  --input-path /path/to/records \
  --input-format csa \
  --output-dir /path/to/output
```

### 2. Data Pre-processing
```bash
poetry run maou pre-process \
  --input-path /path/to/hcpe \
  --output-dir /path/to/processed
```

### 3. Model Training
```bash
poetry run maou learn-model \
  --input-dir /path/to/processed \
  --gpu cuda:0 \
  --epoch 10 \
  --batch-size 256
```

### 4. Multi-Stage Training

**New Feature**: 学習済みバックボーンを固定し，出力ヘッドのみを段階的に学習することで，マルチステージトレーニングを実現する．

#### 基本的なワークフロー

```bash
# Stage 1: 初期学習（フルモデルの学習）
poetry run maou learn-model \
  --input-dir /path/to/initial_data \
  --gpu cuda:0 \
  --epoch 10 \
  --batch-size 256

# 学習結果: model_20251206_120000_resnet-1.2m_10_backbone.pt
#          model_20251206_120000_resnet-1.2m_10_policy_head.pt
#          model_20251206_120000_resnet-1.2m_10_value_head.pt

# Stage 2: バックボーン固定でヘッドのみをfine-tuning
poetry run maou learn-model \
  --input-dir /path/to/new_data \
  --resume-backbone-from model_20251206_120000_resnet-1.2m_10_backbone.pt \
  --freeze-backbone \
  --gpu cuda:0 \
  --epoch 5 \
  --batch-size 256

# Stage 3: 異なるcheckpointからコンポーネントを組み合わせ
poetry run maou learn-model \
  --input-dir /path/to/another_data \
  --resume-backbone-from model_A_backbone.pt \
  --resume-policy-head-from model_B_policy_head.pt \
  --gpu cuda:0 \
  --epoch 3
```

#### コンポーネント別読み込みオプション

- `--resume-backbone-from`: Backbone（embedding，backbone，pool，hand projection）パラメータファイルを指定
- `--resume-policy-head-from`: Policy headパラメータファイルを指定
- `--resume-value-head-from`: Value headパラメータファイルを指定
- `--freeze-backbone`: バックボーンのパラメータを凍結（学習しない）

#### 使用例

**Example 1**: バックボーンのみを事前学習済みモデルから読み込む
```bash
poetry run maou learn-model \
  --input-dir /path/to/data \
  --resume-backbone-from pretrained_backbone.pt \
  --epoch 10
```

**Example 2**: 全コンポーネントを異なるソースから組み立てる
```bash
poetry run maou learn-model \
  --input-dir /path/to/data \
  --resume-backbone-from model_X_backbone.pt \
  --resume-policy-head-from model_Y_policy_head.pt \
  --resume-value-head-from model_Z_value_head.pt \
  --freeze-backbone \
  --epoch 5
```

### 5. Performance Optimization
```bash
# Benchmark Polars + Rust I/O performance
poetry run python -m maou.app.utility.benchmark_polars_io \
  --num-records 50000 \
  --output-dir /tmp/benchmark

# Benchmark DataLoader configurations
poetry run maou utility benchmark-dataloader \
  --input-dir /path/to/processed \
  --gpu cuda:0 \
  --batch-size 256

# Benchmark training performance
poetry run maou utility benchmark-training \
  --input-dir /path/to/processed \
  --gpu cuda:0 \
  --batch-size 256
```

### Cloud Storage Integration

#### S3 Operations
```bash
# Upload with parallel processing
poetry run maou hcpe-convert \
  --output-s3 \
  --bucket-name my-bucket \
  --max-workers 8

# Download with caching
poetry run maou pre-process \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --max-workers 16
```

#### GCS Operations
```bash
# Similar to S3, replace --output-s3 with --output-gcs
poetry run maou hcpe-convert \
  --output-gcs \
  --bucket-name my-bucket \
  --max-workers 8
```

#### Array Bundling for Efficient Caching
**New Feature**: Bundle small numpy arrays into ~1GB chunks for optimal I/O performance.

```bash
# Enable bundling for S3/GCS downloads
poetry run maou pre-process \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --input-enable-bundling \
  --input-bundle-size-gb 1.0 \
  --max-workers 16

# Enable bundling for learning workflows
poetry run maou learn-model \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --input-enable-bundling \
  --input-bundle-size-gb 1.5 \
  --gpu cuda:0
```

**Benefits of Array Bundling:**
- **I/O Efficiency**: Reduces file count from thousands to dozens
- **Cache Management**: 1GB chunks are easier to manage than many small files
- **Memory Optimization**: Uses memory mapping for efficient access
- **Performance**: Significantly faster data loading for training

**How it Works:**
1. Downloads individual arrays from cloud storage
2. Combines arrays into ~1GB bundles locally
3. Creates metadata files for fast array lookup
4. Uses memory mapping for efficient data access during training

## Architecture

The project follows Clean Architecture principles with strict dependency rules:

### Dependency Rules
**Critical**: Dependencies must flow in one direction only:
```
infra → interface → app → domain
```

### Data I/O Architecture

#### Centralized Schema Management
```python
from maou.domain.data.schema import get_hcpe_dtype, get_preprocessing_dtype
from maou.domain.data.io import save_hcpe_array, load_hcpe_array

# Standardized data types
hcpe_dtype = get_hcpe_dtype()
preprocessing_dtype = get_preprocessing_dtype()

# High-performance I/O
save_hcpe_array(array, "output.hcpe.npy", validate=True)
loaded_array = load_hcpe_array("input.hcpe.npy", validate=True)
```

#### Explicit Array Type System
**CRITICAL**: Always specify `array_type` parameter:

```python
# File system data source
datasource = FileDataSource(
    file_paths=paths,
    array_type="hcpe"  # REQUIRED
)

# S3 data source
datasource = S3DataSource(
    bucket_name="my-bucket",
    array_type="preprocessing"  # REQUIRED
)
```

Available types: `"hcpe"` (game records), `"preprocessing"` (training features)

## Neural Network Architecture

### BottleneckBlock Implementation
The project uses optimized BottleneckBlock architecture (1x1→3x3→1x1 convolution):

**Shogi-optimized configuration:**
- Layers: [2, 2, 2, 1] - Wide and shallow
- Bottleneck widths: [24, 48, 96, 144]
- ~40% fewer parameters than ResNet-50

### Mixed Precision Training
Automatic mixed precision (AMP) enabled for CUDA:
- 1.5-2x faster training on GPU
- ~50% GPU memory reduction
- Maintains FP32 accuracy

## Performance Optimization

### Recommended Workflow
1. **DataLoader Benchmarking**: Find optimal settings
2. **Training Performance Analysis**: Identify bottlenecks
3. **Apply Optimizations**: Use recommended settings

### Sample Ratio for Large Datasets
Use `--sample-ratio` for efficient benchmarking:
```bash
poetry run maou utility benchmark-training \
  --input-s3 \
  --sample-ratio 0.1 \
  --gpu cuda:0
```

### GPU Prefetching (Auto-Enabled)
**NEW**: Automatic GPU prefetching dramatically improves training throughput by overlapping data loading with GPU computation.

#### Performance Improvements
- **Data Loading Time**: -93.6% (0.0409s → 0.0026s per batch)
- **Training Throughput**: +53.2% (2,202 → 3,374 samples/sec)
- **GPU Transfer**: Hidden via asynchronous CUDA streams

#### How It Works
1. **Background Loading**: Loads batches in a separate thread
2. **CUDA Streams**: Transfers data to GPU asynchronously
3. **Buffer Queue**: Maintains 3 batches ready on GPU
4. **Pin Memory**: Automatically enabled for faster transfers

#### Configuration
GPU prefetching is **enabled by default** on CUDA devices. No configuration needed!

```python
# Automatically enabled in TrainingLoop for CUDA devices
training_loop = TrainingLoop(
    model=model,
    device=device,
    enable_gpu_prefetch=True,      # Default: True
    gpu_prefetch_buffer_size=3,     # Default: 3 batches
    ...
)
```

To disable (not recommended):
```python
training_loop = TrainingLoop(
    ...
    enable_gpu_prefetch=False,
)
```

**Architecture**: Implemented in `src/maou/app/learning/gpu_prefetcher.py`

### Gradient Accumulation
Simulate larger batch sizes without increasing GPU memory usage.

#### Use Cases
- **Memory-Limited Training**: Increase effective batch size on limited GPU memory
- **Stability**: Larger effective batch sizes can improve training stability
- **Large Models**: Train models that wouldn't fit with desired batch size

#### Configuration
```python
# Example: Simulate batch size of 1024 with 256 physical batch size
training_loop = TrainingLoop(
    model=model,
    device=device,
    gradient_accumulation_steps=4,  # 256 × 4 = 1024 effective batch size
    ...
)
```

#### How It Works
1. Accumulates gradients over N mini-batches
2. Normalizes loss by dividing by accumulation steps
3. Updates weights only after N batches
4. Memory usage stays constant (1× batch size)

**Effective Batch Size** = `batch_size × gradient_accumulation_steps`

**Example**:
```bash
# Physical batch size: 256
# Accumulation steps: 4
# Effective batch size: 1024
# Memory usage: Same as batch_size=256
```

**Note**: Default is 1 (no accumulation). Training time increases proportionally with accumulation steps，but allows much larger effective batch sizes.

## Debugging and Logging

### Log Level Control
```bash
export MAOU_LOG_LEVEL=DEBUG    # Detailed logging
export MAOU_LOG_LEVEL=INFO     # Default
export MAOU_LOG_LEVEL=WARNING  # Minimal

# Or use CLI flag
poetry run maou --debug-mode hcpe-convert ...
```

## Error Resolution

### CI Failure Resolution Order
1. **Code Formatting**: `poetry run ruff format src/ && poetry run ruff check src/ --fix && poetry run isort src/`
2. **Type Errors**: `poetry run mypy src/`
3. **Linting Issues**: `poetry run flake8 src/`
4. **Test Failures**: `poetry run pytest --tb=short`

## Commit Guidelines

### Quality Checks Before Commits
```bash
# Complete pre-commit pipeline
poetry run ruff format src/
poetry run ruff check src/ --fix
poetry run isort src/
poetry run mypy src/
poetry run pytest
```

### Commit Message Format
Use conventional commit format:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Testing
- `perf`: Performance

### Atomic Commits
- One logical change per commit
- Build-passing commits
- Self-contained changes

## Pull Requests

### Mandatory Requirements
1. **Quality Assurance**: All checks must pass
2. **Detailed Description**: Problem, solution, impact, testing
3. **Code Review**: Appropriate reviewers assigned

### Strict Prohibitions
- ❌ `Co-authored-by` trailers
- ❌ AI tool references
- ❌ Generic commit messages
- ❌ Multiple unrelated changes
- ❌ Breaking tests

## 日本語記述規則

コード内日本語使用時の規則:

### 句読点
- **句点**: `，`(全角コンマ)
- **読点**: `．`(全角ピリオド)

### 括弧
- **括弧**: 半角括弧`()`のみ使用

### 例
```python
def process_shogi_game(game_data: str) -> ProcessingResult:
    """
    将棋の棋譜データを処理し，HCPE形式に変換する．

    Args:
        game_data: CSA形式またはKIF形式の棋譜データ

    Returns:
        変換結果を含むProcessingResultオブジェクト
    """
```

## Agent Skills

The project includes specialized Agent Skills that automate common development workflows and reduce context usage in Claude Code sessions. These skills are automatically activated based on your requests.

### Available Skills

#### High Priority Skills (Daily Use)

**1. qa-pipeline-automation**
Executes complete QA pipeline (formatting, linting, type checking, testing).
```
Ask: "Run the QA pipeline"
Triggers: code quality, pre-commit, format code, type checking, run tests
```

**2. pr-preparation-checks**
Comprehensive PR validation including all quality checks and branch status.
```
Ask: "Prepare this branch for a pull request"
Triggers: PR preparation, pull request checks, ready to merge
```

**3. architecture-validator**
Validates Clean Architecture compliance and dependency flow.
```
Ask: "Validate the architecture"
Triggers: architecture compliance, dependency flow, layer validation
```

#### Medium Priority Skills (Feature Development)

**4. type-safety-enforcer**
Enforces type hints and docstring requirements with mypy.
```
Ask: "Check type safety"
Triggers: type hints, type checking, mypy, docstrings
```

**5. cloud-integration-tests**
Executes GCP/AWS integration tests with authentication validation.
```
Ask: "Run cloud integration tests"
Triggers: cloud testing, S3 tests, GCS tests, AWS, GCP
```

**6. feature-branch-setup**
Automates feature branch creation following project conventions.
```
Ask: "Set up a new feature branch for {topic}"
Triggers: create branch, new feature, branch setup
```

**7. dependency-update-helper**
Manages Poetry dependencies with validation (NEVER use pip).
```
Ask: "Add {package} dependency"
Triggers: add package, update dependencies, poetry add
```

#### Specialized Skills (Specific Use Cases)

**8. benchmark-execution**
Executes performance benchmarks for DataLoader and training.
```
Ask: "Benchmark the training performance"
Triggers: benchmark, performance analysis, optimize training
```

**9. japanese-doc-validator**
Validates Japanese punctuation rules (，．and half-width parentheses).
```
Ask: "Validate Japanese documentation"
Triggers: Japanese text, punctuation rules, docstring validation
```

**10. data-pipeline-validator**
Validates data pipeline configuration and array_type parameters.
```
Ask: "Validate the data pipeline configuration"
Triggers: data pipeline, array_type, schema validation, HCPE format
```

### Context Reduction Benefits

Using Agent Skills reduces context usage by 40-50% compared to manual command execution:

- **QA Pipeline**: 5 commands → 1 skill activation (~70% reduction)
- **PR Preparation**: 10 manual steps → 1 skill activation (~80% reduction)
- **Architecture Validation**: Manual inspection → Automated checks (~90% reduction)
- **Cloud Testing**: Environment setup + tests → Single activation (~60% reduction)

### Skill Activation

Skills activate automatically when you:
- Use trigger keywords in your requests
- Describe tasks that match skill capabilities
- Reference specific workflows (e.g., "before committing")

**Example interactions**:
```
You: "I need to commit these changes"
Claude: [Activates qa-pipeline-automation skill]

You: "Is the architecture compliant?"
Claude: [Activates architecture-validator skill]

You: "Add numpy as a dependency"
Claude: [Activates dependency-update-helper skill]
```

### Skill Combinations

Common workflow combinations:

**Before Committing**:
1. `qa-pipeline-automation` - Run quality checks
2. `architecture-validator` - Verify structure
3. `type-safety-enforcer` - Confirm type coverage

**Before PR**:
1. `qa-pipeline-automation` - Quality validation
2. `architecture-validator` - Architecture compliance
3. `pr-preparation-checks` - Final PR validation

**Performance Optimization**:
1. `benchmark-execution` - Measure baseline
2. (Make optimizations)
3. `benchmark-execution` - Validate improvements

**Cloud Development**:
1. `cloud-integration-tests` - Test connectivity
2. `data-pipeline-validator` - Verify configuration
3. `benchmark-execution` - Measure cloud performance

### Skill Documentation

Each skill has detailed documentation in `.claude/skills/{skill-name}/SKILL.md` including:
- Clear usage instructions
- Command examples
- Validation criteria
- Troubleshooting guidance
- Integration with other skills

### Direct Skill Invocation

While skills activate automatically, you can explicitly request them:
```
"Use the qa-pipeline-automation skill"
"Activate the architecture-validator skill"
"Run the pr-preparation-checks skill"
```

Run `poetry run maou --help` for detailed CLI options and examples.
