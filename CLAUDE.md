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

### Critical Development Policies

**IMPORTANT: The following rules must be strictly followed:**

1. **Minimal `__init__.py` Creation**: Do NOT create `__init__.py` files unless absolutely necessary for package functionality. Python 3.3+ supports implicit namespace packages, so `__init__.py` files are only needed when you need to execute package-level initialization code or control what gets imported with `from package import *`.

2. **Pre-commit Hook Enforcement**: NEVER skip pre-commit hooks when running `git push` or `git commit`. The hooks enforce code quality standards and must always run. Do NOT use `--no-verify` flag unless explicitly requested by the user.

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

### Test Organization and Conventions

**CRITICAL:** Tests must strictly mirror the source directory structure following Clean Architecture layers.

#### Directory Structure Rules

**Pattern:**
```
src/maou/{layer}/{module}/file.py
  → tests/maou/{layer}/{module}/test_file.py
```

**Layer Mapping:**
- `src/maou/domain/` → `tests/maou/domain/`
- `src/maou/app/` → `tests/maou/app/`
- `src/maou/interface/` → `tests/maou/interface/`
- `src/maou/infra/` → `tests/maou/infra/`

**Examples:**
```
src/maou/domain/board/shogi.py
  → tests/maou/domain/board/test_shogi.py

src/maou/app/learning/training_loop.py
  → tests/maou/app/learning/test_training_loop.py

src/maou/infra/s3/s3_data_source.py
  → tests/maou/infra/s3/test_s3_data_source.py
```

#### Test File Naming Conventions

**Required format:** `test_{module_name}.py`

**Rules:**
1. **Prefix:** Always start with `test_` for pytest discovery
2. **Module name:** Match the source file name exactly
3. **Descriptive suffixes:** Add clarifying suffixes when testing specific aspects

**Examples:**
- ✅ `test_s3_data_source.py` - Primary S3 DataSource tests
- ✅ `test_file_data_source_stage_support.py` - Stage-specific feature tests
- ❌ `test_validation.py` - Too generic
- ❌ `tests/maou/app/test_training_loop.py` - Wrong location (should be in app/learning/)

#### Special Test Directories

**Integration Tests:** `tests/maou/integrations/`
- Purpose: End-to-end tests spanning multiple layers
- Naming: `test_{workflow}_{scenario}.py`
- Examples: `test_app_hcpe_converter.py`, `test_convert_and_preprocess.py`

**Benchmark Tests:** `tests/benchmarks/`
- Purpose: Performance validation and regression detection
- Naming: `test_{component}_performance.py`
- Run explicitly: `poetry run pytest tests/benchmarks/ -v -s`

#### Test Resource Files

**Rule:** Co-locate test resources with the test files that use them.

**Structure:**
```
tests/maou/{layer}/{module}/
├── test_feature.py
└── resources/
    ├── sample_input.csa
    └── expected_output.feather
```

#### Creating New Tests

**Workflow:**
1. Identify source file: `src/maou/{layer}/{module}/feature.py`
2. Create test file: `tests/maou/{layer}/{module}/test_feature.py`
3. Add test class: `class TestFeatureName:` (optional but recommended)
4. Add test functions: `def test_{specific_behavior}() -> None:`
5. Add resources: Create `resources/` directory if needed

**Template:**
```python
"""Tests for {layer}.{module}.{feature} module."""

from pathlib import Path

import pytest

from maou.{layer}.{module}.{feature} import FeatureClass


class TestFeatureClass:
    """Test suite for FeatureClass."""

    def test_{specific_behavior}(self) -> None:
        """Test that {specific behavior} works correctly."""
        # Arrange
        instance = FeatureClass()

        # Act
        result = instance.method()

        # Assert
        assert result == expected_value
```

#### Running Tests by Layer

```bash
# All tests
poetry run pytest

# Specific layer
poetry run pytest tests/maou/domain/
poetry run pytest tests/maou/app/
poetry run pytest tests/maou/infra/

# Specific module
poetry run pytest tests/maou/app/learning/
poetry run pytest tests/maou/domain/board/

# Integration tests only
poetry run pytest tests/maou/integrations/

# With coverage
poetry run pytest --cov=src/maou --cov-report=html
```

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

#### Interactive Environment (Development Machine / DevContainer)

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

#### Non-Interactive Environment (Google Colab / Jupyter Notebook)

For non-interactive environments like Google Colab, use `-y` flag for automatic acceptance. **Note**: In Colab, environment variables don't persist across cells (`!` commands run in separate shells).

**Recommended approach** (run all in one cell):
```bash
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
  source "$HOME/.cargo/env" && \
  poetry run maturin develop
```

For memory-constrained environments (2-4GB), add build optimizations:
```bash
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
  export PATH="$HOME/.cargo/bin:$PATH" && \
  export CARGO_BUILD_JOBS=1 && \
  export RUSTFLAGS="-C codegen-units=1 -C incremental=1" && \
  poetry run maturin develop
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

### Memory-Constrained Build Configuration

**Problem:** Rust compilation can fail with OOM (Out of Memory) errors on systems with limited RAM (2-4GB) due to:
- Polars `dtype-full` feature compiling 18+ data types (only 6 actually used)
- High parallel codegen units (default: 256) on low-core systems
- Large dependency trees (288 packages) with complex feature interactions

**Solution:** The project is pre-configured with memory-optimized build settings that reduce peak memory usage from 3.0-3.5GB to 1.0-1.5GB (60-70% reduction).

#### Automatic Optimizations

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

#### Expected Build Performance

| Metric | Before Optimization | After Optimization | Improvement |
|--------|--------------------|--------------------|-------------|
| Peak Memory | 3.0-3.5GB | 1.0-1.5GB | **60-70% reduction** |
| OOM Failures | Frequent | Rare (<5%) | **95%+ success rate** |
| Dev Build Time | 1-2 min | 3-5 min | 1.5-2.5x slower (acceptable) |
| Release Build Time | 2-3 min | 5-8 min | ~2x slower (acceptable) |

#### Build Profiles

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

#### Troubleshooting OOM Failures

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


#### Feature Flag Minimization Rationale

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

**Preprocessing Data I/O:**

```python
from maou.domain.data.array_io import save_preprocessing_df, load_preprocessing_df

# Save/load with Rust backend (zero-copy Arrow conversion)
save_preprocessing_df(df, "preprocessed.feather")
loaded_df = load_preprocessing_df("preprocessed.feather")
filtered_df = loaded_df.filter(pl.col("resultValue") > 0.3)
```

**Data Format Summary:**

| Data Type | numpy dtype | Polars Schema | File Format | Rust Backend |
|-----------|-------------|---------------|-------------|--------------|
| HCPE | `get_hcpe_dtype()` | `get_hcpe_polars_schema()` | .feather | ✅ |
| Preprocessing | `get_preprocessing_dtype()` | `get_preprocessing_polars_schema()` | .feather | ✅ |
| Intermediate | `get_intermediate_dtype()` | `get_intermediate_polars_schema()` | .duckdb | ✅ (sparse arrays) |

**Intermediate Store (DuckDB):**

Preprocessing pipeline uses DuckDB for memory-efficient aggregation with Rust sparse array compression (98% reduction). Processes 10M+ positions with 1-5GB RAM.

```python
from maou.domain.data.intermediate_store import IntermediateDataStore

with IntermediateDataStore(db_path=Path("temp.duckdb")) as store:
    for batch_df in hcpe_batches:
        store.add_dataframe_batch(batch_df)
    for chunk_df in store.iter_finalize_chunks_df(chunk_size=1_000_000):
        save_preprocessing_df(chunk_df, output_path)
```

**DataSource with Polars DataFrames:**

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

## Arrow IPC Migration (Completed)

Migration from numpy to Polars + Arrow IPC is complete (Phases 1-3). All data pipeline code now uses Polars DataFrames with Arrow IPC format (.feather files).

**Performance Benchmarks:**
| Data Type | Metric | numpy (.npy) | Polars (.feather) | Improvement |
|-----------|--------|--------------|-------------------|-------------|
| **HCPE** | Load time | 0.0316s | 0.0108s | **2.92x faster** |
| **HCPE** | File size | 29.90 MB | 1.00 MB | **29.78x smaller** |
| **Preprocessing** | Load time | 0.8754s | 0.1092s | **8.02x faster** |

**Key Changes:**
- Use `create_empty_*_df()` instead of numpy arrays
- `store_features()` now accepts `dataframe: pl.DataFrame`
- Schema validation automatic in Arrow IPC (no manual validation needed)
- Legacy numpy conversion kept for PyTorch Dataset compatibility

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

学習済みバックボーンを固定し，出力ヘッドのみを段階的に学習することで，マルチステージトレーニングを実現する．

**CLI Options:**
- `--resume-backbone-from`: Backbone パラメータファイルを指定
- `--resume-policy-head-from`: Policy head パラメータファイルを指定
- `--resume-value-head-from`: Value head パラメータファイルを指定
- `--freeze-backbone`: バックボーンのパラメータを凍結（学習しない）

**Example:**
```bash
poetry run maou learn-model \
  --input-dir /path/to/data \
  --resume-backbone-from pretrained_backbone.pt \
  --freeze-backbone \
  --epoch 10
```

### 5. Performance Optimization
```bash
# Benchmark Polars + Rust I/O performance
poetry run python -m maou.infra.utility.benchmark_polars_io \
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

### cshogi Library Encapsulation

The project uses the `cshogi` C++ library for Shogi game logic, but it is **completely encapsulated within the domain layer** following Clean Architecture.

#### Encapsulation Rules

**Allowed cshogi usage:**
- ✅ `src/maou/domain/board/shogi.py` - Board wrapper (PRIMARY abstraction point)
- ✅ `src/maou/domain/parser/csa_parser.py` - CSA parsing (implementation detail)
- ✅ `src/maou/domain/parser/kif_parser.py` - KIF parsing (implementation detail)
- ✅ `tests/**` - Direct usage allowed for test simplicity (but Board usage preferred)

**Prohibited cshogi usage:**
- ❌ `src/maou/app/**` - MUST use `Board` class, not cshogi directly
- ❌ `src/maou/interface/**` - MUST use domain abstractions
- ❌ `src/maou/infra/**` - MUST use domain abstractions

#### Piece ID Mapping (CRITICAL)

cshogi and PieceId enum use **DIFFERENT orderings**:

| Piece | cshogi ID | PieceId enum | Conversion |
|-------|-----------|--------------|------------|
| 金(GOLD) | 7 | 5 (KI) | Reordered |
| 角(BISHOP) | 5 | 6 (KA) | Reordered |
| 飛(ROOK) | 6 | 7 (HI) | Reordered |
| 白(WHITE) | black+16 | black+14 | Offset difference |

**Conversion methods:**
- `Board._cshogi_piece_to_piece_id()` - Convert piece IDs
- `Board._reorder_piece_planes_cshogi_to_pieceid()` - Reorder feature planes

**IMPORTANT:** All piece ID conversions MUST go through these centralized methods. Never implement conversion logic elsewhere.

#### Replacing cshogi with Another Library

If you need to replace cshogi:

1. **Update Board class** (`src/maou/domain/board/shogi.py`):
   - Replace `self.board = cshogi.Board()` with new library
   - Update `_cshogi_piece_to_piece_id()` for new library's piece IDs
   - Update `_reorder_piece_planes_cshogi_to_pieceid()` if needed
   - Update move utility functions

2. **Update Parsers**:
   - Replace `cshogi.CSA.Parser` in `csa_parser.py`
   - Replace `cshogi.KIF.Parser` in `kif_parser.py`

3. **Verify Constants**:
   - Check `MAX_PIECES_IN_HAND`, `PIECE_TYPES` still match
   - Update static assertions if values differ

4. **Run Tests**:
   - `poetry run pytest tests/maou/domain/board/`
   - Ensure piece ID conversions are correct
   - Verify no regressions in app/interface/infra layers

#### Anti-Patterns (DO NOT DO THIS)

```python
# ❌ BAD: Direct cshogi import in app layer
from cshogi import Board as CshogiBoard

# ✅ GOOD: Use domain Board wrapper
from maou.domain.board.shogi import Board
```

```python
# ❌ BAD: Duplicate piece ID conversion logic
def my_converter(piece):
    if piece == 5:
        return 6  # BISHOP
    # ...

# ✅ GOOD: Use centralized conversion
piece_id = Board._cshogi_piece_to_piece_id(cshogi_piece)
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

## Shogi Visualization Rules

When implementing visualization features for `maou visualize`, Claude Code must understand Shogi-specific conventions to avoid incorrect coordinate mappings, piece placements, and display ordering. This section documents the three critical rules that cause the most implementation errors.

### Critical Conventions

#### 1. Board Coordinate System (CRITICAL)

**Array Structure (Fortran Order):**

The `Board.get_board_id_positions_df()` method returns board positions in **Fortran order (column-major)**:

```python
positions = self.board.pieces.reshape((9, 9), order="F")
# CRITICAL: Fortran order means positions[col][row], NOT positions[row][col]
```

**Array Index Mapping:**

```python
# Array indices to 筋 (file) numbers:
# col: 0 → 筋1 (画面右端, screen right edge)
# col: 1 → 筋2
# col: 2 → 筋3
# ...
# col: 7 → 筋8
# col: 8 → 筋9 (画面左端, screen left edge)
```

**Screen Display Convention:**

When displaying from Black's perspective (先手視点):
- **Screen left edge (画面左端)**: 筋9
- **Screen right edge (画面右端)**: 筋1

**Coordinate Transformation for Rendering:**

```python
# Convert array column index to visual column position
visual_col = 8 - col

# Examples:
# col=0 (筋1) → visual_col=8 → rightmost screen position
# col=8 (筋9) → visual_col=0 → leftmost screen position

# Convert visual column to 筋 number
col_number = col + 1  # or equivalently: 9 - visual_col
```

**IMPORTANT - Incorrect Comment:**

The comment in `board_renderer.py:246` states `"col: 0=右端(筋9), 8=左端(筋1)"` which is **INCORRECT**. The actual mapping is:
- `col: 0 → 筋1 (画面右端)`
- `col: 8 → 筋9 (画面左端)`

The transformation logic in `board_renderer.py:264-265` is correct despite the incorrect comment.

**Common Mistake:**

Assuming `positions[row][col]` (row-major order) leads to transposed or mirrored board displays. Always remember: **Fortran order = `positions[col][row]`**.

**References:**
- `src/maou/domain/board/shogi.py:396-401` - Fortran order reshape
- `src/maou/domain/visualization/board_renderer.py:264-265` - Correct transformation
- `src/maou/domain/visualization/board_renderer.py:436-476` - Coordinate label rendering

#### 2. Initial Position Reference

**Standard SFEN Initial Position:**

```
lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
```

**SFEN Notation:**
- **Uppercase letters** = Black pieces (先手, e.g., `B` = Black Bishop)
- **Lowercase letters** = White pieces (後手, e.g., `b` = White Bishop)
- **Direction**: Left-to-right in SFEN string corresponds to 筋9→筋1 (left-to-right on screen)

**Black's Second Rank (`1B5R1`):**

Reading left-to-right in SFEN:
- Position 0 (筋9): empty
- Position 1 (筋8): **角 (Bishop, `B`)**
- Positions 2-6 (筋7-筋3): empty (5 squares)
- Position 7 (筋2): **飛車 (Rook, `R`)**
- Position 8 (筋1): empty

**Array Representation:**
- **筋8 (col=7)**: 角 (Bishop) → `visual_col=1` → screen left-ish area
- **筋2 (col=1)**: 飛車 (Rook) → `visual_col=7` → screen right-ish area

**Visual Layout (先手視点):**
- **Screen left side**: 角 (Bishop)
- **Screen right side**: 飛車 (Rook)

**cshogi Integration:**

The cshogi library uses square numbering: `square = col * 9 + row` where `col=0→筋1, col=8→筋9`. The `Board` class in `shogi.py` encapsulates cshogi and handles piece ID conversions.

**References:**
- `src/maou/domain/board/shogi.py:425` - SFEN example in docstring
- `src/maou/domain/board/shogi.py` - cshogi encapsulation and piece ID mapping

#### 3. Captured Pieces (持ち駒) Display Order

**Array Structure:**

```python
pieces_in_hand[0:7]   # Black's hand (先手の持ち駒)
pieces_in_hand[7:14]  # White's hand (後手の持ち駒)
```

**Ideal Display Order (価値順 - Value Descending):**

```python
# Display from top to bottom (or left to right):
["飛", "角", "金", "銀", "桂", "香", "歩"]

# Corresponding array indices:
# pieces_in_hand[6]: 飛 (Rook)
# pieces_in_hand[5]: 角 (Bishop)
# pieces_in_hand[4]: 金 (Gold)
# pieces_in_hand[3]: 銀 (Silver)
# pieces_in_hand[2]: 桂 (Knight)
# pieces_in_hand[1]: 香 (Lance)
# pieces_in_hand[0]: 歩 (Pawn)
```

**Current Implementation:**

`board_renderer.py:77-86` defines `HAND_PIECE_NAMES` in **reverse order** (歩→飛), which displays pieces from lowest to highest value. This is acceptable but not the ideal ordering.

**Display Format:**
- Single piece: Display name only (e.g., `"飛"`)
- Multiple pieces: Display name with count (e.g., `"歩×5"`)
- Only show pieces where `count > 0`

**References:**
- `src/maou/domain/visualization/board_renderer.py:77-86` - HAND_PIECE_NAMES definition
- `src/maou/domain/visualization/board_renderer.py:326-434` - Hand piece rendering logic

### Common Implementation Pitfalls

1. **Forgetting Fortran Order**: Accessing `positions[row][col]` instead of `positions[col][row]` causes transposed board display.

2. **Wrong Visual Transformation**: Forgetting `visual_col = 8 - col` or implementing it incorrectly leads to mirrored boards.

3. **Incorrect Initial Positions**: Placing 角 on the right and 飛車 on the left (opposite of correct placement).

4. **Wrong Hand Piece Ordering**: Displaying captured pieces in storage order instead of value-descending order.

5. **Trusting Incorrect Comments**: The comment at `board_renderer.py:246` contradicts the actual implementation. Always verify with the transformation code.

### Detailed Implementation References

**Core Implementation Files:**
- `src/maou/domain/visualization/board_renderer.py` - Complete SVG rendering logic with coordinate transformations
- `src/maou/domain/board/shogi.py` - Board abstraction, cshogi encapsulation, piece ID conversions
- `src/maou/domain/visualization/piece_mapping.py` - Piece rendering and rotation logic

**Design Documentation:**
- `docs/visualization/design.md` - Comprehensive visualization design specification
- `docs/visualization/UI_UX_REDESIGN.md` - Modern UI/UX guidelines and color schemes

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
Automatic GPU prefetching overlaps data loading with computation. **Enabled by default** on CUDA devices.

**Performance**: -93.6% data loading time, +53.2% training throughput (2,202 → 3,374 samples/sec)

```python
# Default: enable_gpu_prefetch=True, gpu_prefetch_buffer_size=3
# To disable: enable_gpu_prefetch=False (not recommended)
```

### Gradient Accumulation
Simulate larger batch sizes without increasing GPU memory. Effective batch size = `batch_size × gradient_accumulation_steps`.

```python
training_loop = TrainingLoop(
    gradient_accumulation_steps=4,  # 256 × 4 = 1024 effective batch
)
# Memory usage: Same as batch_size=256
# Training time: Increases proportionally with steps
```

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

**Pre-commit pipeline**:
```bash
poetry run ruff format src/ && poetry run ruff check src/ --fix && poetry run isort src/ && poetry run mypy src/ && poetry run pytest
```

**Commit format**: `feat|fix|docs|refactor|test|perf: message`

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

The project includes specialized Agent Skills that automate common workflows. Skills activate automatically based on trigger keywords in your requests. See `.claude/skills/{skill-name}/SKILL.md` for detailed documentation.

| Skill | Purpose | Triggers |
|-------|---------|----------|
| **qa-pipeline-automation** | Complete QA pipeline (format, lint, type check, test) | code quality, pre-commit, run tests |
| **pr-preparation-checks** | Comprehensive PR validation and branch status | PR preparation, ready to merge |
| **architecture-validator** | Clean Architecture compliance verification | architecture compliance, dependency flow |
| **type-safety-enforcer** | Enforce type hints and docstring requirements | type safety, mypy, docstrings |
| **cloud-integration-tests** | Execute GCP/AWS integration tests | cloud testing, S3 tests, GCS tests |
| **feature-branch-setup** | Automate feature branch creation | create branch, new feature |
| **dependency-update-helper** | Manage Poetry dependencies (NEVER use pip) | add package, update dependencies |
| **rust-build-optimizer** | Build Rust in memory-constrained environments (2-4GB RAM) | build rust, maturin, OOM error |
| **benchmark-execution** | Performance benchmarks for DataLoader and training | benchmark, performance analysis |
| **japanese-doc-validator** | Validate Japanese punctuation rules (，．) | Japanese text, punctuation rules |
| **data-pipeline-validator** | Validate data pipeline configuration | data pipeline, array_type, schema |

Run `poetry run maou --help` for detailed CLI options and examples.

## Plugins

Claude Code plugins extend functionality with specialized capabilities．

**Installed**: `frontend-design` - Generates distinctive，production-grade frontend interfaces

Configure in `.claude/settings.json`:
```json
{
  "enabledPlugins": {
    "frontend-design@claude-plugins-official": true
  }
}
```
