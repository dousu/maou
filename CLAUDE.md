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

### 4. Performance Optimization
```bash
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

Run `poetry run maou --help` for detailed CLI options and examples.
