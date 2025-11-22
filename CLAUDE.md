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
