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

**Performance Benchmarks (50,000 records):**

| Data Type | Metric | Numpy (.npy) | Polars + Rust (.feather) | Improvement |
|-----------|--------|--------------|--------------------------|-------------|
| **HCPE** | Load time | 0.0316s | 0.0108s | **2.92x faster** |
| | File size | 29.90 MB | 1.00 MB | **29.78x compression** |
| **Preprocessing** | Load time | 0.8754s | 0.1092s | **8.02x faster** |

## Core Development Rules

### Critical Development Policies

**IMPORTANT: The following rules must be strictly followed:**

1. **Minimal `__init__.py` Creation**: Do NOT create `__init__.py` files unless absolutely necessary for package functionality. Python 3.3+ supports implicit namespace packages.

2. **Pre-commit Hook Enforcement**: NEVER skip pre-commit hooks when running `git push` or `git commit`. The hooks enforce code quality standards and must always run. Do NOT use `--no-verify` flag unless explicitly requested by the user.

### Package Management

**ONLY use Poetry. NEVER use pip directly.**

```bash
poetry install                          # Install dependencies
poetry add package-name                 # Add dependency
poetry add --group dev package-name     # Add dev dependency
```

### Code Quality

See [Code Quality Guide](docs/code-quality.md) for detailed tool configuration.

**Quick Pipeline:**
```bash
poetry run ruff format src/ && poetry run ruff check src/ --fix && \
  poetry run isort src/ && poetry run mypy src/ && poetry run pytest
```

## Testing

See [Testing Guide](docs/testing-guide.md) for comprehensive testing standards and conventions.

**Quick Reference:**
```
src/maou/{layer}/{module}/file.py
  → tests/maou/{layer}/{module}/test_file.py
```

**Test Execution:**
```bash
poetry run pytest                           # All tests
poetry run pytest --cov=src/maou            # With coverage
TEST_GCP=true poetry run pytest             # Test GCP features
TEST_AWS=true poetry run pytest             # Test AWS features
```

## Rust Backend Development

The project uses Rust for high-performance I/O operations with Arrow IPC format (Polars + Rust backend). See [Rust Backend Development Guide](docs/rust-backend.md) for comprehensive setup, configuration, and troubleshooting.

### Quick Start
- Development: `poetry run maturin develop`
- Testing: `cargo test --manifest-path rust/maou_io/Cargo.toml`
- Memory-constrained: Automatic optimizations reduce peak RAM to 1.0-1.5GB

See [Memory-Constrained Build Configuration](docs/rust-backend.md#memory-constrained-build-configuration) for detailed setup.

## Environment Setup

```bash
bash scripts/dev-init.sh                 # Initialize
poetry run bash scripts/pre-commit.sh    # Setup hooks
```

**Cloud Auth:**
- GCP: `gcloud auth application-default login`
- AWS: `aws configure sso --use-device-code --profile default`

## CLI Commands

The project provides main CLI commands following the data pipeline. See [Command Reference](docs/commands/) for detailed options.

### Main Commands
1. **hcpe-convert** - Convert game records to HCPE format
2. **pre-process** - Preprocess HCPE data for training
3. **learn-model** - Train the model
4. **utility benchmark** - Performance optimization tools
5. **maou visualize** - Interactive data visualization

**Examples:**
```bash
poetry run maou hcpe-convert --input-path /path/to/records --input-format csa
poetry run maou pre-process --input-path /path/to/hcpe --output-dir /path/to/output
poetry run maou learn-model --input-dir /path/to/data --gpu cuda:0 --epoch 10
```

**Cloud Storage Integration:** See [S3/GCS Operations](docs/commands/) for S3/GCS integration and array bundling.

## Architecture

The project follows Clean Architecture principles with strict dependency rules.

**Dependency Flow:**
```
infra → interface → app → domain
```

See [Architecture Guide](docs/architecture.md) for:
- cshogi library encapsulation and piece ID mapping
- Data I/O architecture and array type system
- Detailed anti-patterns and best practices

## Shogi Visualization Rules

When implementing visualization features, understand critical Shogi conventions to avoid coordinate mapping errors.

See [Shogi Visualization Conventions](docs/visualization/shogi-conventions.md) for:
- Board coordinate system (Fortran order arrays)
- Initial position reference (SFEN notation)
- Captured pieces display ordering
- Common implementation pitfalls

**CRITICAL:** Always use `visual_col = 8 - col` for coordinate transformation.

## Performance Optimization

CUDA devices have GPU prefetching auto-enabled (-93.6% load time, +53.2% throughput).

See [Performance Guide](docs/performance.md) for:
- DataLoader benchmarking workflow
- Gradient accumulation strategies
- GPU prefetching configuration
- Neural network architecture details

**Quick Commands:**
```bash
poetry run maou utility benchmark-dataloader --input-dir /path/to/processed --gpu cuda:0
poetry run maou utility benchmark-training --input-dir /path/to/processed --gpu cuda:0
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

## Git Workflow

See [Git Workflow Guide](docs/git-workflow.md) for detailed commit conventions and PR requirements.

**Pre-commit pipeline:** Format → Lint → Type-check → Test

**Commit format:** `feat|fix|docs|refactor|test|perf: message`

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
| **gradio-screenshot-capture** | Capture Gradio UI screenshots with Playwright | screenshot, capture UI, visual feedback |
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
