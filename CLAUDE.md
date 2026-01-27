# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Maou (魔王) is a Shogi (Japanese chess) AI project implemented in Python following Clean Architecture principles.

### Core Components
- **Domain Layer**: Business logic and entities (network models, loss functions, parsers)
- **App Layer**: Use cases (converter, learning, pre-processing)
- **Interface Layer**: Adapters between app and infrastructure
- **Infrastructure Layer**: External systems (cloud storage, databases, logging)

### Data Pipeline
- **Data Format**: Arrow IPC (.feather) with LZ4 compression
- **Data Processing**: Polars DataFrames
- **I/O Backend**: Rust (PyO3 + maturin) for high-performance file I/O
- **Legacy Support**: Numpy .npy format still supported

## Critical Rules (MUST)

### Architecture
- MUST maintain dependency flow: `infra → interface → app → domain`
- MUST NOT introduce circular dependencies between layers

### Code Quality
- MUST add type hints to all code
- MUST add docstrings to public APIs
- MUST run pre-commit hooks (NEVER use `--no-verify`)

### Forbidden Actions
- MUST NOT use pip directly (use `uv` only)
- MUST NOT create `__init__.py` unless absolutely necessary
- MUST NOT skip pre-commit hooks
- MUST NOT commit secrets (.env, credentials)

## Development Guidelines (SHOULD)

### Tool Priority (Serena MCP)
SHOULD prefer Serena tools for token efficiency:
1. `find_symbol` - Find definitions
2. `find_referencing_symbols` - Find usages
3. Glob/Grep - File/text search
4. Read - Full file (last resort)

### Package Management
- SHOULD use `uv sync` for dependencies
- SHOULD use `uv add` for new packages
- SHOULD use `uv run` for script execution

### Git Workflow
- SHOULD follow commit format: `feat|fix|docs|refactor|test|perf: message`
- SHOULD run QA pipeline before commit:
  `uv run ruff format src/ && uv run ruff check src/ --fix && uv run isort src/ && uv run mypy src/`

### Testing
- SHOULD write tests for new features
- SHOULD write regression tests for bug fixes
- Test path: `src/maou/{layer}/{module}/file.py` → `tests/maou/{layer}/{module}/test_file.py`

## Quick Reference

### Common Commands
```bash
# Dependencies
uv sync                                    # Base install
uv sync --extra cpu                        # CPU PyTorch
uv sync --extra cuda                       # CUDA PyTorch
uv sync --extra cpu --extra visualize      # Testing with visualization
uv sync --extra cuda --extra visualize     # Full development

# Development
uv run pytest                              # Run tests
uv run maturin develop                     # Build Rust extension
uv run maou --help                         # CLI help
```

### Japanese Writing Rules (日本語記述規則)
- 句点: `，` (全角コンマ)
- 読点: `．` (全角ピリオド)
- 括弧: `()` (半角のみ)

## Documentation Links

| Topic | Document |
|-------|----------|
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Testing | [docs/testing-guide.md](docs/testing-guide.md) |
| Code Quality | [docs/code-quality.md](docs/code-quality.md) |
| Rust Backend | [docs/rust-backend.md](docs/rust-backend.md) |
| Performance | [docs/performance.md](docs/performance.md) |
| Git Workflow | [docs/git-workflow.md](docs/git-workflow.md) |
| CLI Commands | [docs/commands/](docs/commands/) |
| Shogi Visualization | [docs/visualization/shogi-conventions.md](docs/visualization/shogi-conventions.md) |
