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

### Serena MCP
- MUST NOT call multiple Serena MCP tools in parallel (to prevent OOM in memory-constrained DevContainer)
- MUST call Serena tools sequentially, one at a time

### Versioning
- MUST bump version in `pyproject.toml` when modifying files under `src/`
- MUST follow semantic versioning: `fix:` → patch, `feat:` → minor, breaking change → major
- MUST NOT push changes to `src/` without a corresponding version bump
- Version is the single source of truth in `pyproject.toml` (`version_provider = "pep621"`)

### Forbidden Actions
- MUST NOT use pip directly (use `uv` only)
- MUST NOT create `__init__.py` unless absolutely necessary
- MUST NOT skip pre-commit hooks
- MUST NOT commit secrets (.env, credentials)

### Documentation
- MUST update `docs/commands/` when modifying CLI commands or options in `src/maou/infra/console/`
- MUST create a new `docs/commands/<command-name>.md` when adding a new CLI command
- MUST remove `docs/commands/<command-name>.md` when removing a CLI command
- MUST follow the existing documentation format (Overview + CLI options tables)

## Code Exploration Policy (MUST)

コードベースの調査・探索には，MUST use `Task` tool with `subagent_type=Explore`.

### Covered Operations
- ファイル検索（Glob/Grep）を複数回行う調査
- 複数ファイルを読んでコードを理解する作業
- エラー原因の調査
- 実装方法を決めるための既存コード調査

### MUST NOT: Direct Multi-file Exploration
以下の操作を直接行うことを禁止:
- 調査目的での連続的なGrep/Glob実行
- 複数ファイルを順次Readして回る探索
- トラブルシューティング時のコード調査

### Exceptions (Direct Access Allowed)
1. **ユーザーが明示的にファイルパスを指定** - 「src/foo.pyを読んで」
2. **単一ファイルの特定行を確認** - エラーメッセージの「file:line」参照
3. **Exploreで特定済みファイルへのアクセス** - 既知の場所への編集

### Decision Criteria
- 「どこにあるか分からない」→ Explore必須
- 「このファイルのこの部分」→ 直接アクセス可

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

### Agent Teams
- SHOULD limit team agents to 2 or fewer in 8GB RAM DevContainer environments
- SHOULD consider memory impact when adding agents (each agent spawns its own MCP server processes)

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
| LR Tuning | [docs/learning-rate-tuning.md](docs/learning-rate-tuning.md) |
| Git Workflow | [docs/git-workflow.md](docs/git-workflow.md) |
| CLI Commands | [docs/commands/](docs/commands/) |
| Shogi Visualization | [docs/visualization/shogi-conventions.md](docs/visualization/shogi-conventions.md) |

### ⚠️ Visualization 実装時の必読ドキュメント

`maou visualize` や将棋盤描画に関する実装を行う前に，
**必ず** [docs/visualization/shogi-conventions.md](docs/visualization/shogi-conventions.md) を読むこと．

将棋の座標系は一般的な row-major 配列とは異なり，`square = col * 9 + row` である．
この規則を理解せずに実装すると，駒の位置や矢印の方向が90度回転するバグが発生する．
