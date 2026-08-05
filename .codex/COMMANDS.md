# Allowed Commands (with uv Prefix)

- Tests
  - `uv run pytest`
  - `uv run pytest --cov=src/maou`
- Type-check
  - `uv run mypy src/`
- Lint & Format
  - `uv run ruff format src/`
  - `uv run ruff check src/ --fix`
  - `uv run ruff check src/`
- Project CLI
  - `uv run maou <subcommand> [options]`
