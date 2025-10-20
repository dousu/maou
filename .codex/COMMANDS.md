# Allowed Commands (with Poetry Prefix)

- Tests
  - `poetry run pytest`
  - `poetry run pytest --cov=src/maou`
- Type-check
  - `poetry run mypy src/`
- Lint & Format
  - `poetry run ruff format src/`
  - `poetry run ruff check src/ --fix`
  - `poetry run isort src/`
  - `poetry run flake8 src/`
- Project CLI
  - `poetry run maou <subcommand> [options]`
