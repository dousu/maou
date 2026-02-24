# Task Completion Checklist

## Before Committing

1. Run QA pipeline:
   ```bash
   uv run ruff format src/
   uv run ruff check src/ --fix
   uv run isort src/
   uv run mypy src/
   ```

2. Run tests if applicable:
   ```bash
   uv run pytest
   ```

3. Ensure pre-commit hooks pass (NEVER use `--no-verify`)

## Commit Message Format
```
feat|fix|docs|refactor|test|perf: message
```
