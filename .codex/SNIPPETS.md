# Common Snippets

## Run unit tests quickly
```bash
uv run pytest -q
```

## Run a single test file / node
```bash
uv run pytest tests/app/test_converter.py::test_convert_basic -q
```

## Full QA before committing
```bash
uv run ruff format src/
uv run ruff check src/ --fix
uv run isort src/
uv run mypy src/
uv run pytest -q
```
