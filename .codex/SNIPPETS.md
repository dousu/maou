# Common Snippets

## Run unit tests quickly
```bash
poetry run pytest -q
```

## Run a single test file / node
```bash
poetry run pytest tests/app/test_converter.py::test_convert_basic -q
```

## Full QA before committing
```bash
poetry run ruff format src/
poetry run ruff check src/ --fix
poetry run isort src/
poetry run mypy src/
poetry run pytest -q
```
