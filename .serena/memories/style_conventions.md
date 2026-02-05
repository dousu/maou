# Code Style and Conventions

## Type Hints
- MUST add type hints to all code

## Docstrings
- MUST add docstrings to public APIs
- Use Japanese for docstrings

## Japanese Writing Rules (日本語記述規則)
- 句点: `，` (全角コンマ)
- 読点: `．` (全角ピリオド)
- 括弧: `()` (半角のみ)

## Architecture (Clean Architecture)
- Dependency flow: `infra → interface → app → domain`
- MUST NOT introduce circular dependencies

## Forbidden Actions
- MUST NOT use pip (use `uv` only)
- MUST NOT create `__init__.py` unless absolutely necessary
- MUST NOT skip pre-commit hooks (NEVER use `--no-verify`)
- MUST NOT commit secrets (.env, credentials)
