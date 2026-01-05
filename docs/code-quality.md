# Code Quality Guide

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

## Code Quality Standards

### Required Standards
- **Type hints**: Required for all functions, methods, and class attributes
- **Docstrings**: Required for all public APIs
- **Line length**: 88 characters maximum
- **Function size**: Functions must be focused and small
- **Architecture**: Follow Clean Architecture dependency rules

## Pre-commit Hook Enforcement

**CRITICAL:** NEVER skip pre-commit hooks when running `git push` or `git commit`. The hooks enforce code quality standards and must always run. Do NOT use `--no-verify` flag unless explicitly requested by the user.

## Error Resolution Order

When encountering CI failures, resolve issues in this order:

1. **Code Formatting**: `poetry run ruff format src/ && poetry run ruff check src/ --fix && poetry run isort src/`
2. **Type Errors**: `poetry run mypy src/`
3. **Linting Issues**: `poetry run flake8 src/`
4. **Test Failures**: `poetry run pytest --tb=short`
