---
name: qa-pipeline-automation
description: Execute complete QA pipeline including code formatting with ruff, linting, import sorting with isort, type checking with mypy, and testing with pytest. Use when preparing code for commits, running pre-commit checks, ensuring code quality standards, or validating changes before pushing.
---

# QA Pipeline Automation

Executes the complete quality assurance workflow for the Maou project following uv and Clean Architecture standards.

## Quick Start

Run the complete QA pipeline:
```bash
uv run ruff format src/ && uv run ruff check src/ --fix && uv run isort src/ && uv run mypy src/ && uv run pytest
```

## Instructions

Execute these steps sequentially:

### 1. Code Formatting
Normalize code style to 88-character line limit:
```bash
uv run ruff format src/
```

### 2. Ruff Linting
Fix linting issues automatically:
```bash
uv run ruff check src/ --fix
```

### 3. Import Sorting
Organize imports with Black profile:
```bash
uv run isort src/
```

### 4. Type Checking
Validate type hints (REQUIRED before commits):
```bash
uv run mypy src/
```

**Critical**: Type hints are required for ALL functions, methods, and class attributes. No exceptions.

### 5. Run Tests
Execute test suite:
```bash
uv run pytest
```

For coverage analysis:
```bash
uv run pytest --cov=src/maou
```

## Success Criteria

All of the following must pass:
- ✓ Code formatted to 88 characters
- ✓ Zero ruff violations
- ✓ Type checking passes with zero errors
- ✓ All tests pass
- ✓ Import ordering correct

## What This Validates

**Type Safety**: All functions must have type hints
**Code Style**: 88-character line limit enforced
**Clean Architecture**: Dependency flow maintained (infra → interface → app → domain)
**Test Coverage**: New features and bug fixes have tests
**Documentation**: Public APIs have docstrings

## When to Use

- Before committing code
- After implementing new features
- Before pushing to remote
- When preparing pull requests
- After resolving merge conflicts
- During code reviews

## CI Compliance

This pipeline matches the CI/CD checks that run on GitHub. Running it locally prevents CI failures.

## Error Resolution

If errors occur, follow this order:

1. **Format first**: `uv run ruff format src/`
2. **Fix lint issues**: `uv run ruff check src/ --fix`
3. **Sort imports**: `uv run isort src/`
4. **Check types**: `uv run mypy src/`
5. **Run tests**: `uv run pytest --tb=short`

## Pre-commit Hooks

This skill replicates the pre-commit hooks configured in `.pre-commit-config.yaml`. To install hooks:

```bash
uv run bash scripts/pre-commit.sh
```

## References

See CLAUDE.md and AGENTS.md for complete development guidelines and code quality standards.
