# Codex Agent Guide (Short & Mandatory)

**🚨 RULE ZERO:** Never run tools directly.
**Always** prefix commands with `poetry run`.

✅ Correct:
```bash
poetry run pytest
poetry run mypy src/
poetry run ruff check src/
poetry run maou --help
```

❌ Incorrect:
```bash
pytest
mypy src/
ruff check
maou --help
```

**Scope:** tests, linters, type-checkers, formatters, project CLIs, CI examples, docs’ code blocks — **everything**.
