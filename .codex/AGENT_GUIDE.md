# Codex Agent Guide (Short & Mandatory)

**🚨 RULE ZERO:** Never run tools directly.
**Always** prefix commands with `uv run`.

✅ Correct:
```bash
uv run pytest
uv run mypy src/
uv run ruff check src/
uv run maou --help
```

❌ Incorrect:
```bash
pytest
mypy src/
ruff check
maou --help
```

**Scope:** tests, linters, type-checkers, formatters, project CLIs, CI examples, docs’ code blocks — **everything**.
