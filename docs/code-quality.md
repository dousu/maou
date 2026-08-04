# Code Quality Guide

## lint / format は ruff に一本化している

整形・lint・import 順のすべてを ruff が担当する．
**isort と flake8 は 2026-08-04 に廃止した** (`reviews/2026-08-04-ruff-single-linter.md`)．

- isort は ruff の `I` ルールと同時に有効化できない
  (19-22 ファイルで整形が食い違い，pre-commit 上で互いに書き換え合う)
- flake8 は pre-commit に未登録で，かつ既に fail していた死んだゲートだった

### ruff の版を上げるときの必須手順

ruff の適用ルールは**デフォルト集合に追随している** (`pyproject.toml` に
`select` を書いていない)．これはデフォルトがファミリ全体ではなく厳選された
サブセットであり (`E` は E722/E902 のみ，`D` は D419 のみ など)，
`select` で再現しようとすると 413 に対し 586 ルールと大幅に超過するため．

デフォルト集合はマイナー更新で変わりうる．実際 0.15 → 0.16 では
同一コードに **484 件**が新規検出された．pre-commit の hook 引数が `--fix`
のため，気付かずに上げると**未レビューの一括自動書き換え**が走る
(0.16 では 245 件が 97 ファイルに適用される状態だった)．

MUST 版を上げる前に新旧の有効ルールを比較し，増分を意図的に採否すること:

```bash
# 有効ルール一覧を出して新旧で diff する
uv tool run ruff@<新版> check --show-settings --isolated \
  | sed -n '/linter.rules.enabled/,/^]/p' | grep -oE '\([A-Z]+[0-9]+\)'

# 実際の検出件数を確認する
uv tool run ruff@<新版> check src/ tests/ scripts/ --statistics
```

## Python Tools

### Essential Commands

```bash
# Type checking (required before commits)
uv run mypy src/

# Code formatting (import 順の整列も ruff が担当する)
uv run ruff format src/
uv run ruff check src/ --fix

# Complete quality pipeline (run before commits)
uv run ruff format src/ && uv run ruff check src/ --fix && uv run mypy src/
```

### Pre-commit Hooks
```bash
uv run bash scripts/pre-commit.sh    # Install hooks
uv run pre-commit run --all-files    # Run manually
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

1. **Code Formatting**: `uv run ruff format src/ && uv run ruff check src/ --fix`
2. **Type Errors**: `uv run mypy src/`
3. **Linting Issues**: `uv run ruff check src/`
4. **Test Failures**: `uv run pytest --tb=short`
