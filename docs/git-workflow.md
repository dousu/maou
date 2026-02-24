# Git Workflow Guide

## Commit Guidelines

**Pre-commit pipeline**:
```bash
poetry run ruff format src/ && poetry run ruff check src/ --fix && poetry run isort src/ && poetry run mypy src/ && poetry run pytest
```

**Commit format**: `feat|fix|docs|refactor|test|perf: message`

### Commit Message Examples

- `feat: add shogi board visualization`
- `fix: correct piece ID mapping in cshogi wrapper`
- `docs: update testing guide with layer structure`
- `refactor: simplify data loading pipeline`
- `test: add regression tests for S3 data source`
- `perf: optimize GPU prefetching buffer size`

## Pull Requests

### Mandatory Requirements
1. **Quality Assurance**: All checks must pass
2. **Detailed Description**: Problem, solution, impact, testing
3. **Code Review**: Appropriate reviewers assigned

### Strict Prohibitions
- ❌ `Co-authored-by` trailers
- ❌ AI tool references
- ❌ Generic commit messages
- ❌ Multiple unrelated changes
- ❌ Breaking tests

### PR Description Template

```markdown
## Problem

[Describe the issue or requirement]

## Solution

[Explain your implementation approach]

## Impact

[List affected components and potential risks]

## Testing

[Describe how you tested the changes]
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added for new features
- [ ] Documentation updated
- [ ] All CI checks pass
```

## Pre-commit Hook Enforcement

**CRITICAL:** NEVER skip pre-commit hooks when running `git push` or `git commit`. The hooks enforce code quality standards and must always run. Do NOT use `--no-verify` flag unless explicitly requested by the user.
