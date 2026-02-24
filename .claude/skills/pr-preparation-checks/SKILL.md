---
name: pr-preparation-checks
description: Perform comprehensive pre-PR validation including code quality checks, test execution, architecture compliance verification, commit message formatting validation, and branch status confirmation. Use when preparing pull requests, validating branches before submission, or ensuring all PR requirements are met.
---

# PR Preparation Checks

Complete validation workflow before submitting pull requests to ensure all mandatory requirements are met.

## Mandatory Requirements

### 1. Quality Assurance Pipeline

All checks MUST pass:

```bash
# Complete QA pipeline
uv run ruff format src/
uv run ruff check src/ --fix
uv run isort src/
uv run mypy src/
uv run pytest
```

### 2. Test Coverage Requirements

**New features**: Must include comprehensive tests
**Bug fixes**: Must include regression tests
**Edge cases**: Test error scenarios and boundary conditions
**Integration tests**: Test cloud providers when applicable

Run with coverage analysis:
```bash
uv run pytest --cov=src/maou
```

Verify coverage meets project standards (aim for >80%).

### 3. Architecture Compliance

Verify Clean Architecture dependency flow:

```bash
# Check for violations: domain should not import from other layers
grep -r "from maou\.(app|interface|infra)" src/maou/domain/

# App should only import from domain
grep -r "from maou\.(interface|infra)" src/maou/app/

# Verify type hints are present
grep -E "def [a-z_]+\(" src/maou/ | grep -v " ->" | grep -v "__" | grep -v "test_"
```

All checks should return empty results (no violations).

### 4. Commit Message Format

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Valid types**: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`

**Example**:
```
feat(cloud): add array bundling for S3 downloads

Implement ~1GB array bundling for efficient cloud caching.
Reduces file count and improves I/O performance.

- Bundle small numpy arrays into chunks
- Create metadata for fast lookup
- Use memory mapping for access
```

### 5. Branch Status Checks

Verify clean state:

```bash
# Check working directory is clean
git status

# Verify branch name follows pattern
git rev-parse --abbrev-ref HEAD

# Review commit history
git log --oneline -10

# Check for merge conflicts
git diff --name-only --diff-filter=U
```

### 6. Documentation Requirements

**Public APIs**: Must have docstrings
**Type hints**: Required for all functions and methods
**Japanese text**: Use 全角コンマ（，）and 全角ピリオド（．）

Verify docstrings exist:
```bash
# Check for missing docstrings in public functions
grep -A5 "^def [a-z]" src/maou/domain/ | grep -B5 'def ' | grep -v '"""'
```

## PR Description Template

Use this template for PR descriptions:

```markdown
## Problem
[Brief description of the issue being fixed or feature being added]

## Solution
[How the change solves the problem]

## Impact
- [What changes, what breaks, what improves]
- [Performance implications if any]
- [Cloud integration effects if applicable]

## Testing
- [New tests added]
- [Existing tests verified]
- [Manual testing performed]

## Checklist
- [ ] Code formatted and linted
- [ ] Type checking passes
- [ ] Tests pass (pytest)
- [ ] Architecture compliant
- [ ] Docstrings updated
- [ ] CLAUDE.md updated if needed
```

## Strict Prohibitions

These will cause PR rejection:

❌ **Co-authored-by trailers** - Never include in commits
❌ **AI tool references** - No mentions of Claude, GPT, etc.
❌ **Generic commit messages** - Be specific and descriptive
❌ **Multiple unrelated changes** - One logical change per PR
❌ **Breaking tests** - All tests must pass
❌ **Missing type hints** - Type hints are mandatory
❌ **Violated dependency flow** - Respect Clean Architecture

## CI Failure Resolution Order

If CI checks fail, debug in this order:

1. **Code Formatting**
   ```bash
   uv run ruff format src/ && uv run ruff check src/ --fix && uv run isort src/
   ```

2. **Type Errors**
   ```bash
   uv run mypy src/
   ```

3. **Linting Issues**
   ```bash
   uv run flake8 src/
   ```

4. **Test Failures**
   ```bash
   uv run pytest --tb=short -v
   ```

## Branch Workflow

Recommended branch naming: `feature/{topic}` or `fix/{issue}`

Example workflow:
```bash
# Create feature branch
git checkout -b feature/add-array-bundling

# Make changes and commit
git add .
git commit -m "feat(cloud): add array bundling support"

# Run PR checks before pushing
# (Use this skill!)

# Push to remote
git push -u origin feature/add-array-bundling
```

## Integration with GitHub Actions

The project uses GitHub Actions for:
- Claude Code integration (`claude.yml`)
- Pre-commit updates (`pre-commit_autoupdate.yml`)

Your local checks should match CI pipeline to prevent failures.

## Custom Slash Commands

Use these commands for additional validation:
- `/security-review` - Complete security review
- `/review` - Review a pull request
- `/pr-comments` - Get comments from GitHub PR

## References

- **CLAUDE.md**: Development guidelines and standards
- **AGENTS.md**: Codex agent rules and conventions
- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `.codex/config.yaml`: QA pipeline configuration
