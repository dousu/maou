---
name: dependency-update-helper
description: Manage Python dependencies using uv exclusively, add new packages, update existing dependencies, remove unused packages, and validate dependency compatibility. Use when adding packages, updating dependencies, resolving conflicts, or managing development dependencies. NEVER use pip directly.
---

# Dependency Update Helper

Manages Python dependencies exclusively through uv for the Maou project.

## Core Rule

**ONLY use uv. NEVER use pip directly.**

All dependency management must go through uv to maintain consistency and avoid conflicts.

## Adding Dependencies

### Add Production Dependency

```bash
# Add package to main dependencies
uv add package-name

# Add specific version
uv add "package-name==1.2.3"

# Add with version constraint
uv add "package-name>=1.2.0,<2.0.0"

# Add from git repository
uv add "package-name @ git+https://github.com/user/repo.git"
```

### Add Development Dependency

```bash
# Add to dev group
uv add --dev package-name

# Common dev dependencies
uv add --dev pytest
uv add --dev mypy
uv add --dev ruff
```

### Add Optional Dependency (Extras)

```bash
# Add to specific extra group
uv add --optional extra-group package-name

# Then configure in pyproject.toml:
# [project.optional-dependencies]
# cuda = ["torch-cuda", "cupy"]
```

## Updating Dependencies

### Update Single Package

```bash
# Update specific package to latest compatible version
uv lock --upgrade-package package-name
uv sync

# Change version constraint and update
uv add "package-name>=2.0.0"
```

### Update All Dependencies

```bash
# Update all packages to latest compatible versions
uv lock --upgrade
uv sync
```

### Update Lock File

```bash
# Regenerate lock file from pyproject.toml
uv lock

# Check if lock file is up to date
uv lock --check
```

## Removing Dependencies

### Remove Package

```bash
# Remove production dependency
uv remove package-name

# Remove dev dependency
uv remove --dev package-name
```

## Installing Dependencies

### Install All Dependencies

```bash
# Sync from lock file
uv sync
```

### Install with Extras

```bash
# Install CPU + GCP extras
uv sync --extra cpu --extra gcp

# Install CUDA + AWS extras
uv sync --extra cuda --extra aws

# Install TPU + GCP extras
uv sync --extra tpu --extra gcp
```

Available extras:
- **Hardware**: `cpu`, `cuda`, `mpu`, `tpu`
- **Inference**: `cpu-infer`, `onnx-gpu-infer`, `tensorrt-infer`
- **Cloud**: `gcp`, `aws`

## Dependency Validation

### Check Configuration

```bash
# Verify lock file is consistent with pyproject.toml
uv lock --check

# Show dependency tree
uv tree
```

### Show Package Information

```bash
# Show all installed packages
uv pip list

# Show specific package details
uv pip show package-name

# Show outdated packages
uv pip list --outdated
```

## Compatibility Testing

After adding or updating dependencies, run QA pipeline:

```bash
# 1. Update dependencies
uv add new-package

# 2. Update lock file
uv lock

# 3. Sync environment
uv sync

# 4. Run QA pipeline
uv run ruff format src/
uv run ruff check src/ --fix
uv run isort src/
uv run mypy src/
uv run pytest
```

All steps must pass for dependency update to be valid.

## Common Scenarios

### Scenario 1: Add New Feature Dependency

```bash
# Add package
uv add transformers

# Verify installation
uv pip show transformers

# Test import
uv run python -c "import transformers; print('✓')"

# Run tests
uv run pytest
```

### Scenario 2: Update Security Vulnerability

```bash
# Check for outdated packages
uv pip list --outdated

# Update specific package
uv lock --upgrade-package vulnerable-package
uv sync

# Verify version
uv pip show vulnerable-package

# Run full test suite
uv run pytest
```

### Scenario 3: Add Type Stubs

```bash
# Add type stubs for third-party library
uv add --dev types-requests
uv add --dev types-PyYAML

# Verify mypy recognizes types
uv run mypy src/
```

### Scenario 4: Remove Unused Dependency

```bash
# Check where package is used
grep -r "import package_name" src/

# If not used, remove
uv remove package-name

# Verify still works
uv run pytest
```

## Environment Management

### Sync Environment

```bash
# Synchronize environment with lock file (removes unneeded packages)
uv sync

# Verify Python version
uv run python --version
```

## Dependency Conflict Resolution

### Issue: Incompatible Version Constraints

```bash
# Check dependency tree for conflicts
uv tree

# Try to resolve by upgrading
uv lock --upgrade

# If conflicts persist, update pyproject.toml constraints
# Example: Change "package-name >= 1.0.0" to "package-name >= 1.5.0"
```

### Issue: Platform-Specific Dependencies

```bash
# Add platform marker in pyproject.toml
# "package-name; sys_platform == 'linux'"
# "windows-package; sys_platform == 'win32'"
```

## Best Practices

### 1. Always Lock After Changes

```bash
# After any pyproject.toml changes
uv lock
```

### 2. Commit Lock File

```bash
# Always commit uv.lock
git add uv.lock pyproject.toml
git commit -m "feat(deps): add new-package"
```

### 3. Use Version Constraints

```bash
# Prefer >= with upper bound for stability
uv add "package-name>=1.2.0,<2.0.0"

# Use exact pin for critical dependencies
uv add "package-name==1.2.3"
```

### 4. Document Extra Requirements

When adding optional dependencies, document in CLAUDE.md:

```markdown
### New Extra: inference

```bash
uv sync --extra inference
```

Includes: onnx, tensorrt, optimization tools
```

### 5. Test Across Environments

```bash
# Test CPU environment
uv sync --extra cpu
uv run pytest

# Test CUDA environment
uv sync --extra cuda
uv run pytest
```

## Pre-commit Hook Integration

After dependency updates, update pre-commit hooks:

```bash
# Update pre-commit dependencies
uv run pre-commit autoupdate

# Run on all files
uv run pre-commit run --all-files
```

## Troubleshooting

### Issue: "No module named X" After Install

```bash
# Verify package is in lock file
grep "name = \"package-name\"" uv.lock

# Reinstall
uv sync

# Check environment
uv run python -c "import sys; print(sys.path)"
```

### Issue: Lock File Out of Date

```bash
# Regenerate lock file
uv lock

# Or update all dependencies
uv lock --upgrade
uv sync
```

## References

- **CLAUDE.md**: Package management rules
- uv documentation: https://docs.astral.sh/uv/
- `pyproject.toml`: Dependency configuration
- `uv.lock`: Locked dependency versions
