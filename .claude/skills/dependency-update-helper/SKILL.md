---
name: dependency-update-helper
description: Manage Python dependencies using Poetry exclusively, add new packages, update existing dependencies, remove unused packages, and validate dependency compatibility. Use when adding packages, updating dependencies, resolving conflicts, or managing development dependencies. NEVER use pip directly.
---

# Dependency Update Helper

Manages Python dependencies exclusively through Poetry for the Maou project.

## Core Rule

**ONLY use Poetry. NEVER use pip directly.**

All dependency management must go through Poetry to maintain consistency and avoid conflicts.

## Adding Dependencies

### Add Production Dependency

```bash
# Add package to main dependencies
poetry add package-name

# Add specific version
poetry add "package-name==1.2.3"

# Add with version constraint
poetry add "package-name>=1.2.0,<2.0.0"

# Add from git repository
poetry add git+https://github.com/user/repo.git
```

### Add Development Dependency

```bash
# Add to dev group
poetry add --group dev package-name

# Common dev dependencies
poetry add --group dev pytest
poetry add --group dev mypy
poetry add --group dev ruff
```

### Add Optional Dependency (Extras)

```bash
# Add to specific extra group
poetry add --optional package-name

# Then add to extras in pyproject.toml:
# [tool.poetry.extras]
# cuda = ["torch-cuda", "cupy"]
```

## Updating Dependencies

### Update Single Package

```bash
# Update specific package to latest version
poetry update package-name

# Update to specific version
poetry add "package-name@^2.0.0"
```

### Update All Dependencies

```bash
# Update all packages to latest compatible versions
poetry update

# Update lock file without upgrading packages
poetry lock --no-update
```

### Update Poetry Lock File

```bash
# Regenerate lock file from pyproject.toml
poetry lock

# Update lock file without installing
poetry lock --no-update
```

## Removing Dependencies

### Remove Package

```bash
# Remove production dependency
poetry remove package-name

# Remove dev dependency
poetry remove --group dev package-name
```

## Installing Dependencies

### Install All Dependencies

```bash
# Install from lock file
poetry install

# Install without dev dependencies
poetry install --without dev
```

### Install with Extras

```bash
# Install CPU + GCP extras
poetry install -E cpu -E gcp

# Install CUDA + AWS extras
poetry install -E cuda -E aws

# Install TPU + GCP extras
poetry install -E tpu -E gcp
```

Available extras:
- **Hardware**: `cpu`, `cuda`, `mpu`, `tpu`
- **Inference**: `cpu-infer`, `onnx-gpu-infer`, `tensorrt-infer`
- **Cloud**: `gcp`, `aws`

## Dependency Validation

### Check Poetry Configuration

```bash
# Verify pyproject.toml is valid
poetry check

# Show current configuration
poetry config --list

# Show dependency tree
poetry show --tree
```

### Show Package Information

```bash
# Show all installed packages
poetry show

# Show specific package details
poetry show package-name

# Show outdated packages
poetry show --outdated
```

### Verify Lock File

```bash
# Check if lock file is consistent with pyproject.toml
poetry lock --check

# Should output: "poetry.lock is consistent with pyproject.toml"
```

## Compatibility Testing

After adding or updating dependencies, run QA pipeline:

```bash
# 1. Update dependencies
poetry add new-package

# 2. Update lock file
poetry lock

# 3. Install
poetry install

# 4. Run QA pipeline
poetry run ruff format src/
poetry run ruff check src/ --fix
poetry run isort src/
poetry run mypy src/
poetry run pytest
```

All steps must pass for dependency update to be valid.

## Common Scenarios

### Scenario 1: Add New Feature Dependency

```bash
# Add package
poetry add transformers

# Verify installation
poetry show transformers

# Test import
poetry run python -c "import transformers; print('✓')"

# Run tests
poetry run pytest
```

### Scenario 2: Update Security Vulnerability

```bash
# Check for outdated packages
poetry show --outdated

# Update specific package
poetry update vulnerable-package

# Verify version
poetry show vulnerable-package

# Run full test suite
poetry run pytest
```

### Scenario 3: Add Type Stubs

```bash
# Add type stubs for third-party library
poetry add --group dev types-requests
poetry add --group dev types-PyYAML

# Verify mypy recognizes types
poetry run mypy src/
```

### Scenario 4: Remove Unused Dependency

```bash
# Check where package is used
grep -r "import package_name" src/

# If not used, remove
poetry remove package-name

# Verify still works
poetry run pytest
```

## Environment Management

### Create New Environment

```bash
# Remove existing environment
poetry env remove python

# Create new environment with specific Python version
poetry env use python3.11

# Install dependencies
poetry install
```

### Show Environment Information

```bash
# Show active environment
poetry env info

# Show environment path
poetry env info --path

# List all environments
poetry env list
```

### Sync Environment

```bash
# Synchronize environment with lock file
poetry install --sync

# This removes packages not in lock file
```

## Dependency Conflict Resolution

### Issue: Incompatible Version Constraints

```bash
# Check dependency tree for conflicts
poetry show --tree

# Try to resolve
poetry update

# If conflicts persist, update pyproject.toml constraints
# Example: Change "package-name = "^1.0.0"" to "package-name = "^1.5.0""
```

### Issue: Solver Takes Too Long

```bash
# Use experimental new installer
poetry config experimental.new-installer true

# Or update Poetry itself
poetry self update
```

### Issue: Platform-Specific Dependencies

```bash
# Add platform marker
poetry add "package-name; sys_platform == 'linux'"
poetry add "windows-package; sys_platform == 'win32'"
```

## Best Practices

### 1. Always Lock After Changes

```bash
# After any pyproject.toml changes
poetry lock
```

### 2. Commit Lock File

```bash
# Always commit poetry.lock
git add poetry.lock pyproject.toml
git commit -m "feat(deps): add new-package"
```

### 3. Use Version Constraints

```bash
# Prefer caret (^) for semantic versioning
poetry add "package-name^1.2.0"  # Allows 1.2.x and 1.y.z where y > 2

# Use tilde (~) for stricter control
poetry add "package-name~1.2.0"  # Only allows 1.2.x
```

### 4. Document Extra Requirements

When adding optional dependencies, document in CLAUDE.md:

```markdown
### New Extra: inference

```bash
poetry install -E inference
```

Includes: onnx, tensorrt, optimization tools
```

### 5. Test Across Environments

```bash
# Test CPU environment
poetry install -E cpu
poetry run pytest

# Test CUDA environment
poetry install -E cuda
poetry run pytest
```

## CI/CD Integration

### GitHub Actions Caching

The project uses Poetry caching in CI:

```yaml
- name: Cache Poetry
  uses: actions/cache@v3
  with:
    path: ~/.cache/pypoetry
    key: poetry-${{ hashFiles('poetry.lock') }}
```

Local changes should invalidate cache appropriately.

## Pre-commit Hook Integration

After dependency updates, update pre-commit hooks:

```bash
# Update pre-commit dependencies
poetry run pre-commit autoupdate

# Run on all files
poetry run pre-commit run --all-files
```

## Troubleshooting

### Issue: "No module named X" After Install

```bash
# Verify package is in lock file
grep "name = \"package-name\"" poetry.lock

# Reinstall
poetry install --sync

# Check environment
poetry run python -c "import sys; print(sys.path)"
```

### Issue: Poetry Command Not Found

```bash
# Install Poetry via pipx (recommended)
pipx install poetry

# Or via script
curl -sSL https://install.python-poetry.org | python3 -
```

### Issue: Lock File Out of Date

```bash
# Regenerate lock file
poetry lock --no-update

# Or update all dependencies
poetry update
```

## References

- **CLAUDE.md**: Package management rules (lines 28-45)
- **AGENTS.md**: Poetry-only policy (lines 33-50)
- Poetry documentation: https://python-poetry.org/docs/
- `pyproject.toml`: Dependency configuration
- `poetry.lock`: Locked dependency versions
