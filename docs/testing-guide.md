# Testing Guide

## Testing Requirements

**Framework**: Use `poetry run pytest`

```bash
poetry run pytest                           # Run all tests
poetry run pytest --cov=src/maou            # Run with coverage
TEST_GCP=true poetry run pytest             # Test GCP features
TEST_AWS=true poetry run pytest             # Test AWS features
```

### Test Requirements
- **New features**: Must include comprehensive tests
- **Bug fixes**: Must include regression tests
- **Edge cases**: Test error scenarios and boundary conditions
- **Integration tests**: Test cloud provider integrations when applicable

## Test Organization and Conventions

**CRITICAL:** Tests must strictly mirror the source directory structure following Clean Architecture layers.

### Directory Structure Rules

**Pattern:**
```
src/maou/{layer}/{module}/file.py
  → tests/maou/{layer}/{module}/test_file.py
```

**Layer Mapping:**
- `src/maou/domain/` → `tests/maou/domain/`
- `src/maou/app/` → `tests/maou/app/`
- `src/maou/interface/` → `tests/maou/interface/`
- `src/maou/infra/` → `tests/maou/infra/`

**Examples:**
```
src/maou/domain/board/shogi.py
  → tests/maou/domain/board/test_shogi.py

src/maou/app/learning/training_loop.py
  → tests/maou/app/learning/test_training_loop.py

src/maou/infra/s3/s3_data_source.py
  → tests/maou/infra/s3/test_s3_data_source.py
```

### Test File Naming Conventions

**Required format:** `test_{module_name}.py`

**Rules:**
1. **Prefix:** Always start with `test_` for pytest discovery
2. **Module name:** Match the source file name exactly
3. **Descriptive suffixes:** Add clarifying suffixes when testing specific aspects

**Examples:**
- ✅ `test_s3_data_source.py` - Primary S3 DataSource tests
- ✅ `test_file_data_source_stage_support.py` - Stage-specific feature tests
- ❌ `test_validation.py` - Too generic
- ❌ `tests/maou/app/test_training_loop.py` - Wrong location (should be in app/learning/)

### Special Test Directories

**Integration Tests:** `tests/maou/integrations/`
- Purpose: End-to-end tests spanning multiple layers
- Naming: `test_{workflow}_{scenario}.py`
- Examples: `test_app_hcpe_converter.py`, `test_convert_and_preprocess.py`

**Benchmark Tests:** `tests/benchmarks/`
- Purpose: Performance validation and regression detection
- Naming: `test_{component}_performance.py`
- Run explicitly: `poetry run pytest tests/benchmarks/ -v -s`

### Test Resource Files

**Rule:** Co-locate test resources with the test files that use them.

**Structure:**
```
tests/maou/{layer}/{module}/
├── test_feature.py
└── resources/
    ├── sample_input.csa
    └── expected_output.feather
```

### Creating New Tests

**Workflow:**
1. Identify source file: `src/maou/{layer}/{module}/feature.py`
2. Create test file: `tests/maou/{layer}/{module}/test_feature.py`
3. Add test class: `class TestFeatureName:` (optional but recommended)
4. Add test functions: `def test_{specific_behavior}() -> None:`
5. Add resources: Create `resources/` directory if needed

**Template:**
```python
"""Tests for {layer}.{module}.{feature} module."""

from pathlib import Path

import pytest

from maou.{layer}.{module}.{feature} import FeatureClass


class TestFeatureClass:
    """Test suite for FeatureClass."""

    def test_{specific_behavior}(self) -> None:
        """Test that {specific behavior} works correctly."""
        # Arrange
        instance = FeatureClass()

        # Act
        result = instance.method()

        # Assert
        assert result == expected_value
```

### Running Tests by Layer

```bash
# All tests
poetry run pytest

# Specific layer
poetry run pytest tests/maou/domain/
poetry run pytest tests/maou/app/
poetry run pytest tests/maou/infra/

# Specific module
poetry run pytest tests/maou/app/learning/
poetry run pytest tests/maou/domain/board/

# Integration tests only
poetry run pytest tests/maou/integrations/

# With coverage
poetry run pytest --cov=src/maou --cov-report=html
```
