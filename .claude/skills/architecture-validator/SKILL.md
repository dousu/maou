---
name: architecture-validator
description: Validate Clean Architecture compliance including dependency flow verification (infra → interface → app → domain), module organization analysis, layer separation checks, and circular dependency detection. Use when reviewing code structure, analyzing imports, ensuring architectural principles, or validating refactoring changes.
allowed-tools: Read, Grep, Glob
---

# Architecture Validator

Validates adherence to Clean Architecture principles and dependency rules for the Maou project.

## Architecture Rules

The Maou project enforces strict dependency flow:

```
infra → interface → app → domain
```

**Critical Rule**: Dependencies must flow in ONE direction only. No circular dependencies allowed.

## Layer Definitions

### Domain Layer (`src/maou/domain/`)
**Purpose**: Business logic and entities
**Dependencies**: NONE - Domain is completely independent
**Contains**:
- Network models and architectures
- Loss functions
- Game format parsers (CSA, KIF)
- Data schemas and I/O operations
- Board representations

**Examples**:
- `maou/domain/model/resnet.py`
- `maou/domain/loss/policy_value_loss.py`
- `maou/domain/data/schema.py`

### App Layer (`src/maou/app/`)
**Purpose**: Use cases and workflows
**Dependencies**: Domain layer ONLY
**Contains**:
- HCPE conversion logic
- Model training pipeline
- Data preprocessing
- Inference workflows
- Benchmarking utilities

**Examples**:
- `maou/app/converter/`
- `maou/app/learning/training_loop.py`
- `maou/app/pre_process/`

### Interface Layer (`src/maou/interface/`)
**Purpose**: Adapters between app and infrastructure
**Dependencies**: App and Domain layers
**Contains**:
- CLI command adapters
- Data source abstractions
- Configuration adapters

**Examples**:
- `maou/interface/converter.py`
- `maou/interface/preprocess.py`

### Infrastructure Layer (`src/maou/infra/`)
**Purpose**: External system integrations
**Dependencies**: Can depend on all other layers
**Contains**:
- CLI commands (Click framework)
- Cloud storage (S3, GCS)
- File system operations
- Logging infrastructure

**Examples**:
- `maou/infra/console/app.py`
- `maou/infra/s3/`
- `maou/infra/gcs/`

## Validation Steps

### 1. Check Domain Layer Purity

Domain must have ZERO dependencies on other layers:

```bash
# Search for imports from app, interface, or infra
grep -r "from maou\.\(app\|interface\|infra\)" src/maou/domain/

# Should return: (empty - no results)
```

If results found: **ARCHITECTURE VIOLATION**

### 2. Check App Layer Dependencies

App should only import from domain:

```bash
# Search for imports from interface or infra
grep -r "from maou\.\(interface\|infra\)" src/maou/app/

# Should return: (empty - no results)
```

If results found: **ARCHITECTURE VIOLATION**

### 3. Verify Type Hints

All functions must have type annotations:

```bash
# Find functions without return type annotations
grep -E "def [a-z_]+\(" src/maou/domain/ | grep -v " ->" | grep -v "__" | grep -v "test_"

# Should return: (empty - no results)
```

If results found: **TYPE SAFETY VIOLATION**

### 4. Check Import Statements

Analyze import patterns in specific files:

```bash
# Example: Check a specific file
grep -n "^import\|^from" src/maou/app/learning/training_loop.py | grep "from maou"
```

Verify imports respect layer boundaries.

### 5. Validate Data I/O Architecture

Check for proper use of centralized schema management:

```bash
# Should use get_hcpe_dtype() or get_preprocessing_dtype()
grep -r "dtype.*=" src/maou/app/ | grep -v "get_.*_dtype"

# Check for array_type parameter usage
grep -r "DataSource" src/maou/ | grep -v "array_type"
```

## Common Violations

### ❌ WRONG: Domain importing from App

```python
# In src/maou/domain/model/resnet.py
from maou.app.learning.training_loop import TrainingLoop  # VIOLATION!
```

**Fix**: Move shared logic to domain layer.

### ❌ WRONG: App importing from Infrastructure

```python
# In src/maou/app/converter/hcpe_converter.py
from maou.infra.s3.client import S3Client  # VIOLATION!
```

**Fix**: Use dependency injection through interface layer.

### ✓ CORRECT: App depending only on Domain

```python
# In src/maou/app/learning/training_loop.py
from maou.domain.data.schema import get_hcpe_dtype
from maou.domain.model.resnet import create_resnet_model
from maou.domain.loss.policy_value_loss import PolicyValueLoss
```

### ✓ CORRECT: Infrastructure depending on App

```python
# In src/maou/infra/console/hcpe_convert.py
from maou.app.converter.hcpe_converter import convert_hcpe
from maou.interface.converter import ConverterInterface
```

## Data I/O Architecture Validation

### Centralized Schema Management

All data I/O MUST use domain layer schemas:

```python
from maou.domain.data.schema import get_hcpe_dtype, get_preprocessing_dtype
from maou.domain.data.io import save_hcpe_array, load_hcpe_array

# CORRECT usage
hcpe_dtype = get_hcpe_dtype()
preprocessing_dtype = get_preprocessing_dtype()

save_hcpe_array(array, "output.hcpe.npy", validate=True)
loaded_array = load_hcpe_array("input.hcpe.npy", validate=True)
```

### Explicit Array Type System

**CRITICAL**: Always specify `array_type` parameter:

```python
# File system data source
datasource = FileDataSource(
    file_paths=paths,
    array_type="hcpe"  # REQUIRED
)

# S3 data source
datasource = S3DataSource(
    bucket_name="my-bucket",
    array_type="preprocessing"  # REQUIRED
)
```

Available types: `"hcpe"` (game records), `"preprocessing"` (training features)

## Validation Report Format

When validating, provide a report:

```
Architecture Validation Report
==============================

Domain Layer: ✓ PASS
- No external dependencies detected
- Type hints present on all public functions
- Docstrings complete

App Layer: ✓ PASS
- Only imports from domain layer
- Use cases properly isolated

Interface Layer: ✓ PASS
- Adapters correctly bridge app and infra

Infrastructure Layer: ✓ PASS
- External integrations isolated

Overall: ✓ ARCHITECTURE COMPLIANT
```

## When to Use

- Before code review
- After refactoring
- When adding new modules
- During architecture discussions
- Before merging pull requests
- When resolving merge conflicts
- After dependency changes

## Integration with Other Skills

Combine with:
- **qa-pipeline-automation** - Run before QA checks
- **pr-preparation-checks** - Include in PR validation
- **type-safety-enforcer** - Complementary type validation

## References

- **CLAUDE.md**: Clean Architecture principles (lines 233-241)
- **AGENTS.md**: Architecture compliance rules (lines 56-66)
- `.codex/config.yaml`: Enforced dependency patterns
