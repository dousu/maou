---
name: type-safety-enforcer
description: Enforce type safety requirements by running mypy type checker, detecting missing type hints, verifying docstring presence on public APIs, and identifying type annotation gaps. Use when validating type coverage, ensuring type safety compliance, checking documentation completeness, or preparing code for strict type checking.
---

# Type Safety Enforcer

Enforces strict type safety requirements and documentation standards for the Maou project.

## Core Requirements

**Type hints**: Required for ALL functions, methods, and class attributes
**Docstrings**: Required for ALL public APIs
**Line length**: 88 characters maximum
**No exceptions**: Type safety is non-negotiable

## Type Checking with mypy

### Full Type Check

Run mypy on entire codebase:

```bash
uv run mypy src/
```

### Per-Module Type Check

Check specific modules:

```bash
uv run mypy src/maou/domain/
uv run mypy src/maou/app/learning/
uv run mypy src/maou/infra/console/
```

### Strict Mode Configuration

The project uses strict mypy configuration (see `pyproject.toml`):

```toml
[tool.mypy]
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
```

## Detecting Missing Type Hints

### Find Functions Without Return Type Annotations

```bash
# Search for functions missing return type hints
grep -rn "def [a-z_]" src/maou/ | grep -v " ->" | grep -v "__" | grep -v "test_" | head -20

# Should return: (empty - no results)
```

### Find Methods Without Parameter Type Hints

```bash
# Search for parameters without types
grep -rn "def .*(.*[a-z_][a-z_]*[,)]" src/maou/domain/ | grep -v ": " | head -20
```

### Check Class Attributes

```bash
# Find class attributes without type annotations
grep -rn "^    [a-z_].*= " src/maou/domain/ | grep -v ": " | head -20
```

## Verifying Docstrings

### Check Public Functions Have Docstrings

```bash
# Find public functions without docstrings
grep -A3 "^def [a-z]" src/maou/domain/ | grep -B3 "def " | grep -v '"""' | grep "def "

# Should return: (empty - no results)
```

### Check Public Classes Have Docstrings

```bash
# Find public classes without docstrings
grep -A3 "^class [A-Z]" src/maou/domain/ | grep -B3 "class " | grep -v '"""' | grep "class "

# Should return: (empty - no results)
```

### Docstring Format Validation

Docstrings should follow this format:

```python
def convert_hcpe(input_path: Path, output_path: Path) -> int:
    """
    将棋の棋譜データを処理し，HCPE形式に変換する．

    Args:
        input_path: 入力ファイルのパス
        output_path: 出力ファイルのパス

    Returns:
        変換されたレコードの数

    Raises:
        ValueError: 入力形式が不正な場合
    """
```

**Note**: Japanese docstrings use 全角コンマ（，）and 全角ピリオド（．）

## Common Type Hint Patterns

### Function Type Hints

```python
# CORRECT: Full type annotations
def calculate_loss(
    policy_output: torch.Tensor,
    value_output: torch.Tensor,
    targets: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calculate policy and value loss."""
    ...
```

```python
# WRONG: Missing return type
def calculate_loss(policy_output, value_output, targets):  # Missing types!
    ...
```

### Class Attribute Type Hints

```python
# CORRECT: Typed class attributes
class TrainingConfig:
    """Training configuration."""

    batch_size: int
    learning_rate: float
    epochs: int
    device: str

    def __init__(self, batch_size: int = 256) -> None:
        self.batch_size = batch_size
```

### Optional and Union Types

```python
from typing import Optional, Union

# CORRECT: Proper Optional usage
def load_model(path: Path, device: Optional[str] = None) -> torch.nn.Module:
    """Load model from path."""
    ...

# CORRECT: Union types for multiple possibilities
def process_input(data: Union[Path, str, list[Path]]) -> list[Path]:
    """Process various input formats."""
    ...
```

### Generic Types

```python
from typing import TypeVar, Generic

T = TypeVar('T')

# CORRECT: Generic class with type parameter
class DataSource(Generic[T]):
    """Generic data source."""

    def load(self) -> list[T]:
        """Load data items."""
        ...
```

## Type Checking Error Resolution

### Common mypy Errors

**1. Missing return type**
```python
# Error: Function is missing a return type annotation
def process() -> None:  # Add return type
    ...
```

**2. Untyped function definition**
```python
# Error: Function is missing a type annotation for one or more arguments
def convert(input: Path, output: Path) -> int:  # Add parameter types
    ...
```

**3. Incompatible return value**
```python
# Error: Incompatible return value type (got "None", expected "int")
def count() -> int:
    return 0  # Fix: return proper type
```

**4. Need type annotation**
```python
# Error: Need type annotation for variable
data: list[str] = []  # Add type annotation
```

## Integration with QA Pipeline

Type checking is part of the QA pipeline:

```bash
# Full QA pipeline includes mypy
uv run ruff format src/
uv run ruff check src/ --fix
uv run isort src/
uv run mypy src/  # Type checking step
uv run pytest
```

## Continuous Type Safety

### Pre-commit Hook

Type checking runs automatically on commit:

```bash
# Install pre-commit hooks
uv run bash scripts/pre-commit.sh

# Hooks will run mypy on staged files
```

### CI/CD Integration

GitHub Actions run mypy on every push and PR.

Local validation prevents CI failures:

```bash
# Run before pushing
uv run mypy src/
```

## Type Stub Files

For third-party libraries without type stubs:

```bash
# Install type stubs
uv add --dev types-requests
uv add --dev types-PyYAML
```

Project already has stubs for:
- numpy (via numpy)
- torch (via torch)
- click (via click)

## Success Criteria

Type safety enforcement passes when:

- ✓ `uv run mypy src/` exits with 0 errors
- ✓ All public functions have type hints
- ✓ All public APIs have docstrings
- ✓ No `# type: ignore` comments (except for justified cases)
- ✓ Generic types properly parameterized

## When to Use

- Before committing code
- After adding new functions or classes
- When refactoring
- During code review
- Before pull requests
- After dependency updates
- When fixing type errors

## References

- **CLAUDE.md**: Type safety requirements (line 10, lines 49-50)
- **AGENTS.md**: Type hints required everywhere (line 14)
- `pyproject.toml`: mypy strict configuration
- Python typing documentation: https://docs.python.org/3/library/typing.html
