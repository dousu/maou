# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Guidelines

When working on the Maou project, please follow these core principles:

1. **Clean Architecture**: Strictly maintain dependency flow: `infra → interface → app → domain`
2. **Type Safety**: Type hints are required for all code. No exceptions.
3. **Code Quality**: Follow established patterns and maintain consistency.
4. **Testing**: New features require tests. Bug fixes need regression tests.
5. **Performance**: Optimize for S3 operations with parallel processing where applicable.
6. **Documentation**: Public APIs must have docstrings.

## Project Overview

Maou (魔王) is a Shogi (Japanese chess) AI project implemented in Python following Clean Architecture principles. The name "maou" translates to "demon king" in Japanese.

### Core Components
- **Domain Layer**: Business logic and entities (network models, loss functions, parsers)
- **App Layer**: Use cases (converter, learning, pre-processing)
- **Interface Layer**: Adapters between app and infrastructure
- **Infrastructure Layer**: External systems (cloud storage, databases, logging)

## Core Development Rules

### Package Management

**ONLY use Poetry. NEVER use pip directly.**

```bash
# Install dependencies
poetry install

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Remove dependency
poetry remove package-name

# Update dependencies
poetry update

# Run commands in poetry environment
poetry run command
```

**Environment-specific installations:**
```bash
# CPU + GCP environment
poetry install -E cpu -E gcp

# CUDA + AWS environment
poetry install -E cuda -E aws

# TPU + GCP environment
poetry install -E tpu -E gcp
```

### Code Quality Standards

#### Required Standards
- **Type hints**: Required for all functions, methods, and class attributes
- **Docstrings**: Required for all public APIs
- **Line length**: 88 characters maximum
- **Function size**: Functions must be focused and small
- **Architecture**: Follow Clean Architecture dependency rules

#### Code Organization
```python
# Good: Clear type hints and docstring
def process_game_record(
    input_path: Path,
    format_type: str,
    min_rating: Optional[int] = None
) -> ProcessingResult:
    """
    Process Shogi game record and convert to HCPE format.

    Args:
        input_path: Path to input game record file
        format_type: Format of input file ('csa' or 'kif')
        min_rating: Minimum rating threshold for filtering

    Returns:
        ProcessingResult containing conversion statistics
    """
    # Implementation here
```

#### Architecture Compliance
- **Domain layer**: No dependencies on other layers
- **App layer**: Only depends on domain layer
- **Interface layer**: Adapts between app and infrastructure
- **Infrastructure layer**: Implements external system integrations

### Testing Requirements

**Framework**: Use pytest with the following patterns:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/path/to/test_file.py

# Run with coverage
pytest --cov=src/maou

# Run with cloud features
TEST_GCP=true pytest
TEST_AWS=true pytest
```

#### Test Requirements
- **New features**: Must include comprehensive tests
- **Bug fixes**: Must include regression tests
- **Edge cases**: Test error scenarios and boundary conditions
- **Integration tests**: Test cloud provider integrations when applicable

#### Test Structure
```python
def test_feature_with_valid_input():
    """Test normal operation with valid input."""
    # Test implementation

def test_feature_with_invalid_input():
    """Test error handling with invalid input."""
    # Test error scenarios

def test_feature_edge_cases():
    """Test boundary conditions and edge cases."""
    # Test edge cases
```

## Python Tools

### Essential Commands

#### Type Checking
```bash
# Run type checking (required before commits)
poetry run mypy src/

# Strict type checking
poetry run mypy src/ --strict
```

#### Code Formatting
```bash
# Format code with Ruff
poetry run ruff format src/

# Check for issues
poetry run ruff check src/

# Auto-fix issues
poetry run ruff check src/ --fix

# Sort imports
poetry run isort src/
```

#### Pre-commit Hooks
```bash
# Install pre-commit hooks
poetry run bash scripts/pre-commit.sh

# Run all pre-commit hooks manually
pre-commit run --all-files
```

#### Linting
```bash
# Run flake8
poetry run flake8 src/

# Run all linters in sequence
poetry run ruff check src/ --fix && poetry run mypy src/ && poetry run flake8 src/
```

## Code Formatting

### Critical Requirements

1. **Line Length**: 88 characters maximum
2. **Type Hints**: Required for all code
3. **Import Organization**: Use isort for consistent import ordering
4. **Code Style**: Follow Ruff formatting standards

### Formatting Commands

```bash
# Complete formatting pipeline (run before commits)
poetry run ruff format src/
poetry run ruff check src/ --fix
poetry run isort src/
poetry run mypy src/
```

### Common Formatting Issues

#### Line Length Violations
```python
# Bad: Line too long
result = some_very_long_function_name(very_long_parameter_name, another_long_parameter, yet_another_parameter)

# Good: Multi-line formatting
result = some_very_long_function_name(
    very_long_parameter_name,
    another_long_parameter,
    yet_another_parameter
)
```

#### Missing Type Hints
```python
# Bad: No type hints
def process_data(data, options):
    return processed_data

# Good: Complete type hints
def process_data(data: List[Dict[str, Any]], options: ProcessingOptions) -> ProcessedData:
    return processed_data
```

## Environment Setup

### Initial Setup

```bash
# Initialize development environment
bash scripts/dev-init.sh

# Get poetry environment path (for VSCode interpreter)
poetry env info --path

# Set up pre-commit hooks
poetry run bash scripts/pre-commit.sh
```

### Cloud Authentication

#### Google Cloud Platform (GCP)
```bash
# Authenticate with GCP
gcloud auth application-default login
gcloud config set project "your-project-id"
gcloud auth application-default set-quota-project "your-project-id"
```

#### Amazon Web Services (AWS)
```bash
# Configure AWS SSO
aws configure sso --use-device-code --profile default

# Renew expired token
aws sso login --use-device-code --profile default
```

## Testing

### Basic Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test patterns
pytest -k "test_converter"
```

### Cloud Testing

```bash
# Test GCP features (requires authentication)
TEST_GCP=true pytest

# Test AWS features (requires authentication)
TEST_AWS=true pytest

# Test specific cloud integration
pytest tests/integrations/test_s3_integration.py
```

### Test Development Guidelines

- **Isolation**: Each test should be independent
- **Clarity**: Test names should clearly describe what is being tested
- **Coverage**: Aim for comprehensive test coverage of business logic
- **Performance**: Integration tests should complete within reasonable time

## CLI Commands

The project provides three main CLI commands following the data pipeline:

### 1. Game Record Conversion

```bash
# Convert Shogi game records to HCPE format
poetry run maou hcpe-convert \
  --input-path /path/to/records \
  --input-format csa \
  --output-dir /path/to/output

# With quality filtering
poetry run maou hcpe-convert \
  --input-path /path/to/records \
  --input-format csa \
  --output-dir /path/to/output \
  --min-rating 1500 \
  --min-moves 50
```

### 2. Data Pre-processing

```bash
# Pre-process HCPE data for training
poetry run maou pre-process \
  --input-path /path/to/hcpe \
  --output-dir /path/to/processed

# With S3 integration
poetry run maou pre-process \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-prefix data \
  --input-data-name hcpe \
  --input-local-cache-dir ./cache \
  --output-dir /path/to/processed
```

### 3. Model Training

```bash
# Train neural network model
poetry run maou learn-model \
  --input-dir /path/to/processed \
  --gpu cuda:0 \
  --epoch 10 \
  --batch-size 256

# With cloud storage for checkpoints
poetry run maou learn-model \
  --input-dir /path/to/processed \
  --gpu cuda:0 \
  --epoch 10 \
  --batch-size 256 \
  --output-s3 \
  --s3-bucket-name my-bucket \
  --s3-base-path models/
```

### Performance Optimization

#### S3 Parallel Processing

For AWS S3 operations, use `--max-workers` to control parallel threads:

```bash
# Upload optimization (hcpe-convert)
poetry run maou hcpe-convert \
  --output-s3 \
  --bucket-name my-bucket \
  --prefix data \
  --data-name features \
  --max-workers 8

# Download optimization (pre-process)
poetry run maou pre-process \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-prefix data \
  --input-data-name hcpe \
  --input-local-cache-dir ./cache \
  --max-workers 16
```

**Performance improvements:**
- **Upload speed**: 4-8x faster with parallel processing
- **Download speed**: 8x faster for large datasets
- **Memory usage**: ~90% reduction (optimized buffering)
- **Large dataset handling**: 100,000 files download time reduced from 20+ minutes to 3-5 minutes

## Architecture

The project follows Clean Architecture principles with strict dependency rules:

### Layer Structure

1. **Domain Layer** (`src/maou/domain/`): Core business logic and entities
   - Network models (ResNet implementation)
   - Loss functions and training logic
   - Shogi game parsers (CSA, KIF formats)
   - Pure business rules with no external dependencies

2. **App Layer** (`src/maou/app/`): Use case implementations
   - Converter: Game record conversion workflows
   - Learning: Neural network training orchestration
   - Pre-processing: Feature extraction pipelines

3. **Interface Layer** (`src/maou/interface/`): Adapters and converters
   - Converts between domain objects and infrastructure representations
   - Protocol definitions for external integrations

4. **Infrastructure Layer** (`src/maou/infra/`): External system integrations
   - Cloud storage (Google Cloud Storage, Amazon S3)
   - Database systems (BigQuery)
   - Console application and CLI
   - Logging and monitoring

### Dependency Rules

**Critical**: Dependencies must flow in one direction only:
```
infra → interface → app → domain
```

- **Domain layer**: Cannot import from any other layer
- **App layer**: Can only import from domain layer
- **Interface layer**: Can import from app and domain layers
- **Infrastructure layer**: Can import from all other layers

## Data Pipeline

### 1. Game Record Conversion (`hcpe_convert`)
- **Input**: Shogi game records in CSA or KIF format
- **Process**: Parse, validate, and filter games based on quality metrics
- **Output**: HCPE (HuffmanCodedPosAndEval) format for training
- **Storage**: Local filesystem, BigQuery, or Amazon S3

### 2. Feature Extraction (`pre_process`)
- **Input**: HCPE format game data
- **Process**: Transform raw game data into neural network input features
- **Output**: Labeled training datasets with board positions and evaluations
- **Optimization**: Memory-efficient processing with streaming

### 3. Model Training (`learn_model`)
- **Input**: Processed feature datasets
- **Process**: Train ResNet-based neural network with PyTorch
- **Output**: Trained model checkpoints and logs
- **Hardware Support**: CPU, CUDA, Apple Silicon (MPS), Google TPU
- **Monitoring**: TensorBoard integration for training visualization

## Storage Options

### Local Filesystem
- **Default**: All operations support local file storage
- **Use case**: Development, small datasets, local testing

### Google Cloud Platform
- **BigQuery**: Structured data storage for HCPE records (requires `-E gcp`)
- **Cloud Storage**: Model checkpoints and large datasets (requires `-E gcp`)
- **Authentication**: `gcloud auth application-default login`

### Amazon Web Services
- **S3**: Object storage for all data types (requires `-E aws`)
- **Parallel processing**: Optimized upload/download with configurable workers
- **Authentication**: AWS SSO or IAM credentials

### Performance Considerations
- **Local**: Fastest for development and small datasets
- **Cloud**: Better for large-scale training and data sharing
- **Hybrid**: Use local cache with cloud storage for optimal performance

## Debugging and Logging

### Log Level Control

Configure logging through the `MAOU_LOG_LEVEL` environment variable:

```bash
# Production logging (INFO level)
export MAOU_LOG_LEVEL=INFO
poetry run maou hcpe-convert --input-path /path/to/records --input-format csa --output-dir /path/to/output

# Debug logging (detailed information)
export MAOU_LOG_LEVEL=DEBUG
poetry run maou pre-process --input-path /path/to/hcpe --output-dir /path/to/processed

# Minimal logging (warnings and errors only)
export MAOU_LOG_LEVEL=WARNING
poetry run maou learn-model --input-dir /path/to/processed --gpu cuda:0 --epoch 10 --batch-size 256
```

**Available log levels:**
- `DEBUG`: Detailed debugging information
- `INFO`: General information messages (default)
- `WARNING`: Warning messages and above
- `ERROR`: Error messages and above
- `CRITICAL`: Only critical error messages

**Priority order:**
1. `--debug-mode` flag: Forces DEBUG level
2. `MAOU_LOG_LEVEL` environment variable: Custom level
3. Default: INFO level

### Debug Mode

Force debug logging with the CLI flag:

```bash
# Enable debug mode (equivalent to MAOU_LOG_LEVEL=DEBUG)
poetry run maou --debug-mode hcpe-convert \
  --input-path /path/to/records \
  --input-format csa \
  --output-dir /path/to/output
```

### Performance Monitoring

Monitor application performance during execution:

```bash
# Monitor S3 transfer performance
MAOU_LOG_LEVEL=DEBUG poetry run maou hcpe-convert \
  --output-s3 \
  --bucket-name my-bucket \
  --prefix data \
  --data-name features \
  --max-workers 8

# Monitor memory usage during processing
MAOU_LOG_LEVEL=INFO poetry run maou pre-process \
  --input-path /path/to/hcpe \
  --output-dir /path/to/processed
```

## Error Resolution

### CI Failure Resolution Order

When tests fail in CI, resolve issues in this order:

1. **Code Formatting**
   ```bash
   poetry run ruff format src/
   poetry run ruff check src/ --fix
   poetry run isort src/
   ```

2. **Type Errors**
   ```bash
   poetry run mypy src/
   # Fix any type-related issues
   ```

3. **Linting Issues**
   ```bash
   poetry run flake8 src/
   # Address any remaining linting issues
   ```

4. **Test Failures**
   ```bash
   pytest --tb=short
   # Fix failing tests
   ```

### Common Issues and Solutions

#### Type Checking Failures
```python
# Problem: Missing Optional type
def process_data(data: str, options: dict):  # Missing Optional

# Solution: Add proper Optional typing
def process_data(data: str, options: Optional[Dict[str, Any]] = None):
```

#### Import Errors
```bash
# Problem: Missing dependencies
ImportError: No module named 'torch'

# Solution: Install with appropriate extras
poetry install -E cuda  # for CUDA support
poetry install -E cpu   # for CPU-only
```

#### Cloud Authentication Errors
```bash
# Problem: GCP authentication failure
# Solution: Re-authenticate
gcloud auth application-default login

# Problem: AWS credentials expired
# Solution: Refresh SSO login
aws sso login --use-device-code --profile default
```

#### Memory Issues During Processing
```python
# Problem: Out of memory with large datasets
# Solution: Reduce batch size or enable streaming
poetry run maou pre-process \
  --input-path /path/to/large/dataset \
  --output-dir /path/to/output \
  --max-cached-bytes 100000000  # Reduce cache size
```

### Development Environment Issues

#### Poetry Environment Problems
```bash
# Reset poetry environment
poetry env remove python
poetry install

# Verify environment
poetry env info
poetry run python --version
```

#### Pre-commit Hook Failures
```bash
# Reinstall pre-commit hooks
poetry run bash scripts/pre-commit.sh

# Run hooks manually to debug
pre-commit run --all-files --verbose
```

## Commit Guidelines

### Commit Message Standards

Write clear, descriptive commit messages that explain the purpose of changes:

```bash
# Good commit messages
git commit -m "feat(converter): add parallel processing for CSA file parsing"
git commit -m "fix(s3): resolve memory leak in large dataset downloads"
git commit -m "docs(api): add docstrings to neural network training functions"
```

### Commit Message Format

Use conventional commit format with appropriate types:

- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without functionality changes
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### Commit Trailers

For specific cases, add trailers to commits:

```bash
# For bug reports from users
git commit --trailer "Reported-by: User Name <email@example.com>"

# For GitHub issues
git commit --trailer "Github-Issue: #123"

# For external contributions
git commit --trailer "Suggested-by: Contributor Name"
```

### Pre-commit Quality Checks

**ALWAYS run quality checks before committing:**

```bash
# Complete pre-commit pipeline
poetry run ruff format src/
poetry run ruff check src/ --fix
poetry run isort src/
poetry run mypy src/
pytest

# Or use pre-commit hooks
pre-commit run --all-files
```

### Atomic Commits

- **One logical change per commit**: Each commit should represent a single, complete change
- **Build-passing commits**: Every commit should pass all tests and quality checks
- **Self-contained**: Commits should not depend on future commits to be functional

## Pull Requests

### Mandatory Requirements

#### 1. Quality Assurance

**BEFORE creating a pull request, ensure all quality checks pass:**

```bash
# Complete quality verification
poetry run ruff format src/
poetry run ruff check src/ --fix
poetry run isort src/
poetry run mypy src/
pytest
pre-commit run --all-files
```

#### 2. Detailed Pull Request Description

**REQUIRED**: Create a detailed message describing:

1. **High-level problem description**: What issue does this solve?
2. **Solution approach**: How is the problem addressed?
3. **Architectural impact**: Any changes to Clean Architecture layers
4. **Performance considerations**: Especially for S3 operations
5. **Testing coverage**: What tests were added or modified

**Example PR Description:**
```markdown
## Problem
The current HCPE converter processes files sequentially, causing
performance bottlenecks when handling large datasets (1000+ files).

## Solution
Implemented parallel processing using ThreadPoolExecutor with
configurable worker count, maintaining Clean Architecture by:
- Adding concurrency logic to App layer
- Keeping domain logic pure and stateless
- Using dependency injection for worker configuration

## Performance Impact
- Processing time reduced from 45 minutes to 8 minutes for 10,000 files
- Memory usage remains constant with streaming approach
- S3 upload utilizes new parallel worker configuration

## Testing
- Added unit tests for parallel processing logic
- Integration tests with various file sizes
- Performance benchmarks included in test suite
```

#### 3. Code Review Assignment

**REQUIRED**: Add appropriate reviewers based on change scope:

- **Architecture changes**: Core maintainers
- **Performance optimizations**: Lead developers with cloud expertise
- **Domain logic changes**: Shogi AI specialists
- **Infrastructure changes**: DevOps team members

#### 4. Strict Prohibitions

**NEVER include in PRs or commits:**
- ❌ `Co-authored-by` trailers or similar
- ❌ References to AI tools used for code generation
- ❌ Generic commit messages like "fix stuff" or "update code"
- ❌ Multiple unrelated changes in single PR
- ❌ Commits that break existing tests

### Pull Request Checklist

Before submitting, verify:

- [ ] All quality checks pass (`poetry run mypy src/ && pytest`)
- [ ] New features include comprehensive tests
- [ ] Public APIs have complete docstrings
- [ ] Clean Architecture principles are maintained
- [ ] Performance considerations are documented
- [ ] Breaking changes are clearly marked
- [ ] Documentation is updated if needed

### Code Review Standards

#### Architecture Review
- **Dependency flow**: Verify `infra → interface → app → domain`
- **Layer boundaries**: No direct dependencies between non-adjacent layers
- **Domain purity**: Domain layer has no external dependencies
- **Interface contracts**: Proper abstraction between layers

#### Performance Review
- **S3 operations**: Use parallel processing with appropriate worker counts
- **Memory usage**: Implement streaming for large datasets
- **Caching**: Utilize local cache for cloud data appropriately
- **Batch processing**: Optimize for training pipeline efficiency

#### Quality Review
- **Type safety**: All code has proper type hints
- **Error handling**: Graceful error handling with informative messages
- **Testing**: Edge cases and error scenarios are covered
- **Documentation**: Clear docstrings for public APIs

#### Security Review
- **Credential handling**: No secrets in code or logs
- **Input validation**: Proper validation of external inputs
- **Authentication**: Cloud authentication is properly handled
- **Data privacy**: No sensitive data in logs or error messages

### Post-Review Process

After review approval:

1. **Squash commits** if multiple commits address the same logical change
2. **Rebase** onto latest main branch
3. **Final quality check** after any merge conflicts
4. **Merge** using appropriate strategy (squash for feature branches)

### Emergency Hotfixes

For critical production issues:

1. **Create hotfix branch** from main
2. **Minimal changes** to address specific issue only
3. **Expedited review** with single reviewer approval
4. **Immediate testing** in staging environment
5. **Post-merge monitoring** for any side effects

## Best Practices

### Development Workflow

1. **Start with Tests**: Write tests before implementing features
2. **Small Commits**: Make focused, atomic commits
3. **Type Safety**: Add type hints as you write code, not after
4. **Architecture**: Respect layer boundaries and dependency rules
5. **Performance**: Profile before optimizing, especially for cloud operations

### Code Organization

- **Single Responsibility**: Each class and function should have one clear purpose
- **Dependency Injection**: Use dependency injection for external services
- **Error Handling**: Handle errors gracefully with informative messages
- **Logging**: Use appropriate log levels for different types of information

### Cloud Integration

- **Efficiency**: Use parallel processing for S3 operations
- **Caching**: Implement local caching for cloud data when appropriate
- **Authentication**: Test authentication before long-running operations
- **Error Recovery**: Implement retry logic for transient cloud failures

Run `poetry run maou --help` for detailed CLI options and examples.
