---
name: data-pipeline-validator
description: Validate data pipeline configuration including array_type parameter verification, HCPE and preprocessing data format validation, storage configuration checks, and schema compliance verification. Use when configuring data sources, validating pipeline setup, debugging data loading issues, or ensuring data type consistency.
allowed-tools: Read, Grep, Glob
---

# Data Pipeline Validator

Validates data pipeline configuration and data type compliance for the Maou project.

## Critical Requirements

### Array Type System

**CRITICAL**: Always specify `array_type` parameter when creating data sources.

Available types:
- `"hcpe"` - Game records in HCPE (Huffman Coded Position Evaluation) format
- `"preprocessing"` - Preprocessed training features

**Never omit this parameter** - it ensures correct schema validation and data loading.

## Data Source Validation

### File System Data Source

```python
from maou.infra.file_system.data_source import FileDataSource

# CORRECT: Explicit array_type
datasource = FileDataSource(
    file_paths=paths,
    array_type="hcpe"  # REQUIRED
)

# WRONG: Missing array_type
datasource = FileDataSource(
    file_paths=paths
)  # ERROR: Missing required parameter
```

### S3 Data Source

```python
from maou.infra.s3.data_source import S3DataSource

# CORRECT: Explicit array_type
datasource = S3DataSource(
    bucket_name="my-bucket",
    prefix="data/hcpe/",
    array_type="preprocessing"  # REQUIRED
)

# WRONG: Missing array_type
datasource = S3DataSource(
    bucket_name="my-bucket",
    prefix="data/hcpe/"
)  # ERROR: Missing required parameter
```

### GCS Data Source

```python
from maou.infra.gcs.data_source import GCSDataSource

# CORRECT: Explicit array_type
datasource = GCSDataSource(
    bucket_name="my-bucket",
    prefix="processed/",
    array_type="preprocessing"  # REQUIRED
)
```

## Schema Validation

### Centralized Schema Management

All data I/O must use domain layer schemas:

```python
from maou.domain.data.schema import get_hcpe_dtype, get_preprocessing_dtype
from maou.domain.data.io import save_hcpe_array, load_hcpe_array

# Get standardized data types
hcpe_dtype = get_hcpe_dtype()
preprocessing_dtype = get_preprocessing_dtype()

# High-performance I/O with validation
save_hcpe_array(array, "output.hcpe.npy", validate=True)
loaded_array = load_hcpe_array("input.hcpe.npy", validate=True)
```

### HCPE Format

HCPE format stores game records:

```python
hcpe_dtype = np.dtype([
    ('hcp', 'u1', (32,)),    # Huffman coded position
    ('eval', 'i2'),          # Evaluation score
    ('bestMove', 'u2'),      # Best move
    ('gameResult', 'u1'),    # Game result
])
```

**Usage**: Game record conversion, self-play data

### Preprocessing Format

Preprocessing format stores training features:

```python
preprocessing_dtype = np.dtype([
    ('features', 'f4', (119, 9, 9)),  # Board features
    ('policy', 'f4', (2187,)),        # Policy targets
    ('value', 'f4'),                  # Value target
])
```

**Usage**: Model training, evaluation

## Validation Methods

### Check for Missing array_type

```bash
# Search for DataSource instantiation without array_type
grep -rn "DataSource(" src/maou/ | grep -v "array_type"

# Should return: (empty - no violations)
```

### Check for Direct dtype Usage

```bash
# Search for direct numpy dtype creation (should use schema functions)
grep -rn "np\.dtype\(\[" src/maou/ | grep -v "test_" | grep -v "schema\.py"

# Should return: (empty - should use get_*_dtype() functions)
```

### Verify Schema Import Patterns

```bash
# Find files using data I/O
grep -l "save.*array\|load.*array" src/maou/app/*.py src/maou/app/**/*.py

# Verify they import from domain.data.schema
grep -l "from maou\.domain\.data\.schema import" src/maou/app/*.py
```

All files using data I/O should import from domain schema.

## Pipeline Configuration Validation

### PreprocessingConfig Validation

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PreprocessingConfig:
    input_path: Path
    output_path: Path
    batch_size: int
    num_workers: int
    array_type: str = "preprocessing"  # REQUIRED
```

**Check configuration**:

```bash
# Search for PreprocessingConfig usage
grep -rn "PreprocessingConfig" src/maou/

# Verify array_type is set
grep -A10 "PreprocessingConfig(" src/maou/ | grep "array_type"
```

### HcpeStorageConfig Validation

```python
@dataclass
class HcpeStorageConfig:
    bucket: str
    prefix: str
    region: str
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None
```

**Validation checklist**:
- [ ] Bucket name is valid
- [ ] Prefix follows convention
- [ ] Region is specified
- [ ] Credentials are configured
- [ ] Session token set if using temporary credentials

## Data Type Mismatch Detection

### Common Mismatches

**Issue**: Using HCPE format where preprocessing expected

```python
# WRONG: Array type mismatch
datasource = FileDataSource(
    file_paths=hcpe_files,
    array_type="preprocessing"  # Files are HCPE format!
)
```

**Fix**:
```python
# CORRECT: Match array_type to actual data
datasource = FileDataSource(
    file_paths=hcpe_files,
    array_type="hcpe"  # Matches file content
)
```

### Detect Type Mismatches

```bash
# Check file extensions vs array_type usage
find src/maou -name "*.py" -exec grep -H "\.hcpe\.npy" {} \; | while read line; do
    file=$(echo "$line" | cut -d: -f1)
    if grep -q 'array_type="preprocessing"' "$file"; then
        echo "WARNING: $file uses .hcpe.npy but specifies preprocessing type"
    fi
done
```

## Storage Configuration Validation

### S3 Configuration

Required parameters:
- `bucket_name`: S3 bucket
- `region`: AWS region (default: "us-east-1")
- `array_type`: Data format type

**Validate configuration**:

```python
# Check S3 bucket accessibility
import boto3

s3 = boto3.client('s3')
try:
    s3.head_bucket(Bucket='my-bucket')
    print("✓ Bucket accessible")
except Exception as e:
    print(f"✗ Bucket error: {e}")
```

### GCS Configuration

Required parameters:
- `bucket_name`: GCS bucket
- `project`: GCP project ID
- `array_type`: Data format type

**Validate configuration**:

```python
# Check GCS bucket accessibility
from google.cloud import storage

client = storage.Client()
try:
    bucket = client.get_bucket('my-bucket')
    print("✓ Bucket accessible")
except Exception as e:
    print(f"✗ Bucket error: {e}")
```

## Array Bundling Validation

### Bundle Configuration

When using array bundling:

```python
bundle_config = {
    'enable_bundling': True,
    'bundle_size_gb': 1.0,
    'cache_dir': './cache',
}
```

**Validation checklist**:
- [ ] `bundle_size_gb` is reasonable (0.5 - 2.0 GB)
- [ ] `cache_dir` exists and is writable
- [ ] Sufficient disk space available
- [ ] Cache directory not in `.gitignore` conflicts

### Verify Bundling Setup

```bash
# Check bundling configuration in CLI calls
grep -rn "enable-bundling" src/maou/infra/console/

# Verify bundle size parameter
grep -rn "bundle-size-gb" src/maou/infra/console/
```

## CLI Validation

### HCPE Conversion Command

```bash
poetry run maou hcpe-convert \
  --input-path /path/to/records \
  --input-format csa \
  --output-dir /path/to/output
```

**Validates**:
- Input format specification
- Output directory exists
- Correct file extensions

### Pre-processing Command

```bash
poetry run maou pre-process \
  --input-path /path/to/hcpe \
  --output-dir /path/to/processed
```

**Validates**:
- Input files are HCPE format
- Output format is preprocessing
- Array type transitions correctly

### Training Command

```bash
poetry run maou learn-model \
  --input-dir /path/to/processed \
  --gpu cuda:0
```

**Validates**:
- Input is preprocessing format
- Data loader configured correctly
- Array type matches expectation

## Common Validation Errors

### Error: "Unknown array type"

```python
# Cause: Invalid array_type value
datasource = FileDataSource(
    file_paths=paths,
    array_type="unknown"  # Not "hcpe" or "preprocessing"
)

# Fix: Use valid array type
datasource = FileDataSource(
    file_paths=paths,
    array_type="hcpe"  # Valid
)
```

### Error: "Schema mismatch"

```python
# Cause: File content doesn't match array_type
# File contains HCPE data, but specified preprocessing

# Fix: Verify file content
import numpy as np
data = np.load('file.npy')
print(data.dtype)  # Check actual dtype

# Use correct array_type
datasource = FileDataSource(
    file_paths=['file.npy'],
    array_type="hcpe"  # Match actual content
)
```

### Error: "Missing required field"

```python
# Cause: Data missing required fields for dtype

# Fix: Validate data schema
from maou.domain.data.schema import get_hcpe_dtype

expected_dtype = get_hcpe_dtype()
actual_dtype = loaded_data.dtype

if expected_dtype != actual_dtype:
    print(f"Schema mismatch!")
    print(f"Expected: {expected_dtype}")
    print(f"Actual: {actual_dtype}")
```

## Validation Checklist

Before running data pipeline:

- [ ] `array_type` specified for all data sources
- [ ] Schema functions imported from `maou.domain.data.schema`
- [ ] Data format matches array_type
- [ ] Storage configuration validated
- [ ] Credentials configured for cloud storage
- [ ] Cache directory exists (if using caching)
- [ ] Bundling configuration reasonable
- [ ] File paths correct and accessible

## Integration Tests

### Test Data Loading

```python
from maou.infra.file_system.data_source import FileDataSource

def test_hcpe_loading():
    """Test HCPE data loading with correct array_type."""
    datasource = FileDataSource(
        file_paths=['test.hcpe.npy'],
        array_type="hcpe"
    )

    data = datasource.load()
    assert data.dtype == get_hcpe_dtype()

def test_preprocessing_loading():
    """Test preprocessing data loading."""
    datasource = FileDataSource(
        file_paths=['test.prep.npy'],
        array_type="preprocessing"
    )

    data = datasource.load()
    assert data.dtype == get_preprocessing_dtype()
```

## Validation Report Format

```
Data Pipeline Validation Report
================================

Data Sources: 3 checked

Configuration:
- array_type specified: ✓ All sources
- Schema imports: ✓ All correct
- Storage config: ✓ Valid

Data Format Validation:
- HCPE files: 150 files
- Preprocessing files: 300 files
- Format mismatches: 0

Cloud Configuration:
- S3 buckets: ✓ Accessible
- GCS buckets: ✓ Accessible
- Credentials: ✓ Valid

Status: ✓ DATA PIPELINE VALIDATED
```

## When to Use

- Before starting data preprocessing
- When configuring new data sources
- After modifying data schemas
- When debugging data loading errors
- Before cloud storage operations
- During pipeline refactoring
- When adding new data formats

## References

- **CLAUDE.md**: Data I/O architecture (lines 244-276)
- **AGENTS.md**: Data pipeline configuration (lines 133-158)
- `src/maou/domain/data/schema.py`: Schema definitions
- `src/maou/domain/data/io.py`: I/O functions
