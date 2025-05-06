# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Maou (魔王) is a Shogi (Japanese chess) AI project implemented in Python following Clean Architecture principles. The name "maou" translates to "demon king" in Japanese.

## Key Commands

### Environment Setup

```bash
# Initialize development environment
bash scripts/dev-init.sh

# Get poetry environment path to set as VSCode interpreter
poetry env info --path

# Set up pre-commit hooks
poetry run bash scripts/pre-commit.sh
```

### Installing Dependencies

The project uses Poetry with "extras" for different environments:

```bash
# Basic installation (minimal configuration)
poetry install

# CPU + GCP environment
poetry install -E cpu -E gcp

# CUDA + GCP environment
poetry install -E cuda -E gcp

# CPU + AWS environment
poetry install -E cpu -E aws

# CUDA + AWS environment
poetry install -E cuda -E aws

# TPU + GCP environment (Google Cloud TPU)
poetry install -E tpu -E gcp

# MPU + AWS environment (Apple Silicon + AWS)
poetry install -E mpu -E aws
```

### Cloud Authentication

```bash
# GCP authentication
gcloud auth application-default login
gcloud config set project "your-project-id"
gcloud auth application-default set-quota-project "your-project-id"

# AWS authentication
aws configure sso --use-device-code --profile default
# Renew token if expired
aws sso login --use-device-code --profile default
```

### Running Tests

```bash
# Run all tests
pytest

# Test with GCP features
TEST_GCP=true pytest

# Test with AWS features
TEST_AWS=true pytest

# Run a specific test file
pytest tests/path/to/test_file.py

# Run a specific test function
pytest tests/path/to/test_file.py::test_function_name

# Run tests with verbose output
pytest -v
```

### Linting and Type Checking

```bash
# Run all pre-commit hooks (includes linting)
pre-commit run --all-files

# Run specific linters
poetry run flake8 src/
poetry run mypy src/
poetry run ruff src/ --fix
poetry run isort src/
```

### CLI Commands

The project provides a CLI with three main commands:

```bash
# Convert Shogi game records to HCPE format
poetry run maou hcpe_convert --input-path /path/to/records --input-format csa --output-dir /path/to/output

# Pre-process HCPE data
poetry run maou pre_process --input-path /path/to/hcpe --output-dir /path/to/processed

# Train model
poetry run maou learn_model --input-dir /path/to/processed --gpu cuda:0 --epoch 10 --batch-size 256
```

Run with `--help` for detailed options for each command.

## Architecture

The project follows Clean Architecture principles with four main layers:

1. **Domain Layer** (`src/maou/domain/`): Core business logic and entities
   - Network models (ResNet implementation)
   - Loss functions
   - Shogi game parsers (CSA, KIF formats)

2. **App Layer** (`src/maou/app/`): Use cases implementation
   - Converter: Converts game records to training data
   - Learning: Neural network training
   - Pre-processing: Feature extraction

3. **Interface Layer** (`src/maou/interface/`): Adapters between app and infrastructure
   - Converts between domain objects and infrastructure

4. **Infrastructure Layer** (`src/maou/infra/`): External systems integration
   - Cloud storage (GCS, S3)
   - Database (BigQuery)
   - Console application
   - Logging

The dependency flow is one-directional: `infra → interface → app → domain`

## Data Pipeline

1. **Game Record Conversion**: `hcpe_convert` command
   - Converts Shogi game records (CSA/KIF) to HCPE format
   - Filters based on game quality metrics

2. **Feature Extraction**: `pre_process` command
   - Transforms HCPE data into neural network inputs
   - Generates labeled training data

3. **Model Training**: `learn_model` command
   - Uses PyTorch with ResNet architecture
   - Supports various hardware (CPU, CUDA, MPS, TPU)
   - Checkpointing and TensorBoard integration

## Storage Options

- **Local Filesystem**: Default for all operations
- **Google Cloud Storage**: For model storage (requires `-E gcp`)
- **Amazon S3**: For model storage (requires `-E aws`)
- **BigQuery**: For storing and retrieving HCPE data (requires `-E gcp`)
