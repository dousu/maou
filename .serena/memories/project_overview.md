# Maou Project Overview

## Purpose
Maou (魔王) is a Shogi (Japanese chess) AI project implemented in Python following Clean Architecture principles.

## Tech Stack
- Python 3.10-3.12
- PyTorch (CPU/CUDA/TPU support)
- Polars DataFrames for data processing
- Arrow IPC (.feather) with LZ4 compression for data format
- Rust (PyO3 + maturin) for high-performance file I/O
- Click for CLI

## Architecture Layers
- **Domain Layer**: Business logic and entities (network models, loss functions, parsers)
- **App Layer**: Use cases (converter, learning, pre-processing)
- **Interface Layer**: Adapters between app and infrastructure
- **Infrastructure Layer**: External systems (cloud storage, databases, logging)

## Test Path Convention
`src/maou/{layer}/{module}/file.py` → `tests/maou/{layer}/{module}/test_file.py`
