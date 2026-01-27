#!/bin/bash

set -eux

# Memory-optimized Rust build configuration for 2-core/3GB environments
# Reduces peak memory usage from 3.0-3.5GB to 1.0-1.5GB
echo "Configuring memory-optimized Rust build settings..."
export CARGO_BUILD_JOBS=1
export RUSTFLAGS="-C codegen-units=1 -C incremental=1"
echo "  CARGO_BUILD_JOBS=1 (sequential builds)"
echo "  RUSTFLAGS=-C codegen-units=1 -C incremental=1 (reduced parallelism)"

# Install rust-analyzer for Serena LSP integration
echo "Installing rust-analyzer..."
rustup component add rust-analyzer

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env"
fi

# Install dependencies with CPU extra (uses lightweight PyTorch from pytorch-cpu index)
uv sync --extra cpu --extra visualize --group dev

# Build Rust extension module (required for CLI commands)
echo "Building Rust extension module..."
uv run maturin develop

# Verify Rust module is built correctly
uv run python -c "from maou._rust.maou_io import hello; print(hello())"

# Clean up uv cache
uv cache clean
