#!/bin/bash

set -eux

# Memory-optimized Rust build configuration for 2-core/3GB environments
# Reduces peak memory usage from 3.0-3.5GB to 1.0-1.5GB
echo "Configuring memory-optimized Rust build settings..."
export CARGO_BUILD_JOBS=1
export RUSTFLAGS="-C codegen-units=1 -C incremental=1"
echo "  CARGO_BUILD_JOBS=1 (sequential builds)"
echo "  RUSTFLAGS=-C codegen-units=1 -C incremental=1 (reduced parallelism)"

pipx install poetry
poetry completions bash >> ~/.bash_completion
poetry sync -E cpu

# shellcheck source=/dev/null
source ~/.bashrc

eval $(poetry env activate)

# Build Rust extension module (required for CLI commands)
echo "Building Rust extension module..."
poetry run maturin develop

# Verify Rust module is built correctly
poetry run python -c "from maou._rust.maou_io import hello; print(hello())"

poetry cache clear --all -q .
