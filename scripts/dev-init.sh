#!/bin/bash

set -eux

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
