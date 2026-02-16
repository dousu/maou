#!/bin/bash

set -eux

# Install Rust if not present
if ! command -v rustup &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

# Memory-optimized Rust build configuration for 2-core/3GB environments
# Uses user-level cargo config so it applies even through PEP 517 build isolation
CARGO_USER_CONFIG="$HOME/.cargo/config.toml"
CARGO_USER_CONFIG_MARKER="# managed by dev-init.sh"
if ! grep -qF "$CARGO_USER_CONFIG_MARKER" "$CARGO_USER_CONFIG" 2>/dev/null; then
    cat > "$CARGO_USER_CONFIG" << CARGO_CONF
$CARGO_USER_CONFIG_MARKER

[build]
target-dir = "/tmp/cargo-target"
jobs = 1

[profile.dev]
codegen-units = 16

[profile.release]
lto = false
incremental = true
codegen-units = 16
CARGO_CONF
    echo "Configured $CARGO_USER_CONFIG for dev environment"
fi

echo "Memory-optimized Rust build settings (via $CARGO_USER_CONFIG):"
echo "  target-dir = /tmp/cargo-target"
echo "  jobs = 1"
echo "  lto = false"
echo "  incremental = true"
echo "  codegen-units = 16"

# Install rust-analyzer for Serena LSP integration
echo "Installing rust-analyzer..."
rustup component add rust-analyzer

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env"
fi

uv generate-shell-completion bash >> ~/.bash_completion

# Install dependencies with CPU extra (uses lightweight PyTorch from pytorch-cpu index)
uv sync --extra cpu --extra visualize --group dev

# Build Rust extension module (required for CLI commands)
echo "Building Rust extension module..."
uv run maturin develop

# Verify Rust module is built correctly
uv run python -c "from maou._rust.maou_io import hello; print(hello())"

# Clean up uv cache
# shellcheck source=/dev/null
source .venv/bin/activate

uv cache clean
