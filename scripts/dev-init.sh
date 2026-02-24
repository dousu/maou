#!/bin/bash

set -eux

# Install Rust if not present
if ! command -v rustup &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

# Install sccache for Rust build caching
SCCACHE_VERSION="0.12.0"
SCCACHE_SHA256="b0e89ead6899224a4ba2b90e9073bf1ce036d95bab30f3dc33c1e1468bc4ad44"
if ! command -v sccache &> /dev/null; then
    echo "Installing sccache v${SCCACHE_VERSION}..."
    curl -L "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" -o /tmp/sccache.tar.gz
    echo "${SCCACHE_SHA256}  /tmp/sccache.tar.gz" | sha256sum -c -
    tar xz -f /tmp/sccache.tar.gz
    # Install to /usr/local/cargo/bin/ (CARGO_HOME set by DevContainer Rust feature, already in PATH)
    mv "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache" /usr/local/cargo/bin/
    rm -rf "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl" /tmp/sccache.tar.gz
    echo "sccache installed: $(sccache --version)"
fi

# Ensure sccache cache directory is writable by current user
# Docker named volumes may create the target directory as root on first mount
if [ -n "${SCCACHE_DIR:-}" ] && [ -d "${SCCACHE_DIR}" ] && [ ! -w "${SCCACHE_DIR}" ]; then
    sudo chown -R "$(id -u):$(id -g)" "${SCCACHE_DIR}"
fi

# Memory-optimized Rust build configuration for 2-core/3GB environments
# Uses user-level cargo config so it applies even through PEP 517 build isolation
CARGO_USER_CONFIG="$HOME/.cargo/config.toml"
CARGO_USER_CONFIG_MARKER="# managed by dev-init.sh"
if ! grep -qF "$CARGO_USER_CONFIG_MARKER" "$CARGO_USER_CONFIG" 2>/dev/null; then
    mkdir -p $(dirname "$CARGO_USER_CONFIG")
    cat > "$CARGO_USER_CONFIG" << CARGO_CONF
$CARGO_USER_CONFIG_MARKER

[build]
target-dir = "/tmp/cargo-target"
jobs = 1
rustc-wrapper = "sccache"

[profile.dev]
codegen-units = 16  # Parallel codegen to reduce peak memory (required for 4GB RAM)
incremental = false

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
echo "  rustc-wrapper = sccache"
echo "  lto = false (release)"
echo "  incremental = false (dev), true (release)"
echo "  codegen-units = 16 (dev), 16 (release)"

# Install rust-analyzer for Serena LSP integration
echo "Installing rust-analyzer..."
rustup component add rust-analyzer

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv generate-shell-completion bash >> ~/.bash_completion

# Install dependencies with CPU extra (uses lightweight PyTorch from pytorch-cpu index)
uv sync --extra cpu --extra visualize --group dev

# Verify Rust module is built correctly
uv run python -c "from maou._rust.maou_io import hello; print(hello())"

# Show sccache statistics
sccache --show-stats

# Clean up uv cache
# shellcheck source=/dev/null
source .venv/bin/activate

uv cache clean
