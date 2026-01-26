#!/bin/bash
# scripts/setup-serena.sh
# Install Serena MCP server for symbol-level code analysis

set -eux

echo "Installing Serena MCP server from GitHub..."
uv tool install git+https://github.com/oraios/serena

echo "Serena setup complete."
echo "  - Serena: $(which serena || echo 'installed via uvx')"
echo ""
echo "Note: Serena uses pyright-langserver (bundled) for Python analysis."
echo "      rust-analyzer is installed separately via rustup."
