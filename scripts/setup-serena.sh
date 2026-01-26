#!/bin/bash
# scripts/setup-serena.sh
# Install Serena MCP server and Python LSP for symbol-level code analysis

set -eu

echo "Installing Serena MCP server..."
uv tool install serena

echo "Installing Python Language Server..."
uv tool install python-lsp-server[all]

echo "Installing pylsp-mypy plugin..."
uv tool install pylsp-mypy

echo "Serena setup complete."
echo "  - Serena: $(which serena || echo 'installed via uvx')"
echo "  - pylsp: $(which pylsp || echo 'installed via uvx')"
