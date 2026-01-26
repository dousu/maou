#!/bin/bash
# scripts/setup-serena.sh
# Install Serena MCP server and Python LSP for symbol-level code analysis

set -eux

echo "Installing Serena MCP server from GitHub..."
uv tool install git+https://github.com/oraios/serena

echo "Installing Python Language Server..."
uv tool install python-lsp-server[all]

echo "Installing pylsp-mypy plugin..."
uv tool install pylsp-mypy

echo "Serena setup complete."
echo "  - Serena: $(which serena || echo 'installed via uvx')"
echo "  - pylsp: $(which pylsp || echo 'installed via uvx')"
