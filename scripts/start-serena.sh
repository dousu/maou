#!/bin/bash
# scripts/start-serena.sh
# Prepare environment for Serena MCP server

set -eu

# Export LSP paths for Serena
export PYLSP_PATH=$(command -v pylsp 2>/dev/null || echo "pylsp")
export RUST_ANALYZER_PATH=$(command -v rust-analyzer 2>/dev/null || echo "rust-analyzer")

echo "Serena environment prepared."
echo "  PYLSP_PATH: $PYLSP_PATH"
echo "  RUST_ANALYZER_PATH: $RUST_ANALYZER_PATH"
