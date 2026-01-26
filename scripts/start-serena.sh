#!/bin/bash
# scripts/start-serena.sh
# Prepare environment for Serena MCP server

set -eux

# Export rust-analyzer path for Serena
export RUST_ANALYZER_PATH=$(command -v rust-analyzer 2>/dev/null || echo "rust-analyzer")

echo "Serena environment prepared."
echo "  RUST_ANALYZER_PATH: $RUST_ANALYZER_PATH"
echo ""
echo "Note: Python LSP (pyright) is bundled with Serena."
