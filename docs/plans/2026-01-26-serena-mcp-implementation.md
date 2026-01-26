# Serena MCP Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

> **⚠️ Implementation Notes (2026-01-26):**
> This was the original plan. Actual implementation differed in key areas:
> - **Python LSP**: Serena uses **pyright** (built-in), not pylsp. Task 6 (pylsp config) was skipped.
> - **Config file**: Serena uses `.serena/project.yml`, not `serena.toml`. Task 4 was adjusted.
> - **MCP config**: Uses `.mcp.json` instead of `.claude/settings.local.json` for MCP servers.
> - **Installation**: `uv tool install git+https://github.com/oraios/serena` (not from PyPI)
>
> See `2026-01-26-serena-mcp-token-reduction-design.md` for the updated design with verification results.

**Goal:** Integrate oraias/serena MCP server to reduce Claude Code token consumption by 50%+ through symbol-level code analysis.

**Architecture:** Serena runs as MCP server inside DevContainer, connecting to pyright (Python, built-in) and rust-analyzer (Rust) LSPs. Claude Code uses Serena's symbol-level tools instead of Grep/Read for code exploration.

**Tech Stack:** Serena MCP, pyright (Serena built-in), rust-analyzer, uv tools

---

## Task 1: Create Serena Setup Script

**Files:**
- Create: `scripts/setup-serena.sh`

**Step 1: Create the setup script**

```bash
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
```

**Step 2: Make script executable and verify syntax**

Run:
```bash
chmod +x scripts/setup-serena.sh
bash -n scripts/setup-serena.sh
```
Expected: No output (syntax OK)

**Step 3: Commit**

```bash
git add scripts/setup-serena.sh
git commit -m "feat: add Serena MCP server setup script"
```

---

## Task 2: Create Serena Start Script

**Files:**
- Create: `scripts/start-serena.sh`

**Step 1: Create the start script**

```bash
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
```

**Step 2: Make script executable and verify syntax**

Run:
```bash
chmod +x scripts/start-serena.sh
bash -n scripts/start-serena.sh
```
Expected: No output (syntax OK)

**Step 3: Commit**

```bash
git add scripts/start-serena.sh
git commit -m "feat: add Serena environment preparation script"
```

---

## Task 3: Update DevContainer Configuration

**Files:**
- Modify: `.devcontainer/devcontainer.json`

**Step 1: Read current devcontainer.json**

Run:
```bash
cat .devcontainer/devcontainer.json
```

**Step 2: Update devcontainer.json with Rust feature and Serena setup**

Replace entire file with:

```json
{
	"name": "Ubuntu",
	"image": "mcr.microsoft.com/vscode/devcontainers/base:ubuntu",
	"postCreateCommand": "bash scripts/dev-init.sh && bash scripts/setup-serena.sh",
	"postStartCommand": "bash scripts/start-serena.sh",
	"features": {
		"ghcr.io/devcontainers/features/python:1": {
			"version": "3.12"
		},
		"ghcr.io/devcontainers/features/github-cli:1": {},
		"ghcr.io/devcontainers/features/rust:1": {
			"version": "latest",
			"profile": "default"
		}
	},
	"customizations": {
		"vscode": {
			"settings": {
				"python.analysis.autoImportCompletions": true,
				"python.analysis.typeCheckingMode": "basic",
				"[python]": {
					"editor.formatOnSave": true,
					"editor.defaultFormatter": "charliermarsh.ruff",
					"editor.codeActionsOnSave": {
						"source.fixAll.ruff": "explicit",
						"source.organizeImports.ruff": "explicit"
					}
				},
				"python.testing.pytestArgs": [
        			"tests"
    			],
    			"python.testing.unittestEnabled": false,
    			"python.testing.pytestEnabled": true
			},
			"extensions": [
				"esbenp.prettier-vscode",
				"charliermarsh.ruff",
				"GitHub.copilot",
				"rust-lang.rust-analyzer"
			]
		}
	}
}
```

**Step 3: Validate JSON syntax**

Run:
```bash
python -c "import json; json.load(open('.devcontainer/devcontainer.json'))"
```
Expected: No output (valid JSON)

**Step 4: Commit**

```bash
git add .devcontainer/devcontainer.json
git commit -m "feat: add Rust feature and Serena hooks to DevContainer"
```

---

## Task 4: Create Serena Configuration

**Files:**
- Create: `serena.toml`

**Step 1: Create serena.toml**

```toml
# serena.toml
# Configuration for Serena MCP server
# https://github.com/oraios/serena

[project]
name = "maou"

[languages.python]
enabled = true
lsp_command = "pylsp"
include = ["src/**/*.py", "tests/**/*.py"]
exclude = ["**/__pycache__/**", "**/.venv/**", "**/node_modules/**"]

[languages.rust]
enabled = true
lsp_command = "rust-analyzer"
include = ["rust/**/*.rs"]
exclude = ["rust/**/target/**"]

[indexing]
on_startup = true
watch = true
```

**Step 2: Verify TOML syntax**

Run:
```bash
python -c "import tomllib; tomllib.load(open('serena.toml', 'rb'))"
```
Expected: No output (valid TOML)

**Step 3: Commit**

```bash
git add serena.toml
git commit -m "feat: add Serena MCP server configuration"
```

---

## Task 5: Create Claude Code MCP Settings

**Files:**
- Create: `.claude/settings.local.json`

**Step 1: Create settings.local.json with MCP server config**

```json
{
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": ["serena", "--project-root", "/workspaces/maou"],
      "env": {
        "SERENA_LOG_LEVEL": "info"
      }
    }
  }
}
```

**Step 2: Validate JSON syntax**

Run:
```bash
python -c "import json; json.load(open('.claude/settings.local.json'))"
```
Expected: No output (valid JSON)

**Step 3: Add to .gitignore (local settings should not be committed)**

Run:
```bash
echo ".claude/settings.local.json" >> .gitignore
```

**Step 4: Commit .gitignore update**

```bash
git add .gitignore
git commit -m "chore: ignore .claude/settings.local.json"
```

---

## Task 6: Add pylsp Configuration to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Check if pylsp config already exists**

Run:
```bash
grep -n "tool.pylsp" pyproject.toml || echo "NOT_FOUND"
```

**Step 2: Add pylsp configuration at end of pyproject.toml**

Append to pyproject.toml:

```toml

# Python Language Server configuration for Serena MCP
[tool.pylsp-mypy]
enabled = true
live_mode = false

[tool.pylsp.plugins]
pyflakes.enabled = false
mccabe.enabled = false
pycodestyle.enabled = false
rope_autoimport.enabled = true
```

**Step 3: Verify TOML syntax**

Run:
```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```
Expected: No output (valid TOML)

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add pylsp configuration for Serena integration"
```

---

## Task 7: Create rust-analyzer Configuration

**Files:**
- Create: `rust/maou_rust/rust-analyzer.toml`

**Step 1: Create rust-analyzer.toml**

```toml
# rust-analyzer.toml
# Configuration for rust-analyzer LSP (used by Serena)

[cargo]
buildScripts.enable = true

[check]
command = "clippy"

[imports]
granularity.group = "module"
prefix = "self"
```

**Step 2: Verify TOML syntax**

Run:
```bash
python -c "import tomllib; tomllib.load(open('rust/maou_rust/rust-analyzer.toml', 'rb'))"
```
Expected: No output (valid TOML)

**Step 3: Commit**

```bash
git add rust/maou_rust/rust-analyzer.toml
git commit -m "feat: add rust-analyzer configuration for Serena integration"
```

---

## Task 8: Update CLAUDE.md with Serena Tool Priority

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Find insertion point (after Rust Backend section)**

Run:
```bash
grep -n "## Rust Backend" CLAUDE.md
```

**Step 2: Add Serena section after line 100 (adjust based on actual file)**

Insert after the "Rust Backend Development" section (around line 105):

```markdown

## MCP Server: Serena (Token Efficiency)

This project uses Serena MCP server for symbol-level code analysis to reduce token consumption.

### Tool Priority for Code Exploration

**When searching for code, use this priority:**

1. **Serena `find_symbol`** - Find class/function/variable definitions
2. **Serena `find_referencing_symbols`** - Find where symbols are used
3. **Serena `get_symbol_definition`** - Get only the definition (not entire file)
4. **Glob/Grep** - File pattern or text search (when not searching for symbols)
5. **Read** - Only when full file context is needed

### Tool Priority for Code Editing

1. **Serena `replace_symbol`** - Replace entire function/class
2. **Serena `insert_after_symbol`** - Insert code after a symbol
3. **Edit** - Line-level modifications

### Examples

**Finding Protocol Implementations:**
```
Bad:  Grep "StorageProtocol" → Read multiple files (high tokens)
Good: find_referencing_symbols("StorageProtocol") (low tokens)
```

**Understanding a Function:**
```
Bad:  Read entire file to find one function (high tokens)
Good: get_symbol_definition("process_hcpe") (low tokens)
```

```

**Step 3: Verify markdown renders correctly**

Run:
```bash
head -150 CLAUDE.md | tail -50
```
Expected: New section visible

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add Serena MCP tool priority guidelines to CLAUDE.md"
```

---

## Task 9: Verify Installation (Manual Test)

**Files:** None (manual verification)

**Step 1: Test Serena installation**

Run:
```bash
uvx serena --help
```
Expected: Serena help output

**Step 2: Test pylsp installation**

Run:
```bash
uvx pylsp --help
```
Expected: pylsp help output

**Step 3: Test rust-analyzer (if installed)**

Run:
```bash
rust-analyzer --version || echo "Not installed (will be installed by DevContainer)"
```
Expected: Version or "Not installed" message

**Step 4: Document verification results**

No commit needed - this is manual verification.

---

## Task 10: Create Verification Documentation

**Files:**
- Update: `docs/plans/2026-01-26-serena-mcp-token-reduction-design.md`

**Step 1: Update implementation task checkboxes**

Replace the "実装タスク" section at the end:

```markdown
## 実装タスク

1. [x] `scripts/setup-serena.sh` 作成
2. [x] `scripts/start-serena.sh` 作成
3. [x] `.devcontainer/devcontainer.json` 更新
4. [x] `serena.toml` 作成
5. [x] `.claude/settings.local.json` にMCPサーバー追加
6. [x] `pyproject.toml` にpylsp設定追加
7. [x] `rust/maou_rust/rust-analyzer.toml` 作成
8. [x] `CLAUDE.md` にツール優先順位追記
9. [ ] DevContainer再構築・動作確認
10. [ ] 効果測定（トークン削減率の検証）

## 次のステップ

DevContainer再構築後，以下で効果を測定:

1. Claude Codeセッションを開始
2. 「StorageProtocolの実装をすべて列挙して」と依頼
3. Serenaツールが使用されることを確認
4. トークン使用量を記録
```

**Step 2: Commit**

```bash
git add docs/plans/2026-01-26-serena-mcp-token-reduction-design.md
git commit -m "docs: update implementation checklist with completed tasks"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Setup script | `scripts/setup-serena.sh` |
| 2 | Start script | `scripts/start-serena.sh` |
| 3 | DevContainer config | `.devcontainer/devcontainer.json` |
| 4 | Serena config | `serena.toml` |
| 5 | Claude MCP settings | `.claude/settings.local.json`, `.gitignore` |
| 6 | pylsp config | `pyproject.toml` |
| 7 | rust-analyzer config | `rust/maou_rust/rust-analyzer.toml` |
| 8 | Documentation | `CLAUDE.md` |
| 9 | Manual verification | - |
| 10 | Update checklist | `docs/plans/...design.md` |

Total commits: 9
