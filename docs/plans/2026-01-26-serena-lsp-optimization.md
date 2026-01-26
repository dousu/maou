# Serena LSP Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** SerenaのRust/Python両言語のシンボル解析を有効化し，トークン消費を50%以上削減する

**Architecture:** `.serena/project.yml`にRust言語を追加し，rust-analyzerのインストールを`scripts/dev-init.sh`に追加．Rustビルドターゲットを`ignored_paths`で除外してインデックス効率を向上．

**Tech Stack:** Serena MCP Server, rust-analyzer, pyright-langserver

---

## Task 1: Update dev-init.sh to install rust-analyzer

**Files:**
- Modify: `scripts/dev-init.sh:24-25`

**Step 1: Add rust-analyzer installation**

Add after line 11 (after RUSTFLAGS export):

```bash
# Install rust-analyzer for Serena LSP integration
echo "Installing rust-analyzer..."
rustup component add rust-analyzer
```

**Step 2: Verify the script syntax**

Run: `bash -n scripts/dev-init.sh`
Expected: No output (syntax OK)

**Step 3: Test rust-analyzer installation command**

Run: `rustup component list --installed | grep rust-analyzer`
Expected: `rust-analyzer-x86_64-unknown-linux-gnu` (already installed from earlier)

**Step 4: Commit**

```bash
git add scripts/dev-init.sh
git commit -m "feat: add rust-analyzer installation for Serena LSP"
```

---

## Task 2: Add Rust to Serena languages configuration

**Files:**
- Modify: `.serena/project.yml:24-25`

**Step 1: Update languages list**

Change:
```yaml
languages:
- python
```

To:
```yaml
languages:
- python
- rust
```

**Step 2: Verify YAML syntax**

Run: `python -c "import yaml; yaml.safe_load(open('.serena/project.yml'))"`
Expected: No error

**Step 3: Commit**

```bash
git add .serena/project.yml
git commit -m "feat: add Rust to Serena languages for symbol analysis"
```

---

## Task 3: Add ignored_paths for Rust build artifacts

**Files:**
- Modify: `.serena/project.yml:36`

**Step 1: Update ignored_paths**

Change:
```yaml
ignored_paths: []
```

To:
```yaml
ignored_paths:
- "**/target/**"
- "**/__pycache__/**"
- "**/.venv/**"
- "**/node_modules/**"
```

**Step 2: Verify YAML syntax**

Run: `python -c "import yaml; yaml.safe_load(open('.serena/project.yml'))"`
Expected: No error

**Step 3: Commit**

```bash
git add .serena/project.yml
git commit -m "feat: add ignored_paths for build artifacts in Serena"
```

---

## Task 4: Clear Serena cache and restart

**Files:**
- None (cache operation)

**Step 1: Clear existing cache**

Run: `rm -rf /workspaces/maou/.serena/cache/*`
Expected: No output

**Step 2: Verify cache is empty**

Run: `ls /workspaces/maou/.serena/cache/`
Expected: Empty or no output

**Step 3: Document cache clearing (no commit needed)**

This step prepares for fresh indexing when Serena restarts.

---

## Task 5: Test Rust symbol detection

**Files:**
- None (verification only)

**Step 1: Restart Claude Code session**

User action: Exit and restart Claude Code to reconnect Serena MCP server

**Step 2: Test Rust symbol search**

Run via Serena MCP:
```
mcp__serena__find_symbol with name_path_pattern="save_feather"
```
Expected: Returns symbol location in `rust/maou_io/src/arrow_io.rs`

**Step 3: Test Rust symbols overview**

Run via Serena MCP:
```
mcp__serena__get_symbols_overview with relative_path="rust/maou_io/src/arrow_io.rs"
```
Expected: Returns functions like `save_feather`, `load_feather`, etc.

---

## Task 6: Test Python symbol detection

**Files:**
- None (verification only)

**Step 1: Test Python symbol search**

Run via Serena MCP:
```
mcp__serena__find_symbol with name_path_pattern="CloudStorage"
```
Expected: Returns class definition in `src/maou/domain/cloud_storage.py`

**Step 2: Test Python referencing symbols**

Run via Serena MCP:
```
mcp__serena__find_referencing_symbols with name_path="CloudStorage" relative_path="src/maou/domain/cloud_storage.py"
```
Expected: Returns GCS, S3 implementations and usage locations

---

## Task 7: Update design document with results

**Files:**
- Modify: `docs/plans/2026-01-26-serena-mcp-token-reduction-design.md`

**Step 1: Mark tasks as completed**

Update the implementation tasks section to mark all tasks as complete.

**Step 2: Add verification results**

Add a new section documenting:
- Rust symbol analysis: working/not working
- Python symbol analysis: working/not working
- Token reduction estimate

**Step 3: Commit**

```bash
git add docs/plans/2026-01-26-serena-mcp-token-reduction-design.md
git commit -m "docs: update Serena design with verification results"
```

---

## Task 8: Remove obsolete serena.toml

**Files:**
- Delete: `serena.toml`

**Step 1: Verify file is not used**

The `.serena/project.yml` is the actual configuration file used by Serena.
`serena.toml` was created based on incorrect assumptions and is not recognized.

**Step 2: Delete the file**

Run: `rm serena.toml`

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove unused serena.toml (use .serena/project.yml)"
```

---

## Verification Checklist

After completing all tasks:

- [ ] `rustup component list --installed` shows `rust-analyzer`
- [ ] `.serena/project.yml` contains `rust` in languages list
- [ ] `.serena/project.yml` contains ignored_paths for target/**
- [ ] `mcp__serena__find_symbol("save_feather")` returns Rust symbol
- [ ] `mcp__serena__find_symbol("CloudStorage")` returns Python symbol
- [ ] `serena.toml` is deleted from project root
