# AGENTS.md

Guidance for OpenAI Codex agents working in this repository.

> **For Codex agents:** Follow `.codex/config.yaml`, `.codex/AGENT_GUIDE.md`,
> and `.codex/COMMANDS.md`.
> All shell commands **must** use `uv run`.

## CLAUDE.md is the source of truth

**Read [CLAUDE.md](CLAUDE.md) and follow it.** It holds the project's
rules for every agent working here, not only Claude Code:

- Project overview and Clean Architecture layering
  (`infra → interface → app → domain`)
- Critical MUST rules: architecture, type hints, docstrings, pre-commit
- Versioning for the Python package and each Rust crate
- Forbidden actions (no `pip`, no `--no-verify`, no committed secrets)
- Documentation requirements for `docs/commands/`
- Repository-centric memory architecture (`reviews/`, `audits/`,
  `scratchpad/`, `worklog/`)
- Code exploration policy
- Development guidelines: package management, git workflow, testing
- 日本語記述規則 — 読点 `，` / 句点 `．` / 半角括弧 `()`
- Quick reference commands and the documentation index

This file previously restated those rules in its own words. That produced
drift — including a 日本語記述規則 section that had 読点 and 句点 swapped
relative to CLAUDE.md. Shared rules now live in exactly one place. To
change one, edit `CLAUDE.md` through the `reviews/*.md` approval process
described there; do not add a local override here.

## Codex-specific notes

Only these are specific to Codex agents; everything else comes from
CLAUDE.md.

- Configuration and command references live under `.codex/`.
- Prefix shell commands with `uv run` (see `.codex/AGENT_GUIDE.md`).
- Create a dedicated feature branch for your work and open a Pull Request
  for every change.

## Attribution (required)

This repository **does** attribute agent-assisted work. Verify against
`git log` / `gh pr view` rather than assuming:

- ✅ Commits end with `Co-Authored-By: <model> <noreply@anthropic.com>`
- ✅ PR bodies end with
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`

An earlier revision prohibited both. That contradicted the repository's own
history (17 of the 20 commits on `main` preceding 2026-07-29 carry the
trailer, and every recent PR body carries the footer), so following it
produced work inconsistent with the rest of the project.
Corrected 2026-07-29 (`reviews/2026-07-29-csa-floodgate-client.md`).

## Quick start

```bash
bash scripts/dev-init.sh                 # Initialize environment
uv run bash scripts/pre-commit.sh        # Setup hooks
uv run maou --help                       # CLI help
```

See CLAUDE.md § Quick Reference for the full command set, and
`uv run maou --help` for CLI options.
