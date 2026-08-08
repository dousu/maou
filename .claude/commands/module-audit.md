---
description: Run the bug-detection + simplification + docstring/type + CLI-doc-sync audit cycle against one path or crate at a chosen effort level, then version-bump and commit. Designed so each module/crate can be audited as an independent, resumable unit across separate sessions.
argument-hint: <path-or-crate> [effort-level: low|medium|high|max, default medium]
---

You are auditing one module, layer directory, or Rust crate for bugs,
simplification opportunities, and documentation gaps — one bounded unit
of the repo-wide refactor/bugfix/doc-audit effort, sized so it can be
handed off to a fresh session without losing continuity.

`$ARGUMENTS` is `<path> [level]`:
- First token: target path, e.g. `src/maou/domain/model`,
  `src/maou/app/learning`, `rust/maou_shogi`, `docs/commands`.
- Second token (optional): review effort level `low|medium|high|max`.
  Default `medium`. Bug-hunting benefits from `high`/`max`; a small,
  already-stable module can use `low`/`medium`.

If the first token is missing or the path does not exist, stop and ask
the user for a valid target instead of guessing one.

## Hard constraints

- **Never `--no-verify`.** Pre-commit must run on every commit this
  command makes.
- **Never bump version for docs-only changes.** Only `src/` changes bump
  `pyproject.toml`; only `rust/<crate>/` changes bump that crate's
  `Cargo.toml` (independently, per crate).
- **Don't blind-apply bug fixes.** Correctness findings from the
  code-review pass are triaged: obvious, low-risk fixes are applied
  directly; ambiguous or architecturally significant ones are surfaced to
  the user before touching code (same posture as reviewing someone else's
  PR — see CLAUDE.md's dependency-flow and layer rules before fixing
  anything that crosses `infra → interface → app → domain`).
- **Respect the Code Exploration Policy.** If the CLI-doc-sync step (6)
  needs to read multiple unfamiliar files to figure out current CLI
  options, delegate that to an `Explore` agent rather than looping
  Read/Grep directly.
- **Serena MCP tools, if used, are called one at a time** — never in
  parallel (memory-constrained DevContainer).
- **Stay inside the target path.** Don't let a bug fix in one module spill
  into unrelated refactors elsewhere; if the audit surfaces an issue
  outside `<path>`, note it for a future `/module-audit <that-path>` run
  instead of fixing it inline.

## Steps

### 0. Resolve scope

- Confirm `<path>` exists.
- Classify it: `python-src` (under `src/`), `rust-crate` (under
  `rust/<crate>/`), or `docs` (under `docs/`). This decides which
  version file (if any) gets bumped in step 8, and whether step 4
  (type-safety) applies.
- `git status --short -- <path>` — if already dirty from unrelated prior
  work, tell the user and ask whether to proceed on top of it or stash
  first. Don't silently mix this audit's diff with pre-existing changes.

### 1. Bug detection

Run `/code-review <path> <level>`. This covers correctness bugs *and*
reuse/simplification/efficiency findings in one pass.

- Apply directly: fixes that are unambiguous, low-risk, and contained to
  `<path>`.
- Defer to the user: anything ambiguous, cross-layer, or that changes a
  public API other code calls.
- Record deferred items verbatim (file:line + the finding) so they don't
  get lost at session handoff — see step 10.

### 2. Simplification cleanup

Run `/simplify <path>` to apply the quality-only cleanups (reuse,
simplification, efficiency, altitude) that step 1 didn't already fix.
This one auto-applies — no bug-hunting, so lower risk.

### 3. Docstring & type-hint completeness (python-src only)

Invoke the `type-safety-enforcer` skill scoped to `<path>`:
- `uv run mypy <path>`
- missing type hints on functions/methods/class attributes
- missing docstrings on public APIs (CLAUDE.md: "MUST add docstrings to
  all public APIs")

Fix trivial gaps directly (add a type hint, add a docstring stub).
Flag anything that needs real design thought (e.g. an ambiguous return
type) rather than guessing.

### 4. Japanese writing-rule check (if `<path>` contains Japanese prose)

If `<path>` includes Japanese docstrings, comments, or `.md` docs,
invoke the `japanese-doc-validator` skill: 全角コンマ／ピリオド (，．),
半角括弧 compliance.

### 5. CLI doc-sync check (only if `<path>` touches CLI surface)

Applies when `<path>` is under `src/maou/infra/console/`, or touches an
app/interface module backing a CLI command.

- Identify which `docs/commands/<command-name>.md` files correspond to
  the touched command(s).
- Check: does the doc exist for every live command reachable from this
  path? Does its CLI-options table match the actual `click`/argparse
  options in code? (Use an `Explore` agent for this cross-reference if it
  requires reading more than the file already open.)
- Fix drift directly (update the options table). If a command was added
  without a doc, create `docs/commands/<command-name>.md` following the
  existing format. If a command was removed, delete its doc.

### 6. QA pipeline

Run the `qa-pipeline-automation` skill (or manually):
```bash
uv run ruff format <path>
uv run ruff check <path> --fix
uv run mypy <path>          # python-src only
uv run pytest <corresponding test path>
```
For `rust-crate` scope, run `cargo test -p <crate>` (add
`--release --ignored` only for tests flagged `[SLOW]` per CLAUDE.md).

### 7. Versioning

- `python-src` scope, any `src/` file changed → bump `pyproject.toml`
  per semver (`fix:` patch / `feat:` minor / breaking major).
- `rust-crate` scope, any file under that crate changed → bump that
  crate's own `Cargo.toml` independently of the Python version.
- `docs` scope → no version bump.

### 8. Commit

Stage only the files touched by this audit (the target path + the
version file bumped in step 7). Never `git add -A`. Commit message:

```
fix|refactor|docs: <module-audit summary for <path>>
```

Run pre-commit; never skip it.

### 9. Report

Print a compact summary (~10 lines):
- Target path, effort level
- Bugs found / fixed / deferred (with file:line for deferred ones)
- Simplifications applied (count)
- Docstring/type-hint gaps fixed / flagged
- Doc-sync fixes (files created/updated/deleted under `docs/commands/`)
- QA result (pass/fail)
- Version bump (old → new, file)
- Commit SHA

### 10. Handoff

This command does **not** call `/checkpoint-context` itself — commits
already happened in step 8, satisfying the dirty-tree gate. But before
ending the session (or switching to the next module), remind the user to
run `/checkpoint-context`, and make sure any **deferred** findings from
step 1/3 are stated plainly in that turn so they land in
`scratchpad/current.md`'s "Next concrete step" / open items — that's
what lets a *different* session pick up `/module-audit <next-path>`
without re-discovering what this one already found and chose not to fix.

## Usage

- `/module-audit src/maou/domain/model` — medium effort, default.
- `/module-audit src/maou/app/learning high` — broader bug hunt.
- `/module-audit rust/maou_shogi max` — thorough crate-level pass.
- `/module-audit docs/commands` — docs-only, skips versioning and
  type-safety steps.
