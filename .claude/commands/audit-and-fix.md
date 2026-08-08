---
description: Audit one path (source module, Rust crate, or doc tree) for correctness bugs, simplification opportunities, and documentation drift — then APPLY the code fixes, bump the version, and commit. Documentation drift is detected but never edited directly; it is filed as a reviews/ proposal for approval. Sized so each path is an independent, resumable unit across sessions.
argument-hint: <path-or-crate> [effort-level: low|medium|high|max, default medium]
---

You are auditing **one path** — a source module, a layer directory, a Rust
crate, or a doc subtree — and repairing what you find. This is one bounded
unit of the repo-wide refactor / bugfix / doc-audit effort, sized so it can
be handed to a fresh session without losing continuity.

This command **changes code**. Code fixes are applied and committed here.
Documentation fixes are **not** applied here — see the routing rule below.

`$ARGUMENTS` is `<path> [level]`:
- First token: target path, e.g. `src/maou/domain/model`,
  `src/maou/app/learning`, `rust/maou_shogi`, `docs/design/tsume-solver`.
- Second token (optional): effort level `low|medium|high|max`, default
  `medium`. Bug hunting benefits from `high`/`max`; a small, stable path
  can use `low`/`medium`.

If the first token is missing or the path does not exist, stop and ask for
a valid target instead of guessing one.

## The routing rule (MUST — read before step 1)

Findings split into two streams that are handled **differently**:

| Finding lands in | Handling | Committed by |
|---|---|---|
| `src/`, `rust/`, `tests/` — including docstrings and Japanese comments *inside* source files | Fix applied directly | this command (step 8) |
| `docs/**`, `CLAUDE.md`, `README.md` and any other committed durable doc | **Detected and reported only — never edited here** | filed as `reviews/*.md` `status: pending` (step 9); applied later by `/checkpoint-context` step 5 after user approval |

CLAUDE.md: "MUST NOT edit `CLAUDE.md` / `docs/` without an **approved**
`reviews/*.md` proposal." A docstring inside `src/foo.py` is source, so it
is fixable here; `docs/architecture.md` is not, no matter how obvious the
fix looks. Do not rationalize an exception for "trivial" doc drift — the
approval gate is the safeguard against silent durable-doc edits, and a
one-word fix bypasses it exactly as much as a rewrite.

## Hard constraints

- **Never `--no-verify`.** Pre-commit runs on every commit.
- **Version bumps follow the scope.** `src/` changes bump
  `pyproject.toml`; changes under `rust/<crate>/` bump *that crate's*
  `Cargo.toml` independently. Doc-only work bumps nothing.
- **Don't blind-apply bug fixes.** Obvious, low-risk, contained fixes are
  applied directly; ambiguous or architecturally significant ones are
  surfaced to the user before touching code. Check CLAUDE.md's
  `infra → interface → app → domain` rule before any fix that crosses
  layers.
- **Respect the Code Exploration Policy.** Steps that need to read
  multiple unfamiliar files (especially step 5) are delegated to an
  `Explore` agent, not run as a direct Grep/Read loop.
- **Serena MCP tools are called one at a time** — never in parallel
  (memory-constrained DevContainer).
- **Stay inside the target path.** If the audit surfaces an issue outside
  `<path>`, record it for a future `/audit-and-fix <that-path>` run rather
  than fixing it inline. Scope creep is what makes these units
  unresumable.

## Steps

### 0. Resolve scope

- Confirm `<path>` exists.
- Classify it, which decides which later steps apply:
  - `python-src` (under `src/`) — all steps.
  - `rust-crate` (under `rust/<crate>/`) — skip step 3 (mypy/docstrings);
    step 6 uses `cargo test`.
  - `docs` (under `docs/`) — steps 1–4 and 6–8 are mostly inert; the run
    is essentially step 5 + step 9, producing proposals, not edits.
- `git status --short -- <path>` — if already dirty from unrelated prior
  work, say so and ask whether to build on it or stash first. Never
  silently blend this audit's diff into pre-existing changes.

### 1. Bug detection

Run `/code-review <path> <level>` — this covers correctness bugs *and*
reuse/simplification/efficiency findings in one pass.

Triage each finding:
- **Apply** — unambiguous, low-risk, contained to `<path>`.
- **Defer to user** — ambiguous, cross-layer, or changing a public API
  other code calls.
- Record every deferred item verbatim (`file:line` + the finding) so it
  survives the session handoff (step 11).

### 2. Simplification cleanup

Run `/simplify <path>` for the quality-only cleanups (reuse,
simplification, efficiency, altitude) that step 1 did not already fix.
Quality-only means lower risk, so this one auto-applies.

### 3. Docstring & type-hint completeness (`python-src` only)

Invoke `type-safety-enforcer` scoped to `<path>`:
- `uv run mypy <path>`
- missing type hints on functions, methods, class attributes
- missing docstrings on public APIs (CLAUDE.md: "MUST add docstrings to
  all public APIs")

Fix trivial gaps directly — these live in source, so the routing rule
allows it. Flag anything needing real design thought (e.g. an ambiguous
return type) instead of guessing an annotation.

### 4. Japanese writing rules

If `<path>` contains Japanese prose, invoke `japanese-doc-validator`
(全角コンマ／ピリオド `，．`, 半角括弧).

Apply the routing rule: violations **inside source files** (docstrings,
comments) are fixed here; violations in `docs/**.md` are collected for
step 9.

### 5. Documentation accuracy (the drift hunt)

This is the step with no other home — nothing else in the repo checks
whether the prose still describes the code. Delegate the cross-referencing
to an `Explore` agent, then judge the findings yourself.

**5a. Which docs describe this path?** Do not rely on a hardcoded map (it
would rot exactly like the docs it checks). Discover them:
- CLAUDE.md's "Documentation Links" table and the `docs/design/` index
  files (`docs/design/*/index.md`).
- Grep `docs/` for the module, crate, class, and command names that live
  under `<path>`.
- `docs/architecture.md` applies to any layer/module structural change;
  `docs/adr-*.md` applies when `<path>` implements a decision an ADR
  recorded.

**5b. Verify each concrete claim.** Design docs go stale in specific,
checkable ways — check the claims that have a truth value in code:
- named modules, types, functions, CLI flags that no longer exist or were
  renamed
- data flow / layer assignments that no longer match the imports
- file formats, schemas, defaults, and tuning parameters quoted with
  specific values
- performance numbers or benchmark claims attributed to code that has
  since changed
- documented invariants the code no longer upholds

Classify each: **accurate** / **stale** (was true, code moved on) /
**wrong** (never true, or now actively misleading). Only stale and wrong
ones go to step 9.

**5c. CLI option sync** (when `<path>` is under
`src/maou/infra/console/` or backs a CLI command): does a
`docs/commands/<command-name>.md` exist for every live command reachable
from this path, is its options table consistent with the actual
`click` options in code, and does a doc linger for a removed command?

**5d. Link integrity**: do the paths referenced from CLAUDE.md's
Documentation Links table and from the docs under review actually exist?
A link to a moved or deleted file is a doc defect worth reporting.

**Report only. Edit nothing under `docs/` or `CLAUDE.md` in this step.**

### 6. QA pipeline

For `python-src`:
```bash
uv run ruff format <path>
uv run ruff check <path> --fix
uv run mypy <path>
uv run pytest <corresponding test path>     # tests/maou/{layer}/{module}/
```
For `rust-crate`:
```bash
cargo test -p <crate> -- --test-threads=1
```
Add `--release --ignored` only for tests flagged `**[SLOW]**` per
CLAUDE.md. `--test-threads=1` is mandatory — the default parallelism OOMs
the 8GB DevContainer and surfaces as `signal: 15 SIGTERM`, which reads
like a code regression but is not one.

If QA fails, fix it before committing. Never commit a red tree.

### 7. Versioning

- `python-src`, any `src/` file changed → bump `pyproject.toml` per semver
  (`fix:` patch / `feat:` minor / breaking major).
- `rust-crate`, any file under that crate changed → bump that crate's
  `Cargo.toml` independently of the Python version.
- Doc-only run → no bump.

### 8. Commit the code changes

Stage only the files this audit touched (the target path + the version
file from step 7). Never `git add -A` — `scratchpad/` and `worklog/` are
gitignored, but unrelated in-flight work is not. Commit:

```
fix|refactor: <what changed under <path>>
```

Run pre-commit; never skip it.

### 9. File the documentation drift as a proposal

If step 4 or step 5 found anything stale or wrong, create
`reviews/$(TZ=Asia/Tokyo date '+%Y-%m-%d')-<kebab-title>.md` with
`status: pending` and the shape in `docs/memory-architecture.md`
§ "Review proposal shape" — frontmatter (`status`, `applied_in:` empty,
`date`, `target:` listing the doc files, `risk`, `reversibility`), then
Trigger / Proposed change / Motivation / Alternatives considered / What
this enables / What this constrains / Rollback plan.

Write the **exact before/after text** for each doc fix in "Proposed
change". The value of the proposal is that the user can approve it during
`/checkpoint-context` without re-deriving what was wrong — a vague "update
the architecture doc" forces the whole investigation to happen twice.

One proposal per audit run, listing every doc target, is preferred over
one proposal per file — the drift shares a trigger and gets approved as a
unit.

Commit `reviews/` **separately** from the step 8 code commit (doc-only
commit, per `/checkpoint-context` step 5.5):
```
docs(reviews): propose <one-line summary> (from audit of <path>)
```

If nothing drifted, skip this step and say so explicitly in step 10 —
"docs verified accurate" is a real result and stops the next session from
re-checking.

### 10. Report

Print a compact summary (~12 lines):
- Target path, effort level, scope class
- Bugs: found / fixed / **deferred** (with `file:line` for each deferred)
- Simplifications applied (count)
- Docstring & type-hint gaps: fixed / flagged
- Doc drift: N stale, M wrong → `reviews/<file>` (or "docs verified
  accurate")
- QA result (pass/fail, what ran)
- Version bump (old → new, which file) or "none (doc-only)"
- Commit SHAs: code commit, reviews commit
- Out-of-scope issues noticed, as `/audit-and-fix <path>` suggestions

### 11. Handoff

Do **not** call `/checkpoint-context` from here — step 8 already
committed, so the dirty-tree gate is satisfied and the user may want to
audit several paths before checkpointing.

Before the session ends or the next path starts, remind the user to run
`/checkpoint-context`, and restate the **deferred** findings (step 1/3)
and the **pending proposal** (step 9) in that same turn, so they land in
`scratchpad/current.md`. That is what lets a *different* session run
`/audit-and-fix <next-path>` without re-discovering what this run already
found and deliberately did not fix.

## Usage

- `/audit-and-fix src/maou/domain/model` — medium effort, default.
- `/audit-and-fix src/maou/app/learning high` — broader bug hunt.
- `/audit-and-fix rust/maou_shogi max` — thorough crate-level pass.
- `/audit-and-fix docs/design/tsume-solver` — doc-accuracy pass only;
  produces a `reviews/` proposal, no direct edits, no version bump.
