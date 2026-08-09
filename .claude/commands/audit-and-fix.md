---
description: Audit one path (source module, Rust crate, or doc tree) for correctness bugs, simplification opportunities, and documentation drift — then APPLY the code fixes, bump the version, and commit. Documentation drift is never edited silently: it is filed as a reviews/ proposal and reconciled with the user in the same run, then applied on approval. Records coverage in audits/ so a path can be resumed across sessions.
argument-hint: [path-or-crate | omit to resume from the audits/ ledger] [effort-level: low|medium|high|max, default medium]
---

You are auditing **one path** — a source module, a layer directory, a Rust
crate, or a doc subtree — and repairing what you find. This is one bounded
unit of the repo-wide refactor / bugfix / doc-audit effort, sized so it can
be handed to a fresh session without losing continuity.

This command **changes code**. Code fixes are applied and committed here.
Documentation fixes are never applied *silently* — they are proposed,
approved by the user in this same run, and only then applied. See the
routing rule below.

`$ARGUMENTS` is `[path] [level]`, both optional:
- First token: target path, e.g. `src/maou/domain/model`,
  `src/maou/app/learning`, `rust/maou_shogi`, `docs/design/tsume-solver`.
  **Omit it to resume** — step 0 reads the `audits/` ledger and offers the
  unfinished work.
- Second token (optional): effort level `low|medium|high|max`, default
  `medium`. Bug hunting benefits from `high`/`max`; a small, stable path
  can use `low`/`medium`. When resuming, default to the level recorded for
  that path.

If a path is given but does not exist, stop and ask for a valid target
instead of guessing one.

## Standing principle: derive, never enumerate

This command outlives the tree it audits. Crates, design docs, layers, and
validators all get added over time, and any list baked in here goes stale
silently — becoming the same class of defect the command exists to find.
(Live proof: CLAUDE.md's versioning section enumerates five crates while
`rust/` holds seven.)

So every step below resolves its targets **by looking at the repository**,
not by consulting a list written here. Where a list is unavoidable it is
labelled *examples, not exhaustive*, and paired with the discovery rule
that supersedes it. When you extend this command, preserve that property —
see "Extending this command" at the end.

## The routing rule (MUST — read before step 1)

Findings split into two streams that are handled **differently**:

| Finding lands in | Handling | Committed by |
|---|---|---|
| Source: anything under `src/`, `rust/`, `tests/`, `scripts/` — including docstrings and Japanese comments *inside* source files | Fix applied directly | this command (step 7) |
| Durable docs: `CLAUDE.md`, `AGENTS.md`, and every tracked prose file under `docs/` or the repo root | **Never edited before approval.** Filed as a `reviews/*.md` proposal, then reconciled with the user in step 8 — applied in this run if approved | this command (step 8), or a later `/checkpoint-context` if deferred |

To decide which stream a file is in, ask whether it is **tracked prose
that documents the system** (durable doc) or **code and its inline
comments** (source). `git ls-files <path>` settles trackedness;
`scratchpad/` and `worklog/` are gitignored working memory and are neither
— never audit them.

CLAUDE.md: "MUST NOT edit `CLAUDE.md` / `docs/` without an **approved**
`reviews/*.md` proposal." A docstring inside `src/foo.py` is source, so it
is fixable here; `docs/architecture.md` is not, no matter how obvious the
fix looks. Do not rationalize an exception for "trivial" doc drift — the
approval gate is the safeguard against silent durable-doc edits, and a
one-word fix bypasses it exactly as much as a rewrite.

What the gate requires is **approval**, not a particular command. This run
files the proposal *and* takes the approval (step 8), so a doc fix found
here can be applied here. Deferring to `/checkpoint-context` is a fallback
for when the user does not want to decide now, not the normal path.

## Hard constraints

- **Never `--no-verify`.** Pre-commit runs on every commit.
- **Version bumps follow the manifest that owns the file** (step 6).
- **Don't blind-apply bug fixes.** Obvious, low-risk, contained fixes are
  applied directly; ambiguous or architecturally significant ones are
  surfaced to the user before touching code. Check CLAUDE.md's
  `infra → interface → app → domain` rule before any fix that crosses
  layers.
- **Respect the Code Exploration Policy.** Steps that need to read
  multiple unfamiliar files (especially step 4) are delegated to an
  `Explore` agent, not run as a direct Grep/Read loop.
- **Serena MCP tools are called one at a time** — never in parallel
  (memory-constrained DevContainer).
- **Stay inside the target path.** If the audit surfaces an issue outside
  `<path>`, record it for a future `/audit-and-fix <that-path>` run rather
  than fixing it inline. Scope creep is what makes these units
  unresumable.

## Steps

### 0. Pick up prior coverage, then resolve scope

Assume **no memory of any previous run**. Everything you need is in
`audits/`, and it is committed precisely so a cold session can read it.

**0a. Sync the ledger before reading it.** Another session — possibly on
another machine — may have pushed audit records after this working copy
was created:
```bash
git fetch origin <current-branch>
git status -sb          # report if behind; the ledger you are about to
                        # read is stale until it is caught up
```
Do not auto-merge; report and let the user decide if there is in-flight
work. A stale ledger silently re-audits paths someone else finished.

**0b. Read the whole ledger, not just one row.** Read
`audits/coverage.md` in full (shape and protocol: `audits/README.md`) and
report:
- every `in-progress` row — **including paths other than `<path>`**. A
  half-finished path left by an earlier session is the single thing most
  likely to be lost, and it will never surface if you only look up the
  row you were asked about. Name them even when you are not resuming them.
- every `blocked` row and its blocker.
- **`coverage.md`'s § "Open findings backlog"**, in full — **both** the
  Deferred backlog and the Out-of-scope backlog tables. Any row whose
  target falls inside `<path>` is **in scope for this run** — fold it in
  and say so. Those rows are findings an earlier audit confirmed but was
  not allowed or not ready to fix; picking them up here is the whole point
  of the backlog, and skipping them strands work that is already diagnosed.

**Do not gather open work from the record files.** A record is an
immutable account of one run at one time — its Deferred section stays
true forever, including after the finding ships. Treating it as a worklist
re-surfaces resolved findings on every run with no way to remove them.
`coverage.md` is the authority on what is open (a row is deleted when the
finding is consumed); records are the authority on what happened. Opening
a record to recover the full reasoning behind a row you are about to act
on is fine and often useful — rows are condensed. Opening it to discover
work is the error.

The one legitimate exception is resuming your *own* unfinished path
(0c, `in-progress`), where the record holds the resume point and the
triage of the run still in flight.

**0c. Resolve the target.**

If `<path>` was given:
- **`in-progress`** → open its record file and resume from the recorded
  resume point. Do **not** restart the path — the record's Deferred
  section lists findings already triaged, and re-deriving them wastes the
  session the ledger exists to save.
- **`done`** → decide staleness *concretely* rather than by feel:
  ```bash
  git log --oneline <last_sha>..HEAD -- <path>
  ```
  No commits → say so and ask whether a re-audit is really wanted.
  Commits since → report how many and re-audit.
  Either way, the already-known findings for this path are its rows in
  `coverage.md`'s backlog tables (0b) — those are current, because they are
  deleted when consumed. The record's Deferred section is the *original
  reasoning* behind those rows and is worth reading for depth, but it is
  not the list of what is still open.
- **`blocked`** → surface the blocker and ask how to proceed. Never retry
  a blocked path silently.
- **no row** → fresh path.

If `<path>` was **omitted**, do not stop immediately. Report the ledger
state from 0b and offer the natural continuations — resume an
`in-progress` path, revisit a `blocked` one, or start a named fresh path —
then wait for the user to choose. "Where was I?" is the expected way to
open a new session, and the ledger can answer it.

**0d. Classify the target** **by inspecting the tree**, in this order:

1. **Rust crate** — walk up from `<path>` to the nearest ancestor
   containing a `Cargo.toml` with a `version` field. That manifest is the
   crate root and the version file for step 6. This resolves any crate,
   including ones added after this command was written.
2. **Python source** — under `src/`, file suffix `.py`. Version file is
   the nearest ancestor `pyproject.toml`.
3. **Durable docs** — matches the routing-rule test above.
4. **Other tracked source** — anything else tracked and executable or
   compiled: web assets under `src/**/static/`, `scripts/`, CI workflows.
   These have no mypy/pytest path and usually no version file; audit them
   for correctness and consistency, run whatever linter the repo actually
   configures for them (check `.pre-commit-config.yaml` — it is the source
   of truth for which tools apply to which file types), and skip the
   inapplicable steps explicitly rather than silently.
5. **Unclassified** — stop and ask the user how to treat it rather than
   forcing it into a class above. A new language or a new top-level
   directory is a signal this command needs extending, not a case to
   improvise through.

A path may span classes (e.g. `src/maou/infra/visualization` holds both
`.py` and `static/*.css`/`*.js`). Handle each class present, and say in
step 10 which classes were found — a silently skipped asset type is how
these audits leave holes.

**0e. Check the working tree.** Run `git status --short -- <path>`. If it
is already dirty from unrelated work, say so and ask whether to build on
it or stash first. Never silently blend this audit's diff into
pre-existing changes.

**0f. Claim the path.** Before running step 1, write the `in-progress`
row and record file for `<path>` (step 9's shape) and commit them. This
costs one commit and buys two things: a session that dies mid-run still
leaves a resume point, and a concurrent session reading the ledger sees
the path is taken. Do not defer this to the end — a crash before step 9
is exactly the case the ledger exists for.

### 1. Bug detection

Run `/code-review <path> <level>` — this covers correctness bugs *and*
reuse/simplification/efficiency findings in one pass.

Triage each finding:
- **Apply** — unambiguous, low-risk, contained to `<path>`.
- **Defer to user** — ambiguous, cross-layer, or changing a public API
  other code calls.
- Record every deferred item verbatim (`file:line` + the finding) so it
  survives the session handoff — step 9 records it.

### 2. Simplification cleanup — the whole path, never the diff

Run `/simplify <path>` for the quality-only cleanups (reuse,
simplification, efficiency, altitude) that step 1 did not already fix.
Quality-only means lower risk, so this one auto-applies.

**Scope is `<path>` in full — every file under it, changed or not.**

This needs stating because `/simplify` is a general-purpose command whose
*default* scope is the current diff: its own instructions say "treat this
diff as the review scope", and three of its four angles are worded around
what "the diff adds", "the diff introduces", or what is "new code".
Passing a path is meant to override that, but the angle wording pulls the
review back toward changed lines anyway.

Left at its default here it reviews **whatever step 1 just committed** —
the smallest slice of the path, and the one slice already reviewed
minutes earlier. That inverts the point of the step. A path is audited to
find accumulated complexity nobody has revisited, which by definition
lives in the files this run did *not* touch.

So when handing the work to review agents, override the framing
explicitly: name `<path>`, tell each agent to read the files under it,
and phrase its angle as "existing code in this module" rather than "new
code" or "the diff". If a review returns citing only lines this run
changed, its scope was wrong — re-run it with the path restated.

The same caution applies to `/code-review <path> <level>` in step 1: it
also falls back to a branch/working-tree diff when one exists, which on a
**resumed** audit is this run's own earlier commits. Check its scope note
names `<path>`, not a commit range, before trusting its findings as a
path-wide bug hunt.

Record in step 9 which files the pass actually covered. "Ran /simplify"
is not evidence the path was covered; a run that reviewed only the step 1
diff must be reported as **not done**, not as a clean pass.

### 3. Applicable project validators

The repo ships validator skills under `.claude/skills/`, and that set
grows. **List the available skills and pick every one whose description
matches this scope** — do not work from a fixed list. As of writing, the
ones that commonly apply (*examples, not exhaustive*):

| When `<path>` … | Skill | Checks |
|---|---|---|
| is Python source | `type-safety-enforcer` | `uv run mypy <path>`, missing type hints, missing docstrings on public APIs |
| is a layer/module boundary, or the fix moved imports | `architecture-validator` | `infra → interface → app → domain` flow, circular dependencies, layer separation |
| contains Japanese prose | `japanese-doc-validator` | 全角コンマ／ピリオド `，．`, 半角括弧 |
| touches `learn-model` options | `benchmark-training-sync` | `benchmark-training` option parity with `learn-model` |
| touches data sources or array types | `data-pipeline-validator` | `array_type`, HCPE/preprocessing format, schema compliance |

Fix trivial gaps directly **when they live in source** (a missing type
hint, a missing docstring, a punctuation fix in a docstring) — the routing
rule allows it. Flag anything needing real design thought (an ambiguous
return type, a dependency inversion) instead of guessing. Violations in
durable docs go to step 8, never fixed here.

### 4. Documentation accuracy (the drift hunt)

This is the step with no other home — nothing else in the repo checks
whether the prose still describes the code. Delegate the cross-referencing
to an `Explore` agent, then judge the findings yourself.

**4a. Which docs describe this path?** Discover them; never rely on a
stored map, which would rot exactly like the docs it checks:
- CLAUDE.md's "Documentation Links" table and every `docs/design/*/index.md`.
- Grep `docs/` and the root `*.md` files for the module, crate, class, and
  command names that live under `<path>`.
- `docs/architecture.md` applies to any layer/module structural change;
  `docs/adr-*.md` applies when `<path>` implements a decision an ADR
  recorded.
- Check `AGENTS.md` alongside `CLAUDE.md` — the repo carries both, so a
  rule can drift between them as well as away from the code.

**4b. Verify each concrete claim.** Design docs go stale in specific,
checkable ways — check the claims that have a truth value in code:
- named modules, types, functions, CLI flags that no longer exist or were
  renamed
- **enumerations that the tree has outgrown** — a documented list of
  crates, commands, layers, or supported formats that is missing a member
  present on disk. These fail silently and are the most common drift in a
  growing repo; check every list in the doc against the tree
- data flow / layer assignments that no longer match the imports
- file formats, schemas, defaults, and tuning parameters quoted with
  specific values
- performance numbers or benchmark claims attributed to code that has
  since changed
- documented invariants the code no longer upholds

Classify each: **accurate** / **stale** (was true, code moved on) /
**wrong** (never true, or now actively misleading). Only stale and wrong
ones go to step 8.

**4c. CLI option sync** (when `<path>` is under
`src/maou/infra/console/` or backs a CLI command): does a
`docs/commands/<command-name>.md` exist for every live command reachable
from this path, is its options table consistent with the actual `click`
options in code, and does a doc linger for a removed command?

**4d. Link integrity**: do the paths referenced from CLAUDE.md's
Documentation Links table, from `AGENTS.md`, and from the docs under
review actually exist? A link to a moved or deleted file is a doc defect
worth reporting.

**Report only. Edit no durable doc in this step.**

### 5. QA pipeline

Run what the scope classes from step 0 actually call for:

- **Python source**: `uv run ruff format <path>`,
  `uv run ruff check <path> --fix`, `uv run mypy <path>`, then
  `uv run pytest` on the mirrored test path
  (`src/maou/{layer}/{module}/file.py` → `tests/maou/{layer}/{module}/test_file.py`).
  If no mirrored test exists, say so in step 10 — a missing test path is a
  finding, not a pass.
- **Rust crate**: `cargo test -p <crate>`, observing CLAUDE.md's
  §"重いテスト (Rust dfpn)" for `--test-threads=1`, `--release`, and
  `[SLOW]`/`#[ignore]` handling. Follow that section rather than a copy of
  it here; it is the maintained source of truth and copies drift.
- **Other tracked source**: whatever `.pre-commit-config.yaml` configures
  for those file types.

If QA fails, fix it before committing. Never commit a red tree.

### 6. Versioning

Bump the manifest that **owns** the changed files, resolved in step 0 —
not a manifest named in a list:

- Any changed file under a Rust crate → that crate's own `Cargo.toml`,
  bumped independently of every other crate and of the Python package.
- Any changed file under `src/` → the owning `pyproject.toml`.
- Durable-doc-only run → no bump.

Semver from the nature of the change: `fix:` patch, `feat:` minor,
breaking major. If a run changed files under two manifests, bump both.

### 7. Commit the code changes

Stage only the files this audit touched (the target path + the version
file from step 6). Never `git add -A` — `scratchpad/` and `worklog/` are
gitignored, but unrelated in-flight work is not. Commit:

```
fix|refactor: <what changed under <path>>
```

Run pre-commit; never skip it.

### 8. File the documentation drift — and reconcile it now

**8a. File the proposal.**
If step 3 or step 4 found anything stale or wrong, create
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

When the drift is an **outgrown enumeration** (4b), prefer proposing a
rule that cannot go stale again — "every crate under `rust/`" instead of a
refreshed five-item list — and say so in "What this enables". Re-listing
the current members just resets the same clock.

One proposal per audit run, listing every doc target, is preferred over
one proposal per file — the drift shares a trigger and gets approved as a
unit.

Commit `reviews/` **separately** from the step 7 code commit (doc-only
commit, per `/checkpoint-context` step 5.5):
```
docs(reviews): propose <one-line summary> (from audit of <path>)
```

If nothing drifted, skip 8a–8b entirely and say so explicitly in step 10 —
"docs verified accurate" is a real result and stops the next session from
re-checking.

**8b. Reconcile it with the user, in this run.**
Do not end the run leaving your own proposal dangling. Present it —
filename, title, target files, risk, and the concrete before/after — and
ask: **approve / reject / defer**?

- **approve** → apply the edits yourself, exactly as written in "Proposed
  change". Run pre-commit (never `--no-verify`) and commit:
  `docs: <proposal title>`. Then set the proposal's frontmatter to
  `status: applied` + `applied_in: <sha of that doc commit>` and commit
  the frontmatter change. This is the authorized path — CLAUDE.md ties the
  gate to approval, not to `/checkpoint-context`.
- **reject** → ask for a one-line reason, set `status: rejected` with that
  reason in the body, and commit. The file is **retained** as committed
  do-not-redo provenance; delete it only if it was never substantive.
- **defer** → leave `status: pending`. It resurfaces at the next
  `/checkpoint-context`. Record in step 9's record file that it is
  outstanding, so the next audit of a neighbouring path does not re-file
  the same drift.

If the run produced many separate findings, print the list first and let
the user pick which to decide now; the rest stay `pending`.

### 9. Write the audit record and ledger

Finish the record and row **claimed in step 0f** — update them, do not
create a second set. `audits/YYYY-MM-DD-<path-slug>.md` (JST date;
`<path-slug>` is `<path>` with separators flattened) follows the record
shape in `audits/README.md`: frontmatter (`path`, `scope`, `level`,
`status`, `started`, `last_sha`), then Resume point / Applied / Deferred /
Doc findings / Out of scope.

Set `status: done` and clear the resume point when the path is complete;
leave `status: in-progress` with a sharpened resume point otherwise.
Update the `audits/coverage.md` row with status, level, last SHA, record
link, and open-item count.

**Also update `coverage.md`'s § "Open findings backlog" — both tables.**
This is what makes a finding survive; the record alone does not, because
records are never read as a worklist (0b).

- **Append a row for every finding this run leaves open**, in the matching
  table: **Deferred backlog** for findings confirmed *inside* `<path>` but
  not fixed, **Out-of-scope backlog** for findings *outside* it. Each row
  carries the record that found it, the path that should fix it, and enough
  of the finding to act on without reopening the record — including *why*
  it was not fixed, since that is what the next run must decide.
  A deferred finding written only into the record's Deferred section is
  invisible to every future run: nothing reads it. Appending the row is not
  bookkeeping, it is the handoff.
- **Delete** the rows this run resolved (including any it folded in at
  step 0b), from whichever table held them. Do not delete a row that was
  merely re-triaged — sharpen its text instead.
- Keep the row's `Open items` count in the main table consistent with the
  number of backlog rows that name this path.

The record's own Deferred section is still written in full — it is the
durable reasoning, and the backlog row is its condensed, deletable index
entry. Both, not one.

Commit both:
```
docs(audits): record audit of <path>
```

**This step also runs when the audit stops early.** If the session is
ending, the context is filling, or the path turned out too large to
finish, write the record with `status: in-progress` and a resume point
naming the next step and the sub-paths still uncovered, then commit. An
interrupted run that leaves no record is indistinguishable from one that
never happened, and the next session pays for it by starting over.

### 10. Report

Print a compact summary (~12 lines):
- Target path, effort level, **scope classes found** (flag any skipped)
- Bugs: found / fixed / **deferred** (with `file:line` for each deferred)
- Simplifications applied (count)
- Validators run (step 3) and what each found
- Doc drift: N stale, M wrong → `reviews/<file>` **and its resolved
  status** (applied `<sha>` / rejected / deferred-pending), or "docs
  verified accurate"
- QA result (pass/fail, what ran, any missing test path)
- Version bump(s) (old → new, which manifest) or "none (doc-only)"
- Commit SHAs: code, reviews, doc-edit (if approved), audits
- Ledger: `audits/coverage.md` row status for this path
- Out-of-scope: N appended to the backlog, M folded in from it and
  resolved

### 11. Handoff

Do **not** call `/checkpoint-context` from here — steps 7–9 already
committed, so the dirty-tree gate is satisfied and the user may want to
audit several paths before checkpointing.

The `audits/` record written in step 9 is what carries this run across
sessions — it is committed, so it survives container reclamation, and it
holds the deferred findings and the resume point. That is what lets a
*different* session run `/audit-and-fix` on this path or a neighbouring
one without re-discovering what this run already found and deliberately
did not fix.

`/checkpoint-context` remains the campaign memory's checkpoint and is
worth running at session end, but it is **not** where audit coverage
lives; do not duplicate the audit record into `worklog/` or
`scratchpad/compass.md`.

## Extending this command

When the repo grows a new kind of thing, extend here — and keep the
derive-never-enumerate property:

- **New Rust crate** — nothing to do. Step 0's nearest-`Cargo.toml` walk
  and step 6's owning-manifest rule already cover it.
- **New design doc** — nothing to do, if it is reachable from CLAUDE.md's
  Documentation Links table or a `docs/design/*/index.md`. If it is
  reachable from neither, that itself is a step 4d finding.
- **New validator skill** — nothing to do; step 3 enumerates
  `.claude/skills/` at run time. Optionally add a row to its example table.
- **New language or top-level directory** — add a scope class to step 0
  and its QA path to step 5. Prefer keying off a discoverable marker (a
  manifest file, a `.pre-commit-config.yaml` entry) over a hardcoded path.
- **New durable-doc location** — verify the routing-rule *test* still
  classifies it correctly before adding it to the table; if the test
  works, the table needs no edit.

## Usage

- `/audit-and-fix src/maou/domain/model` — medium effort, default.
- `/audit-and-fix src/maou/app/learning high` — broader bug hunt.
- `/audit-and-fix rust/maou_shogi max` — thorough crate-level pass.
- `/audit-and-fix docs/design/tsume-solver` — doc-accuracy pass only;
  produces a `reviews/` proposal, no direct edits, no version bump.
