---
description: Consume the deferred and out-of-scope findings that /audit-and-fix left behind. Gathers them from audits/coverage.md's backlog tables, re-verifies each against HEAD, ranks them by priority, and lets the user pick which to resolve — then applies the fixes on the normal route (version bump, regression test, reviews/ proposal for durable docs), deletes the resolved backlog rows, and records the run in audits/.
argument-hint: [item-selector or tier | omit to list everything] [effort-level: low|medium|high|max, default medium]
---

`/audit-and-fix` deliberately leaves work behind. It defers findings that
need a decision, and it records findings outside its target path without
fixing them. Both are *confirmed diagnoses that nobody is scheduled to
act on*. This command is that schedule.

It is the **normal route** for that backlog — not a special case. Anything
it fixes goes through the same gates as any other change: applied fix,
owning-manifest version bump, regression test, `reviews/` proposal for
durable docs, committed audit record.

`$ARGUMENTS` is `[selector] [level]`, both optional:
- **Omit the selector** to list the whole backlog, ranked, and stop for the
  user to choose. This is the intended way to open the command.
- A selector may name a tier (`T1`), item IDs from the listing (`T1-2,T2-1`),
  or a target path (`src/maou/interface`) to take everything aimed at it.
- Second token: effort level `low|medium|high|max`, default `medium`.

## What this command is *not*

It is **not** a path audit. `/audit-and-fix <path>` sweeps a whole path and
earns that path a `done` row in the ledger's main table. This command
resolves individual findings inside paths that are mostly still unaudited.

**MUST NOT write a `done` row in `coverage.md`'s main table.** A row there
claims the path was audited; writing one after fixing one finding inside it
would mark unaudited code as covered, which is precisely the silent-staleness
defect the ledger exists to prevent. This command's account lives in its own
record file, linked from the backlog section (step 5).

## Standing principle: a recorded finding is a hypothesis

Every item in the backlog was true **as of some SHA**, written by a session
that no longer exists. Between then and now the code may have been fixed,
moved, renamed, or changed shape enough that the recorded fix is wrong.

**MUST re-verify every candidate against HEAD before presenting it** (step
2), and again before fixing it. Read the actual `file:line`. A record that
says "X has zero callers" is a claim to re-check, not a fact to act on.

This is not a formality. A worked example, from the run that motivated this
command: a backlog row said `get_opening_name` failed to check
`_initial_sfen`, and named the sibling that "does branch on `_initial_sfen`"
as proof. The obvious fix — return early when `_initial_sfen is not None` —
would have **disabled the feature entirely in production**, because the
producer resolves 平手 to a concrete SFEN string rather than leaving it
`None`, so the field is never `None` on the real path. The record was right
about the bug and wrong about the fix. Only reading the producer surfaced
that.

So: trust a record for **where to look**, never for **what to do**.

## Hard constraints

- **Never `--no-verify`.** Pre-commit runs on every commit.
- **Durable docs are never edited before approval.** `CLAUDE.md`,
  `AGENTS.md`, and tracked prose under `docs/` or the repo root go through a
  `reviews/*.md` proposal reconciled in this run (step 4d) — identical to
  `/audit-and-fix`'s routing rule. Source files, including their docstrings
  and Japanese comments, are fixed directly.
- **One item may not silently grow into its neighbours.** If fixing a
  selected item requires changing something the user did not select, stop and
  say so (step 4a). Scope creep is what makes a backlog unresumable.
- **Respect the Code Exploration Policy.** Verification that needs to read
  multiple unfamiliar files is delegated to an `Explore` agent.
- **Serena MCP tools are called one at a time** — never in parallel.
- **Every selected item ends in one of three states**: resolved (fixed +
  committed), re-triaged (still open, with sharpened reasoning recorded), or
  rejected (won't do, recorded as such). Silently dropping a selected item is
  the one outcome that is not allowed.

## Steps

### 1. Gather the backlog — from `coverage.md`, and only from there

Assume no memory of any previous run.

**1a. Sync first.**
```bash
git fetch origin <current-branch>
git status -sb
```
Report if behind; do not auto-merge. A stale ledger silently re-proposes
work another session already finished.

**1b. Read `audits/coverage.md` § "Open findings backlog" in full** — both
the Deferred backlog and the Out-of-scope backlog tables. That is the
complete worklist.

**1c. MUST NOT read the records' `## Deferred` / `## Out of scope` sections
to decide what work remains.** A record is an immutable account of one run
at one time: its Deferred section says "as of that run, this was deferred",
and that stays true forever even after the finding ships. Gathering from
records therefore re-surfaces resolved findings on every run with no way to
remove them — the backlog would never shrink, and step 2 would burn its
verification budget re-confirming work that is already done.

Deleting a row from `coverage.md` is what makes a finding **consumed**, and
it is the only mechanism that does. So `coverage.md` is the authority on
what is open, and records are the authority on what happened.

You may still *open* a record — to recover the full reasoning behind a row
you are about to act on (rows are condensed; records are not). Reading a
record for context is fine. Reading it for the worklist is the error.

**1d. Reconcile, don't second-guess.** If a record documents an open
finding that has **no row** in `coverage.md`, that is a **retrieval bug in
the ledger**: the finding is invisible to every future run. Report it, add
the missing row, and say so in step 5 — but treat this as a ledger repair,
not as a licence to gather from records generally. The reverse case (a row
whose record is gone) is a broken link worth reporting too.

### 2. Re-verify each candidate against HEAD

Per the standing principle above. For each item, before it reaches the user:

- Open the cited `file:line`. Does the code still look like the finding
  describes?
- `git log --oneline <record last_sha>..HEAD -- <target path>` — commits
  since the record was written are where a finding goes stale.
- For "zero callers" / "dead code" claims, re-run the search across `src/`
  **and** `tests/` **and** `docs/` — a documented public API is not dead in
  the sense that matters.
- For any fix that touches a value's origin, **read the producer**, not only
  the consumer. This is where the worked example above went wrong.

Mark each item: **confirmed** (still true) / **stale** (already fixed or
moved — delete its row in step 5 and say so) / **changed shape** (still real,
different fix than recorded).

Do not present an unverified item as actionable. Verification is cheap
relative to writing the wrong fix.

### 3. Rank, then let the user choose

Rank every confirmed item into tiers. The rubric is **impact first, cost
second** — a cheap fix to a cosmetic problem does not outrank a real defect:

| Tier | Meaning |
|---|---|
| **T1** | Wrong behavior a user or an operator can see — wrong output, a CLI option that cannot work, silently corrupted results — **and** the fix is contained. |
| **T2** | Wrong instructions that cause wrong work: agent-facing files (`.claude/`), or doc drift that a reader would act on. Cheap, high leverage. |
| **T3** | Real but needs a decision before code can move: dead code (keep vs. delete), an unread parameter, a protocol/type gap. Contained once decided. |
| **T4** | Large refactors — duplication spanning files, base-class extraction. Correct to do, wrong to fold into a backlog run; each deserves its own change. |
| **T5** | Cannot be validated in this environment (needs GPU hardware, a long measurement, production data). Report as un-consumable *here* rather than guessing. |
| **T6** | New documentation to author, or findings the original record could not confirm. |

Present the ranked list with, for each item: an ID (`T1-1`), the source
record, the target path, the one-line finding, verification result from step
2, what decision (if any) the user must make, and the expected blast radius.

Then **stop and ask** which to consume, via `AskUserQuestion`. Offer the
tiers as coarse choices (e.g. "all of T1", "T1+T2") alongside individual
IDs. Recommend a batch and say why — usually all of T1 plus any T2, since
those are contained and their cost is dominated by the QA cycle they share.

If the selector was given in `$ARGUMENTS`, skip the question and confirm the
resolved item set in one line before proceeding.

**T3 items need their decision taken here, before any code changes.** Ask it
as a real question with the consequences of each branch (deleting a public
function removes its tests too; keeping it means fixing the defects that
made it a finding). Do not pick the default yourself.

### 4. Fix, on the normal route

Group the selected items **by owning path** so each commit is one coherent
fix rather than a mixed bag.

**4a. Check the scope boundary first.** If a selected item cannot be fixed
without touching something unselected, stop and report before editing.
Two legitimate resolutions: pull the neighbour in with the user's agreement,
or re-triage the item as still-open with the coupling recorded. Silently
widening is not one of them.

**4b. Apply source fixes.** Same triage as `/audit-and-fix` step 1: contained
and unambiguous → apply. Check the `infra → interface → app → domain` rule
before any fix crossing layers. Prefer the fix that cannot regress the same
way twice — a derivation over a refreshed hardcoded list, a name-keyed lookup
over a positional one.

**4c. Test, QA, version, commit** — per item group:
- **Regression test is mandatory, not optional.** CLAUDE.md asks for one on
  every bug fix, and here the bug was already diagnosed once and survived; a
  test is what stops it being re-filed a third time. Where the correct fix
  differed from the recorded one (step 2 "changed shape"), **test the trap**
  — pin the behavior that the naive fix would have broken.
- Verify the test is non-vacuous: neuter the fix, confirm the test fails,
  restore.
- QA per scope class: Python → `ruff format`, `ruff check --fix`, `mypy`,
  `pytest` on the mirrored test path. Rust crate → `cargo test -p <crate>`
  under CLAUDE.md §"重いテスト (Rust dfpn)". Other → whatever
  `.pre-commit-config.yaml` configures.
- Bump the **owning** manifest (nearest ancestor `Cargo.toml` with a
  `version`, else the owning `pyproject.toml`). Semver from the change:
  `fix:` patch, `feat:` minor, breaking major. One bump per commit that
  touches `src/` or `rust/`.
- Commit per group, naming the backlog origin in the body so the fix is
  traceable back to the record that filed it.

If QA cannot run in this environment, say which tool was blocked and why,
and record it in step 5's Environment notes — never report an unrun check as
passing.

**4d. Durable-doc drift → propose, then reconcile in this run.** A source
fix routinely invalidates prose: a doc that accurately described the old
(broken) behavior becomes wrong the moment the behavior changes. Check for
that on every fix — `docs/commands/<command>.md` when a CLI option changes,
and any doc describing behavior the fix altered.

File one `reviews/$(TZ=Asia/Tokyo date '+%Y-%m-%d')-<kebab-title>.md`,
`status: pending`, with exact before/after text per the shape in
`docs/memory-architecture.md` § "Review proposal shape". Commit it
separately from the code. Then present it and take the decision:

- **approve** → apply exactly as written, commit `docs: <title>`, then set
  `status: applied` + `applied_in: <sha>` and commit that.
- **reject** → `status: rejected` with the reason, retained as do-not-redo
  provenance.
- **defer** → leave `pending`; record it as outstanding in step 5.

### 5. Update the ledger and write the record

Three separate bookkeeping duties. All are required; the first is the one
that keeps the backlog from growing forever.

**5a. Delete the resolved rows** from `coverage.md` — from **either** the
Deferred backlog or the Out-of-scope backlog, whichever table held them.
This is the step that makes consumption real; without it the finding is
still open as far as every future run is concerned.

Also delete rows that step 2 found **stale**, noting in 5c that they were
already fixed elsewhere rather than fixed here. Do **not** delete a row that
was merely re-triaged — sharpen its text instead, so the next run inherits
the better reasoning rather than the original impasse.

Update the `Open items` count on the affected main-table row to match.

**5b. Leave the source records alone.** They are immutable accounts; their
Deferred sections stay as written, describing what that run decided at that
time. Do **not** add resolution markers, do **not** renumber, do **not**
move an item into Applied. The row deletion in 5a is the resolution record,
and 5c is its account.

One narrow exception: when step 2 proved a record's **diagnosis or proposed
fix wrong** (not merely resolved — *wrong*), append a short correction to
that record, because leaving it uncorrected makes it actively misleading to
the next reader. State the correction, never the worklist state:

```markdown
   **Correction** (2026-08-09, `<sha>`): the fix suggested above would
   have <consequence>, because <what the record missed>.
```

If you find yourself wanting to write "RESOLVED" or "done" in a record,
that is worklist state — it belongs in 5a's deletion, not here.

**5c. Write this run's record.**
`audits/$(TZ=Asia/Tokyo date '+%Y-%m-%d')-backlog-<slug>.md`, where `<slug>`
names the batch (e.g. `tier-a`, `interface-openings`). Frontmatter mirrors
the record shape in `audits/README.md`, with `path:` naming the **targets
touched** and a `kind: backlog` line marking it as a consumption record
rather than a path audit. Body:

- **Consumed** — one row per item: source record, target, and what shipped.
- **Applied** — the fixes, with `file:line` and commit SHA.
- **Re-triaged** — items selected but left open, with the sharpened reason.
  This is the section that earns the run its keep: a second impasse on the
  same item is far more informative than the first.
- **Corrections to the source records** — where step 2 found a record's
  diagnosis or proposed fix wrong. Record the *reason*, so the next reader
  learns to distrust that class of claim.
- **Doc findings** — `reviews/<file>` and its resolved status.
- **Out of scope** — anything new this run noticed; append it to
  `coverage.md`'s backlog too, or it is lost.
- **Environment notes** — what could not be run here, and why.

Link the record from `coverage.md` under the backlog table so a future run
can find the account of a row that is no longer there. Then commit:
```
docs(audits): record backlog consumption (<slug>)
```

**This step also runs when the run stops early.** A selected-but-unfinished
item must leave a record with its resume point, or it is indistinguishable
from one never attempted.

### 6. Report

Compact summary:
- Backlog size found: N deferred + M out-of-scope (from `coverage.md`)
- Verification: confirmed / stale / changed-shape counts
- Selected and resolved (IDs + one line each), with commit SHAs
- Re-triaged / rejected, with the reason
- Doc drift → `reviews/<file>` and its status
- QA: what ran, what passed, what was blocked and why
- Version bump(s), old → new, which manifest
- Ledger: rows deleted (which table), rows added by 1d's repair, any
  record correction from 5b, new record path
- Backlog remaining, by tier — so the next run knows what is left

### 7. Handoff

Do **not** call `/checkpoint-context` from here; step 5 already committed.

The committed record is what carries this run forward. Do not mirror it into
`worklog/` or `scratchpad/compass.md` — per CLAUDE.md, `audits/` and the
campaign memory are deliberately separate systems.

## Extending this command

Keep the derive-never-enumerate property:

- **New audited path** — nothing to do, provided `/audit-and-fix` appended
  its open findings to `coverage.md`. If a path's findings never reach the
  backlog tables, no amount of work here recovers them; that is a defect in
  `/audit-and-fix` step 9, not in this command.
- **New scope class** (new language, new top-level dir) — add its QA path to
  4c, keyed off a discoverable marker (a manifest, a
  `.pre-commit-config.yaml` entry), not a hardcoded path.
- **New durable-doc location** — verify 4d's routing test classifies it
  correctly; if it does, no edit is needed here.
- **A tier that keeps coming up empty** — that is a signal about the repo,
  not about the rubric. Leave the tier; report the emptiness.

## Usage

- `/audit-backlog` — list the whole backlog, ranked, and stop for a choice.
  The intended entry point.
- `/audit-backlog T1` — consume every Tier 1 item.
- `/audit-backlog T1-1,T2-3 high` — consume two specific items, broader
  verification.
- `/audit-backlog src/maou/interface` — consume everything targeting that
  path (without auditing the path itself — use `/audit-and-fix` for that).
