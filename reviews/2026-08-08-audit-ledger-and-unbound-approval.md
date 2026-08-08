---
status: applied
applied_in: 3904559
date: 2026-08-08
target: [CLAUDE.md, AGENTS.md, docs/memory-architecture.md, reviews/README.md, audits/README.md]
risk: medium
reversibility: moderate
---

# audits/ ledger, approval unbound from /checkpoint-context, and two doc-consistency fixes

## Trigger

Designing `/audit-and-fix` (`.claude/commands/audit-and-fix.md`, commits
`40f30c6` / `5158122`) surfaced four durable-doc problems that the command
cannot work around on its own.

1. **Approval is bound to one command.** CLAUDE.md § MUST rules reads
   "on user approval in `/checkpoint-context` step 5, the model applies
   the edit itself" — and `.claude/commands/checkpoint-context.md` step 5c
   calls itself "the ONLY path by which the model may edit `CLAUDE.md` /
   `docs/architecture.md`". That forces every doc fix found by an audit to
   wait for a separate `/checkpoint-context` run. The user states this
   constraint was never the intent: the safeguard is *approval*, not the
   command it happens in.
2. **No coverage ledger for a repo-wide audit.** ~60k lines of Python
   across 133 files, 7 Rust crates, and 22 design docs cannot be audited
   in one session. Nothing records which paths are done, which are
   partially done, or what was deliberately deferred. `scratchpad/` is
   gitignored, so on a remote container it is destroyed on reclamation and
   cannot carry this state.
3. **CLAUDE.md's crate enumeration is stale.** § Versioning (Rust crates)
   lists five crates; `rust/` contains seven — `maou_convert` and
   `maou_usi` are absent, so neither is currently covered by the
   version-bump MUST.
4. **`AGENTS.md` duplicates CLAUDE.md and has drifted.** It restates
   architecture, type-safety, testing, package-management, commit-format,
   and 日本語記述規則 rules in its own words, and its 日本語 section
   inverts the definitions (labels `，` as 句点 and `．` as 読点; CLAUDE.md
   has 読点 `，` / 句点 `．`). Two sources for one rule set is a standing
   drift generator.

## Proposed change

### 1. CLAUDE.md § MUST rules — unbind approval from the command

Before:
```
- MUST NOT edit `CLAUDE.md` / `docs/` without an **approved** `reviews/*.md`
  proposal. Draft it `status: pending`; **on user approval in
  `/checkpoint-context` step 5, the model applies the edit itself and
  commits** (approval is the safeguard against *silent* edits).
```
After:
```
- MUST NOT edit `CLAUDE.md` / `docs/` without an **approved** `reviews/*.md`
  proposal. Draft it `status: pending`; **on explicit user approval the
  model applies the edit itself and commits**, then sets `status: applied`
  + `applied_in: <sha>` (approval is the safeguard against *silent* edits
  — it is not tied to any one command). `/checkpoint-context` step 5 and
  `/audit-and-fix` step 8 both reconcile proposals; either may take the
  approval.
```

### 2. CLAUDE.md § Versioning (Rust crates) — derive, do not enumerate

Before: a five-item bullet list naming each crate's `Cargo.toml`.

After:
```
- MUST bump version in the owning `Cargo.toml` when modifying files under
  `rust/<crate>/`. The owning manifest is the nearest ancestor
  `Cargo.toml` with a `version` field — resolve it from the changed file
  rather than from a list, so crates added later are covered automatically
  (`ls rust/` is the current set; as of 2026-08-08: `maou_convert`,
  `maou_index`, `maou_io`, `maou_rust` (PyO3 bindings), `maou_search`,
  `maou_shogi`, `maou_usi`).
```

### 3. CLAUDE.md § Repository-Centric Memory Architecture — register `audits/`

Add to the Files table:
```
| `audits/coverage.md` | Repo-wide audit ledger: per-path status, resume point, deferred findings. | yes |
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. | yes |
```

Change the `.claude/commands/checkpoint-context.md` row's Role from
"The only writer." to "Writer of campaign working memory
(`worklog/`, `scratchpad/`)." — it was never the only writer of
`reviews/`, and it is not the writer of `audits/`.

Add MUST rules:
```
- MUST record every `/audit-and-fix` run in `audits/` (ledger row +
  record file) and commit it — `audits/` is the only cross-session
  record of repo-wide audit coverage, and unlike `scratchpad/` it
  survives container reclamation.
- MUST keep `audits/` independent of the campaign memory
  (`worklog/` / `scratchpad/compass.md`). The campaign layer tracks one
  stable-environment measurement campaign; `audits/` tracks traversal of
  the tree. Do not mirror one into the other.
```

### 4. `audits/README.md` — new file

Defines the ledger and record shapes, the status vocabulary
(`in-progress` / `done` / `blocked`), and the resume protocol. Full text
in the file itself as committed.

### 5. `AGENTS.md` — reference instead of restate

Replace the duplicated rule sections (Development Guidelines, Core
Development Rules, Code Quality Standards, Testing Requirements, Python
Tools, Commit Guidelines, Pull Requests, 日本語記述規則) with a pointer to
`CLAUDE.md` as the single source, keeping only what is genuinely
Codex-specific: the `.codex/` config pointers and the `uv run` shell
requirement. Retain the Attribution section's correction note verbatim —
it is provenance for a rule that was already wrong once
(`reviews/2026-07-29-csa-floodgate-client.md`).

### 6. `docs/memory-architecture.md` and `reviews/README.md` — consistency

Update both status-lifecycle blocks so approval is not attributed solely
to `/checkpoint-context`, and fix `reviews/README.md`'s stale "user
applies the edit + commits" (CLAUDE.md and `docs/memory-architecture.md`
both say the model applies it). Add `rejected` to `reviews/README.md`'s
frontmatter comment, which still lists only `pending | approved |
applied` while the lifecycle section documents rejection.

## Motivation

Items 1 and 2 block the repo-wide audit outright: without (1) every doc
fix stalls behind an unrelated command, and without (2) the work cannot
span sessions, which for a tree this size means it cannot be finished at
all. Items 3 and 4 are live drift — (3) leaves two crates outside a MUST
rule, and (4) has already produced a 日本語 rule that contradicts
CLAUDE.md.

## Alternatives considered

**For the ledger (2):**
- *Keep coverage in `scratchpad/current.md`.* Ruled out: gitignored, so it
  is lost when a remote container is reclaimed — exactly the
  cross-session case this must serve.
- *Track coverage in `worklog/` checkpoints.* Ruled out: worklogs are
  immutable per-checkpoint snapshots, so current coverage would have to be
  reconstructed by replaying them, and they are gitignored too. The user
  also asked to keep the campaign memory unmerged with audit traversal.
- *A GitHub issue or project board.* Ruled out: state would live outside
  the repo, breaking the repository-centric memory principle and
  offline/derived-clone work.

**For approval (1):**
- *Leave approval in `/checkpoint-context` only.* Ruled out by the user:
  the coupling was incidental, not intended.
- *Let `/audit-and-fix` edit docs without a proposal.* Ruled out: deletes
  the audit trail and the safeguard against silent durable-doc edits.

**For AGENTS.md (4):**
- *Delete `AGENTS.md`.* Ruled out: it is the discovery entry point for
  Codex agents and carries `.codex/` pointers with no other home.
- *Keep both and add a "keep in sync" note.* Ruled out: that is the
  current state minus the honesty; it has already drifted.

## What this enables

- A doc fix found during an audit can be proposed, approved, and applied
  in the same session, with the `reviews/` audit trail intact.
- A repo-wide audit can stop mid-path and resume in a fresh session — on
  a different machine or after container reclamation — without
  re-discovering what was already done or deliberately skipped.
- `maou_convert` and `maou_usi` come under the version-bump MUST, and
  future crates are covered on creation with no doc edit.
- One source of truth for shared rules, so a Codex agent and a Claude
  agent cannot be following contradictory 日本語 conventions.

## What this constrains

- `/audit-and-fix` may no longer silently fix docs — the proposal step is
  mandatory even when approval follows immediately, which costs a round
  trip per run that has doc findings.
- Every audit run must write and commit `audits/` state, so an audit
  cannot be left entirely unrecorded.
- `audits/` becomes committed history: deferred findings are public and
  must be written to be understood by a reader who was not in the session.
- `AGENTS.md` can no longer be edited in isolation to change a shared
  rule; such a change goes through `CLAUDE.md` and this same review
  process.
- The `audits/` ledger is one file, so concurrent audits in parallel
  sessions can conflict on it. Accepted: the alternative (per-path status
  files) trades a merge conflict for an unreadable coverage view.

## Rollback plan

- Revert this commit: `CLAUDE.md`, `AGENTS.md`,
  `docs/memory-architecture.md`, `reviews/README.md` return to their
  prior text; `audits/` is deleted.
- `/audit-and-fix` then loses its step 0 ledger read and step 8 approval
  branch — revert `.claude/commands/audit-and-fix.md` to `5158122`
  alongside, or those steps reference a directory that no longer exists.
- No code depends on any of this; nothing to rebuild and no version bump
  to undo. Audit records already written would be lost, so capture any
  outstanding deferred findings before reverting.
