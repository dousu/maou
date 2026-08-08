# audits/

Cross-session coverage record for the repo-wide audit driven by
`/audit-and-fix` (`.claude/commands/audit-and-fix.md`).

The tree is too large to audit in one session — ~60k lines of Python
across four layers, 7 Rust crates, and 22 design documents. This
directory is what lets a run stop mid-path and resume later, in a
different session, on a different machine, or after a remote container
has been reclaimed.

**Committed on purpose.** `scratchpad/` and `worklog/` are gitignored, so
they cannot carry state across container reclamation. `audits/` can.

**Independent of the campaign memory.** `worklog/` and
`scratchpad/compass.md` track one measurement campaign in a stable
environment; `audits/` tracks traversal of the tree. They answer
different questions and are deliberately not merged — do not mirror one
into the other.

## Files

| Path | Role |
|---|---|
| `audits/coverage.md` | The ledger. One row per audited path: status, resume point, open items. |
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. Immutable once the run's status is `done`. |

`<path-slug>` is the target path with separators flattened:
`src/maou/domain/model` → `src-maou-domain-model`.

## The ledger holds only what was audited

`coverage.md` does **not** enumerate the paths that still need auditing.
Such a list would go stale on every new module — the same defect the
audit exists to find. Rows are added as paths are worked. To see what is
left, compare the ledger against the tree at read time (`ls`, `find`),
which is always current.

## Status vocabulary

| Status | Meaning |
|---|---|
| `in-progress` | Started, not finished. **Resume point is mandatory** — which step, and which sub-paths are already covered. |
| `done` | Every step completed for this path. Code fixes committed; doc findings either proposed-and-resolved or recorded as none. |
| `blocked` | Cannot proceed without a decision or an external change. The blocker must name what would unblock it. |

An absent row means "never audited" — no row is written speculatively.

A `done` row is not permanent truth: it is `done` **as of a commit**. When
the path changes materially afterwards, the row's `Last SHA` is what tells
a later reader the audit predates the change.

## Ledger row shape

```markdown
| Path | Scope | Status | Level | Last SHA | Record | Open items |
|---|---|---|---|---|---|---|
| `src/maou/domain/model` | python | done | high | `abc1234` | [2026-08-08](2026-08-08-src-maou-domain-model.md) | 0 |
| `rust/maou_shogi` | rust | in-progress | max | `def5678` | [2026-08-09](2026-08-09-rust-maou-shogi.md) | 3 |
```

## Record shape

```markdown
---
path: src/maou/domain/model
scope: python | rust | docs | other
level: low | medium | high | max
status: in-progress | done | blocked
started: YYYY-MM-DD          # JST
last_sha: <short sha>
---

# Audit — <path>

## Resume point
<Required while `in-progress`. Which step of /audit-and-fix is next, and
which files or sub-paths under <path> are already covered. Written for a
reader with no memory of the session — "step 4, remaining: loss/, move/"
not "continue where I left off".>

## Applied
<Code fixes made, with file:line and commit SHA.>

## Deferred
<Findings deliberately not fixed, verbatim, with file:line and WHY —
ambiguous, cross-layer, needs a decision. This is the section that stops
the next session from re-deriving the same finding and reaching the same
impasse.>

## Doc findings
<Drift found, and where it went: reviews/<file> (with its status), or
"none — docs verified accurate".>

## Out of scope
<Issues noticed outside <path>, as /audit-and-fix <path> suggestions.>
```

## Protocol

1. `/audit-and-fix` reads `coverage.md` before starting, to pick up an
   `in-progress` row for the same path rather than restarting it.
2. It writes the record and the ledger row, and commits them, at the end
   of the run — and also when stopping early, which is what makes an
   interrupted run resumable.
3. `blocked` rows surface to the user; they are not silently retried.

## See also

- `.claude/commands/audit-and-fix.md` — the command that writes this.
- `reviews/README.md` — where doc findings go for approval.
- `docs/memory-architecture.md` — the separate campaign memory system.
