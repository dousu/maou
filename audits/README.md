# audits/

Cross-session coverage record for the repo-wide audit, driven by
`/audit-and-fix` (`.claude/commands/audit-and-fix.md`) for whole paths and
`/audit-backlog` (`.claude/commands/audit-backlog.md`) for the individual
findings those runs leave open.

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
| `audits/coverage.md` | The ledger. One row per audited path: status, resume point, open items. Also carries the **out-of-scope backlog**. |
| `audits/YYYY-MM-DD-<path-slug>.md` | One record per `/audit-and-fix` run. Immutable once the run's status is `done` — it is an account, never a worklist (see below). |
| `audits/YYYY-MM-DD-backlog-<slug>.md` | One record per `/audit-backlog` run (`kind: backlog`). Consumes individual findings; gets no main-table row. |

`<path-slug>` is the target path with separators flattened:
`src/maou/domain/model` → `src-maou-domain-model`.

## The ledger holds only what was audited

`coverage.md` does **not** enumerate the paths that still need auditing.
Such a list would go stale on every new module — the same defect the
audit exists to find. Rows are added as paths are worked. To see what is
left, compare the ledger against the tree at read time (`ls`, `find`),
which is always current.

## The out-of-scope backlog

An audit stays inside its target path, so it regularly notices real
problems it must not fix. Those go in `coverage.md`'s **Out-of-scope
backlog**, not only in the run's own record.

The reason is retrieval, not bookkeeping. A per-run record is read only
when someone opens that specific path — so a finding filed there is
visible exactly to the audit least able to act on it. `coverage.md` is
read at the start of *every* run, which is the only place a cross-path
finding reliably resurfaces.

Each row carries the record that found it, the path that should fix it,
and enough of the finding to act on without reopening the record.

- **Before auditing a path**, check the backlog for rows whose target
  falls inside it, and fold them into the run.
- **At the end of a run**, append any new out-of-scope findings.
- **When an item is resolved**, delete its row. The resolving audit's
  record is the durable account; the backlog is a worklist, not an
  archive. Do not delete a row that was merely re-triaged elsewhere.

## Deferred findings

A record's `## Deferred` section holds findings the audit **confirmed but
deliberately did not fix** — ambiguous, cross-layer, or needing a
decision. A deferred finding is a diagnosis with the fix withheld pending
a decision, **not** a decision never to fix it.

Deferred findings therefore get a row in `coverage.md`'s **Deferred
backlog**, exactly as out-of-scope findings get one in the Out-of-scope
backlog. The retrieval argument above applies to both classes in full:
what is written only into a record is visible only to whoever opens that
record.

`coverage.md` is the authority on **what is open**; records are the
authority on **what happened**. The row is the condensed, deletable index
entry; the record's Deferred section is the durable reasoning behind it.
Both are written, and only the row is ever deleted.

## Records are accounts, not worklists

A `done` record is the account of one run at one time. Its Deferred
section says "as of that run, this was deferred" — and that stays true
forever, **including after the finding has shipped**.

That is why no command reads a record to decide what work remains. Doing
so would re-surface every resolved finding on every run, with no way to
remove it: a record cannot be "cleared" without destroying the account,
so the list would only ever grow. Deleting a row from `coverage.md` is
what marks a finding consumed, and it is the only mechanism that does.

So a record is **never amended to carry state**: no `RESOLVED` markers,
no moving an item from Deferred into Applied, no renumbering. Commit
`916e874` did move a deferred item into Applied — that predates the
Deferred backlog, when the record was the only place to record it, and it
is not the pattern to follow now.

The one narrow exception is a **correction**: when a later run proves a
record's diagnosis or proposed fix *wrong*, append a short note saying
so, because an uncorrected record actively misleads the next reader. A
correction states what the record got wrong, never whether the work is
done:

```markdown
   **Correction** (YYYY-MM-DD, `<sha>`): the fix suggested above would
   have <consequence>, because <what the record missed>.
```

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

## Cross-module sweep
<step 2.5 で導出した sweep key と，各 key の結果．finding だけでなく
**clean だった key も書く** — 「調べて一貫していた」は次の隣接 path 監査が
同じ Explore sweep を再実行しないための結果である．意図的な分岐は理由と
ともにここに記録する．>

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

A run assumes **no memory of any previous run**. Everything needed to
resume is here.

1. **Sync first.** `git fetch origin <branch>` before reading the ledger —
   another session, possibly on another machine, may have pushed records
   after this working copy was created. A stale ledger silently re-audits
   finished paths.
2. **Read the whole ledger, not one row.** Every `in-progress` row is
   reported, *including paths other than the one being audited* — a
   half-finished path left by an earlier session is the thing most likely
   to be lost, and it never surfaces if only the requested row is checked.
   Open findings come from `coverage.md`'s two backlog tables, **not** from
   the record files (see "Records are accounts, not worklists").
3. **Claim before working.** The `in-progress` row and record are written
   and committed *before* the audit starts, not after. A session that dies
   mid-run then still leaves a resume point, and a concurrent session sees
   the path is taken.
4. **Finish or hand off.** At the end of the run the same record and row
   are updated — `done` with the resume point cleared, or `in-progress`
   with a sharpened one. Stopping early takes the same path; that is what
   makes an interrupted run resumable.
5. **Staleness is decided concretely.** For a `done` row,
   `git log <last_sha>..HEAD -- <path>` answers whether a re-audit is
   warranted. What is still open for that path is its rows in the backlog
   tables; the record's Deferred section is worth reading for the *reasoning*
   behind those rows, but it is not the list of what remains.
6. `blocked` rows surface to the user; they are not silently retried.

Running `/audit-and-fix` with **no path** is the intended way to open a
cold session: it reports ledger state and offers the unfinished work.

## See also

- `.claude/commands/audit-and-fix.md` — the command that writes this.
- `reviews/README.md` — where doc findings go for approval.
- `docs/memory-architecture.md` — the separate campaign memory system.
