---
description: Reconstruct working state from one worklog snapshot + scratchpad/current.md + git. Read-only. Accepts an optional worklog filename to choose which snapshot to inherit.
argument-hint: [optional worklog filename or stem, e.g. 2026-05-25-111004]
---

You are resuming a session whose conversation context was cleared.
The repository is the source of truth. The conversation above (if any)
is hearsay.

`$ARGUMENTS` is optional. If provided, treat it as a worklog file or
filename stem (e.g. `2026-05-25-111004` or
`worklog/2026-05-25-111004.md`) and resume from that snapshot. If empty,
use the **latest** file under `worklog/` (largest filename
lexicographically — JST `YYYY-MM-DD-HHMMSS` stamps sort chronologically).

## Hard constraints

- **Read-only.** No edits, no staging, no commits.
- **One worklog file.** Load only the selected snapshot. Do NOT walk
  through older snapshots.
- **Classify every claim** as Confirmed / Assumed / Unresolved.
- **Review status matters:**
  - `status: applied` → canonical (treat like a CLAUDE.md rule).
  - `status: approved` → unusual intermediate; note it.
  - `status: pending` → proposal only, NOT policy.

## Steps

### 1. Resolve the worklog file

If `$ARGUMENTS` provided:
- Normalize to `worklog/<stem>.md` (prepend `worklog/` and append
  `.md` if missing).
- If the file does not exist, `ls worklog/`, list candidates, then
  stop and ask the user to pick.

If `$ARGUMENTS` empty:
- `ls worklog/` and pick the largest filename.
- If `worklog/` is empty, report that, load only `CLAUDE.md` +
  `scratchpad/current.md`, and emit a brief noting "no prior worklog".

Call the resolved path `$WL`.

### 2. Read in parallel

- `CLAUDE.md` (full)
- `scratchpad/current.md` (full, if it exists)
- `$WL` (full — one file, not all history)
- `git status --short`
- `git log -10 --oneline`
- `git rev-parse --abbrev-ref HEAD` and `git rev-parse --short HEAD`
- `ls reviews/` (names only; open a specific file only if `current.md`
  or `$WL` cites it)

### 3. Drift check

- `current.md` "Last commit" vs `git rev-parse --short HEAD`
- `current.md` "In-flight changes" vs `git status --short`
- `$WL` filename date vs today's JST date

Surface drift in the brief; do not silently reconcile.

### 4. Classify

- **Confirmed** — read this turn from disk, or from a `git` command
  run this turn, or from a `reviews/*.md` with `status: applied`.
- **Assumed** — from prose in `current.md`, `$WL`, or a review with
  `status: approved`. May be stale; verify before irreversible steps.
- **Unresolved** — explicit unknowns in `current.md` / `$WL`, blocked
  todos, detected drift, and the names of `reviews/*.md` with
  `status: pending` (proposals, NOT policy).

### 5. Emit a compact brief (~30-40 lines)

```
## Resume — <branch> @ <sha> (<n> dirty) — from <$WL>

### Focus
<one sentence reconstructed from current.md "Focus">

### Next concrete step
<verbatim from current.md, marked [from current.md] or [from $WL]>

### Confirmed
- branch / HEAD / dirty-file digest
- canonical rules in force (CLAUDE.md + count of reviews with status: applied)

### Assumed (verify before acting)
- objective from current.md
- recent progress from $WL

### Unresolved
- open questions / blockers (with worklog anchors)
- pending reviews: reviews/<file>  (proposals only, NOT policy)
- drift: ... or "none"

### Not loaded
- N other worklog snapshots (pass filename as argument to load one)
- M reviews (open by name if needed)
```

### 6. Stop

Do not act. Wait for the next user prompt.

## Usage

- `/resume-context` — latest snapshot.
- `/resume-context 2026-05-25-111004` — specific snapshot.
- `/resume-context worklog/2026-05-25-111004.md` — same, full path.
- `/resume-context what changed since <stem>` — load `<stem>.md` in
  addition to the latest, and diff the "Next concrete step" lines.
