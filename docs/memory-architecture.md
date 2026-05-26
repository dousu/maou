# Repository-Centric Memory Architecture

Long-term memory lives in the repository, not in conversation history
or `/compact` summaries. Treat the conversation as ephemeral, the repo
as the authoritative store, and the system as model-agnostic
(Claude / GPT / future).

The architecture has **two slash commands** and **three persistence
layers**.

## Files

| Path | Role | Committed |
|---|---|---|
| `CLAUDE.md` | Durable project rules. | yes |
| `docs/memory-architecture.md` | Full spec (this file). | yes |
| `reviews/YYYY-MM-DD-<title>.md` | Proposals + audit trail. Status in frontmatter. | yes |
| `scratchpad/current.md` | Authoritative current working state. | no — `.gitignore`d |
| `worklog/YYYY-MM-DD-HHMMSS.md` | One file per checkpoint, JST. Immutable. | no — `.gitignore`d |
| `.claude/commands/checkpoint-context.md` | The only writer. | yes |
| `.claude/commands/resume-context.md` | Read-only resume. | yes |

`scratchpad/` and `worklog/` are per-developer working memory and are
gitignored: they keep noise out of PRs and let each developer
checkpoint freely. `reviews/` is the durable audit trail — committed
because it answers "why does `CLAUDE.md` say what it says?".

## Flow

```
session start
  └─ /resume-context [filename]
        └─ reads CLAUDE.md + current.md + one worklog snapshot
        └─ emits a Confirmed / Assumed / Unresolved brief, then stops

session work
  └─ work normally; current.md stays whatever the last checkpoint left

session end / context heavy / handoff
  └─ /checkpoint-context [reason]
        ├─ writes a new worklog/$(TZ=Asia/Tokyo date +%Y-%m-%d-%H%M%S).md
        ├─ overwrites scratchpad/current.md
        ├─ if architecture changed: drafts reviews/<date>-<title>.md
        │                            (frontmatter status: pending)
        └─ reconciles all pending reviews with the user:
              approve → user applies + gives SHA
                     → frontmatter becomes status: applied, applied_in: <sha>
              reject  → delete file
              defer   → leave as pending; resurfaces next checkpoint
```

## Worklog entry shape

One file per checkpoint. Filename: `worklog/YYYY-MM-DD-HHMMSS.md` (JST).
Immutable — corrections go in a new file that references the old one.

````markdown
# Checkpoint YYYY-MM-DD HH:MM:SS JST — <reason>

## Objective
<one paragraph: the problem this session is solving and why now>

## Recent changes (this session)
- <file:line> — <what changed and why>
- ...

## Modified files (uncommitted)
```
<verbatim `git status --short`>
```

## What was tried
1. <action> → <outcome>: <verbatim error / observation / test result>
2. ...

## Failed experiments
- <approach>: <signal that killed it — error, wrong output, broken test>
- ...

## Reasoning preserved
<why the current direction was chosen. Name alternatives ruled out and
the criterion that ruled them out. Do not collapse to a conclusion.>

## Unresolved
- <question or unknown> — <what would resolve it>
- ...

## Risks
- <risk>: <likelihood> / <blast radius> / <mitigation>
- ...

## Uncertainty
- <claim NOT verified> — confidence: low | medium
- ...

## Next actions
1. <concrete next step, file-level if possible>
2. ...

## Architectural decisions
<"none" OR bullet list with link to reviews/<file>.md>
````

Empty sections must be written as `_(none)_`, not omitted — absence is
information.

## current.md shape

Overwritten on every checkpoint. Target ~100-150 lines.

```markdown
# Current Working State

## Branch
- Branch: <name>
- Last commit: <short-sha> <subject>
- Dirty files: <n>

## Focus
<one paragraph: problem + why now>

## In-flight changes
- <file> — <what is changing and why>

## Todo
- [ ] open item
- [~] in progress
- [!] blocked — see worklog/<file>.md
- [x] done (will be cleared at next checkpoint)

## Open questions / blockers
- <question> — see worklog/<file>.md

## Next concrete step
<single unambiguous action; if unclear, write the question to resolve>

## Session handoff
- 3-5 bullets a cold session needs to start
- Including "what NOT to redo"
```

## Review proposal shape

One file per proposal. Filename: `reviews/YYYY-MM-DD-<kebab-title>.md`.
Single flat folder; lifecycle is tracked by the `status:` frontmatter
field.

```markdown
---
status: pending          # pending | approved | applied
applied_in:              # commit SHA, filled when status becomes applied
date: YYYY-MM-DD
target: [CLAUDE.md, docs/architecture.md]
risk: low | medium | high
reversibility: trivial | moderate | hard
---

# <Title>

## Trigger
worklog/<file>.md — what specifically forced this proposal.

## Proposed change
Concrete before/after. For a `CLAUDE.md` edit, paste the exact paragraph.

## Motivation
What in the worklog forced this. Cite failed attempts or unresolved
issues — not a vague "to improve quality".

## Alternatives considered
At least two. For each: what it would look like and why ruled out.

## What this enables
A capability or guarantee that did not exist before.

## What this constrains
Things that become harder or forbidden. ("Nothing" is usually a sign
the proposal is too vague — sharpen it.)

## Rollback plan
How to undo: which files revert, what code breaks, what to redo.
```

### Status transitions

```
pending  → applied   (user approves during /checkpoint-context;
                      user applies + commits;
                      model writes status: applied, applied_in: <sha>)
pending  → deleted   (user rejects during /checkpoint-context)
```

`status: applied` is terminal — applied proposals are never modified.
The intermediate value `approved` is permitted for the rare case where
approval and apply are split in time; normal flow skips it.

## Confirmed / Assumed / Unresolved

Discipline applied by `/resume-context` so the model never confuses
audited fact with model-authored prose.

- **Confirmed** — file content read this turn, `git` command output run
  this turn, or a `reviews/*.md` with `status: applied`.
- **Assumed** — prose in `current.md`, a worklog file, or a review with
  `status: approved`. Safe for small things; verify before any
  irreversible step.
- **Unresolved** — explicit unknowns in `current.md` / loaded worklog,
  blocked todos, drift between `current.md` and `git`, and names of
  pending reviews.

## Token budget guidance

| Source | Approx lines | Note |
|---|---|---|
| `CLAUDE.md` | ~300 | always loaded |
| `scratchpad/current.md` | ~150 | always loaded |
| one `worklog/*.md` snapshot | ~100-200 | one file only |
| `git status / log -10 / rev-parse` | ~30 | always |
| `ls reviews/` | ~20 names | open files on demand |

Target: ≤ ~600 lines of file I/O for a typical resume. Above ~1,500
you are over-loading history — stop and re-evaluate.

## What was deliberately left out

| Idea | Why not |
|---|---|
| `/memory-sync` (lighter end-of-session) | Subset of `/checkpoint-context`. One writer keeps state coherent; empty sections in the strict template collapse to `_(none)_` cheaply. |
| `/propose-knowledge` (ad-hoc draft) | Proposals are either filed automatically by `/checkpoint-context` step 4 or written by hand into `reviews/`. A dedicated command added surface without unique value. |
| Hooks (`.claude/hooks/`) | The system is human-triggered by design — keeps it model-agnostic. If "I forgot to checkpoint" becomes a recurring failure, re-introduce a `Stop` hook. |
| `pending/` vs `approved/` subdirs | Replaced by `status:` frontmatter; collapses the `git mv` + `Applied-in:` append into a single in-place edit driven by `/checkpoint-context` step 5. |
| `scratchpad/todo.md`, `scratchpad/decisions.md` | Todos fold into `current.md`'s `## Todo` section. Small decisions live in the worklog; durable decisions live in `reviews/`. |

## Anti-patterns

- Loading multiple worklog files at resume time.
- Treating `current.md` as immutable history (it is *current* state).
- Treating worklog files as mutable (they are append-only by creation,
  not by editing).
- Silently editing `CLAUDE.md` without a `reviews/` trail.
- Generic checkpoint summaries that hide failed experiments.
- Treating `status: pending` reviews as policy.
