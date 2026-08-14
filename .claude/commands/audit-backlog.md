---
description: Consume the deferred and out-of-scope findings that /audit-and-fix left behind. Gathers them from audits/coverage.md's backlog tables, re-verifies each against HEAD, then classifies each by DECISION COST on a mechanical six-class ladder — P1-P3 need no user judgment and are fixed, PR'd and merged without asking; P4-P6 need judgment, are built into unmerged PRs, and are then put to the user as ONE AskUserQuestion listing every outstanding decision with the PR that implements it — so what is being asked, and what is waiting, is never left to be reconstructed from diffs. An item is left unwritten before that question only when the branches produce materially different diffs. Every PR of a run is chained into one stack with exactly one merge into main, so siblings never conflict and the wheel builds once. Applies fixes on the normal route (version bump, regression test, reviews/ proposal for durable docs), deletes the resolved backlog rows, and records the run in audits/.
argument-hint: [item-selector or class (P1..P6 / auto / judgment) | omit to take everything] [effort-level: low|medium|high|max, default medium]
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
- **Omit the selector** to do the whole run: consume the auto band
  (step 3b), build the judgment band into unmerged PRs (step 3c), put every
  one of those decisions to the user as a single `AskUserQuestion`
  (step 3d), and act on the answers (step 3e). This is the intended way to
  open the command. It asks exactly once, and only when the judgment band
  is non-empty.
- A selector may name a class (`P4`), a band (`auto`, `judgment`), item IDs
  from the listing (`P3-2,P4-1`), or a target path
  (`src/maou/interface`) to take everything aimed at it.
- Second token: effort level `low|medium|high|max`, default `medium`.

## What this command is *not*

It is **not** a path audit. `/audit-and-fix <path>` sweeps a whole path and
earns that path a `done` row in the ledger's main table. This command
resolves individual findings inside paths that are mostly still unaudited.

**MUST NOT write a `done` row in `coverage.md`'s main table.** A row there
claims the path was audited; writing one after fixing one finding inside it
would mark unaudited code as covered, which is precisely the silent-staleness
defect the ledger exists to prevent. This command's account lives in its own
record file, linked from the backlog section (step 6).

## Standing principle: a recorded finding is a hypothesis

Every item in the backlog was true **as of some SHA**, written by a session
that no longer exists. Between then and now the code may have been fixed,
moved, renamed, or changed shape enough that the recorded fix is wrong.

**MUST re-verify every candidate against HEAD before classifying it** (step
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

Verification is also what makes the classification in step 3 mean anything.
An unverified item cannot be classified — you would be classifying the
record's *description* of a fix rather than the fix.

## Standing principle: priority is decision cost, not severity

This command used to rank findings by impact and stop for the user on every
one of them. That was wrong on the cheap end. Correcting a doc that
contradicts the code, deleting a branch that cannot execute, fixing a log
line that globs the wrong extension — none of these have a second version
the user would have preferred. Asking about them buys nothing and delays
the findings that genuinely need a decision behind a round-trip.

So the ranking axis is **who has to decide**, applied mechanically:

| Class | Band | The fix… |
|---|---|---|
| **P1** | 自動 | touches no shipped file. `.claude/`, `audits/`, `reviews/`, `.github/`, tooling config (`.pre-commit-config.yaml`, ruff/mypy settings), `tests/`, and formatting/lint-only edits. Nothing under `src/` or `rust/` — so there is nothing to version-bump. |
| **P2** | 自動 | touches only tracked prose (`*.md` under `docs/`, the repo root, `CLAUDE.md`), **and** the corrected text follows uniquely from the code as it exists. A drift correction, not an authored opinion. |
| **P3** | 自動 | changes code without changing behavior: for every input the program already accepts, the artifacts it writes and the values it returns are unchanged. Only diagnostics, log text, timing, and memory may differ. |
| **P4** | 要判断 | changes observable behavior, but existing data stays readable and existing invocations stay valid. |
| **P5** | 要判断 | breaks data compatibility: artifacts written by the current code stop being readable, or the new code cannot read what the current code wrote. Schema/dtype/field/file-layout changes live here. |
| **P6** | 要判断 | breaks a contract callers depend on: a CLI option removed or re-meaninged, a public API deleted or re-signatured, a documented output path or format moved. |

**Evaluate P6 first and descend.** An item takes the **highest** class it
triggers, so a fix that corrects a doc *and* changes a dtype is P5, never
P2. Classifying top-down is what makes that automatic rather than a thing
you have to remember.

**The fail-safe direction is up, and it is not negotiable.** If a class's
question cannot be answered without reading more code than the item
justifies — or the honest answer is "probably" — the item is **P4 at
minimum**. A misclassification into the auto band merges without anyone
looking at it; a misclassification into the judgment band costs one
question. Those are not symmetric errors, so do not treat them as one.

### Gates — these demote out of the auto band regardless of class

A gate does not change an item's class; it removes the item from the band
that ships unattended. Report the class *and* the gate.

| Gate | Trigger |
|---|---|
| **G1 unverifiable here** | Correctness cannot be established in this environment — needs GPU hardware, a long measurement, or production data. |
| **G2 scope bleed** | The fix cannot be made without touching something outside the item (step 4a). |
| **G3 no QA** | The QA for this scope class cannot run here, so "behavior unchanged" is an assertion rather than a result. |
| **G4 the record says so** | The row's own stated reason for deferral is a design decision (`見送り理由: …設計判断が要る`, "two directions", "keep vs delete"), and step 2 did not eliminate it. |

G4 is the cheapest signal in the list and the most frequently correct: an
earlier run already looked at this and concluded a human had to choose.
Re-verification can retire a G4 — the decision may have been taken since,
or the alternatives may have collapsed to one — but it must be retired
*explicitly*, with the reason, not by forgetting to look.

### Worked classifications, from rows in the current backlog

These are the calibration set. When a new item's class is unclear, find the
row here it most resembles.

- **O11** — `docs/commands/pre_process.md` says pre-process writes `.npy`
  shards; the code writes `transformed_chunk{NNNN}.feather`. The corrected
  text is dictated by the code, and the diff touches one `.md`. → **P2**.
- **O6** — `bq_data_source.py:483` validates the local cache with
  `glob("*.npy")` while the writer writes `.feather`, so it always logs
  "Created 0 local cache files". Fixing the glob changes a log line and
  nothing else. → **P3** (log text may differ; artifacts and return values
  do not).
- **D12(b)** — a `try/except ImportError` nested inside a `try/except
  Exception`, where the inner arm re-raises only to be caught by the outer.
  Removing the nesting leaves the same exception reaching the same handler.
  → **P3**.
- **O4** — `learn_model.py:876` uses `test_ratio or 0.1`, so
  `--test-ratio 0.0` silently becomes 10%. Fixing it changes what the
  program does for an invocation that is still valid, and no artifact
  becomes unreadable. → **P4**. Note it is *not* P3: the whole point is
  that behavior changes.
- **D1** — `moveWinRate` missing from `get_preprocessing_dtype()`. Adding
  the field changes the structured dtype, so arrays and `.npy` written
  before the change no longer match what the reader expects. → **P5**,
  and the row also carries **G4** (the record names two opposed
  directions, "add to the dtype" vs "drop it right after conversion").
- **D8/D9** — `file_level_split` has zero production callers. "Delete it"
  and "keep it and fix it" are different products; deleting also removes
  its tests. → **P6** if deleted (a public name disappears), **P4** if
  repaired, and **G4** either way. The class is a *consequence* of the
  decision, which is why the decision cannot be skipped.

Note what the calibration set does *not* contain: an item whose class was
decided by how urgent it felt. A user-visible wrong result is not thereby
auto-mergeable, and a cosmetic cleanup is not thereby safe — D2's "just add
a default seed" is one line and lands in P4, because it changes which rows
train and which validate.

Severity has not stopped mattering; it has stopped being the *tier*. It is
the sort key **inside** a class (step 3a).

## Standing principle: every judgment-band item is *asked*, once, with its PR in hand

Classifying a finding as P4/P5/P6 says a human has to decide. The run's job
is to make that decision **legible and answerable** — not merely to leave
it somewhere the user could find it.

An earlier version of this file said the PR *was* the question: write the
fix, open it unmerged, put the alternatives in the body, and let the user
answer by merging or closing. That under-delivered in practice, and the
runs through 2026-08-14 are the evidence. A PR body explains itself to
whoever already opened it, but nothing tells the user **what is being
asked** or **what is waiting on them**. They are left to open each PR, read
a diff, and reverse-engineer which of them contains a real fork and which
is routine — which is exactly the reconstruction the ledger exists to
remove.

So the shape is both, in order:

1. **Build the fix and open its PR** (unmerged). The PR carries the diff,
   the QA output, the compatibility statement, and the alternative. It is
   the *evidence*.
2. **Raise one `AskUserQuestion`** naming, per item, the choice and the PR
   that implements it. It is the *interface*.

The question is what the user answers; the PR is what they consult when the
one-line summary is not enough. Neither replaces the other — a question
without a PR asks about prose instead of a diff, and a PR without a
question leaves the user unsure whether anything is waiting at all.

**The run does not end with an unasked judgment item.** If a judgment-band
PR is open, either the question has been raised, or the run must say
explicitly in its report that it could not raise one and why (see
"When the question cannot be raised" below).

### What the single question contains

Ask **once per run**, batching every outstanding decision, raised *after*
the auto band is settled and the judgment-band PRs are open. A second
question in the same run means the first one was under-specified.

Entries come in two kinds, and one question may hold both:

- **受理を問うもの** — the fix is written and PR'd. The choice is whether to
  take it. Give the option that merges it, and the alternative that was
  rejected: "PR #N: `test_ratio or 0.1` を直す (`0.0` を渡していた実行は
  結果が変わる) / 現状維持".
- **向きを問うもの** — the fix is *not* written, because the branches
  produce **materially different diffs** and implementing the wrong one
  wastes work the user would have to review and discard. Give the branches
  and what each costs.

The second kind is the narrow case, and both halves of its test must hold:

- D8/D9 — "delete `file_level_split`" and "repair it" share no lines: one
  removes a public name and its tests, the other rewrites the constructor.
  Guessing wrong throws away the whole change. **Ask before writing.**
- D1 — "add `moveWinRate` to the dtype" and "drop it right after
  conversion" are opposed designs of comparable size, each touching several
  layers. **Ask before writing.**
- O4 — `test_ratio or 0.1` has exactly one repair; the judgment is only
  whether to accept that results change for anyone passing `0.0`. There is
  nothing to write differently. **Write it, PR it, then ask for
  acceptance.**
- N9 (2026-08-14) — "delete the `.npy` remnant scripts" vs "port them to
  `.feather`". Re-verification showed the port was a rewrite against four
  changed APIs, so the branches collapsed to one in substance; the
  deletion was still written, PR'd, and **asked**, because whether the
  user wants their own tooling gone is theirs to say.
- P5 and P6 items are *not* automatically direction questions. A dtype
  change with one sensible form is written and asked for acceptance; two
  credible forms is asked before writing.

When the branches differ but the diff is small and cheap to redo,
implement the one you would recommend and make it an acceptance question
with the alternative named. A 30-line diff the user rejects costs less
than a round-trip spent describing it.

### Phrase each entry as the decision, not as permission

Each entry carries **the decision itself**: what the branches are, what
each costs, and — for P5/P6 — the concrete compatibility break. Put the
option you would recommend first and say so.

> "`file_level_split` を削除するか修理するか？ 削除は公開名とテストが
> 消える．修理は構築時の全ロードを直すことになる" — これは質問である．
>
> "D8 に着手してよいですか？" — これは質問ではない．

Name the PR number in the entry whenever one exists, so the user can go
from the question to the diff in one step.

### When the question cannot be raised

`AskUserQuestion` may be unavailable, or the run may be executing
unattended. That does not license silence: leave the PRs open, and make
the report state — in its own words, not by linking — **which items are
waiting and what each is waiting on**. The next run picks them up through
step 1e, and the question is raised then.

## Standing principle: the run narrates itself in the conversation

Every artifact this command produces — the classification, the PRs, the
record — is written for someone who has already decided to open it. None
of them tell the user *what the run is doing* or *which of its outputs
needs them*. That job belongs to the conversation, and it cannot be
delegated to a link.

Three moments carry the whole weight:

- **Before any file is touched** (3a): the classification table, plus a
  plain list of what this session will work and what it will leave. A
  reader must be able to see the run's scope without opening anything.
- **When the judgment band is ready** (3d): the single `AskUserQuestion`.
  This is the moment the run tells the user, in as many words, that a
  decision is theirs and what it is. Nothing else in the run does that
  job — a PR left open is a state, not a question.
- **When PRs are opened** (5f): a table mapping each backlog row to the PR
  and commit that answers it. A PR number on its own forces the user to
  read a diff just to learn which finding it belongs to — which is
  precisely the reconstruction the ledger exists to remove.

The failure this prevents is not a missing artifact; it is a run whose
artifacts are all correct and whose user still has to reverse-engineer
what happened. Treat "it's in the PR body" and "it's in the record" as
*non-answers* to "what did this run do?" — those are where the detail
lives, not where the orientation lives.

Both lists are cheap: a dozen lines each, written from material step 3
already produced. Skipping them buys the run nothing and costs the user a
manual pass over every diff.

## Hard constraints

- **Never `--no-verify`.** Pre-commit runs on every commit.
- **Durable docs still go through `reviews/`.** `CLAUDE.md`, `AGENTS.md`,
  and tracked prose under `docs/` or the repo root get a `reviews/*.md`
  proposal recording what changed and why — the audit trail is not what
  P2 removes. What P2 removes is the *round-trip* on a pure drift
  correction, which CLAUDE.md's standing approval covers. Anything beyond
  a drift correction — new guidance, a restructure, a rule — fails P2's
  second test, lands in the judgment band, and waits for a real approval
  (step 4d).
- **Source files, including their docstrings and Japanese comments, are
  fixed directly.** They are code, not durable docs — but a docstring edit
  touches `src/`, so it is classified P3 (or higher) and carries a version
  bump, never P2.
- **One item may not silently grow into its neighbours.** If fixing a
  selected item requires changing something the user did not select, stop
  and say so (step 4a). Scope creep is what makes a backlog unresumable.
- **Respect the Code Exploration Policy.** Verification that needs to read
  multiple unfamiliar files is delegated to an `Explore` agent.
- **Serena MCP tools are called one at a time** — never in parallel.
- **Every selected item ends in one of four states**: resolved (fixed +
  merged), in flight (fixed, PR open, awaiting the user), re-triaged (still
  open, with sharpened reasoning recorded), or rejected (won't do, recorded
  as such). Silently dropping a selected item is the one outcome that is
  not allowed.
- **A backlog row is deleted in the PR that carries its fix**, whenever
  the two cannot merge apart. A row is kept only when its fix could land
  — or fail to land — independently of the deletion, which in practice
  means a judgment-band PR stacked above the `audits/` PR (step 6a).
- **Ask exactly once per run, and cover every judgment-band item** (step
  3d). Once, because a second question means the first was
  under-specified; every item, because an open PR is a state and not a
  question — leaving one out is what forces the user to reconstruct the
  decision from diffs. The only run that asks nothing is one whose
  judgment band is empty, or one that could not raise the question at all
  (say which, in the report).
- **Every PR of the run is chained into one stack** (step 5b), and
  **exactly one of them targets `main`**. There is no "independent, based
  on `main`" case: the version bump and `uv.lock` make siblings conflict by
  construction, and every merge into `main` triggers a wheel build. The run
  lands on `main` once.
- **One merge into `main` per run — bookkeeping included.** The ledger
  update is part of the run, not a sequel to it: a row deletion, a
  `reviews/` status transition, and the run's record all ride in the run's
  single PR (6a). **Never open a follow-up PR into `main` to finish the
  paperwork** — if a decision arrives too late for this run's PR, the next
  run picks it up through step 2 (stale) and step 1e (merged-after-the-fact)
  at no extra cost.

## Steps

### 1. Gather the backlog — from `coverage.md`, and only from there

Assume no memory of any previous run.

**1a. Sync first.**
```bash
git fetch origin main
git status -sb
```
Report if behind; do not auto-merge into the working branch without saying
so. A stale ledger silently re-proposes work another session already
finished — and with an auto band that merges unattended, it would re-open
PRs for findings that are already on `main`.

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
the missing row, and say so in step 6 — but treat this as a ledger repair,
not as a licence to gather from records generally. The reverse case (a row
whose record is gone) is a broken link worth reporting too.

**1e. Check what earlier runs left in flight.** Judgment-band PRs are the
normal end state of a run, so a cold session inherits them. List them
(`mcp__github__list_pull_requests`) and, for each, report it alongside the
backlog row it belongs to — an open PR and its still-present row are the
same finding counted once, not two units of work. **Do not open a second PR
for a row that already has one**; update the existing PR instead.

Three follow-ups belong here rather than to a later step, because each is
work an earlier run could not finish itself:

- **Stale stacks.** A PR still based on a merged or deleted branch shows a
  meaningless diff. Retarget and refresh it per 5b before anything else
  reads it.
- **Merged PRs whose rows are still present.** A row kept under 6a's
  separability test outlives its PR when the merge happens after the run
  ended. Confirm against `main`, delete the row **in this run's own PR**,
  and note it in this run's record. This is the designed recovery path —
  it is why a late decision never justifies a bookkeeping-only PR.
- **`reviews/` proposals left `pending` whose decision has since been
  taken** (the PR carrying them merged, or the user answered on it). Apply
  the proposal, commit, set `status: applied` + `applied_in: <sha>`. This
  is the tail of 4d's non-drift branch, and nobody else picks it up.

### 2. Re-verify each candidate against HEAD

Per the standing principle above. For each item, before it is classified:

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
moved — delete its row in step 6 and say so) / **changed shape** (still real,
different fix than recorded).

Do not classify an unverified item, and do not present one as actionable.
Verification is cheap relative to writing the wrong fix — and far cheaper
than merging one unattended.

### 3. Classify, run the auto band, build the judgment band, then ask once

**3a. Assign every confirmed item a class and its gates.** Apply the ladder
top-down (P6 → P1), then the four gates. Record, per item: ID (`P3-1`),
source record, target path, the one-line finding, verification result from
step 2, **class + the specific test that decided it**, gates, and expected
blast radius.

Sort **within** each class by severity — wrong results a user or operator
can see, ahead of internal cleanups. That is where the old impact rubric
went: it orders work inside a class, it no longer decides which class.

**Print this table in the conversation before touching a single file, and
follow it with the run's plan.** Not after the first fix, not alongside
it — before. This is the first of the two moments in the standing
principle above. It is the user's chance to stop a misclassification
before it merges, and it is the artifact that makes the classification
auditable after the fact. A table printed once work is already underway
has lost the half of its job that mattered.

The plan is a second, shorter list, and it answers a different question
than the table: **what is this session actually going to do?** The
classification says what each item *is*; the plan says which items get
worked, which get left, and in what order. Write it as two blocks:

```markdown
### このセッションで処理する
1. D3+D4 (P3, 自動帯) — columnar 変換表の 1 本化 + structured array 復元の統合
2. D2 (P4) — 行分割の既定 seed
3. N-2 (P6) — `.npy` ベンチの扱い (向きはユーザに確認してから書く)
4. N-3 (doc) — `reviews/` 提案のみ，doc は編集しない

### このセッションでは処理しない (残す理由)
- Deferred 5/6/7, O9 — G1: GPU / BigQuery がこの環境に無い
- Deferred 2/4 — G3: ~400 行の学習経路リファクタで等価性を示せない
- …
```

The "処理しない" half is not padding. A run that lists only what it will
do leaves the user unable to tell a deliberate deferral from an
oversight — and that distinction is the entire value of a backlog that
has already been triaged once.

Do not begin step 3b until both blocks are on screen.

**3b. Consume the auto band (P1 → P2 → P3) without asking.** The auto band
never appears in 3d's question — that is what the classification bought.
No pause here. Work the classes in ascending order — P1 is
both the cheapest and the least able to break anything, so a failure there
surfaces before the run has spent effort on P3.

Ungated P1/P2/P3 items are consumed in full and settled (step 5d) — fixed,
QA'd, and placed at the bottom of the run's chain, needing no further input.
They reach `main` with the run's single merge (5b), not on their own. Gated
ones are not settled: a gate moves the item into the judgment band, where it
becomes a PR whose body names the gate as the reason it is unresolved.

If the auto band is empty, say so in one line and go straight to 3c.

**3c. Build the judgment band (P4/P5/P6 + everything gated) into PRs.**
Every item here ends up in step 3d's question; what 3c decides is whether
it arrives there **with a diff attached**. For each item:

1. Apply the split test from the standing principle — do the branches
   produce materially different diffs, *and* would guessing wrong waste
   work worth reviewing? If both hold, the item becomes a **向き** entry
   and is left unwritten. Otherwise it gets the fix you would recommend
   and becomes a **受理** entry.
2. Fix it on the normal route (step 4), including the regression test and
   the version bump. A judgment-band item is not a draft: the PR has to be
   mergeable the moment the user says yes.
3. Open its PR **unmerged** (step 5e), with the decision at the top of the
   body: what changes for the user, what the alternative was, and why this
   branch was chosen.

**A P5 or P6 PR body MUST state the compatibility break in concrete
terms** — which existing artifacts stop loading, which existing command
lines stop working, what the user has to regenerate. That sentence is the
entire reason the item is not in the auto band; a PR that omits it has
turned the decision back into a rubber stamp.

**3d. Ask — one `AskUserQuestion` covering every judgment-band item.**
Raised after the auto band is settled and the 3c PRs are open, so the user
answers with the safe work already out of the way and the diffs available
to consult.

Include **every** judgment-band item, not only the forks: an item whose
fix is written and PR'd still needs the user to accept it, and leaving it
out of the question is what made previous runs opaque. Shape each entry
per the standing principle — the decision, its cost, the recommended
option first, and the PR number when one exists.

Skip the question only when the judgment band is empty. Say so in one
line if it is.

**3e. Act on the answers in the same run.**

- A **向き** answer decides the diff: implement it on the normal route
  (step 4), open its PR, and treat it as accepted for merge — the user
  chose the branch knowing what it costs, so do not ask a second time.
- A **受理** answer decides the merge: merge under 5d's conditions,
  folding the ledger update into the same PR **before** merging (6a).
- A rejection closes that PR; re-base anything stacked above it (5b) and
  record the rejection in step 6c's Re-triaged section with the user's
  stated reason.

If a selector was given in `$ARGUMENTS`, honor it: confirm the resolved
item set in one line and skip the parts of 3b/3c/3d it excludes.

### 4. Fix, on the normal route

Group the selected items **by class first, then by owning path**, so each
commit is one coherent fix and each class is its own review unit in the chain (step 5).

**4a. Check the scope boundary first.** If a selected item cannot be fixed
without touching something unselected, that is gate **G2**, and it applies
even mid-fix, after classification. An auto-band item that hits G2 leaves
the auto band; it does not get a quiet waiver because the run had already
started.

Two legitimate resolutions, chosen by the same split test as any other
judgment: **pull the neighbour in** and PR the widened fix unmerged, with
the coupling stated at the top of the body and the widening put to the
user as a 3d entry — or **re-triage** the item as still-open with the
coupling recorded, when the neighbour is large enough that guessing wrong
wastes the work. Silently widening is not one of them, and neither is
widening quietly *and* auto-merging it.

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
- For a **P3** item the test has a second job: it is the evidence for the
  classification. "Behavior unchanged" is a claim about outputs, so pin the
  output — a characterization test that passes before *and* after the fix,
  in addition to whatever the fix needs. Without it, P3 is an assertion, and
  an assertion is not enough to merge unattended.
- Verify the test is non-vacuous: neuter the fix, confirm the test fails,
  restore.
- QA per scope class: Python → `ruff format`, `ruff check --fix`, `mypy`,
  `pytest` on the mirrored test path. Rust crate → `cargo test -p <crate>`
  under CLAUDE.md §"重いテスト (Rust dfpn)". Other → whatever
  `.pre-commit-config.yaml` configures.
- Bump the **owning** manifest (nearest ancestor `Cargo.toml` with a
  `version`, else the owning `pyproject.toml`). Semver from the change:
  `fix:` patch, `feat:` minor, breaking major. One bump per commit that
  touches `src/` or `rust/`. P1 items touch no shipped file and so need no
  bump — if a "P1" item wants a bump, it was misclassified.
- Commit per group, naming the backlog origin in the body so the fix is
  traceable back to the record that filed it.

If QA cannot run in this environment, that is gate **G3**: say which tool
was blocked and why, record it in step 6's Environment notes, and move the
item out of the auto band. Never report an unrun check as passing, and
never merge on the strength of one.

**4d. Durable-doc drift → `reviews/`, with P2 deciding whether it waits.**
A source fix routinely invalidates prose: a doc that accurately described
the old (broken) behavior becomes wrong the moment the behavior changes.
Check for that on every fix — `docs/commands/<command>.md` when a CLI
option changes, and any doc describing behavior the fix altered.

File one `reviews/$(TZ=Asia/Tokyo date '+%Y-%m-%d')-<kebab-title>.md`,
`status: pending`, with exact before/after text per the shape in
`docs/memory-architecture.md` § "Review proposal shape". Then route by the
P2 test — *does the corrected text follow uniquely from the code as it
exists?*

- **Yes → it is a drift correction.** CLAUDE.md's standing approval covers
  it: apply it in this run, commit `docs: <title>`, set `status: applied` +
  `applied_in: <sha>`. The proposal is still written and still committed —
  it is the audit trail, and the next reader needs to see that a durable
  doc changed and why. Only the round-trip is skipped, and it is skipped
  because there was no alternative text to choose between.
- **No → something is being decided**, so the doc is **not edited**. Leave
  `status: pending` and ship the proposal alone: the proposal already
  contains the exact before/after text, so the PR carries the full change
  for the user to read without the edit having been made. Do not put the
  doc edit in the PR "ready to merge" — CLAUDE.md's gate is approval
  *before* the edit, and the standing approval covers drift corrections
  only. The proposal is what the user approves; a later run (this one, if
  they answer here, or the next `/audit-backlog` or `/checkpoint-context`)
  applies it, commits, and sets `status: applied` + `applied_in: <sha>`.

  This is the one place the run deliberately leaves a second round-trip in
  place. It is rare — most doc drift found from a re-verified backlog row
  passes the uniqueness test — and the alternative is writing durable-doc
  text nobody chose.

The test is about the *text*, not about the size of the diff. "The doc says
`.npy`, the writer writes `.feather`" has one correct replacement. "This
section should also explain when to use streaming" has as many as there are
authors — a one-line addition can still fail the test.

### 5. Ship: one PR per class, auto band merged without waiting

Committing is not shipping. This step turns the run's commits into review
units the user can look at afterwards, and merges the ones that had nothing
to decide.

**5a. One PR per class — as a review unit inside the run's single stack.**
`P1`, `P2`, `P3`, `P4`… each becomes at most one PR. That is the
"ある程度の単位" the batching exists for: a reviewer opening the P4 PR
knows every commit in it changes behavior and none of it breaks data,
because that is what the class means.

**A class PR is a unit of review, not a unit of merging into `main`.** All
of them are chained (5b) and the run lands on `main` once. Splitting by
class costs nothing when they are stacked; it costs a conflict per merge
when they are not.

Split a class into more than one PR only when:
- it spans unrelated scope classes (Python and Rust, say) — then split by
  scope, because their QA and version bumps are separate anyway; or
- it has grown past roughly ten items or a diff nobody will read in one
  sitting.

Never merge two classes into one PR. The class *is* the merge policy, so a
mixed PR has no policy.

If the session was handed a designated branch it must develop on, that
mandate wins: commit everything there, open the single PR from it, and say
in the report that class-per-PR was collapsed and why.

**A collapsed run is all-or-nothing, and that changes 6a.** With one PR
there is no way for the user to accept some items and reject others — the
whole branch merges or it does not. So the ledger update belongs *inside*
that PR: delete the rows for everything the PR ships, in the PR that ships
it. Keeping them "until it merges" is what forces a second PR into `main`
afterwards, which 5b exists to prevent. The commits are the review units
here, so make each one a coherent class and say so in the body.

**5b. Chain EVERY PR of the run into one stack. Exactly one of them
targets `main`.**

This is GitHub's stacked pull requests
(<https://docs.github.com/en/pull-requests/how-tos/stacked-pull-requests>):
PR 1 is based on `main`, PR 2 is based on PR 1's head branch, PR 3 on
PR 2's, and so on. The base link *is* the dependency — naming related PRs
in the body is not stacking and does not prevent a single conflict.

**There is no "independent, based on `main`" case.** An earlier version of
this file had one, gated on a per-PR dependency test. That test asked the
wrong question, and the run of 2026-08-12 is the evidence: five PRs opened
"independently" off `main`, and every merge into `main` forced a manual
conflict resolution in the next one. Two properties of this repo make
sibling PRs conflict *by construction*, whatever their diffs touch:

1. **Every PR that changes `src/` or `rust/` bumps a version** (CLAUDE.md)
   and regenerates `uv.lock`. Two siblings off `main` therefore edit the
   same `version = ...` line and the same lock entry. This is not
   incidental — it is guaranteed by the versioning rule.
2. **`.github/workflows/build-wheel.yml` triggers on `push: branches:
   [main]`.** N merges into `main` build N wheels. The wheel build is the
   most expensive job in the repo, and N−1 of those builds are waste.

So the shape to produce is one chain, and **one merge into `main` for the
whole run**.

**Order, bottom to top:**

1. the `audits/` ledger + record PR (P1 — see 6e),
2. the auto band, classes ascending (P1 → P2 → P3),
3. the judgment band, classes ascending (P4 → P5 → P6),
4. **the item the user is least likely to accept, last.**

Rule 4 is what makes the stack safe to reject piecewise: the user drops an
item by simply *not* merging that PR into its parent, and everything below
it is unaffected. Anything placed under a contentious PR is held hostage to
it, so put a judgment call the run genuinely cannot predict at the top even
if its class would sort it lower, and say why in the body.

**Version bumps run along the chain, not in parallel.** Each PR bumps from
the version its parent left, so the stack carries one increasing sequence.
Two PRs of a run must never claim the same version — that is the conflict
this step exists to prevent.

**Merging the stack:**

- Merge **top down, child into parent**. Those merges target a branch, not
  `main`: no `push: main`, no wheel build, and no conflict, because each
  child is already based on its parent.
- Then merge the **bottom** PR into `main`, **once**. By then it contains
  everything above it that was accepted.
- To reject an item: do not merge it into its parent, and close it. Its
  children (if any) must be re-based onto its parent before they can flow
  down — this is the cost that rule 4 above is designed to keep rare.

Do **not** merge a child into `main` directly, and do not retarget children
onto `main` as parents merge. Both re-create the N-merges-N-wheels problem
the chain removes.

**Keeping the chain healthy is part of the run.** After any push to a
branch in the stack, its descendants are stale. Merge the parent branch
into each descendant in order (bottom → top) and push, so every PR shows
only its own incremental diff. Verify each PR's base with
`mcp__github__pull_request_read`; set it with
`mcp__github__update_pull_request` when GitHub has retargeted a child
behind your back (it does this when a parent's head branch is deleted).

**5c. PR body — written for the user reading it after the fact.** State,
for each item in the PR: the backlog row and record it came from, its class
**and the test that decided the class**, what shipped, and the QA that ran.

For judgment-band PRs the decision goes at the **top**, because it is the
whole reason the PR is unmerged and it is the only part the user must
supply: what changes for them, what the alternative was, why this branch
was taken, and — for P5/P6 — the concrete compatibility break. Write it so
that merging is a complete answer.

Every PR of the run carries the same **stack map**, listing the chain
bottom-first and marking its own position, so the order and the single
`main` merge are visible from whichever PR the user opens first. State each
PR's base explicitly — that is what tells the reader it is a real stack and
not a list of related links:

```markdown
## Stack (2026-08-12 /audit-backlog) — main へのマージは #481 の 1 回だけ
1. #481 P1 audits/ ledger + record   (base: main)     ← 最後に main へ
2. #482 P3 behavior-preserving fixes (base: #481)
3. #483 P4 test-ratio / cache glob   (base: #482)  ← このPR
4. #484 P5 moveWinRate dtype         (base: #483)
5. #485 P6 file_level_split 削除     (base: #484)  ← 一番不確実なので最上段

上から順に親へマージし，最後に #481 を main へ 1 回だけマージする．
不要な段は「親へマージせず close」するだけでよく，下の段は影響を受けない．
```

Follow the repository's PR template if one exists.

**5d. Land the run in one merge. Do not wait for the user on the auto
band.** The auto band still ships without a question — what changed is
*where* it ships to. Its PRs are settled as soon as all three conditions
hold, and settled means "imposes no further wait on the run's single merge":

1. **The QA in 4c ran here and passed.** This is the real gate. The repo
   deliberately runs no test suite on PRs — `claude-code-review.yml` is
   `workflow_dispatch` only, with "PR push ごとの自動レビューはトークン
   浪費のため廃止 — ローカルでの確認を正とする" written into it. So a
   green PR is **not** evidence the tests ran; step 4c's output is. Do not
   let a checkmark stand in for a check you did not run.
2. **Every check GitHub reports on the PR is green or neutral.** Read them
   off the PR rather than assuming which ones exist — today `main` gets
   only `Check Version Bump` on `pull_request`, and that set changes
   without this file changing. A red check blocks the merge, full stop,
   including when you believe it is unrelated: investigate, fix, or move
   the PR to the judgment band and say so.
3. **Everything above it in the stack has been resolved** — merged into
   it, or closed. The auto band is at the bottom, so this is normally
   satisfied only once the judgment band above it has been answered. Merging
   the auto band into `main` *first* would split the run across two merges
   and two wheel builds, which is exactly what 5b removes.

**What "merge the auto band" means under a chain.** It does not mean
pushing the auto band to `main` on its own. It means: the auto-band PRs are
settled and need no further input, so they impose no wait — the run's single
merge into `main` can proceed as soon as the judgment band above them is
answered. If the user answers nothing this session, the whole stack stays
open and the run hands off as in 5e.

Use `mcp__github__merge_pull_request` for the child-into-parent merges and
for the final merge into `main`. Match the repository's merge style
(currently merge commits — `git log --merges` settles it at read time).

**MUST NOT merge into `main`**: a stack with an unresolved P4/P5/P6 PR, a
gated item, a classification you could not decide without hedging, a red or
still-running check, or an open PR above the one being merged. When in doubt
the stack stays open — that costs a comment, while a wrong merge costs a
revert on `main`.

Child-into-parent merges inside the stack are not subject to the wait: they
target a branch, land no code on `main`, and are how an accepted item flows
down toward the single merge.

**5e. Judgment-band PRs stay open until 3d's question is answered.** They
hold the fix, the QA, and the reasoning; the question (3d) is what tells
the user a decision is theirs. Opening the PR without asking is the
failure this step used to permit — do not repeat it.

The normal path is that the user answers **in this session**, because 3d
raised the question here: apply what they decided (3e), re-run the QA, and
merge under 5d's conditions — folding the ledger update into the same PR
**before** merging it, never into a follow-up (6a).

When no answer arrives — the question could not be raised, or the session
ends first — the PR **is** the handoff, and whether its row stays put is
6a's separability test: kept when the PR can merge apart from the ledger,
deleted with the fix when it cannot. Say plainly in the report which items
are in that state and what each is waiting on; step 1e of the next run
picks them up and raises the question then.

**5f. Hand the PRs over as a row → PR table in the conversation.** The
third moment in the standing principle above, and the one that gives 3d's
question its context. The moment a judgment-band PR is opened, print — in
chat, not only in the PR body — one row per finding:

| backlog 行 | 由来の記録 | クラス | PR | commit | 何を出荷したか |
|---|---|---|---|---|---|

Opening a PR is not the same as telling the user what to look at. The
body explains itself to whoever already opened it; the chat table is what
lets the user decide *which* PR to open, and in what order, without
paging through diffs to reconstruct which backlog row each one answers.
That reconstruction is exactly the work the ledger exists to remove.

Mark the auto-band rows in the table as needing no decision, so the user
can see at a glance how much of the run is actually theirs to judge. When
the whole run collapsed into one PR (the designated-branch case in 5a),
the table's commit column carries the review units instead — say so in
the same breath, since "one PR" otherwise reads as "one decision."

### 6. Update the ledger and write the record

Three separate bookkeeping duties. All are required; the first is the one
that keeps the backlog from growing forever.

**6a. Delete the rows in the PR that carries their fix** — from **either**
the Deferred backlog or the Out-of-scope backlog, whichever table held
them. This is the step that makes consumption real; without it the finding
is still open as far as every future run is concerned.

**The test is separability, not merge timing.** The reason a row is not
deleted on the strength of an unmerged branch is that the PR can be
rejected, reworked, or closed — leaving the ledger claiming a fix that
`main` does not have. That risk exists only when the deletion and the fix
can **diverge**: when they sit in different PRs that merge independently.

So apply this test, not a calendar:

- **The deletion rides in the same PR as its fix** → delete the row **in
  that PR**. Merging accepts both; closing discards both. They cannot
  diverge, so there is nothing to protect against. This is always the case
  for a run collapsed into one PR (the designated-branch case in 5a), and
  it is the normal case.
- **The deletion would ride in a PR that can merge without its fix** →
  keep the row and append the PR link to its text, so the next run sees
  the work is in flight instead of starting it again. In a stack this is
  the bottom (`audits/`) PR versus a judgment-band PR above it: the user
  can close the upper PR and the bottom one still merges. Write those
  deletions into the bottom branch only once the item is accepted, before
  the single merge into `main`.

**MUST NOT open a PR into `main` whose only content is ledger
bookkeeping.** A row deletion is not worth a merge commit and a wheel
build (5b), and it never needs one — when a decision lands after the run
has ended, the **next** run collects it for free: step 2 re-verifies every
row and marks one whose fix already shipped as **stale**, and step 1e
deletes rows for PRs that merged after the fact. Both fold the deletion
into that run's own single PR.

A row that is one run stale costs one re-verification, which step 2
performs anyway. A second merge into `main` costs a wheel build and
breaks the one-merge-per-run property outright. Those are not close.

The failure this prevents is concrete: the run of 2026-08-13 was collapsed
into a single PR, kept its rows because that PR was unmerged, and so
needed a **second** PR after the merge purely to delete them — two merges
into `main` and two wheel builds for one run's worth of work.

Also delete rows that step 2 found **stale**, noting in 6c that they were
already fixed elsewhere rather than fixed here. Do **not** delete a row that
was merely re-triaged — sharpen its text instead, so the next run inherits
the better reasoning rather than the original impasse.

Update the `Open items` count on the affected main-table row to match.

**6b. Leave the source records alone.** They are immutable accounts; their
Deferred sections stay as written, describing what that run decided at that
time. Do **not** add resolution markers, do **not** renumber, do **not**
move an item into Applied. The row deletion in 6a is the resolution record,
and 6c is its account.

One narrow exception: when step 2 proved a record's **diagnosis or proposed
fix wrong** (not merely resolved — *wrong*), append a short correction to
that record, because leaving it uncorrected makes it actively misleading to
the next reader. State the correction, never the worklist state:

```markdown
   **Correction** (2026-08-09, `<sha>`): the fix suggested above would
   have <consequence>, because <what the record missed>.
```

If you find yourself wanting to write "RESOLVED" or "done" in a record,
that is worklist state — it belongs in 6a's deletion, not here.

**6c. Write this run's record.**
`audits/$(TZ=Asia/Tokyo date '+%Y-%m-%d')-backlog-<slug>.md`, where `<slug>`
names the batch (e.g. `auto-band`, `interface-openings`). Frontmatter mirrors
the record shape in `audits/README.md`, with `path:` naming the **targets
touched** and a `kind: backlog` line marking it as a consumption record
rather than a path audit. Body:

- **Classification** — every item this run touched, with its class, the
  test that decided the class, and its gates. This is the section a later
  reader checks when a class turns out to have been wrong, so record the
  *reasoning*, not just the verdict.
- **Consumed** — one row per item: source record, target, what shipped, and
  the PR it merged in.
- **Applied** — the fixes, with `file:line` and commit SHA.
- **Decisions asked** — 3d's question, verbatim enough to be re-read: each
  entry, the options offered, and what the user chose. When the run could
  not raise the question, say so here and why. This is the section that
  makes a judgment-band outcome auditable — "the user approved it" is not
  a record; the options they were shown are.
- **In flight** — judgment-band PRs left open, with their PR number, their
  base (`main`, or the PR they are stacked on and why), and the question
  outstanding. Say for each whether its row was deleted here (the PR
  carries both) or kept (the PR can merge apart from the ledger) — that is
  6a's test, and this section is what step 1e of the next run reads to
  pick the work back up. A run whose question was answered in-session
  normally leaves this section empty; say "なし" rather than omitting it.
- **Re-triaged** — items selected but left open, with the sharpened reason.
  This is the section that earns the run its keep: a second impasse on the
  same item is far more informative than the first.
- **Corrections to the source records** — where step 2 found a record's
  diagnosis or proposed fix wrong. Record the *reason*, so the next reader
  learns to distrust that class of claim.
- **Doc findings** — `reviews/<file>`, its status, and whether P2's standing
  approval or a real decision applied it.
- **Out of scope** — anything new this run noticed; append it to
  `coverage.md`'s backlog too, or it is lost.
- **Environment notes** — what could not be run here, and why (the G1/G3
  evidence).

**6d. Reconcile findings to rows before committing (MUST).** Same closing
check as `/audit-and-fix` step 9x, and it applies here for the same
reason: 6c tells you to *append* new out-of-scope items but never makes
you verify the mapping is total. Assign every item this run touched, and
every new finding it noticed, exactly one disposition — **resolved** (row
deleted in the PR that carries its fix) / **in flight** (row kept because
its PR can merge apart from the ledger, PR linked) / **re-triaged** (row
kept, text sharpened) / **new row** (id) / **not a finding** (reason).
An item you cannot assign is the defect the check exists to catch.

Note that **resolved** is decided by 6a's separability test, not by
whether the merge button has been pressed yet. A collapsed single-PR run
resolves its items *in that PR*; expecting to come back later and delete
the rows is what produces a second merge into `main`.

The two failure modes named in step 9x apply verbatim: consolidating
several findings into one readable row drops the ones the merged prose
stops naming, and **this record's prose sections are not a worklist** —
`/audit-backlog` and `/audit-and-fix` both gather work from
`coverage.md` and only from there, so a finding written only into
`## Re-triaged` or `## Out of scope` here is invisible to every future
run. Read 9x for the full rationale and the worked example rather than a
copy of it; a copy would drift.

Print the reconciliation as an equation in step 7, not as an assurance.

**6e. Ship the record itself, at the bottom of the stack.** The ledger
update and the record are P1 by construction — `audits/` ships nothing — so
they go in their own PR, and it is the **bottom** of the chain: the one PR
whose base is `main` (5b). Everything else in the run is stacked on it, so
whatever the user accepts flows down into it and reaches `main` with it.

It must **not** ride inside a judgment-band PR. Being the bottom is what
guarantees the account survives: if the user rejects the fixes above it,
each of those PRs is closed and the record still merges — carrying the
Re-triaged section that explains why they were rejected.

Link the record from `coverage.md` under the backlog table so a future run
can find the account of a row that is no longer there. Then commit:
```
docs(audits): record backlog consumption (<slug>)
```

**This step also runs when the run stops early.** A selected-but-unfinished
item must leave a record with its resume point, or it is indistinguishable
from one never attempted.

### 7. Report

Compact summary:
- Backlog size found: N deferred + M out-of-scope (from `coverage.md`),
  plus any PRs already open from earlier runs (1e)
- Verification: confirmed / stale / changed-shape counts
- **Classification**: the count per class, and every gate applied with its
  item — so a reader can see what was kept out of the auto band and why
- Auto band: what was fixed, which PRs, **merged or blocked** (with the
  blocking check named)
- Judgment band: one line per PR — number, class, the decision it carries,
  and its base. Print the **stack map** once, in the same shape 5c puts in
  the PR bodies, so the report and the PRs agree
- The **row → PR table from 5f**, repeated here if the report is not
  already adjacent to it. The report is what a user re-reads days later to
  remember what a PR was for; a PR number without its backlog row forces
  them back into the diff
- **The question and its answers (3d)** — what was asked, per entry, and
  what the user chose. A run with a non-empty judgment band that reports no
  question has skipped a required step: say which of the two exemptions
  applies (empty judgment band, or the question could not be raised) and
  why. Never report an open judgment-band PR as if leaving it open were
  itself the ask
- Re-triaged / rejected, with the reason
- Doc drift → `reviews/<file>`, its status, and which P2 branch applied
- QA: what ran, what passed, what was blocked and why
- **Reconciliation (6d)**: `<items touched + new findings> = <resolved> +
  <in flight> + <re-triaged> + <new rows> + <not-a-finding>`, printed as
  the equation. Backlog rows before → after, so the ledger's movement is
  visible
- Version bump(s), old → new, which manifest
- Ledger: rows deleted (which table), rows added by 1d's repair, any
  record correction from 6b, new record path
- Backlog remaining, by class — so the next run knows what is left, and
  how much of it will need the user

### 8. Handoff

Do **not** call `/checkpoint-context` from here; step 6 already committed.

The committed record is what carries this run forward. Do not mirror it into
`worklog/` or `scratchpad/compass.md` — per CLAUDE.md, `audits/` and the
campaign memory are deliberately separate systems.

A judgment-band PR that the user answered in-session (3d → 3e) is not a
handoff at all — it merged, or it was closed, and step 6 recorded which.

A judgment-band PR left **open** is a handoff, and only arises when 3d's
question went unanswered or could not be raised. Each holds a finished
fix, its QA, and the decision it needs, so a later session can pick it up
cold. Leave them listed in the report with their stack map and **what each
is waiting on**; step 1e of the next run reads them back from
`list_pull_requests` plus the still-present backlog rows, and raises the
question that this run could not.

Do not end a run by re-asking a decision the user already made in it —
3d asks once, and 3e acts on the answer.

## Extending this command

Keep the derive-never-enumerate property:

- **New audited path** — nothing to do, provided `/audit-and-fix` appended
  its open findings to `coverage.md`. If a path's findings never reach the
  backlog tables, no amount of work here recovers them; that is a defect in
  `/audit-and-fix` step 9, not in this command.
- **New scope class** (new language, new top-level dir) — add its QA path to
  4c, keyed off a discoverable marker (a manifest, a
  `.pre-commit-config.yaml` entry), not a hardcoded path. Then check the P1
  test: is the new directory shipped? That answer, not a list here, decides
  whether its changes need a version bump.
- **New durable-doc location** — verify 4d's routing test classifies it
  correctly; if it does, no edit is needed here.
- **New CI check on PRs** — nothing to do. 5d reads the checks off the PR
  rather than naming them, so a new required check gates the auto band
  automatically. Adding its name here would be the enumeration bug.
- **A repo that starts running tests on PRs** — 5d's first condition stops
  being the only real gate, but it does not become redundant: it is what
  covers the checks CI still does not run. Add nothing here; the condition
  is already written as "the QA in 4c ran here", not "no CI exists".
- **A run that produces exactly one PR** — fine, and it is already the
  shape 5b converges on: one chain, one merge into `main`. Do not split a
  single coherent change into a stack to make the stack look used; do not
  un-chain a multi-class run to avoid a stack. Remember that a single PR
  is all-or-nothing, so its ledger deletions go **in** it (6a) — the one
  way to turn a one-PR run into a two-merge run is to leave the paperwork
  for afterwards.
- **A new kind of bookkeeping** (a new status file, a new index) — route
  it by 6a's separability test, not by adding a case here: if it can only
  be correct once the run's fixes land, it ships in the same PR as those
  fixes. Bookkeeping never earns its own merge into `main`.
- **A repo that stops building a wheel on `push: main`** — 5b's second
  reason weakens but the first does not. The version-bump rule alone still
  makes siblings off `main` conflict pairwise, so keep the chain.
- **A class that keeps coming up empty** — that is a signal about the repo,
  not about the rubric. Leave the class; report the emptiness.
- **A run where the question feels like a formality** — every entry an
  obvious yes — that is not a reason to skip 3d. The user cannot know an
  answer was obvious until they have been shown the options, and the
  judgment classes exist precisely because the run is not entitled to
  decide. If a *class* keeps producing formalities, that is evidence its
  test is drawn too high: sharpen the ladder (below), do not quietly stop
  asking.
- **A question that keeps getting answered "whichever you recommend"** —
  the entries are under-specified, not unnecessary. Give the cost of each
  branch concretely enough that the two are distinguishable; "delete vs
  repair" without what each removes is a coin flip dressed as a choice.
- **A class that keeps being wrong** — that is a signal about the *ladder*,
  and it is worth acting on. If items keep being merged as P3 and turning
  out to change behavior, sharpen P3's test with the case that fooled it
  and add the item to the calibration set. The calibration set is the part
  of this file designed to grow.

## Usage

- `/audit-backlog` — classify everything, settle the auto band unattended,
  build the judgment band into PRs, then ask the user once about all of
  them — all of it as one chain that merges into `main` once. The intended
  entry point.
- `/audit-backlog auto` — the auto band only. Its chain has nothing above
  it, so it merges into `main` immediately, and it asks nothing because
  the judgment band is empty by construction. The narrowest run.
- `/audit-backlog judgment` — the judgment band only, when the auto band
  is already clear. Always ends in the question.
- `/audit-backlog P4` — the P4 items: fix them, open their PR, and ask.
- `/audit-backlog P3-1,P4-3 high` — two specific items, broader
  verification.
- `/audit-backlog src/maou/interface` — everything targeting that path,
  auto band merged and judgment band PR'd as usual (without auditing the
  path itself — use `/audit-and-fix` for that).
