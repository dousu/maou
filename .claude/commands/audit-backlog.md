---
description: Consume the deferred and out-of-scope findings that /audit-and-fix left behind. Gathers them from audits/coverage.md's backlog tables, re-verifies each against HEAD, then classifies each by DECISION COST on a mechanical six-class ladder — P1-P3 need no user judgment and are fixed, PR'd and merged without asking; P4-P6 need judgment and stop for the user. Applies fixes on the normal route (version bump, regression test, reviews/ proposal for durable docs), deletes the resolved backlog rows, and records the run in audits/.
argument-hint: [item-selector or class (P1..P6 / auto / judgment) | omit to run the auto band and then ask] [effort-level: low|medium|high|max, default medium]
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
- **Omit the selector** to do the whole run: consume the auto band without
  asking (step 3b), then stop and ask about the judgment band (step 3c).
  This is the intended way to open the command.
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
- **Every selected item ends in one of three states**: resolved (fixed +
  merged), re-triaged (still open, with sharpened reasoning recorded), or
  rejected (won't do, recorded as such). Silently dropping a selected item
  is the one outcome that is not allowed.
- **A backlog row is deleted when its fix has merged — not when it is
  written.** An unmerged PR is an open finding (step 6a).

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

**1e. Check for open PRs from earlier runs.** A previous run may have left
judgment-band PRs open. List them (`mcp__github__list_pull_requests`) and
report them alongside the backlog — an open PR and its still-present row
are the same finding counted once, not two units of work. Do not open a
second PR for a row that already has one.

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

### 3. Classify, then run the auto band before asking anything

**3a. Assign every confirmed item a class and its gates.** Apply the ladder
top-down (P6 → P1), then the four gates. Record, per item: ID (`P3-1`),
source record, target path, the one-line finding, verification result from
step 2, **class + the specific test that decided it**, gates, and expected
blast radius.

Sort **within** each class by severity — wrong results a user or operator
can see, ahead of internal cleanups. That is where the old impact rubric
went: it orders work inside a class, it no longer decides which class.

Print this table before doing anything with it. It is the user's chance to
stop a misclassification before it merges, and it is the artifact that
makes the classification auditable after the fact.

**3b. Consume the auto band (P1 → P2 → P3) without asking.** No
`AskUserQuestion`, no pause. Work the classes in ascending order — P1 is
both the cheapest and the least able to break anything, so a failure there
surfaces before the run has spent effort on P3.

Ungated P1/P2/P3 items are consumed in full. Gated ones are not: a gate
moves the item into the judgment band, where it is reported in 3c with the
gate as the reason.

If the auto band is empty, say so in one line and go straight to 3c.

**3c. Stop and ask about the judgment band (P4/P5/P6 + everything gated).**
Ask via `AskUserQuestion`, after the auto band's PRs exist, so the user is
answering with the safe work already visible and mergeable.

Offer classes as coarse choices (e.g. "all of P4", "P4+P5") alongside
individual IDs, and recommend a batch with the reason. For each item the
question must carry **the decision itself**, not a request for permission:
what the two branches are and what each costs. "Delete `file_level_split`
or repair it? Deleting removes its tests and a public name; repairing means
fixing the全ロード it forces at construction" is a question. "May I work on
D8?" is not.

**A P5 or P6 item's question MUST state the compatibility break in
concrete terms** — which existing artifacts stop loading, which existing
command lines stop working. That sentence is the entire reason the item is
not in the auto band; omitting it turns the question back into a
formality.

If a selector was given in `$ARGUMENTS`, honor it: confirm the resolved
item set in one line and skip the parts of 3b/3c it excludes. A selector
naming judgment-band items still requires their decisions to be taken here,
before any code changes.

### 4. Fix, on the normal route

Group the selected items **by class first, then by owning path**, so each
commit is one coherent fix and each class can ship independently (step 5).

**4a. Check the scope boundary first.** If a selected item cannot be fixed
without touching something unselected, stop and report before editing —
that is gate **G2**, and it applies even mid-fix, after classification.
Two legitimate resolutions: pull the neighbour in with the user's agreement,
or re-triage the item as still-open with the coupling recorded. Silently
widening is not one of them. An auto-band item that hits G2 leaves the auto
band; it does not get a quiet waiver because the run had already started.

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
- **No → something is being decided.** Leave `status: pending`, present it
  in 3c's question (or a follow-up if the drift only surfaced during 4b),
  and take the decision: **approve** → apply, commit, `status: applied` +
  `applied_in: <sha>`; **reject** → `status: rejected` with the reason,
  retained as do-not-redo provenance; **defer** → leave `pending` and
  record it as outstanding in step 6.

The test is about the *text*, not about the size of the diff. "The doc says
`.npy`, the writer writes `.feather`" has one correct replacement. "This
section should also explain when to use streaming" has as many as there are
authors — a one-line addition can still fail the test.

### 5. Ship: one PR per class, auto band merged without waiting

Committing is not shipping. This step turns the run's commits into review
units the user can look at afterwards, and merges the ones that had nothing
to decide.

**5a. One PR per class, off `main`.** `P1`, `P2`, `P3`, `P4`… each becomes
at most one PR, branched from the current `main`. That is the "ある程度の
単位" the batching exists for: a reviewer opening the P4 PR knows every
commit in it changes behavior and none of it breaks data, because that is
what the class means.

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

**5b. PR body — written for the user reading it after the fact.** State,
for each item in the PR: the backlog row and record it came from, its class
**and the test that decided the class**, what shipped, and the QA that ran.
For judgment-band PRs, put the decision the user took (or the one still
outstanding) at the top — that is what the reviewer is actually checking.

Follow the repository's PR template if one exists.

**5c. Merge the auto band. Do not wait for the user.** P1/P2/P3 PRs with no
gates merge as soon as both conditions hold:

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

Prefer `mcp__github__enable_pr_auto_merge` so GitHub merges when the checks
settle; if the repository does not allow auto-merge, read the checks and
call `mcp__github__merge_pull_request`. Match the repository's merge style
(currently merge commits — `git log --merges` settles it at read time).

**MUST NOT auto-merge**: any P4/P5/P6 PR, any PR carrying a gated item, any
PR whose classification you could not decide without hedging, and any PR
whose checks are red or still running. When in doubt the PR stays open —
that costs a comment, while a wrong merge costs a revert on `main`.

**5d. Judgment-band PRs stay open**, linked in the report next to the
question from 3c. If the user answers in this session, merge them the same
way once their decision is applied and QA is green. If they do not, the
PR *is* the handoff: it holds the fix, the reasoning, and the question, and
the backlog row stays put until it merges.

### 6. Update the ledger and write the record

Three separate bookkeeping duties. All are required; the first is the one
that keeps the backlog from growing forever.

**6a. Delete the rows whose fixes have MERGED** — from **either** the
Deferred backlog or the Out-of-scope backlog, whichever table held them.
This is the step that makes consumption real; without it the finding is
still open as far as every future run is concerned.

Merged, not written. A row whose fix sits in an open judgment-band PR is
**still open**: the PR can be rejected, reworked, or closed, and a row
deleted on the strength of an unmerged branch would take the finding out of
the ledger with nothing on `main` to show for it. Keep the row and append
the PR link to its text, so the next run sees the work is already in
flight instead of starting it again.

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
- **In flight** — judgment-band PRs left open, with their PR number and the
  question outstanding. Their backlog rows are still present by design
  (6a); this section says why.
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
deleted, fix merged) / **in flight** (row kept, PR linked) / **re-triaged**
(row kept, text sharpened) / **new row** (id) / **not a finding** (reason).
An item you cannot assign is the defect the check exists to catch.

The two failure modes named in step 9x apply verbatim: consolidating
several findings into one readable row drops the ones the merged prose
stops naming, and **this record's prose sections are not a worklist** —
`/audit-backlog` and `/audit-and-fix` both gather work from
`coverage.md` and only from there, so a finding written only into
`## Re-triaged` or `## Out of scope` here is invisible to every future
run. Read 9x for the full rationale and the worked example rather than a
copy of it; a copy would drift.

Print the reconciliation as an equation in step 7, not as an assurance.

**6e. Ship the record itself.** The ledger update and the record are P1 by
construction — `audits/` ships nothing — so they go in their own PR and
merge under 5c like any other P1 work. They must **not** ride in a
judgment-band PR: the account of a run has to reach `main` even when the
fixes it describes are still under review, or an unmerged PR takes the
run's only record down with it.

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
- Judgment band: the question asked, the answer taken, PRs left open
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

Open judgment-band PRs are the other half of the handoff. Leave them
listed in the report with their questions; a later session picks them up
from `list_pull_requests` plus the still-present backlog rows.

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
- **New CI check on PRs** — nothing to do. 5c reads the checks off the PR
  rather than naming them, so a new required check gates the auto band
  automatically. Adding its name here would be the enumeration bug.
- **A class that keeps coming up empty** — that is a signal about the repo,
  not about the rubric. Leave the class; report the emptiness.
- **A class that keeps being wrong** — that is a signal about the *ladder*,
  and it is worth acting on. If items keep being merged as P3 and turning
  out to change behavior, sharpen P3's test with the case that fooled it
  and add the item to the calibration set. The calibration set is the part
  of this file designed to grow.

## Usage

- `/audit-backlog` — classify everything, ship the auto band unattended,
  then ask about the rest. The intended entry point.
- `/audit-backlog auto` — the auto band only. Stop after 5c; do not ask
  about P4-P6 at all. Useful when the user is away.
- `/audit-backlog P4` — take the P4 decisions and fix what they resolve.
- `/audit-backlog P3-1,P4-3 high` — two specific items, broader
  verification.
- `/audit-backlog src/maou/interface` — everything targeting that path,
  auto band merged and judgment band asked as usual (without auditing the
  path itself — use `/audit-and-fix` for that).
