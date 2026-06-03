---
description: Externalize working memory. Writes a new immutable worklog snapshot, refreshes scratchpad/current.md, drafts architecture proposals, and reconciles pending reviews with the user (filling applied_in: SHA on approval).
argument-hint: [optional one-line reason for the checkpoint]
---

You are checkpointing the current working context. This is the **only**
writer of working memory. Run before any context reset, long break,
handoff, or whenever the conversation grows heavy.

`$ARGUMENTS` is an optional reason ("near context limit", "task swap",
"end of day"). If empty, use "routine".

## Hard constraints

- **JST only.** All timestamps and filenames use `Asia/Tokyo` time.
- **Worklog files are immutable.** Each checkpoint creates a **new**
  file. Never edit a previous worklog file.
- **No silent CLAUDE.md edits.** Architectural changes go through
  `reviews/`; the user applies the actual edit.
- **Preserve failures, reasoning, uncertainty.** Do not over-summarize.

## Anti-patterns — do not do these

- Generic summaries ("did some refactoring") → name files, lines,
  errors verbatim.
- Conclusions without reasoning → preserve alternatives ruled out.
- Hiding failed attempts → enumerate each with its killing signal.
- False confidence → tag unverified claims as uncertain.
- Treating tool output as memory → tool output disappears at
  compaction; copy the relevant lines verbatim into the worklog file.

## Steps

### 1. Gather raw state (parallel)

Run these independently:
- `TZ=Asia/Tokyo date '+%Y-%m-%d-%H%M%S'` → call the result `$STAMP`
- `git status --short`
- `git diff --stat`
- `git log -10 --oneline`
- Read `scratchpad/current.md` (may not exist yet on a fresh repo)
- `ls reviews/` (note every file whose frontmatter has
  `status: pending` — those are reconciled in step 5)

### 2. Write a new worklog snapshot

Create `worklog/$STAMP.md` — a **new** file. Never edit an existing
worklog. Use the shape defined in
`docs/memory-architecture.md` § "Worklog entry shape".

Fill every section. If a section is genuinely empty, write `_(none)_`
— absence is information. Do not collapse sections to look tidy.

### 3. Refresh `scratchpad/current.md`

Overwrite. Target ~100-150 lines. A cold reader must be able to pick
up without reading any worklog. Use the shape defined in
`docs/memory-architecture.md` § "current.md shape".

### 3.5. Curate `scratchpad/compass.md`

上書きではなく**剪定**する（`docs/memory-architecture.md` § "compass.md — the
curated layer"）:
- North-star: active metric の現在値・現状最良を更新．変化なしなら
  "unchanged" を明記する（指標を失わないため）．
- Invariants: 今セッションが do-not-redo 結論を確立したら追加（Est. に worklog/SHA）．
  今セッションが既存不変則を **覆した** なら該当行を delete/edit（append しない）．
- 上限 ~45 行 / 不変則 ~12 件を超えたら，最も load-bearing でない項目を evict．
- 末尾 `## Last curated:` を現 SHA @ JST 日付に更新．

### 4. Detect architectural decisions

A change is **architectural** if ANY of:
- Introduces or removes a layer / module / cross-cutting convention.
- Changes dependency direction.
- Changes a public API surface other code is expected to call.
- Adds, removes, or changes a MUST/SHOULD rule in `CLAUDE.md`.
- Would require updating `docs/architecture.md` to stay accurate.
- Establishes a new convention future code should follow.

If detected, create
`reviews/$(TZ=Asia/Tokyo date '+%Y-%m-%d')-<kebab-title>.md`
with frontmatter `status: pending` and the shape in
`docs/memory-architecture.md` § "Review proposal shape". Cite the
worklog file (`worklog/$STAMP.md`) in the proposal's Trigger section.

### 5. Reconcile pending reviews with the user

For every `reviews/*.md` with frontmatter `status: pending` — including
any new one filed in step 4 — do the following **interactively**:

a. Print a one-line summary: filename, title, target files, risk.
b. Ask: "**approve / reject / defer**?"
c. On **approve**:
   - Ask the user to apply the change themselves (edit `CLAUDE.md` /
     `docs/` per the proposal) and commit it.
   - Ask for the resulting commit short SHA.
   - Update the proposal frontmatter — single `Edit` per file:
     ```
     status: applied
     applied_in: <sha>
     ```
   - This is the **only** path by which `CLAUDE.md` / `docs/architecture.md`
     get changed. The model never edits them directly.
d. On **reject**: ask for a one-line reason, then delete the file.
e. On **defer**: leave as `status: pending`. It surfaces again next
   checkpoint.

If there are many pending items, you may print the full list first and
let the user pick which to handle now; the rest stay pending.

### 6. Print compact status (max ~8 lines)

- New worklog: `worklog/$STAMP.md`
- `scratchpad/current.md` updated (line count)
- `compass.md` curated: scoreboard updated / N invariants added / N removed
- New proposal filed: path or "none"
- Pending reviews handled: N approved / N rejected / N deferred / N left pending
- One-line **resume hint** for the next `/resume-context`

Stop here. Do not begin further work without a new user prompt.

## Usage

- `/checkpoint-context` — routine checkpoint.
- `/checkpoint-context context near limit` — before an expected reset.
- `/checkpoint-context end of day`
- `/checkpoint-context before risky refactor of dfpn solver`
