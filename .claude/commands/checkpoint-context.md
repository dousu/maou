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
- **No UNAPPROVED CLAUDE.md edits.** Durable-doc changes go through
  `reviews/`. The model MUST NOT edit `CLAUDE.md` / `docs/` without an
  **approved** proposal — but **on user approval in step 5, the model
  applies the edit itself** (and commits). Approval is the safeguard
  against *silent* edits; it is no longer "the user hand-applies".
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

### 0. Dirty-tree gate (HARD — refuse on uncommitted src/rust)

Run `git status --short` first. **If any tracked file under `src/` or
`rust/` is modified/added/deleted-uncommitted, STOP.** Do not write the
worklog. Print:

> Working tree has uncommitted `src/`/`rust/` changes — commit first
> (pre-commit MUST run, never `--no-verify`; bump `pyproject.toml` /
> `Cargo.toml` per the versioning rules). `/checkpoint-context` will not
> snapshot a dirty src tree. (feedback_commit_before_checkpoint)

`scratchpad/` and `worklog/` dirtiness is fine (gitignored). Only an
explicit user `--allow-dirty` argument overrides this gate. Rationale:
HEAD sat non-compiling across 5+ consecutive checkpoints because the rule
lived only as prose; this makes it an active refusal.

### 1. Gather raw state (parallel)

Run these independently:
- `TZ=Asia/Tokyo date '+%Y-%m-%d-%H%M%S'` → call the result `$STAMP`
- `git status --short`
- `git diff --stat`
- `git log -10 --oneline`
- Read `scratchpad/current.md` (may not exist yet on a fresh repo)
- `ls reviews/` (note every file whose frontmatter has
  `status: pending` OR `status: approved` — both are reconciled in step 5)

### 2. Write a new worklog snapshot

Create `worklog/$STAMP.md` — a **new** file. Never edit an existing
worklog. Use the shape defined in
`docs/memory-architecture.md` § "Worklog entry shape".

Fill every section. If a section is genuinely empty, write `_(none)_`
— absence is information. Do not collapse sections to look tidy.

**Do NOT re-emit stable reproduction state.** SFENs, build/rebuild
commands, env stacks, the `--test-threads=1`/OOM note — these live ONCE
in `compass.md` § "環境リファレンス". The worklog writes
`repro: see compass §環境リファレンス` and notes only what CHANGED this
session. (Today the 39te SFEN is duplicated across 23/69 worklogs and the
`--test-threads=1` boilerplate across 48/69 — that is the single largest
write-side duplication.)

**No-op fast path:** if git + code state is unchanged since the last
worklog, write a 3-line pointer worklog (not a full snapshot).

### 3. Refresh `scratchpad/current.md`

Overwrite. Target ~100-150 lines. A cold reader must be able to pick
up without reading any worklog. Use the shape defined in
`docs/memory-architecture.md` § "current.md shape".

### 3.5. Curate `scratchpad/compass.md`

上書きではなく**剪定**する（`docs/memory-architecture.md` § "compass.md — the
curated layer"）．compass は always-loaded な唯一の binding 層であり，
記録の **発火力は場所で決まる**（postmortem: 既記録の教訓が 74% 再発したが，
compass にあった 5 件は ~0.7 セッションしか浪費せず 2 回は再試行を能動抑止した）．
ゆえに binding な do-not-redo は compass にのみ置き，achievement は置かない．

compass は以下の **固定セクション**を維持する（順序も固定）:
1. `## 🚫 VETOES` — user-set「絶対に覆さない」指示．各行 `<directive> — (user, YYYY-MM-DD)`．**上限 ~5 行**．user veto は expire しない（evict 対象外）．
2. `## 🚦 TRIPWIRES` — 動詞に束縛した発火ゲート（benign 結論前／lever 提案前／KH 比掲示前／STRICT-None 時）．**各 1 行・本数固定**．肥大したら die するので増やさない．
3. `## North-star` — goal / active metric / 現状最良 / 残ギャップ．数値は更新するか "unchanged" を明記．
4. `## Invariants — Measured do-not-redo` — 各行末に **evidence-scope タグ `[single]`（単発測定＝組合せに非外挿）/ `[bundle]`（bundle 内で測定）** を付ける．「単独で棄却」を「組合せでも棄却」と誤読させない（dominance×look-ahead, CHUAI が反例）．`✅ 完了 / 達成` は **achievement であって guardrail でない** → ここに置かず worklog/current.md の changelog へ．mid_v4 確定 lever 群は 1 行のポインタに圧縮．
5. `## ❌ REFUTED — do-not-re-derive` — append-once．各行 `<idea> — killed by <one-line 信号> @ <SHA>`．**上限 ~8 行・oldest evict**．invariant を supersede=削除する際，それが *refutation* なら killing-signal をここへ移す（消さない＝再提案の loop を断つ）．
6. `## 環境リファレンス` — stable な repro 一式を **ここに 1 回だけ**（bin/build/rebuild cmd・29te/39te SFEN・KH driver・`--test-threads=1`/OOM note・env stack）．worklog はここを参照する．
7. `## Last curated:` を現 SHA @ JST 日付に更新．

**剪定規則**:
- 既存不変則を覆したら該当行を delete/edit（append しない）．refutation は REFUTED へ移す．
- 単発測定からの結論を `[bundle]` と誤タグしない（default 保守=`[single]`）．
- **HARD BYTE CAP: compass ≤ ~9KB**（env-reference 込; knowledge 部 ~7KB + env ~2KB．
  line 上限 ~50 行とどちらか先に効く方）．元の ~12KB は achievement-changelog と repro 重複が
  原因だった．超過したら ① `✅ 完了` 行を changelog へ ② 密な多節パラグラフを one-liner へ
  ③ 最も load-bearing でない項目を evict（user veto は除外）．

### 3.6. Reconcile VETOES（standing orders）

今セッションで user が新たな veto/correction を出したか（「X するな」「X は棄却」「先に
TARGET を確立せよ」等）を確認し，あれば `## 🚫 VETOES` に `<directive> — (user, $DATE)`
を 1 行追加．既存 veto が今セッションで覆されていないか確認．rejected な lever は VETOES /
REFUTED に居るので「open lever」へ silently 再昇格させない．

**Campaign log は repo (compass + worklog) が所有する．auto-memory (`~/.claude` の
`project_*.md` / `MEMORY.md` campaign 行) を新規に書かない**（§
`docs/memory-architecture.md` の dual-memory 規則; binding な do-not-redo を
"background, may-be-outdated" チャネルに mirror すると再litigate を許してしまう）．

### 4. Detect architectural decisions

File a `reviews/*.md` **ONLY when the change touches a COMMITTED
durable-doc target** — `CLAUDE.md`, `docs/architecture.md`,
`docs/memory-architecture.md`, `docs/commands/` — OR a layer/dependency/
cross-cutting convention rule. Concretely, a change is **architectural** if
ANY of:
- Introduces or removes a layer / module / cross-cutting convention.
- Changes dependency direction.
- Changes a public API surface other code is expected to call.
- Adds, removes, or changes a MUST/SHOULD rule in `CLAUDE.md`.
- Would require updating `docs/architecture.md` to stay accurate.
- Establishes a new convention future code should follow.

**EXPLICITLY EXCLUDE** `rust/`/`src/` algorithmic tuning, param-gated
experiments, and lever rejections — a rust-only finding is NOT an
architectural decision. Those belong in `worklog/` + `compass.md`
(Invariants / REFUTED list), which already own them with cap + eviction +
provenance. (Routing algorithmic work here is what created the
`approved`/`pending` limbo and stale third-copy reviews.)

If detected, create
`reviews/$(TZ=Asia/Tokyo date '+%Y-%m-%d')-<kebab-title>.md`
with frontmatter `status: pending` and the shape in
`docs/memory-architecture.md` § "Review proposal shape". Cite the
worklog file (`worklog/$STAMP.md`) in the proposal's Trigger section.

### 5. Reconcile pending/approved reviews with the user

For every `reviews/*.md` with frontmatter `status: pending` **OR
`status: approved`** — including any new one filed in step 4 — do the
following **interactively**:

a. Print a one-line summary: filename, title, target files, risk.
b. Ask: "**approve / reject / defer**?"
c. On **approve**:
   - **The model applies the edit itself** — edit `CLAUDE.md` / `docs/`
     exactly per the proposal's "Proposed change" section. (Approval in
     step 5 is the authorization; this is the ONLY path by which the model
     may edit `CLAUDE.md` / `docs/architecture.md`.)
   - Run pre-commit (never `--no-verify`) and commit the durable-doc edit:
     `docs: <proposal title>` + trailers. Note the resulting short SHA.
   - Update the proposal frontmatter — single `Edit` per file:
     ```
     status: applied
     applied_in: <sha of the durable-doc commit>
     ```
   - Commit the frontmatter change per step 5.5.
   - If the user prefers to hand-apply instead, they may say so; then ask
     for their commit SHA and only write the frontmatter.
d. On **reject**: ask for a one-line reason, then set
   `status: rejected` with that reason in the body (terminal — file
   RETAINED as committed do-not-redo provenance). Delete the file only
   if it was never substantive.
e. On **defer**: leave as `status: pending`. It surfaces again next
   checkpoint.

**For `status: approved` items specifically** (split-state — approval and
apply separated in time): an approved review with no remaining doc-target,
or whose experiment was rejected-after-approval, must NOT dangle — ask:
finalize as `applied` (give SHA) / mark `rejected` (terminal, retain) /
relocate to `worklog`/`docs/plans`. `rejected` is a terminal status for
"architectural experiment that completed but was measured-rejected" or
"findings doc with no doc to apply".

If there are many items, you may print the full list first and let the
user pick which to handle now; the rest stay as-is.

### 5.5. Commit review status changes

`reviews/` is the **committed** audit trail — a `status:` transition is
only durable once committed. After reconciling, if step 4 filed a new
proposal or step 5 changed any `status:` (`applied` / `rejected` /
`approved`), **commit the touched `reviews/*.md` immediately**:

- Stage ONLY `reviews/` (doc-only; never bundle `src/`/`rust/` code — those
  commit separately under the versioning rules).
- `git commit` with `docs(reviews): reconcile statuses (<one-line summary>)`
  and the required trailers. Run pre-commit; **never** `--no-verify`.
- A `pending → applied` transition is committed together with (or right
  after) the user's own durable-doc commit; record that doc commit's SHA in
  `applied_in` and commit the frontmatter update.

### 6. Print compact status (max ~8 lines)

- New worklog: `worklog/$STAMP.md`
- `scratchpad/current.md` updated (line count)
- `compass.md` curated: byte size (≤ ~9KB) / scoreboard updated / N invariants ± / N VETOES / N REFUTED
- New proposal filed: path or "none"
- Reviews handled: N approved / N rejected / N deferred / N left pending|approved
- Canonical soundness this session: 29te/39te STRICT-VERIFY = ran(Some/None) / NOT-run
- One-line **resume hint** for the next `/resume-context`

Stop here. Do not begin further work without a new user prompt.

## Usage

- `/checkpoint-context` — routine checkpoint.
- `/checkpoint-context context near limit` — before an expected reset.
- `/checkpoint-context end of day`
- `/checkpoint-context before risky refactor of dfpn solver`
