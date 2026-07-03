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
| `scratchpad/compass.md` | Curated campaign invariants + north-star metrics. Always loaded; prunable. | no — `.gitignore`d |
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

Do NOT re-emit stable reproduction state (SFENs, build/rebuild commands,
env stacks). Write `repro: see compass §環境リファレンス` and note only what
changed. **No-op fast path:** if git + code state is unchanged since the
last worklog, write a 3-line pointer worklog, not a full snapshot.

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

## compass.md — the curated layer

`current.md`（毎回全上書き）でも worklog（immutable・1 つだけロード）でもない
**第 3 のモード = curated**．常時ロードされ，編集・削除で剪定される．

設計空間 (mutable↔immutable) × (常時ロード↔オンデマンド) の，これまで空いていた
「mutable・常時ロード・campaign を貫く durable」セルを埋める．

**固定セクション（順序も固定）**．記録の発火力は *場所* で決まる（postmortem:
既記録の教訓が 74% 再発したが，always-loaded な compass の項だけは再試行を能動抑止
した）．ゆえに binding な do-not-redo は compass にのみ置き，achievement は置かない:

1. **🚫 VETOES** — user-set「絶対に覆さない」指示．`<directive> — (user, YYYY-MM-DD)`．
   ~5 行上限．**resume が最初に逐語出力**．user veto は expire しない（evict 対象外）．
2. **🚦 TRIPWIRES** — 動詞に束縛した発火ゲート（benign 結論前 / lever 提案前 /
   KH 比掲示前 / STRICT-None 時）．各 1 行・**本数固定**（肥大したら散文同様 die する）．
   resume が VETOES と並べて最初に逐語出力．
3. **North-star** — goal / active metric / 現状最良 / 残ギャップ．毎 checkpoint で
   数値を更新するか "unchanged" を明記する．
4. **Invariants — Measured do-not-redo** — 各行末に evidence-scope タグ
   `[single]`（単発測定＝組合せに非外挿）/ `[bundle]`（bundle 内で測定）．
   「単独で棄却」を「組合せでも棄却」と誤読させない．evidence で覆ったら delete/edit．
   `✅ 完了/達成` は achievement ＝ guardrail でない → worklog/current.md へ．
5. **❌ REFUTED — do-not-re-derive** — append-once．`<idea> — killed by <信号> @ <SHA>`．
   ~8 行上限・oldest evict．invariant を supersede=削除する際それが *refutation* なら
   killing-signal をここへ移す（消さない＝再提案 loop を断つ）．
6. **環境リファレンス** — stable な repro 一式（bin/build/SFEN/driver/env stack）を
   **ここに 1 回だけ**．worklog はここを参照する（再出力しない）．

肥大化対策（必須）:
1. **HARD BYTE CAP ~9KB**（env-reference 込; knowledge 部 ~7KB + env ~2KB．line 上限
   ~50 行とどちらか先に効く方）．line のみの上限は byte bloat を防げない（旧 compass は
   ~45 行内のまま 12KB / ~324 B/行 に肥大した）．超えたら `✅ 完了` 行を worklog へ →
   密パラグラフを one-liner へ → 最も load-bearing でない項目を evict（user veto は除外）．
2. スロット正当化: 「破ると 1 セッション無駄になる」もののみ．豆知識は worklog 行き．
3. curation は checkpoint の必須ステップ（§ "Steps" step 3.5）．

## Review proposal shape

One file per proposal. Filename: `reviews/YYYY-MM-DD-<kebab-title>.md`.
Single flat folder; lifecycle is tracked by the `status:` frontmatter
field.

```markdown
---
status: pending          # pending | approved | applied | rejected
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
                      the MODEL applies the edit + commits;
                      model writes status: applied, applied_in: <sha>)
pending  → rejected  (user rejects; file RETAINED as do-not-redo
                      provenance — set status: rejected with a reason)
pending  → deleted   (rejected AND never substantive)
approved → applied | rejected  (split-state, resolved next /checkpoint-context)
```

`status: applied` and `status: rejected` are terminal — never modified.
`rejected` covers an architectural experiment that completed but was
measured-rejected, or a findings doc with no doc to apply (retained for
provenance, never re-promoted). The intermediate value `approved` is for
when approval and apply are split in time; step 5 reconciles `approved`
as well as `pending` so it cannot dangle. **On approval the model itself
applies the durable-doc edit** (approval is the safeguard against silent
edits) — it no longer waits for the user to hand-apply.

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

Cap by **bytes**, not lines — a line cap lets dense multi-clause prose
bloat invisibly (compass once hit 12KB / ~324 B/line while "within ~45 lines").

| Source | Byte cap | Note |
|---|---|---|
| `CLAUDE.md` | — | always loaded |
| `scratchpad/current.md` | ≤ ~6KB | always loaded |
| `scratchpad/compass.md` | ≤ ~9KB | always loaded (env-reference 込) |
| `~/.claude/.../memory/MEMORY.md` | ≤ ~2KB | always loaded; `feedback_*.md` index ONLY |
| one `worklog/*.md` snapshot | — | one file only; no re-emitted repro |
| `git status / log -10 / rev-parse` | — | always |
| `ls reviews/` | — | open files on demand |

Do NOT re-emit stable repro state in worklogs — it lives once in
`compass.md` § 環境リファレンス (the 39te SFEN was once duplicated in 23/69
worklogs). Above ~2× these caps you are over-loading history — prune.

## What was deliberately left out

| Idea | Why not |
|---|---|
| `/memory-sync` (lighter end-of-session) | Subset of `/checkpoint-context`. One writer keeps state coherent; empty sections in the strict template collapse to `_(none)_` cheaply. |
| `/propose-knowledge` (ad-hoc draft) | Proposals are either filed automatically by `/checkpoint-context` step 4 or written by hand into `reviews/`. A dedicated command added surface without unique value. |
| Hooks (`.claude/hooks/`) | The system is human-triggered by design — keeps it model-agnostic. If "I forgot to checkpoint" becomes a recurring failure, re-introduce a `Stop` hook. |
| `pending/` vs `approved/` subdirs | Replaced by `status:` frontmatter; collapses the `git mv` + `Applied-in:` append into a single in-place edit driven by `/checkpoint-context` step 5. |
| `scratchpad/todo.md`, `scratchpad/decisions.md` | Todos fold into `current.md`'s `## Todo` section. Small decisions live in the worklog; durable decisions live in `reviews/`. |

## Dual-memory division of labor

Two memory systems coexist: the **repo** system (this spec —
compass + worklog + current + reviews) and the Claude Code **auto-memory**
(`~/.claude/.../memory/`: `MEMORY.md` index + `feedback_*.md` +
`project_*.md`). They MUST divide labor, not duplicate:

- **Repo system OWNS the campaign narrative** end-to-end. It has the
  discipline auto-memory lacks: byte cap, eviction, immutability,
  single-writer, Confirmed/Assumed/Unresolved provenance.
- **Auto-memory holds ONLY the `feedback_*.md`** — cross-session
  process/preference rules (advisory, rarely overturned) that must fire
  even when `/resume-context` is skipped (survive `/clear`). Do NOT author
  new `project_*.md`; archive existing ones out of the always-loaded path.
  `MEMORY.md` indexes only the `feedback_*.md`.

**Decisive reason** campaign do-not-redo lives SOLELY in compass: auto-memory
arrives as a `<system-reminder>` framed "background, may-be-outdated" — the
*weakest* channel. Mirroring a binding conclusion there licenses the model
to re-litigate it. compass Invariants/VETOES are binding; that is where
do-not-redo belongs.

## Anti-patterns

- Mirroring campaign do-not-redo into auto-memory (the weak, "may-be-outdated"
  channel) instead of compass — licenses re-litigation.
- "Record more" to fix recurring user corrections — the postmortem showed
  74% of re-instructions recurred *despite* being recorded. Fix is
  enforcement/load-order (compass VETOES/TRIPWIRES), not volume.
- Promoting a single-measurement verdict to an Invariant without a
  `[single]`/`[bundle]` scope tag — "rejected alone" gets misread as
  "rejected in combination" and the always-loaded layer goes stale.
- Loading multiple worklog files at resume time.
- Treating `current.md` as immutable history (it is *current* state).
- Treating worklog files as mutable (they are append-only by creation,
  not by editing).
- Silently editing `CLAUDE.md` without a `reviews/` trail.
- Generic checkpoint summaries that hide failed experiments.
- Treating `status: pending` reviews as policy.
- `compass.md` を append-only で扱う（剪定せず積み上げる）．
- 覆れた不変則を残して「墓場」を作る．
- North-star の数値を checkpoint で更新し忘れる（"unchanged" すら書かない）．
