---
status: approved  # 2026-06-22 reconcile: user approved; awaiting user apply of docs/memory-architecture.md + CLAUDE.md edits + commit SHA → then applied
applied_in:
date: 2026-06-22
target: [docs/memory-architecture.md, CLAUDE.md, reviews/2026-06-20-dfpn-incremental-effect-tables.md, reviews/2026-06-14-dfpn-stage0-kh-searchimpl-port-map.md]
risk: low
reversibility: trivial
---

# Memory-system enforcement reform — convert recorded prose into active gates

## Trigger

A workflow postmortem mined all 69 `worklog/*.md` for re-instruction
episodes (the user re-pointing-out corrections in later sessions). Finding:
**50 episodes, 37 (74%) recurred even though the lesson was ALREADY
recorded.** Stickiness tracks *record location* almost perfectly:
compass (always-loaded) = 5 episodes / ~0.7 sessions wasted; passive
feedback-memory = 21 / ~12.5; immutable worklog = 10 / ~6.4. The fix is
not "record more" — it is "convert recorded prose into active gates
positioned at the decision moment". Worst classes: divergence-hunting
methodology not engaged (9/10 recorded), `benign/faster/rejected`
premature verdicts (12/14 recorded), wrong-unit metrics (the 4-session
`190K vs 19K` error), soundness shortcuts hidden behind `#[ignore]`.

Worklog: `worklog/2026-06-22-175500.md` (postmortem + reform session).

The **command-file + compass + auto-memory** changes have already been
applied directly (Tier 1+2, user-approved); this proposal covers only the
**durable-doc** edits that the project rules require the user to apply.

## Proposed change

### A. `docs/memory-architecture.md`

1. **§ "compass.md — the curated layer"**: replace the single flat
   `Invariants` model with the fixed-section structure now implemented in
   `.claude/commands/checkpoint-context.md` step 3.5:
   `## 🚫 VETOES` (≤5, user-set, never-undo, exempt from eviction) →
   `## 🚦 TRIPWIRES` (fixed-count, verb-bound gates) →
   `## North-star` → `## Invariants — Measured do-not-redo` (each line
   carries an evidence-scope tag `[single]`/`[bundle]`) →
   `## ❌ REFUTED — do-not-re-derive` (append-once, ~8 cap, oldest-evict;
   a superseded refutation moves here instead of vanishing) →
   `## 環境リファレンス` (stable repro state, single home).
2. Add a **HARD BYTE CAP** beside the existing line cap:
   `compass.md ≤ ~9KB (env-reference 込; knowledge 部 ~7KB + env ~2KB) —
   line 上限 ~50 行とどちらか先に効く方`. Rationale: the line-only cap let
   compass bloat to 11,997 B (~324 B/line) while reporting itself "within
   ~45 lines"; the env-reference fold-in legitimately adds ~1.5KB that
   previously lived (duplicated) in current.md.
3. **§ "Token budget guidance"**: replace the line-based table with
   per-file BYTE caps (compass ≤ ~9KB env 込, current.md ≤ ~6KB) and note
   the "no re-emit stable repro / point to compass §環境リファレンス" rule.
4. **§ "Status transitions" + Review proposal shape**: add terminal
   `status: rejected` = "architectural experiment that completed but was
   measured-rejected, OR findings-doc with no doc to apply; file RETAINED
   as committed do-not-redo provenance". Note that `/checkpoint-context`
   step 5 now reconciles `approved` as well as `pending`.
5. **New § "Dual-memory division of labor"**: the repo system
   (compass + worklog + current + reviews) OWNS the campaign narrative;
   the `~/.claude` auto-memory holds ONLY the 5 `feedback_*.md`
   process/preference rules. Do NOT author new `project_*.md` or prepend
   campaign lines to `MEMORY.md`. Decisive reason: auto-memory arrives as
   "background, may-be-outdated" — mirroring binding do-not-redo there
   licenses re-litigation.
6. **§ "Worklog entry shape"**: add the `repro: see compass §環境リファレンス`
   rule and the no-op fast-path (3-line pointer worklog when state unchanged).

### B. `CLAUDE.md` — "Repository-Centric Memory Architecture (MUST)"

- Update the **Files table** note: compass now has fixed sections + byte
  cap; auto-memory `MEMORY.md` indexes only `feedback_*.md`.
- Add MUST rules: (1) `/checkpoint-context` MUST refuse on an uncommitted
  `src/`/`rust/` tree (active form of the existing commit-before-checkpoint
  rule); (2) campaign do-not-redo lives SOLELY in `compass.md`, never
  mirrored to auto-memory; (3) reviews/ fires ONLY on committed durable-doc
  targets — `rust/`/`src/` algorithmic tuning + lever rejections go to
  worklog + compass REFUTED, never reviews/.

### C. Reclassify the two stuck reviews (need `rejected` status from A.4 first)

- `reviews/2026-06-20-dfpn-incremental-effect-tables.md`: `approved` →
  `rejected` (effect tables experimentally refuted, §10; compass already
  records `❌ 棄却確定`).
- `reviews/2026-06-14-dfpn-stage0-kh-searchimpl-port-map.md`: `approved` →
  `rejected` (a findings doc with no doc target; conclusions absorbed into
  compass), or relocate to `docs/plans/`.

## Motivation

74% recorded-but-recurred proves the bottleneck is enforcement/loading,
not writing. The two worst classes recurred because the lesson sat in
passive prose with no trigger tied to the about-to-act moment; the
always-loaded compass is the ONE record class that demonstrably worked
(its items twice actively suppressed a re-try). Codifying the new
structure in the spec keeps the authoritative doc in sync with the
already-applied command behavior, and the `rejected` status clears the
two-file `approved` limbo the user noticed.

## Alternatives considered

1. **Abolish reviews/ entirely** (user floated this). Ruled out: reviews/
   is the ONLY git-committed provenance layer (compass/worklog are
   gitignored, so in a clone/PR only reviews/ answers "why does CLAUDE.md
   say X"); it has 100% coverage of the 3 real CLAUDE.md edits in 27 days.
   "Rarely files" is correct behavior once the rust-only mis-fire stops —
   not a failure. Reform (narrow trigger + terminal `rejected`) keeps the
   irreducible value and removes the limbo.
2. **Record more / longer worklogs.** Ruled out by the central finding:
   74% already recorded. More passive prose adds tokens without firing.
3. **Hooks (Stop / pre-commit) to hard-enforce gates.** Deferred: the
   system is human-triggered + model-agnostic by design. The dirty-tree
   gate + verb-bound tripwires get most of the benefit without a hook;
   memory-architecture.md already notes a Stop hook may be reintroduced if
   skipping recurs.

## What this enables

A binding user veto / methodology gate fires at the decision point
(printed FIRST at resume, phrased on the conclusion verb) instead of
sitting as "one dense line among twelve". A measured `[single]` verdict
can no longer be misread as `[bundle]`-rejected. Rejected experiments get
a terminal home. Always-loaded tokens go DOWN (compass 12KB→9.3KB,
MEMORY.md 23KB→~1.5KB, env block de-duplicated).

## What this constrains

No reviews/ for rust/src tuning. No new `project_*.md`. compass ≤ ~7KB and
no `✅ done` changelog lines in Invariants. `/checkpoint-context` blocked
on a dirty src tree. Campaign CONCLUDE/pivot becomes user-gated. These
make the cheap-but-wrong terminal actions structurally harder.

## Rollback plan

All edits are docs/frontmatter — revert the `CLAUDE.md` /
`docs/memory-architecture.md` paragraphs and the two `status:` fields.
The command-file + compass changes are independent and gitignored
(compass) or trivially revertible (commands). Archived `project_*.md` are
in `memory/archive/` (moved, not deleted) — restore with `mv` if needed.
