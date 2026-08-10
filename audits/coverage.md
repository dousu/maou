# Audit coverage ledger

One row per path touched by `/audit-and-fix`. Shape, status vocabulary,
and protocol: [README.md](README.md).

**This table lists only what has been audited.** It is not a plan and not
an inventory of remaining *paths* — to see which paths are left, compare
against the tree (`ls src/maou/*/`, `ls rust/`, `find docs -name '*.md'`),
which is always current where a checked-in list would not be.

Remaining *findings* are a different question, and they **are** inventoried
here: see § "Open findings backlog" below. That is the live worklist; the
per-run records are immutable accounts and are never read to decide what
work remains.

| Path | Scope | Status | Level | Last SHA | Record | Open items |
|---|---|---|---|---|---|---|
| `src/maou/domain/game_graph` | python | done | high | `2686689` | [2026-08-08](2026-08-08-src-maou-domain-game-graph.md) | 0 |
| `src/maou/app/learning` | python | done | high | `52d9bd2` | [2026-08-08](2026-08-08-src-maou-app-learning.md) | 6 deferred |
| `src/maou/infra/file_system` | python | in-progress | high | `4433871` | [2026-08-10](2026-08-10-src-maou-infra-file-system.md) | — |

## Blocked

_(none)_

<!-- Rows move here only while status is `blocked`, with the blocker and
     what would unblock it. Keep the main table for in-progress/done. -->

## Open findings backlog — the single live worklist

The two tables below are **the** authority on what audit work remains.
Both `/audit-and-fix` and `/audit-backlog` gather candidate work from
here and **only** from here.

**Why this file and not the records.** A per-run record is read only when
someone opens that specific path — so a finding left there is visible
exactly to the audit least able to act on it. This file is read at the
start of every run.

**Why the records are not also consulted.** A record is an *immutable
account of one run at one time*: its Deferred section says "as of that
run, this was deferred", and that stays true forever even after the
finding is fixed. Reading records for open work therefore re-surfaces
resolved findings on every run, with no way to remove them — the ledger
would never shrink. Deleting a row here is what makes a finding
*consumed*, and it is the only mechanism that does.

**Protocol (both tables).**
- **Before auditing a path**, check both tables for rows whose target
  falls inside it and fold them into the run.
- **At the end of a run**, append a row for every finding left open —
  deferred (inside the path) and out-of-scope (outside it) alike.
  Writing it only into the run's record buries it.
- **When a finding is resolved, delete its row.** The resolving record is
  the durable account. Do not delete a row that was merely re-triaged —
  sharpen its text instead.

Records of runs that resolved rows deleted from here:

- [2026-08-09 backlog tier-a](2026-08-09-out-of-scope-tier-a.md)
- [2026-08-09 backlog contained-fixes](2026-08-09-backlog-contained-fixes.md)
- [2026-08-09 backlog streaming-len-and-docs](2026-08-09-backlog-streaming-len-and-docs.md)
- [2026-08-09 backlog tier-3-contracts](2026-08-09-backlog-tier-3-contracts.md)

### Deferred backlog

Findings an audit **confirmed inside** its target path but deliberately
did not fix — ambiguous, cross-layer, architecturally significant, or
needing a decision. A deferred finding is a diagnosis with the fix
withheld pending a decision, **not** a decision never to fix it.

| Found by | Target | Item |
|---|---|---|
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 2 | `src/maou/app/learning` | Stage 1 / Stage 2 pipeline cloned across five files (three of four review angles reported it independently). `run_stage1_with_training_loop` / `run_stage2_with_training_loop` (`multi_stage_training.py:422`/`:571`, ~150 lines each) differ only in head class, callback class, metric getter and two log strings — the loop class is already shared. `_build_stage1_model_and_optimizer` / `_build_stage2_model_and_optimizer` (`stage_component_factory.py:646`/`:735`) have byte-identical 38-line tails. Also `dataset.py:202`/`:279` (file untouched since the record, so those still hold) and `_yield_stage1_batches`/`_yield_stage2_batches` (`streaming_dataset.py:851`/`:911`). **~400-line refactor of the multi-stage training path — architecturally significant.** (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 3 | `src/maou/app/learning` | Six adapter classes are three duplicated pairs. `Stage1ModelAdapter`/`Stage2ModelAdapter` (`multi_stage_training.py:111`/`:240`) differ in **zero** characters; `Stage1DatasetAdapter`/`Stage2DatasetAdapter` (`:151`/`:183`) in one type annotation; `Stage1StreamingAdapter`/`Stage2StreamingAdapter` (`streaming_dataset.py:721`/`:686`) in a redundant `hasattr` guard. Merging also deletes the `isinstance` dispatch + `TypeError` arm at `stage_component_factory.py:876-882`, which exists only to choose between two identical classes. Six public names referenced from tests — should land as its own reviewed change. (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 4 | `src/maou/app/learning` | `callbacks.py` — `_ensure_device` written six times (`:238`, `:362`, `:1044`, `:1433`, `:1558`, `:1705`), plus three copies of the loss-accumulator scaffolding (in `Stage2F1Callback`, `Stage1AccuracyCallback`, `Stage3LossCallback` — the record's `:1375`/`:1499`/`:1652` are now those class bodies, +37 lines). `ValidationCallback` hand-lists the same 13 accumulator tensors in three places (`__init__` / `_ensure_device` / `reset`) — the exact shape that produces "new metric added, never moved to GPU, never reset" defects. Base-class extraction across the module's metric hub (~250 → ~120 lines). (Line numbers re-verified 2026-08-09 at `ff5bbaa`.) |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 5 | `src/maou/app/learning` | `training_loop.py:1110` per-batch host-device sync — `if not has_legal.all():` calls `Tensor.__bool__` on a CUDA tensor, a full pipeline stall once per batch, to guard a warning. **Now dormant, not fixed** (2026-08-09, backlog `tier-3-contracts`): the record's premise "Stage 3 always ships a `legal_move_mask`, so the branch is always taken" no longer holds — no data path supplies a mask, so `_compute_policy_loss` never enters the masking arm and the sync does not execute. The code is unchanged and the stall returns the moment a real legal-move mask is wired in; fix it **then**, together with whatever produces the mask, and measure on GPU. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 6 | `src/maou/app/learning` | `training_loop.py:460` `stream.synchronize()` blocks the host, defeating much of the prefetch it implements. `wait_stream()` gives the same ordering guarantee device-side without stalling the CPU, and the `record_stream()` added in `073adbd` already covers the allocator hazard. **Second untested GPU-semantics change stacked on the first** — validate both together on real hardware. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 7 | `src/maou/app/learning` | `gradient_noise_scale.py:150,189-192,247` — one GPU sync per parameter tensor per micro-batch (`.item()` inside `for param in model.parameters()`): 60-300 syncs per micro-batch on a ResNet/ViT backbone whenever adaptive batch is on. Accumulating into a device scalar changes when the value materializes, and GNS feeds the adaptive batch controller — needs a numerical equivalence check. |

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|

_(none — all out-of-scope findings recorded so far have been consumed.
An empty table is not a claim that the tree is clean: it means every path
audited so far surfaced nothing outside itself that is still open.)_
