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
| `src/maou/domain/game_graph` | python | done | high | `2686689` | [2026-08-08](2026-08-08-src-maou-domain-game-graph.md) | 1 deferred, 3 out-of-scope |
| `src/maou/app/learning` | python | done | high | `52d9bd2` | [2026-08-08](2026-08-08-src-maou-app-learning.md) | 9 deferred, 1 out-of-scope |

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

### Deferred backlog

Findings an audit **confirmed inside** its target path but deliberately
did not fix — ambiguous, cross-layer, architecturally significant, or
needing a decision. A deferred finding is a diagnosis with the fix
withheld pending a decision, **not** a decision never to fix it.

| Found by | Target | Item |
|---|---|---|
| [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) Deferred 2 | `src/maou/domain/game_graph` | `openings.py:115-139` `find_opening` takes a bare `moves: list[str]` with no way to express "this list starts from 平手初期局面", though every `_DEFAULT_OPENINGS` entry is valid only from startpos. The **symptom** was fixed caller-side in `cc10790`; the domain API still cannot state its own precondition. Right shape (require an initial SFEN argument? return None unless root is startpos? leave it to the caller?) is a design decision on a public domain API the interface layer calls. **Re-verified 2026-08-09 (`cdc4031`)**: the interface guard `_root_is_startpos()` (`game_graph_visualization.py:717-733`) is regression-tested *and* documented (`docs/commands/visualize.md:184-186`), so a domain-level parameter would now duplicate a shipped guard rather than replace one — weigh closing this as won't-do. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 1 | `src/maou/app/learning` | `streaming_dataset.py:602-607` `StreamingStage2Dataset.__len__` overestimates — `_compute_total_batches` sums per-file `ceil(rows/batch)` but `__iter__` concatenates `_FILES_PER_CONCAT = 10` files before batching. **Not just tqdm**: the inflated count reaches the scheduler as `total_steps`, so cosine decay never completes. **Corrected 2026-08-09 (`cdc4031`)**: the record's chain via `dl.py:318,498` is wrong — those are the *kif* loaders, whose `__len__` is correct. The real path is `stage_component_factory.py:713,803` (`steps_per_epoch=len(pipeline.train_dataloader)`), reached through `Stage2StreamingAdapter.__len__` (`:636`). Exact fix needs `num_workers` (grouping is per worker via `_resolve_worker_files`, which also **shuffles**, so the exact count depends on the seed) — how to model sharding is a design decision. `_compute_total_batches` is shared by all three datasets (`:309`, `:440`, `:605`) and must not be mutated in place; `StreamingKifDataset.__len__` (`:306`) and `StreamingStage1Dataset.__len__` (`:437`) are both correct. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 2 | `src/maou/app/learning` | Stage 1 / Stage 2 pipeline cloned across five files (three of four review angles reported it independently). `run_stage1_with_training_loop` / `run_stage2_with_training_loop` (`multi_stage_training.py:436`/`:585`, ~150 lines each) differ only in head class, callback class, metric getter and two log strings — the loop class is already shared. `_build_stage1_model_and_optimizer` / `_build_stage2_model_and_optimizer` (`stage_component_factory.py:636`/`:724`) have byte-identical 38-line tails. Also `dataset.py:202`/`:279`, `streaming_dataset.py:775`/`:835`. **~400-line refactor of the multi-stage training path — architecturally significant.** |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 3 | `src/maou/app/learning` | Six adapter classes are three duplicated pairs. `Stage1ModelAdapter`/`Stage2ModelAdapter` (`multi_stage_training.py:111`/`:240`) differ in **zero** characters; `Stage1DatasetAdapter`/`Stage2DatasetAdapter` (`:151`/`:183`) in one type annotation; `Stage1StreamingAdapter`/`Stage2StreamingAdapter` (`streaming_dataset.py:645`/`:610`) in a redundant `hasattr` guard. Merging also deletes the `isinstance` dispatch + `TypeError` arm at `stage_component_factory.py:866-872`, which exists only to choose between two identical classes. Six public names referenced from tests — should land as its own reviewed change. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 4 | `src/maou/app/learning` | `callbacks.py` — `_ensure_device` written six times (`:238`, `:362`, `:1007`, `:1396`, `:1521`, `:1668`), plus three copies of the loss-accumulator scaffolding (`:1375`, `:1499`, `:1652`). `ValidationCallback` hand-lists the same 13 accumulator tensors in three places (`__init__` / `_ensure_device` / `reset`) — the exact shape that produces "new metric added, never moved to GPU, never reset" defects. Base-class extraction across the module's metric hub (~250 → ~120 lines). |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 5 | `src/maou/app/learning` | `training_loop.py:1093` per-batch host-device sync — `if not has_legal.all():` calls `Tensor.__bool__` on a CUDA tensor, a full pipeline stall once per training *and* validation batch, to guard a warning. Stage 3 always ships a `legal_move_mask`, so the branch is always taken. The branchless rewrite changes the loss path — **measure, don't assume**. Needs GPU hardware. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 6 | `src/maou/app/learning` | `training_loop.py:460` `stream.synchronize()` blocks the host, defeating much of the prefetch it implements. `wait_stream()` gives the same ordering guarantee device-side without stalling the CPU, and the `record_stream()` added in `073adbd` already covers the allocator hazard. **Second untested GPU-semantics change stacked on the first** — validate both together on real hardware. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 7 | `src/maou/app/learning` | `gradient_noise_scale.py:150,188-192,246` — one GPU sync per parameter tensor per micro-batch (`.item()` inside `for param in model.parameters()`): 60-300 syncs per micro-batch on a ResNet/ViT backbone whenever adaptive batch is on. Accumulating into a device scalar changes when the value materializes, and GNS feeds the adaptive batch controller — needs a numerical equivalence check. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 8 | `src/maou/app/learning` | `dataset.py:91` / `streaming_dataset.py:754` — an all-ones `legal_move_mask` is built per sample and shipped over PCIe per batch (~9 MB/batch at B=1024), then consumed by five kernels that are no-ops for an all-ones mask. `callbacks.py:493` keys the `policy_move_label_ce` metric off `legal_move_mask is not None`, so it must stay non-`None`; the fix is "build once on device", which **changes the dataset contract**. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) Deferred 9 | `src/maou/app/learning` | `polars_datasource.py:204-266` `_PolarsField` fakes numpy flags to get past validation it cannot satisfy — guesses dtypes from Python value shape rather than the schema, and synthesizes a `FakeFlags` asserting `c_contiguous=True, writeable=True` purely so `dataset.py:186-198`'s zero-copy guards pass (the guards are structurally unreachable on this path). **The recorded fix is no longer available (2026-08-09, `1c714db`)**: the `domain/data/polars_tensor.py` tensor helpers it proposed switching to had zero callers repo-wide and have been deleted, and the "documented public API (`docs/rust-backend.md:704`)" justification was verified false. Whoever takes this must design the replacement rather than wire up existing helpers — either build the tensors from the schema here, or drop the numpy mimicry by giving `dataset.py` a non-numpy path. |

`/audit-and-fix src/maou/app/learning` Deferred 10 is **not** listed: its
fix requires editing `src/maou/interface`, so it lives in the
out-of-scope table below instead of being duplicated here.

### Out-of-scope backlog

Findings an audit surfaced **outside** the path it was auditing, and was
therefore not allowed to fix.

| Found by | Target | Item |
|---|---|---|
| [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) | `src/maou/app/game_graph` | `query.py:174-200` `get_path_to_root` breaks out of a broken parent chain at three points and still returns the partial path reversed, so `path[0]` is not guaranteed to be the root. Still **unconfirmed** as of 2026-08-09 (`cdc4031`) — no reachable input was constructible from the builder's output. Seven interface-layer call sites (`game_graph_visualization.py:443,648,697,944,1011`). |
| [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) | `docs/` (new) | `src/maou/domain/game_graph/openings.py` has no module documentation — `OpeningDatabase`, `find_opening`, and the 9 opening names return zero hits across `docs/`, `CLAUDE.md`, `AGENTS.md`, `README.md`. **Narrowed 2026-08-09 (`cdc4031`)**: the *user-visible* behavior (定跡 row, and its startpos-only restriction) is now documented at `docs/commands/visualize.md:183-186` (`3600b32`), so what remains is the module/API doc plus the list of supported 定跡 — smaller than "entirely undocumented" implies. |
| [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) | `docs/architecture.md` | `game_graph` has no architectural home: `CLAUDE.md`, `AGENTS.md`, `README.md`, `docs/architecture.md` and every `docs/adr-*.md` contain zero references to it. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `src/maou/interface` + `src/maou/app/learning` | `stage2_test_ratio` is unread on the **streaming** path. **Corrected 2026-08-09 (`cdc4031`)**: the record names the wrong callee — `learn.py:829-834` logs "ignored" and then calls `create_stage2_streaming_components` (`:836`, forwarding at `:848`), which forwards again at `stage_component_factory.py:612` into `create_stage2_streaming_data_pipeline` (`:255`), where the body (`:277-320`) never reads it. So **three** signatures are involved, not two. **Trap**: the parameter is not dead overall — `learn.py:682` feeds the same CLI value into the *non-streaming* Stage 2 pipeline, where `stage_component_factory.py:196-216` genuinely uses it, and `docs/loss-functions.md:87-91` recommends `--stage2-test-ratio 0.1`. Scope must stay inside the streaming functions. Also note `utility.py:1217` and `utility_interface.py:498` coerce an explicit `0.0` to `0.2` via `or 0.2`. |
