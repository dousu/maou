---
path: src/maou/app/learning
scope: python
level: high
status: done
started: 2026-08-08
last_sha: 52d9bd2
---

# Audit — src/maou/app/learning

## Resume point

_(complete — no resume point)_

Scope resolved: python-src only — 17 `.py` files, 11,544 lines, no
non-Python assets (`git ls-files` returns `.py` exclusively). Owning
manifest `pyproject.toml`. Mirrored test dir `tests/maou/app/learning/`
exists with 31 test files; every source file has coverage.

**Files actually covered by the step 2 simplify pass** (all 17, four
independent whole-module review angles): `adaptive_batch.py`,
`callbacks.py`, `compilation.py`, `dataset.py`, `dl.py`,
`gradient_noise_scale.py`, `model_io.py`, `multi_stage_training.py`,
`network.py`, `onnx_verifier.py`, `polars_datasource.py`,
`policy_targets.py`, `resource_monitor.py`, `setup.py`,
`stage_component_factory.py`, `streaming_dataset.py`,
`training_loop.py`.

## Applied

### Code commit `073adbd` (`pyproject.toml` 0.82.1 → 0.82.2, patch)

Step 1 bug fixes. 9 of 12 `/code-review` findings, plus one the review
missed.

1. `multi_stage_training.py:352` — `TruncatedStageModel._compute_output_channels`
   built its probe with `torch.zeros(...)` on CPU while
   `StageComponentFactory` passes a backbone that
   `create_shogi_backbone` already did `.to(device)` on
   (`setup.py:721`), and `model.to(device)` only runs *after*
   construction (`stage_component_factory.py:873`). So
   resnet + `--trainable-layers N` + `--gpu` died at model construction.
   Now derives device/dtype from `partial.parameters()`, with a CPU
   fallback for a parameterless partial.
2. `onnx_verifier.py` — `success` was computed *after* the sample loop
   from leaked loop variables, so only the last sample's tolerance was
   ever checked and `num_test_samples=0` raised `NameError`. Now
   accumulated per sample.
3. `onnx_verifier.py` — every "random test sample" was
   `create_empty_preprocessing_array(1)`, i.e. the all-zero empty board.
   Verification was vacuous beyond one position no matter how large
   `num_test_samples` was. Added `_random_sample()` generating random
   board IDs (bounded by the model's own `board_vocab_size`) and random
   hand counts (bounded per piece type by `shogi.MAX_PIECES_IN_HAND * 2`),
   seeded for reproducibility. **This is reachable production code** —
   `model_io.py:508,556` calls it during ONNX export verification.
   The existing fp32/fp16 equivalence tests still pass with real random
   inputs, which is the first time they actually proved anything.
4. `onnx_verifier.py:132` — `GraphStructureReport.summary` compared
   `input_names != ["input"]` while the producer requires
   `["board", "hand"]` (`:511-512`), so a correct graph was always
   reported as having unexpected inputs.
5. `polars_datasource.py` — the `preprocessing` branch mapped columns by
   hard-coded tuple position against a 5-column comment, but
   `get_preprocessing_polars_schema()` has 7 columns: `row_tuple[4]` is
   `moveWinRate` (a 1496-element list), not `resultValue`, so
   `dataset.py:88`'s `data["resultValue"].item()` raised. Replaced with
   `_row_by_names()` resolving through a `_col_idx` name→position map
   built once in `__init__`, so the mapping cannot silently drift from
   the schema again. `moveWinRate`/`bestMoveWinRate` are now surfaced as
   optional columns, enabling `KifDataset`'s 4-element target path.
6. `training_loop.py:440` — `_iterate_cuda_overlap` transferred on a
   side stream without `record_stream()`, so the caching allocator could
   recycle a freed batch's memory for the next H2D copy while the
   compute stream was still reading it. Added `_record_stream()` walking
   the context's CUDA tensors. **Unverified on GPU hardware** — no CUDA
   device in this container; the fix is the standard PyTorch pattern and
   is a no-op on CPU paths.
7. `resource_monitor.py` — `time.sleep` sat inside the `try`, so a
   sampling exception skipped the sleep and turned the monitor thread
   into a full-speed loop spamming `logger.error`. Fixed in **both**
   `GPUResourceMonitor._monitor_loop` (the reported one) and
   `SystemResourceMonitor._monitor_loop` (**not reported by the review —
   found by checking the sibling**).
8. `callbacks.py:949` — `on_forward_pass_end` unconditionally subtracted
   `_temp_timings["loss_computation"]`, but two of the four training
   paths (`training_loop.py:803`, `:1015`) close the forward-pass window
   *before* loss computation, so it subtracted the **previous batch's**
   loss time and could go negative. Now only subtracts when
   `loss_start >= forward_start`, which is correct for both orderings.
9. `setup.py:1068` — the unsupported-scheduler error listed display
   names ("Warmup+CosineDecay") while only snake_case keys are accepted,
   so following the message reproduced the error.
10. `setup.py:142` — `torch.cuda.get_device_name(device)` ran for any
    `--gpu` value that was not literally `"cpu"`, raising on e.g. `mps`.
    Now guarded on `device.type == "cuda"`.
11. `dl.py:1218` — `hasattr(self.model, "_hand_projection")` is always
    True because `HeadlessNetwork` assigns the attribute as `None` when
    `hand_projection_dim <= 0` (`network.py:75-82`), so
    `None.parameters()` raised. Now `getattr(..., None) is not None`.

### Code commit `3e2cd13` (`pyproject.toml` 0.82.2 → 0.82.3, patch)

Step 2 simplify — only the findings whose blast radius is contained.

- `model_io.py:171-278` — five `load_*` staticmethods identical apart
  from one log noun, collapsed into `_load_component(..., label=)`.
  The five public names are kept as delegations because
  `interface/learn.py:1111`, `dl.py:1155-1171` and four tests call them
  by name.
- `callbacks.py:852` — deleted `TimingData`; repo-wide grep across
  `src/` and `tests/` found only its own definition.
- `setup.py:121` — `LR_SCHEDULER_DISPLAY_NAMES` values were read
  nowhere (the canonical display-name table is
  `interface/learn.py:SUPPORTED_LR_SCHEDULERS`); replaced with the
  `SUPPORTED_LR_SCHEDULER_KEYS` tuple.
- `dataset.py:165` — `_numpy_to_tensor` reconstructed `np.dtype`
  objects on every call (4-5× per sample, in every worker); added
  `_resolve_expected_dtypes` with a module-level cache.

### Code commit `52d9bd2` (`pyproject.toml` 0.82.3 → 0.82.4, patch)

Applied on a follow-up request after the initial run — this was the one
deferred item whose blast radius was already contained, held back only
for session budget.

- `training_loop.py` — the 29-line non-finite-loss guard was duplicated
  verbatim between `_train_batch_mixed_precision` and
  `_train_batch_full_precision` (differing only in the GNS-reset
  comment). Extracted to `_abort_on_nonfinite_loss()`; both paths now
  call it. This is the numerical safety net, so the duplication meant a
  hardening change applied to one path silently missed the other.
- `tests/maou/app/learning/test_training_loop.py` — coverage was
  **asymmetric in exactly the way the duplication predicted**:
  `test_nan_loss_does_not_call_scaler_update` exercised only the AMP
  path. Added
  `test_nan_loss_full_precision_skips_batch_and_resets_gns` asserting
  the batch is skipped and `GradientNoiseScaleEstimator.reset_cycle()`
  fires. Verified non-vacuous by neutering the guard — both tests then
  fail.

## Deferred

Verified but **not** applied. Each is real; the reason for deferring is
given.

1. **`streaming_dataset.py:604` — `__len__` overestimates, and it feeds
   the LR scheduler.** `StreamingStage2Dataset.__len__` uses
   `_compute_total_batches` (sum of per-file `ceil(rows/batch)`), but
   its `__iter__` concatenates `_FILES_PER_CONCAT = 10` files before
   batching, so the real count is lower. This is **not just tqdm**:
   `dl.py:318` and `:498` pass `steps_per_epoch=len(loader)` into the
   scheduler, so `total_steps` is inflated and cosine decay never
   completes. **Not applied because** an exact fix needs `num_workers`
   (grouping happens per worker over `_resolve_worker_files`' subset),
   which is not available in `__len__`; chunking the global
   `row_counts` by 10 is exact only for `num_workers <= 1`. Choosing
   how to model sharding is a design decision. **Magnitude is small** —
   at ~100K rows/file and batch 256 the error is ~3 batches per 10
   files (~0.08%), so it is not urgent; it matters for small shards.
   `StreamingKifDataset.__len__` (`:306`) is **correct** — that class
   batches per file, so the per-file ceiling sum is exact.

2. **Stage 1 / Stage 2 pipeline is cloned across five files.** Reported
   independently by three of the four review angles.
   `run_stage1_with_training_loop` / `run_stage2_with_training_loop`
   (`multi_stage_training.py:436` / `:585`, ~150 lines each) differ only
   in head class, callback class, metric getter
   (`get_epoch_accuracy` / `get_epoch_f1`) and two log strings — even
   the loop class is shared (`Stage1TrainingLoop = RawLogitsTrainingLoop`,
   `training_loop.py:1159`). `_build_stage1_model_and_optimizer` /
   `_build_stage2_model_and_optimizer` (`stage_component_factory.py:636`
   / `:724`) have byte-identical 38-line tails differing only in
   `stage_name=`. Also `dataset.py:202`/`:279` and
   `streaming_dataset.py:775`/`:835`. **Not applied because** it is a
   ~400-line refactor of the multi-stage training path — architecturally
   significant, not a contained cleanup.

3. **Six adapter classes are three duplicated pairs.**
   `Stage1ModelAdapter`/`Stage2ModelAdapter`
   (`multi_stage_training.py:111`/`:240`) differ in **zero** characters;
   `Stage1DatasetAdapter`/`Stage2DatasetAdapter` (`:151`/`:183`) differ
   only in a type annotation; `Stage1StreamingAdapter`/
   `Stage2StreamingAdapter` (`streaming_dataset.py:645`/`:610`) differ
   only in a redundant `hasattr` guard. Merging them also deletes the
   `isinstance` dispatch and its `TypeError` arm at
   `stage_component_factory.py:866-872`, which exists *only* to choose
   between two identical classes. **Not applied because** the six names
   are public and referenced from tests; the merge is worth doing but
   should land as its own reviewed change, not folded into a bug-fix
   audit.

4. **`callbacks.py` — `_ensure_device` written six times** (`:238`,
   `:362`, `:1007`, `:1396`, `:1521`, `:1668`), plus three copies of the
   loss-accumulator scaffolding (`Stage2F1Callback:1375`,
   `Stage1AccuracyCallback:1499`, `Stage3LossCallback:1652`).
   `ValidationCallback` hand-lists the same 13 accumulator tensors in
   three places (`__init__` / `_ensure_device` / `reset`), which is the
   exact shape that produces "new metric added, never moved to GPU,
   never reset" defects. **Not applied because** it is a base-class
   extraction across the module's metric hub (~250 lines → ~120);
   contained in principle but large enough to deserve its own change.

5. **`training_loop.py:1093` — per-batch host-device sync.**
   `if not has_legal.all():` calls `Tensor.__bool__` on a CUDA tensor,
   a full pipeline stall once per training *and* validation batch, to
   guard a warning. Stage 3 always ships a `legal_move_mask`, so the
   branch is always taken. **Not applied because** the branchless
   rewrite changes the loss path; it should be measured, not assumed.

6. **`training_loop.py:460` — `stream.synchronize()` blocks the host,
   defeating much of the prefetch it implements.** `wait_stream()` gives
   the same ordering guarantee device-side without stalling the CPU, and
   the `record_stream()` added in `073adbd` already covers the allocator
   hazard that would otherwise make this unsafe. **Not applied because**
   it is a second untested GPU-semantics change stacked on the first,
   with no CUDA device here to verify either. These two should be
   validated together on real hardware.

7. **`gradient_noise_scale.py:150,188-192,246` — one GPU sync per
   parameter tensor per micro-batch** (`.item()` inside
   `for param in model.parameters()`): 60-300 syncs per micro-batch on a
   ResNet/ViT backbone whenever adaptive batch is on. **Not applied
   because** accumulating into a device scalar changes when the value
   materializes, and GNS feeds the adaptive batch controller — needs a
   numerical equivalence check.

8. **`dataset.py:91` / `streaming_dataset.py:754` — an all-ones
   `legal_move_mask` is built per sample and shipped over PCIe per
   batch** (~9 MB/batch at B=1024), then consumed by five kernels that
   are no-ops for an all-ones mask. **Not applied because**
   `callbacks.py:493` keys the `policy_move_label_ce` metric off
   `legal_move_mask is not None`, so it must stay non-`None`; the fix is
   "build once on device", which changes the dataset contract.

9. **`polars_datasource.py:205-268` — `_PolarsField` fakes numpy flags
   to get past validation it cannot satisfy.** It guesses dtypes from
   Python value shape rather than the schema, and synthesizes a
   `FakeFlags` asserting `c_contiguous=True, writeable=True` purely so
   `dataset.py:186-198`'s zero-copy guards pass — the guards are
   structurally unreachable on this path. Separately,
   `domain/data/polars_tensor.py` already ships
   `polars_row_to_{preprocessing,stage1,stage2}_tensors` producing the
   finished tensors, and **all three have zero callers**. **Not applied
   because** switching to them activates never-exercised code and this
   is a documented public API (`docs/rust-backend.md:704`).

10. **`stage_component_factory.py:255` —
    `create_stage2_streaming_data_pipeline(test_ratio=...)` is documented
    but never read**; `interface/learn.py:830` already logs that it is
    ignored. **Not applied because** removing the parameter requires
    editing `interface/learn.py`, outside this path.

11. **`multi_stage_training.py:399-417` — `TruncatedStageModel.forward`
    re-drives `HeadlessNetwork`'s private preprocessing** (`_separate_inputs`,
    `_prepare_inputs`, `_hand_projection`, `_combine_board_and_hand`,
    `_embedding_channels`, `_board_size`), copying `network.py:164-172`
    verbatim. A public `HeadlessNetwork.embed_inputs()` would fix it.
    Related: `FreezableBackbone` (`domain/model/protocol.py`) does not
    declare `preprocess_for_blocks`, which the app layer depends on.

Smaller confirmed items, not individually filed: `dl.py:296`/`:1034`
emit the "Both freeze_backbone and trainable_layers specified" warning
twice per run; `dl.py:609` assigns a dead `epoch_number = 0`;
`callbacks.py:416,500,511` recompute `log_softmax` three times and
`topk` twice on identical logits per validation batch;
`setup.py:229-235` `stat()`s every data file on the startup path just
to average sizes; `setup.py:713-726`/`750-765` restate defaults that
`network.py:47-60` already has.

## Doc findings

Filed as `reviews/2026-08-08-learning-docs-drift.md` (`fa0701e`) —
**9 WRONG / 5 STALE**. **Approved by the user and applied in `5b61444`**
(10 files); proposal frontmatter set to `status: applied`.
Highest-impact:

- **WRONG** `docs/performance.md:35-43` — a "GPU Prefetching
  (Auto-Enabled)" section documenting `enable_gpu_prefetch=True` with
  "+53.2% training throughput (2,202 → 3,374 samples/sec)".
  `gpu_prefetcher.py` does not exist and `TrainingLoop.__init__` has no
  such parameter; the numbers are attributed to deleted code.
- **WRONG** `README.md:185-206` — documents
  `--tensorboard-histogram-frequency` / `--tensorboard-histogram-module`,
  removed 2026-08-04 and absent from `learn_model.py`.
- **WRONG** `docs/commands/utility_benchmark_training.md:31` — lists
  `cosine_annealing` and `step` as `--stage12-lr-scheduler` choices;
  both reach `ValueError` because the alias table
  (`interface/learn.py:66-82`) does not contain them. Doc *and* code
  defect — the code half is in the out-of-scope backlog.
- **WRONG** `README.md:208-218` — documents `mmap_mode="c"` and a
  `preprocessing_mmap_mode` argument; neither exists anywhere in `src/`.
- **WRONG** `docs/loss-functions.md:204` — attributes loss construction
  to the interface layer; `interface/learn.py` has zero loss references,
  it is entirely app-layer.
- **WRONG** `docs/loss-functions.md:94` — cites `--streaming`; the flag
  is `--no-streaming` (streaming is the default), so the note inverts.
- **STALE** `docs/architecture.md:142` — `array_type` listed as 2
  members; it is a 4-member `Literal`. Proposal adds a pointer to the
  canonical definition rather than just refreshing the list.

Also verified **accurate** (do not re-check): every `learn-model` CLI
flag and quoted default in `docs/commands/learn_model.md`; all its
enumerations (`--stage`, `--early-stopping-metric`,
`--policy-target-mode`, `--optimizer`, `--lr-scheduler`,
`--model-architecture`); `docs/learning-rate-tuning.md`'s sqrt-scaling
base 256, the 10%/1-epoch warmup rule, and "Stage 3 では sqrt scaling が
適用されない"; `docs/design/training-quality/index.md` in full;
`docs/rust-backend.md`'s `PolarsDataFrameSource` signature and
`array_type` set; the "legal_move_mask はダミー" invariant
(`docs/loss-functions.md:136-146`); every symbol named in
`docs/stage2-speed-investigation.md:240-248`. **Link integrity: no
broken links** — all CLAUDE.md Documentation Links targets and all
AGENTS.md paths resolve.

## Out of scope

Recorded for future `/audit-and-fix` runs; **not** fixed here.

1. `/audit-and-fix src/maou/infra/console` — `utility.py:487-495`
   declares `click.Choice(["warmup_cosine_decay", "cosine_annealing",
   "step"])` for `--stage12-lr-scheduler`, but `cosine_annealing` and
   `step` are absent from the alias table
   (`interface/learn.py:66-82`), so both always raise `ValueError`. The
   CLI advertises two options that can never work.
2. `/audit-and-fix src/maou/interface` — `learn.py:830` passes
   `stage2_test_ratio` into
   `create_stage2_streaming_data_pipeline` after logging that it is
   ignored; the parameter is unread (see Deferred 10). Removing it
   needs both sides.
3. `/audit-and-fix src/maou/domain/model` — `FreezableBackbone`
   (`protocol.py`) does not declare `preprocess_for_blocks`, which all
   three concrete backbones implement and `multi_stage_training.py:413`
   calls. Also `DomainResNet` lacks `forward_features` (ViT and
   MLP-Mixer have it), forcing the `getattr` probe + `RuntimeError` at
   `network.py:179-188`.
4. `/audit-and-fix src/maou/domain/data` —
   `polars_tensor.py`'s `polars_row_to_preprocessing_tensors`,
   `polars_row_to_stage1_tensors`, `polars_row_to_stage2_tensors` and
   `dataframe_to_tensor_batch` have **zero callers** repo-wide. Either
   wire them in (see Deferred 9) or delete them.

## Environment notes

Unlike the `src/maou/domain/game_graph` run, the network was usable:
`uv sync --extra cpu --no-install-project` installed torch 2.11.0+cpu,
polars 1.38.1, ruff 0.16.1, mypy 1.19.1, onnxruntime 1.24.2.

`maou._rust` is **not** importable from a bare sync, and
`app/learning/network.py` → `domain/board/shogi.py` needs it, so pytest
on this path requires `python -m maturin develop --release` first
(**24m34s** in this container; a `patchelf` rpath warning is harmless).

The git pre-commit hook is **not installed** (`.git/hooks/pre-commit`
absent); hooks were run manually via `pre-commit run --files`. Results:
`trim trailing whitespace`, `fix end of files`, `check toml`,
`check for added large files`, **`check-cli-docs`** passed;
`uv-lock`, `test`, `mypy`, `ruff-check`, `ruff-format` **failed**, all
for the same environment reason — they are `uv run`-based, and `uv run`
re-resolves the project, which pulls `maou[tensorrt-infer]` →
`tensorrt-cu12-libs` from `pypi.nvidia.com`, unreachable here. The same
tools run directly against `.venv` all pass:

- `ruff format src/ tests/`: 285 files already formatted
- `ruff check src/ tests/`: all checks passed
- `mypy src/ tests/`: no issues in 284 source files
- `pytest tests/maou/app/learning`: **476 passed, 1 skipped**

A stale `.mypy_cache` produced a spurious
`AssertionError: Cannot find module for google`; `rm -rf .mypy_cache`
cleared it.

**Step 2 note.** The first attempt at the simplify pass lost all four
review agents to an API session limit and produced nothing; it was
re-run after the limit reset. This exposed a real defect in
`/audit-and-fix` itself — step 2 delegated to `/simplify`, whose default
scope is the current diff, so it would have reviewed step 1's own commit
rather than the path. Fixed in `5a2c0b0`
(`reviews/2026-08-08-audit-and-fix-simplify-scope.md`, applied).
