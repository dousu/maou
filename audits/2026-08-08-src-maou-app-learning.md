---
path: src/maou/app/learning
scope: python
level: high
status: in-progress
started: 2026-08-08
last_sha: ecc0905
---

# Audit — src/maou/app/learning

## Resume point

Claimed at step 0f; audit not yet started. Next: step 1 (`/code-review
src/maou/app/learning high`).

Scope resolved: python-src only — 17 `.py` files, 11,544 lines, no
non-Python assets (`git ls-files` returns `.py` exclusively). Owning
manifest `pyproject.toml` (0.82.1). Mirrored test dir
`tests/maou/app/learning/` exists with 31 test files.

Sub-paths not yet covered: all of them —
`adaptive_batch.py`, `callbacks.py`, `compilation.py`, `dataset.py`,
`dl.py`, `gradient_noise_scale.py`, `model_io.py`,
`multi_stage_training.py`, `network.py`, `onnx_verifier.py`,
`polars_datasource.py`, `policy_targets.py`, `resource_monitor.py`,
`setup.py`, `stage_component_factory.py`, `streaming_dataset.py`,
`training_loop.py`.

## Applied

_(none yet)_

## Deferred

_(none yet)_

## Doc findings

_(none yet)_

## Out of scope

_(none yet)_

## Environment notes

`uv sync --extra cpu --no-install-project` succeeded in this container
(torch 2.11.0+cpu, polars 1.38.1, ruff 0.16.1, mypy 1.19.1) — unlike the
`src/maou/domain/game_graph` run, `download.pytorch.org` was reachable.
The Rust extension `maou._rust` is **not** importable from a bare sync,
and `maou.app.learning.network` imports `maou.domain.board.shogi` which
imports it, so pytest on this path needs `maturin develop` first.
