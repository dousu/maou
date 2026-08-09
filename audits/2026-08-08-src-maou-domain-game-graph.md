---
path: src/maou/domain/game_graph
scope: python
level: high
status: done
started: 2026-08-08
last_sha: 2686689
---

# Audit — src/maou/domain/game_graph

## Resume point

_(complete — no resume point)_

Scope resolved: python-src only — `model.py`, `openings.py`, `schema.py`
(no non-Python assets under this path). Owning manifest `pyproject.toml`.
Mirrored tests all present.

## Applied

Code commit `82342c6` (`pyproject.toml` 0.82.0 → 0.82.1, patch):

- `openings.py:8,11` — removed the unused module-level `logger` and its
  `import logging`.
- `openings.py:103-107` — hoisted the longest-first sort of
  `_DEFAULT_OPENINGS` out of `OpeningDatabase.__init__` into module-level
  `_ENTRIES_LONGEST_FIRST`. The ordering is a property of the table, not
  of an instance; re-sorting on every construction was pointless work.
  Safe to share: `OpeningEntry` is frozen and the list is never mutated.
- `model.py:1`, `schema.py:1` — Japanese module docstrings ended with a
  half-width `.`; changed to 全角 `．` per CLAUDE.md 日本語記述規則.
  `openings.py:1` was already correct, so this was inconsistent *within*
  the module.

Doc commit `2686689` (see Doc findings).

## Deferred

1. **`schema.py:41-79` — dead code, needs a keep/delete decision.**
   `_create_empty_df`, `create_empty_nodes_df`, `create_empty_edges_df`
   have **no production callers** — repo-wide grep finds them only in
   `tests/maou/domain/game_graph/test_schema.py`. The `size` parameter
   (the `extend_constant(None, size)` branch) exists solely to satisfy
   those tests. Additional defects if kept: `create_empty_nodes_df(5)`
   yields 5 all-null rows *including `position_hash`*, which the schema
   treats as the unique key; a negative `size` surfaces a raw polars
   `InvalidOperationError` rather than a `ValueError`.
   **Not applied because** deleting three public functions plus their
   tests is a public-API removal, not an unambiguous contained fix.
   Deciding to keep them means fixing the null-key and negative-size
   behaviour instead.

2. **`openings.py:109-133` — `find_opening` cannot express its own
   precondition.** It takes a bare `moves: list[str]` with no way to say
   "this list starts from 平手初期局面", but every entry in
   `_DEFAULT_OPENINGS` is only valid from startpos. That missing
   precondition is the root cause of the out-of-scope bug recorded below.
   **Not applied because** fixing it changes a public domain API that the
   interface layer calls, and the right shape (require an initial SFEN
   argument? return None unless root is startpos? move the check to the
   caller?) is a design decision, not a cleanup.

## Doc findings

Filed as `reviews/2026-08-08-game-graph-command-docs-drift.md` —
**approved and applied in `2686689`**; proposal frontmatter updated to
`status: applied` in `615cb56`.

Two docs carry substantive coverage of this module:
`docs/commands/build_game_graph.md` (producer) and
`docs/commands/visualize.md:147-194` (consumer). 7 fixes applied:

- **WRONG** `visualize.md:172` — node fill colour comes from
  `sente_best_move_win_rate` (`static/game_graph_canvas.js:241`), not
  `result_value`. `result_value` reaches the client as
  `sente_result_value` but only feeds the tooltip.
- **WRONG** `visualize.md:174` — node size is affine in **√p**
  (`canvas.js:64-66`), not proportional to `probability`.
- **STALE** `visualize.md:182` — 局面統計 is 6 items, not 4; the missing
  `定跡` row is the only user-visible output of `openings.py`.
- **STALE** `visualize.md:186-190` — 表示深さ max is 20, not 10
  (`game_graph_server.py:765-770`); 更新 / ルートに設定 / パンくずリスト /
  エクスポート were all undocumented.
- **STALE** `build_game_graph.md:6,48-71` — `metadata.json` was
  undocumented. Load-bearing: `game_graph_server.py:287,297` reads
  `initial_sfen` back to establish root turn for `_to_sente_perspective`.
  A doc-following third-party writer would render every win rate from
  the wrong side, silently.
- **STALE** `build_game_graph.md:20` — "Epic 2" was the only occurrence
  of that label in the entire repo; replaced with the real reference.
- **added** `build_game_graph.md:19` — `--max-depth ≤ 65535` (UInt16 in
  `schema.py:19`, enforced at `builder.py:74-78`) was undocumented and is
  not guarded by click, so it fails only after the full preprocess load.

Verified **accurate** (do not re-check): `num_branches` dual semantics
(the subtlest claim in the doc set — matches `builder.py:234-244,323-329`
exactly); `is_leaf`; `move_label` 0–1495 vs `MOVE_LABELS_NUM = 1496`;
"cshogi 互換 move16" wording (deliberate, per
`reviews/2026-07-15-cshogi-vocabulary-docs-sync.md`); Arrow IPC/LZ4
claim; BFS shortest-distance `depth`; all 7 CLI options and their
defaults; both column tables (6 node + 7 edge columns, every dtype);
55%/45% colour thresholds; 分岐分析 top-10. Link integrity: **no broken
links** — all 15 CLAUDE.md Documentation Links targets and all 7
Implementation-references paths resolve.

Fragile-but-correct (not changed, noted for later): the CLI option table
and the two column tables are hand-maintained enumerations with no test
guarding them. The column tables matter most — `_validate_schema`
(`game_graph_io.py:193-207`) makes the schema a hard load-time contract,
so a doc-driven writer with one wrong dtype produces files that fail to
load rather than degrade.

## Out of scope

Recorded for future `/audit-and-fix` runs; **not** fixed here.

1. `/audit-and-fix src/maou/interface` —
   `game_graph_visualization.py:718` `get_opening_name` matches the
   root-relative move list against `_DEFAULT_OPENINGS` without checking
   that the graph root is 平手初期局面. A graph built with
   `--initial-sfen <middlegame>` whose first edge is `5g5f` gets labelled
   「先手中飛車 / 振り飛車」. The sibling `export_sfen_path` (line 738)
   *does* branch on `_initial_sfen`, confirming the omission. Related to
   deferred item 2 above.
   **Correction** (2026-08-09, `cc10790`): the fix suggested above (branch
   on `_initial_sfen` like the sibling) would have disabled 定跡 display
   entirely. `build_game_graph.py:184-189` resolves 平手 to a concrete
   SFEN (`Board().get_sfen()`) before writing `metadata.json`, so
   `_initial_sfen` is **never `None`** on the production path — the
   sibling's `is not None` branch always takes the `position sfen` arm.
   The shipped fix compares the board/turn/hand SFEN fields against
   平手 instead.
2. `/audit-and-fix src/maou/app/game_graph` —
   `query.py:184-194` `GameGraphQuery.get_path_to_root` `break`s out of a
   broken parent chain and still returns the partial path reversed, so
   `path[0]` is not guaranteed to be the root. No reachable input was
   constructible from the builder's output (which always produces a
   connected chain), so this is unconfirmed.
3. `.claude/skills/type-safety-enforcer/SKILL.md:14` states
   "Line length: 88 characters maximum". The project uses
   `line-length = 64` (`pyproject.toml:220`), and
   `docs/code-quality.md:96-100` documents 64 correctly, noting the old
   88 lapsed with flake8's removal on 2026-08-04. The skill file is
   stale and agents follow it. Needs its own `reviews/` proposal.
   **Correction** (2026-08-09, `3600b32`): this finding was incomplete —
   the same 88 桁 staleness is also in
   `.claude/skills/qa-pipeline-automation/SKILL.md` (3 places), which this
   record did not mention. A reader acting only on the row above would
   have left the sibling skill stale.
4. `openings.py` is entirely undocumented — no mention of
   `OpeningDatabase`, `find_opening`, or any of the 9 opening names
   anywhere under `docs/`, `CLAUDE.md`, `AGENTS.md`, `README.md`.
   Recorded as follow-up in the applied proposal; needs new
   documentation drafted, which is separate work from drift correction.
5. `game_graph` has no architectural home: `CLAUDE.md`, `AGENTS.md`,
   `README.md`, `docs/architecture.md` and every `docs/adr-*.md` contain
   zero references to it.

## Environment notes

`uv run` is unusable in this container, so QA ran via
`.venv/bin/python` + `PYTHONPATH=src` after `uv sync --no-install-project`:

- ruff format: 3 files already formatted (project `line-length = 64`).
- ruff check: all checks passed.
- mypy: no issues in 3 source files.
- pytest `tests/maou/domain/game_graph`: **20 passed**.

Two distinct network blockers, neither caused by this audit:
- `ort-sys v2.0.0-rc.10`'s build script fails, so any `uv run` that
  rebuilds the editable project (maturin → cargo → `--features onnx`)
  dies before running the requested tool.
- pre-commit's `uv run`-based hooks (`test`, `mypy`, `ruff-check`,
  `ruff-format`, `uv-lock`) fail resolving torch from
  `download-r2.pytorch.org`, which is unreachable, while the configured
  index `download.pytorch.org` (`pyproject.toml:116,121`) is reachable.

The git pre-commit hook is **not installed** in this container
(`.git/hooks/pre-commit` absent), so commits did not trigger it; hooks
were run manually instead. On the doc-only commit every applicable hook
passed, including `check-cli-docs`.
