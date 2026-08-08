# Audit coverage ledger

One row per path touched by `/audit-and-fix`. Shape, status vocabulary,
and protocol: [README.md](README.md).

**This table lists only what has been audited.** It is not a plan and not
an inventory of remaining work — to see what is left, compare against the
tree (`ls src/maou/*/`, `ls rust/`, `find docs -name '*.md'`), which is
always current where a checked-in list would not be.

| Path | Scope | Status | Level | Last SHA | Record | Open items |
|---|---|---|---|---|---|---|
| `src/maou/domain/game_graph` | python | done | high | `2686689` | [2026-08-08](2026-08-08-src-maou-domain-game-graph.md) | 2 deferred, 5 out-of-scope |
| `src/maou/app/learning` | python | done | high | `fa0701e` | [2026-08-08](2026-08-08-src-maou-app-learning.md) | 12 deferred, 4 out-of-scope, 1 reviews pending |

## Blocked

_(none)_

<!-- Rows move here only while status is `blocked`, with the blocker and
     what would unblock it. Keep the main table for in-progress/done. -->

## Out-of-scope backlog

Findings an audit surfaced *outside* the path it was auditing. They live
here, not only in the per-run record, because a record is read only when
someone opens that specific path — while this file is read at the start of
every run. That is what makes them recoverable.

**Protocol.** Before auditing a path, check this table for rows whose
target falls inside it and fold them into the run. When a row is
resolved, delete it — the resolving audit's record is the durable
account. Do not delete a row that was merely re-triaged.

| Found by | Target | Item |
|---|---|---|
| [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) | `src/maou/interface` | `game_graph_visualization.py:718` `get_opening_name` matches the root-relative move list against `_DEFAULT_OPENINGS` without checking the graph root is 平手初期局面, so a graph built with `--initial-sfen <middlegame>` starting `5g5f` is mislabelled 「先手中飛車」. Sibling `export_sfen_path` (:738) does branch on `_initial_sfen`. |
| [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) | `src/maou/app/game_graph` | `query.py:184-194` `get_path_to_root` breaks out of a broken parent chain and still returns the partial path reversed, so `path[0]` is not guaranteed to be the root. Unconfirmed — no reachable input was constructible from the builder's output. |
| [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) | `.claude/skills/type-safety-enforcer/SKILL.md` | Line 14 states "Line length: 88 characters maximum"; the project uses `line-length = 64` (`pyproject.toml:220`). `docs/code-quality.md:96-100` is correct and notes the 88 lapsed with flake8's removal on 2026-08-04. Agents follow this file. |
| [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) | `docs/` (new) | `src/maou/domain/game_graph/openings.py` is entirely undocumented — `OpeningDatabase`, `find_opening`, and all 9 opening names return zero hits across `docs/`, `CLAUDE.md`, `AGENTS.md`, `README.md`, despite being user-visible via the 定跡 row. Needs a new doc drafted. |
| [2026-08-08 game_graph](2026-08-08-src-maou-domain-game-graph.md) | `docs/architecture.md` | `game_graph` has no architectural home: `CLAUDE.md`, `AGENTS.md`, `README.md`, `docs/architecture.md` and every `docs/adr-*.md` contain zero references to it. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `src/maou/infra/console` | `utility.py:487-495` declares `click.Choice(["warmup_cosine_decay", "cosine_annealing", "step"])` for `--stage12-lr-scheduler`, but `cosine_annealing` / `step` are absent from the alias table (`interface/learn.py:66-82`), so both always raise `ValueError`. The CLI advertises two options that can never work. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `src/maou/interface` | `learn.py:830` passes `stage2_test_ratio` into `create_stage2_streaming_data_pipeline` right after logging that it is ignored; the parameter is never read in the callee (`stage_component_factory.py:255,277-320`). Removing it needs both sides edited together. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `src/maou/domain/model` | `FreezableBackbone` (`protocol.py`) does not declare `preprocess_for_blocks`, which all three concrete backbones implement and `multi_stage_training.py:413` calls. Also `DomainResNet` lacks `forward_features` (ViT/MLP-Mixer have it), forcing the `getattr` probe + `RuntimeError` at `network.py:179-188`. |
| [2026-08-08 app/learning](2026-08-08-src-maou-app-learning.md) | `src/maou/domain/data` | `polars_tensor.py`'s `polars_row_to_preprocessing_tensors`, `polars_row_to_stage1_tensors`, `polars_row_to_stage2_tensors`, `dataframe_to_tensor_batch` have **zero callers** repo-wide. Either wire them into `PolarsDataFrameSource` (which reimplements them via numpy-structured-array mimicry) or delete them. |
