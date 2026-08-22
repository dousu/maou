# `maou analyze-gui`

## Overview

- Launches a **Gradio browser UI for reviewing and interactively analyzing a
  game record** (CSA / KIF) together with an `maou analyze-game` JSON report.
  Design document: `docs/design/game-analysis/gui.md`.
- The UI is a **three-column workbench** rendered as a single `gr.HTML`
  panel — no tabs. Left rail: blunder jump list (win-rate loss above an
  adjustable threshold) over the move list; centre: the board with branch
  breadcrumb and navigation; right rail: eval graph, candidate moves, and the
  analysis panel. Everything is visible at once. Designed for laptop widths
  (1280-1440px); the workbench has a `min-width` of 1280px.
- Viewer features:
  - Board view (SVG) with last-move highlight and **candidate-move arrows**
    (best move in red with rank labels, others in blue with opacity scaled by
    visit count; optional PV chain arrows for the best move).
  - **Win-rate / eval graph with selectable perspective** (sente / gote ×
    win rate / eval; the JSON stores side-to-move values and the UI
    converts them — the gote view is the mirror of the sente view), with
    blunder markers (●), mate markers (★) and a current-position indicator.
    Rendered as inline SVG; **clicking the plot jumps to that ply**, and the
    ● / ★ markers jump individually.
  - Move list (Japanese notation, engine match ✓, sente win rate / eval,
    win-rate loss, mate ★) — clicking a row jumps to the position after that
    move. Moves at or above the blunder threshold are shown in red.
  - **Blunder jump list**: moves whose win-rate loss is at or above an
    adjustable threshold (default `0.10`, editable in place). This is a
    single filter, not a good/bad classification — no 疑問手 / 悪手 badges,
    numbers only. Clicking a row jumps to the position **before** that move,
    so it lines up with the candidate-move arrows. The threshold drives both
    this list and the red highlighting in the move list.
  - Per-position candidate table (rank / move / visits / win rate
    (side-to-move) / prior / proven value) — clicking a row plays that move
    as a variation.
  - Position info: SFEN and USI `position` string for hand-off to other
    tools, per-move notes (engine best vs played, recorded time / comments),
    and the engine evaluation of the current position (win rate / eval in
    side-to-move and sente perspectives) whenever an analysis is cached —
    including freshly analyzed variation positions.
  - Files can be loaded at startup (CLI flags) or uploaded from the UI.
- Interactive analysis features:
  - **Board click input**: click a piece (or a piece in hand) then a
    destination square to play a move; legal destinations are highlighted
    and a promote / no-promote confirmation is shown when both are legal.
    A legal-move dropdown is provided as a fallback input.
  - **Variation branching (継ぎ盤)**: playing a move that differs from the
    mainline automatically creates a branch. A breadcrumb shows the branch
    point and moves; "本譜へ戻る" returns to the mainline. Branches persist
    for the session.
  - **Single-position analysis**: analyze the current position (mainline or
    branch) with the resident engine. Results are cached per position;
    "再解析" overwrites the cache. "PV を分岐で再生" replays the analyzed PV
    as a branch.
  - **Whole-game analysis**: analyze every mainline position with progress
    display and cooperative cancellation; the result updates the graph /
    move list and is downloadable as an analyze-game compatible JSON report.
    The analysis panel keeps a progress bar and a status line visible at all
    times.
  - The engine is loaded once per server process and search events are
    serialized (`concurrency_limit=1`). Without `--model-path` a
    deterministic **mock evaluator** is used (development verification only —
    clearly labeled in the UI).
- **Keyboard shortcuts**:

  | Key | Action |
  | --- | --- |
  | `←` / `→` | Previous / next move |
  | `⇧←` / `⇧→` | Previous / next blunder |
  | `Home` / `End` | Initial / final position |
  | `B` | Back to the mainline |
  | `Space` | Analyze the current position |
  | `1`-`5` | Branch on that candidate move |
  | `L` | Toggle the legal-move list |
  | `Esc` | Clear the board selection |

- Requires the `visualize` extra (`uv sync --extra visualize`). Viewing a
  pre-computed report needs no model or GPU; in-GUI analysis quality requires
  a real ONNX model (`--model-path`).

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--input-path PATH` | | Game record file (CSA / KIF) loaded at startup. Files can also be uploaded from the UI. |
| `--report PATH` | | analyze-game JSON report (the `--output` file) matching the game record. Requires `--input-path`. The report is validated against the record (move count, played moves, per-position SFEN). |
| `--model-path PATH` | | ONNX model file for in-GUI analysis. Uses a deterministic mock evaluator when omitted (development only; labeled in the UI). |
| `--time-ms INT` | | Default time budget per in-GUI analysis in milliseconds (default `1000`). Mutually exclusive with `--playouts`. |
| `--playouts INT` | | Default playout budget per in-GUI analysis. Mutually exclusive with `--time-ms`. |
| `--num-candidates INT` | default `5` | Maximum number of candidate moves shown in the UI (table rows and arrows) and recorded per position. |
| `--threads INT` | default `1` | Number of search threads. |
| `--batch-size INT` | default `8` | Evaluation batch size. |
| `--root-dfpn/--no-root-dfpn` | default on | Run dfpn mate search on each root position in parallel. |
| `--root-dfpn-nodes INT` | default `2000000` | Node budget for the root dfpn mate search. |
| `--root-dfpn-depth INT` | default `2047` | Search depth limit for the root dfpn mate search (max 2047). |
| `--leaf-mate/--no-leaf-mate` | default on | Enable short mate search at MCTS leaves (async). |
| `--leaf-mate-nodes INT` | default `50` | Node budget per leaf-mate df-pn call. |
| `--leaf-mate-threads INT` | default `1` | Number of dedicated leaf-mate threads. |
| `--defensive-mate/--no-defensive-mate` | **default on** | Prove whether the side to move is *being* mated (root + leaves that are in check), and drop root moves that let the opponent force mate. Without it the engine only asks whether it can mate, so a forced mate against it stays invisible until the last ply. |
| `--defensive-mate-threads INT` | default `1` | Parallelism of the defensive filter that screens root moves. Each root move is judged independently, so raising this uses spare CPU to reach a larger budget within the same wall clock (memory scales with it, ~88MB/thread). |
| `--cuda/--no-cuda` | default off | Enable CUDA Execution Provider (requires `--model-path`). |
| `--tensorrt/--no-tensorrt` | default off | Enable TensorRT Execution Provider (requires `--model-path`). |
| `--trt-cache-dir PATH` | | TensorRT engine cache directory. |
| `--tensorrt/--no-tensorrt` (exit behaviour) | | With TensorRT enabled the command exits **without running destructors** once the server stops, for the same reason as [`analyze_game.md`](analyze_game.md). |
| `--port INT` | | Gradio server port. Auto-selected by Gradio when omitted. |
| `--share` | flag | Create a public Gradio link (auto-enabled on Google Colab). |
| `--server-name HOST` | default `127.0.0.1` | Server bind address. |

## Example invocation

```bash
# 1. Analyze a game (e.g. on Colab GPU) to produce the JSON report
uv run maou analyze-game \
  --input-path game.csa --model-path model.onnx \
  --time-ms 1000 --output report.json

# 2. Review it in the browser (no model needed)
uv run maou analyze-gui --input-path game.csa --report report.json

# Interactive analysis with a real model (per-position 1000ms)
uv run maou analyze-gui --input-path game.csa --model-path model.onnx

# Board replay / branching only (no report, mock engine labeled in UI)
uv run maou analyze-gui --input-path game.csa

# Start empty and upload files from the UI
uv run maou analyze-gui
```

## Implementation references

- CLI: `src/maou/infra/console/analyze_gui.py`
- Gradio server (Blocks wiring, `data-action` bridges, engine events):
  `src/maou/infra/visualization/analysis_gui_server.py`
- Interface adapter (board SVG / tables / perspective conversion /
  click state machine / breadcrumb): `src/maou/interface/analysis_gui.py`
- Workbench HTML assembly (three-column layout, eval-graph SVG, blunder
  list): `src/maou/interface/analysis_workbench.py`
- Workbench styling and browser wiring (delegated listeners, shortcuts):
  `src/maou/infra/visualization/static/analysis_workbench.css` /
  `analysis_workbench.js`
- Use cases (kifu → per-ply snapshots, report validation, variation tree):
  `src/maou/app/analysis/analysis_session.py`; resident engine:
  `src/maou/app/analysis/interactive_analyzer.py`
- Multi-arrow board rendering, click targets and theming: `ArrowSpec` /
  `interactive` / `BoardTheme` in
  `src/maou/domain/visualization/board_renderer.py`. The default
  `SVGBoardRenderer()` output is unchanged; the workbench passes
  `MODERNIST_BOARD_THEME`.
- Design document: `docs/design/game-analysis/gui.md`
