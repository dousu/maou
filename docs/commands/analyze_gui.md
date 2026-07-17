# `maou analyze-gui`

## Overview

- Launches a **Gradio browser UI for reviewing and interactively analyzing a
  game record** (CSA / KIF) together with an `maou analyze-game` JSON report.
  Design document: `docs/design/game-analysis/gui.md`.
- Viewer features:
  - Board view (SVG) with last-move highlight and **candidate-move arrows**
    (best move in red with rank labels, others in blue with opacity scaled by
    visit count; optional PV chain arrows for the best move).
  - **Win-rate / eval graph with selectable perspective** (sente / gote ×
    win rate / eval; the JSON stores side-to-move values and the UI
    converts them — the gote view is the mirror of the sente view), with
    mate markers (★) and a current-position indicator. Plotly-based.
  - Move list (Japanese notation, engine match ✓, sente win rate / eval,
    win-rate loss, mate ★) — clicking a row jumps to the position after that
    move.
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
  - The engine is loaded once per server process and search events are
    serialized (`concurrency_limit=1`). Without `--model-path` a
    deterministic **mock evaluator** is used (development verification only —
    clearly labeled in the UI).
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
| `--cuda/--no-cuda` | default off | Enable CUDA Execution Provider (requires `--model-path`). |
| `--tensorrt/--no-tensorrt` | default off | Enable TensorRT Execution Provider (requires `--model-path`). |
| `--trt-cache-dir PATH` | | TensorRT engine cache directory. |
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
- Gradio server (Blocks wiring, click bridge, engine events):
  `src/maou/infra/visualization/analysis_gui_server.py`
- Interface adapter (board SVG / graph / tables / perspective conversion /
  click state machine / breadcrumb): `src/maou/interface/analysis_gui.py`
- Use cases (kifu → per-ply snapshots, report validation, variation tree):
  `src/maou/app/analysis/analysis_session.py`; resident engine:
  `src/maou/app/analysis/interactive_analyzer.py`
- Multi-arrow board rendering and click targets: `ArrowSpec` /
  `interactive` in `src/maou/domain/visualization/board_renderer.py`
- Design document: `docs/design/game-analysis/gui.md`
