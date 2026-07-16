# `maou analyze-gui`

## Overview

- Launches a **Gradio browser UI for reviewing a game record** (CSA / KIF)
  together with an `maou analyze-game` JSON report. Design document:
  `docs/design/game-analysis/gui.md`.
- Features (viewer milestone):
  - Board view (SVG) with last-move highlight and **candidate-move arrows**
    (best move in red with rank labels, others in blue with opacity scaled by
    visit count; optional PV chain arrows for the best move).
  - **Win-rate / eval graph from the sente perspective** (the JSON stores
    side-to-move values; the UI converts them), with mate markers (★) and a
    current-position indicator. Plotly-based.
  - Move list (Japanese notation, engine match ✓, sente win rate / eval,
    win-rate loss, mate ★) — clicking a row jumps to the position after that
    move.
  - Per-position candidate table (rank / move / visits / win rate
    (side-to-move) / prior / proven value).
  - Position info: SFEN and USI `position` string for hand-off to other
    tools, per-move notes (engine best vs played, recorded time / comments).
  - Files can be loaded at startup (CLI flags) or uploaded from the UI.
- Interactive analysis (board move input, variation branching, on-demand
  engine search from the GUI) is designed but **not implemented yet** — see
  `docs/design/game-analysis/gui.md` §13 milestones.
- Requires the `visualize` extra (`uv sync --extra visualize`); analysis
  reports are produced separately by `maou analyze-game` (any environment,
  e.g. Colab GPU), so the GUI itself needs no model or GPU.

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--input-path PATH` | | Game record file (CSA / KIF) loaded at startup. Files can also be uploaded from the UI. |
| `--report PATH` | | analyze-game JSON report (the `--output` file) matching the game record. Requires `--input-path`. The report is validated against the record (move count, played moves, per-position SFEN). |
| `--num-candidates INT` | default `5` | Maximum number of candidate moves shown in the UI (table rows and arrows). |
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

# Board replay only (no report)
uv run maou analyze-gui --input-path game.csa

# Start empty and upload files from the UI
uv run maou analyze-gui
```

## Implementation references

- CLI: `src/maou/infra/console/analyze_gui.py`
- Gradio server (Blocks wiring): `src/maou/infra/visualization/analysis_gui_server.py`
- Interface adapter (board SVG / graph / tables / perspective conversion):
  `src/maou/interface/analysis_gui.py`
- Use case (kifu → per-ply snapshots, report validation):
  `src/maou/app/analysis/analysis_session.py`
- Multi-arrow board rendering: `ArrowSpec` in
  `src/maou/domain/visualization/board_renderer.py`
- Design document: `docs/design/game-analysis/gui.md`
