# `maou analyze-game`

## Overview

- Analyzes a whole game record (CSA / KIF) by running the MCTS engine — the
  same engine as `maou search` — on the position **before each move**, and
  produces a machine-readable JSON report plus a human-readable summary
  (best-move match rate per player, biggest win-rate losses, mates found).
  Design document: `docs/design/game-analysis/index.md`.
- The evaluator (ONNX model or the deterministic mock) is loaded **once** via
  the persistent engine binding (`maou._rust.maou_search.SearchEngine`) and
  reused across all positions, so per-position model loading is avoided. With
  TensorRT, the engine cache (`--trt-cache-dir`) additionally makes warmup
  nearly free from the second process on.
- Time management lives in this command (a budget allocator decides the
  per-position budget); the search itself only consumes the budget it is
  given. Three allocation modes are available: fixed time per position
  (`--time-ms`), total time divided equally (`--total-time-ms`), and fixed
  playouts per position (`--playouts`). These are mutually exclusive; when
  none is given, 1000 ms per position is used (same default as `maou search`).
- Only single-game files are supported: a CSA file containing multiple games
  is rejected with an error (split it into one game per file, e.g. the output
  of `maou fetch-floodgate`). File encoding is auto-detected (strict UTF-8
  first, then cp932).

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--input-path PATH` | ✅ | Game record file (CSA / KIF). |
| `--input-format csa\|kif` | | Record format. Auto-detected from the file extension (`.csa` → csa, `.kif` / `.kifu` → kif) when omitted; unknown extensions require this flag. |
| `--model-path PATH` | | ONNX model file path. When omitted, a deterministic mock evaluator is used (API verification only — analysis quality is meaningless). Published wheels (Release `latest`) support this out of the box; local development builds need the `onnx` cargo feature (see `docs/commands/search.md`). |
| `--time-ms INT` | | Time budget per position in milliseconds. Mutually exclusive with `--total-time-ms` / `--playouts`. Defaults to 1000 when no budget option is given. |
| `--total-time-ms INT` | | Total time budget for the whole game, divided equally across positions (floor division, min 1 ms per position). |
| `--playouts INT` | | Playout budget per position. |
| `--num-candidates INT` | default `5` | Number of candidate moves recorded per position in the JSON (best move first, then by visit count). |
| `--output PATH` | | Write the JSON report to this file and print the human-readable summary to stdout. When omitted, the JSON itself is printed to stdout (pipe-friendly); progress goes to stderr either way. |
| `--threads INT` | default `1` | Number of search threads. |
| `--batch-size INT` | default `8` | Evaluation batch size. |
| `--root-dfpn/--no-root-dfpn` | **default on** | Run dfpn mate search on each root position in parallel (same as `maou search`). Proven mates appear as `stop=root_proven` / `mate_found=true`. |
| `--root-dfpn-nodes INT` | default `2000000` | Node budget for the root dfpn mate search. |
| `--root-dfpn-depth INT` | default `2047` | Search depth limit for the root dfpn mate search (max 2047). |
| `--leaf-mate/--no-leaf-mate` | **default on** | Enable short mate search at MCTS leaves (async, dedicated threads). |
| `--leaf-mate-nodes INT` | default `50` | Node budget per leaf-mate df-pn call. |
| `--leaf-mate-threads INT` | default `1` | Number of dedicated leaf-mate threads. |
| `--cuda/--no-cuda` | default `--no-cuda` | Enable the CUDA Execution Provider. Requires `--model-path` and a wheel built with `onnx-cuda`. |
| `--tensorrt/--no-tensorrt` | default `--no-tensorrt` | Enable the TensorRT Execution Provider. Requires `--model-path` and a wheel built with `onnx-tensorrt`. |
| `--trt-cache-dir PATH` | | TensorRT engine cache directory. |

## Outputs

JSON report (schema details: `docs/design/game-analysis/index.md` §7):

- `input` — file path, format, player names, ratings, result (`win`:
  0=draw / 1=black / 2=white), endgame marker, number of moves.
- `engine` — model path (null = mock), threads, batch size, EP flags, mate
  search options.
- `budget` — allocation mode, its parameter, and the resolved per-position
  budget.
- `positions[]` — one record per move: `ply`, `side_to_move` (`b`/`w`),
  `sfen` (position before the move), `played_move` / `best_move` (USI) and
  `match`, `winrate` / `eval_cp` (side-to-move view; Ponanza-style
  `600 × logit`), `played_move_winrate` / `winrate_loss` (from the same
  position's root statistics; null when the played move was not visited),
  `pv`, `candidates[]` (usi / visits / winrate / prior / proven),
  `mate_found` (`stop == "root_proven"`), `playouts` / `elapsed_ms` / `stop`,
  and the record's own metadata echo (`record_time_s` / `record_score` /
  `record_comment`).
- `summary` — per-player best-move `match_rate` and `mean_winrate_loss`,
  `worst_moves` (top 5 by `winrate_loss`), `mates_found`,
  `total_elapsed_ms` / `total_playouts`.

The stdout summary (with `--output`) shows the players, result, match rates,
mean win-rate losses, the worst moves, mates found, and total elapsed time.

## Example invocation

```bash
# Quick check with the mock evaluator (development only)
uv run maou analyze-game \
  --input-path game.csa --playouts 100 --output report.json

# Real analysis: 1 second per position with an ONNX model
uv run maou analyze-game \
  --input-path game.csa --model-path model.onnx \
  --time-ms 1000 --threads 4 --batch-size 8 \
  --output report.json

# Whole-game budget: 60 seconds divided equally across positions
uv run maou analyze-game \
  --input-path game.kif --model-path model.onnx \
  --total-time-ms 60000 --output report.json

# Pipe the JSON (no --output): summary is omitted, progress on stderr
uv run maou analyze-game --input-path game.csa --playouts 800 | jq '.summary'
```

## Implementation references

- CLI: `src/maou/infra/console/analyze_game.py`
- Interface (validation / allocator construction / formatting):
  `src/maou/interface/analyzer.py`
- Use case (game walk, per-position search, summary):
  `src/maou/app/analysis/game_analyzer.py`
- Budget allocation strategies: `BudgetAllocator` and implementations in
  `src/maou/app/analysis/game_analyzer.py`
- Persistent engine binding: `SearchEngine` in
  `rust/maou_rust/src/maou_search.rs`
- Design document: `docs/design/game-analysis/index.md`
