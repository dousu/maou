# `maou selfplay`

## Overview

- Runs **in-process self-play games** with the same game-playing agent as
  [`maou usi`](usi.md) (design:
  [docs/design/usi-engine/index.md §9](../design/usi-engine/index.md)).
- One game drives **two agents** (independent search trees for Black and
  White, each with subtree reuse across its own turns) directly in Rust —
  no stdio, no subprocesses, not even protocol string parsing.
- The **evaluator is shared process-wide**: the ONNX session (and TensorRT
  engine cache) is loaded and warmed up exactly once, then shared by every
  game. ONNX/GPU calls serialize on the shared session internally.
- **Game termination uses the same implementation as the USI engine**
  (semantic parity): nyugyoku declaration (27-point rule, re-validated by
  the driver; a false declaration loses), sennichite by the real four-fold
  repetition rule with perpetual-check classification (the perpetually
  checking side loses), max-moves draw (a valid declaration at the limit
  still wins), resignation (mate and threshold resign recorded
  separately), and illegal moves (a bug indicator; the offender loses).
- Per-game records (`sfen`, USI move list, winner, reason, playouts,
  elapsed time) can be written as **JSON Lines** with `--output`. A results
  summary is printed to stdout; per-game progress lines go to stderr.
- Parallelism is "games at a time" (`--parallel`); each game additionally
  uses `--threads` search threads per move. Node pools are pre-allocated
  per agent (two per game), so keep `--node-capacity` modest when running
  many games concurrently.
- Game diversification: `--opening-random-plies N` makes the driver play
  the first N plies uniformly at random among legal moves (deterministic
  per `--seed`, mixed with the game index). `--opening-script` applies a
  forced opening to both agents instead.
- HCPE (training data) generation from self-play records is a later
  campaign; this command produces the driver + records + smoke layer only.

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--model-path PATH` | | ONNX model file path. When omitted, a deterministic mock evaluator is used (development only). |
| `--games INT` | default `1` | Number of games to play. |
| `--parallel INT` | default `1` | Number of games played concurrently (worker threads). |
| `--playouts INT` | | Per-move playout budget (mutually exclusive with `--movetime-ms`; defaults to `800` when neither is given). |
| `--movetime-ms INT` | | Per-move thinking time in milliseconds (mutually exclusive with `--playouts`; no network-delay margin is applied in self-play). |
| `--max-moves INT` | default `512` | Move count for a drawn game (Denryu-sen rule). A valid nyugyoku declaration at the limit still wins. Also enables the agents' in-search draw terminal. |
| `--sfen "SFEN"` | | Starting position (default: even initial position). |
| `--opening-random-plies INT` | default `0` | Play this many uniformly random legal moves at the start of each game (diversification). |
| `--seed INT` | default `0` | Random seed for the opening randomization (mixed with the game index). |
| `--output PATH` | | Write per-game records as JSON Lines. |
| `--threads INT` | default `1` | Search threads per move. |
| `--batch-size INT` | default `8` | Evaluation batch size (use around 256 on GPU). |
| `--node-capacity INT` | default `65536` | Node pool capacity per agent (two agents per game; pools are pre-allocated). |
| `--draw-value-black INT` | default `500` | Draw value for Black in permille (Denryu-sen `400`). |
| `--draw-value-white INT` | default `500` | Draw value for White in permille (Denryu-sen `600`). |
| `--resign-value INT` | default `0` | Resign when the root win rate stays below this permille for `--resign-consecutive` moves (`0` = never). |
| `--resign-consecutive INT` | default `3` | Consecutive below-threshold moves required to resign. |
| `--opening-script "MOVES"` | | Forced opening move sequence in USI notation, applied to both agents. |
| `--root-dfpn/--no-root-dfpn` | **default on** | Root-parallel dfpn mate search. |
| `--root-dfpn-nodes INT` | default `2000000` | Node budget for the root dfpn mate search. |
| `--root-dfpn-depth INT` | default `2047` | Depth limit for the root dfpn mate search. |
| `--leaf-mate/--no-leaf-mate` | **default on** | Short mate search at MCTS leaves. |
| `--leaf-mate-nodes INT` | default `50` | Node budget per leaf-mate df-pn call. |
| `--leaf-mate-threads INT` | default `1` | Dedicated leaf-mate threads. |
| `--cuda/--no-cuda` | default off | CUDA Execution Provider (requires a wheel built with `onnx-cuda`). |
| `--tensorrt/--no-tensorrt` | default off | TensorRT Execution Provider (requires a wheel built with `onnx-tensorrt`). |
| `--trt-cache-dir PATH` | | TensorRT engine cache directory. |
| `--quiet` | default off | Suppress per-game progress lines on stderr. |

## Output

- `--output` writes one JSON object per game:
  `{"game_index": 0, "sfen": "...", "moves": ["7g7f", ...], "winner":
  "black"|"white"|null, "reason": "checkmate"|"resign"|"declaration"|
  "repetition"|"perpetual_check"|"max_moves"|"illegal_move",
  "black_player": "a"|"b", "plies": N, "playouts": N, "elapsed_ms": N}`.
  (`black_player` records color alternation in A/B matches driven via the
  Rust harness `cargo run -p maou_usi --example selfplay_ab`; plain CLI
  self-play always reports `"a"`.)
- stdout prints a summary: game count, black/white/draw results, a reason
  histogram, and totals for plies / playouts / summed game time.

## Example

```bash
# Mock smoke run (development only)
maou selfplay --games 2 --playouts 16 --max-moves 64 \
  --no-root-dfpn --no-leaf-mate --output records.jsonl

# Real-model self-play, 4 games at a time, varied openings
maou selfplay --model-path model.onnx --games 20 --parallel 4 \
  --playouts 800 --opening-random-plies 6 --seed 1 \
  --output selfplay.jsonl
```
