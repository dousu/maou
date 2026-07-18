# `maou usi`

## Overview

- Runs maou as a **USI (Universal Shogi Interface) engine** on stdin/stdout,
  so it can play against humans or other USI engines from Shogi GUIs
  (Shogidokoro / ShogiGUI / ShogiHome). Design:
  [docs/design/usi-engine/index.md](../design/usi-engine/index.md).
- The protocol loop, game-playing agent, and time management all run in Rust
  (`maou_usi` crate) with the GIL released; the Python layer only passes the
  configuration (`maou._rust.maou_usi.run_usi`). A dedicated reader thread
  keeps `stop` / `quit` responsive even while the engine is thinking.
- **stdout is reserved for the USI protocol**; logs go to stderr.
- CLI flags provide **initial values**, which USI `setoption` commands from
  the GUI override. Heavy initialization (ONNX model load, TensorRT engine
  build, warmup inference) happens on `isready`.
- When no model is configured, a deterministic **mock evaluator** is used and
  announced with `info string mock evaluator (development only) ...` on
  `isready` (development/verification only — move quality is meaningless).
- Supported through milestone M3: full game loop (`usi` / `isready` /
  `setoption` / `usinewgame` / `position` / `go` with
  `btime wtime byoyomi binc winc`, `go infinite`, `go nodes`, `go movetime` /
  `stop` / `gameover` / `quit`); time strategy with soft/hard budgets and
  best-move-instability extension (byoyomi / Fischer / sudden death, network
  delay margin); streaming `info` during search; draw-value strategy
  (`DrawValueBlack` / `DrawValueWhite`); nyugyoku declaration win
  (`bestmove win`, 27-point rule); resign threshold (`ResignValue` /
  `ResignConsecutive`, off by default); `MaxMovesToDraw` (declaration check +
  budget narrowing near the limit); root-dfpn + leaf-mate search; **pondering**
  (`USI_Ponder`, `go ponder` / `ponderhit`, and `bestmove <move> ponder
  <reply>` with the predicted reply = PV's 2nd move) — a ponder hit *continues*
  the same unbounded search under a fresh time budget, so the tree built while
  pondering carries over (the main ponder benefit); and **subtree reuse across
  moves** — when the game advances along an explored line, the retained search
  tree is rerooted to the new position so its subtree warm-starts the next
  search instead of rebuilding from scratch (a ponder *miss*, or any advance
  the tree did not explore, falls back to a fresh search).
  Not yet implemented (later milestones): tournament opening scripts and the
  self-play driver (M4), the in-search `MaxMovesToDraw` draw terminal (M4),
  `go mate` (answers `checkmate notimplemented`).

## Engine registration in a GUI

- The `maou-usi` console script starts the engine **without arguments**
  (settings via `setoption`), for GUIs that cannot pass command-line
  arguments (e.g. Shogidokoro):
  - Linux / macOS: register `<venv>/bin/maou-usi`.
  - Windows: register `<venv>\Scripts\maou-usi.exe` (in Shogidokoro's file
    dialog, switch the filter to "all files" if needed).
- Alternatively register a one-line wrapper script that runs
  `maou usi --model-path ... --threads ...` to bake in CLI defaults.
- Configure `ModelPath` (and `UseCuda` / `UseTensorRT` on GPU machines) in
  the GUI's engine-options dialog, then start a game. The first `isready`
  performs model load and warmup (TensorRT engine build can take minutes on
  first run; use `TrtCacheDir` to cache).

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--model-path PATH` | | ONNX model file path. When omitted, the mock evaluator is used (announced via `info string`). Can also be set from the GUI via `setoption name ModelPath`. |
| `--threads INT` | default `1` | Number of search threads. |
| `--batch-size INT` | default `8` | Evaluation batch size (use around 256 on GPU; also the TensorRT padding size). |
| `--node-capacity INT` | | Node pool capacity (default 2^20 nodes). |
| `--network-delay-ms INT` | default `1000` | Communication overhead margin in milliseconds. The GUI/server measures elapsed time including transport, so the per-move budget is reduced by this amount. |
| `--min-think-ms INT` | default `100` | Minimum thinking time in milliseconds. |
| `--draw-value-black INT` | default `500` | Draw value for Black in permille. Repetition / max-moves draw terminals are valued at this (root side-to-move view). Denryu-sen Black 0.4 win = `400`. |
| `--draw-value-white INT` | default `500` | Draw value for White in permille (Denryu-sen White 0.6 win = `600`). |
| `--resign-value INT` | default `0` | Resign when the root win rate stays below this permille for `--resign-consecutive` moves. `0` = never resign. |
| `--resign-consecutive INT` | default `3` | Consecutive below-threshold moves required to resign (with `--resign-value > 0`). |
| `--max-moves-to-draw INT` | default `0` | Move count for a drawn game (`0` = disabled; Denryu-sen `512`). At/near the limit the engine always checks nyugyoku declaration and narrows its search budget. |
| `--usi-ponder/--no-usi-ponder` | **default on** | Enable pondering (thinking on the opponent's turn). When on, the engine declares `USI_Ponder` and appends the predicted reply to `bestmove` so the GUI sends `go ponder`. |
| `--root-dfpn/--no-root-dfpn` | **default on** | Run dfpn mate search on the root position in parallel with MCTS. |
| `--root-dfpn-nodes INT` | default `2000000` | Node budget for the root dfpn mate search. |
| `--root-dfpn-depth INT` | default `2047` | Search depth limit for the root dfpn mate search (max 2047). |
| `--leaf-mate/--no-leaf-mate` | **default on** | Enable short mate search at MCTS leaves (async, dedicated threads). |
| `--leaf-mate-nodes INT` | default `50` | Node budget per leaf-mate df-pn call. |
| `--leaf-mate-threads INT` | default `1` | Number of dedicated leaf-mate threads. |
| `--cuda/--no-cuda` | default off | Enable CUDA Execution Provider (requires a wheel built with `onnx-cuda`). |
| `--tensorrt/--no-tensorrt` | default off | Enable TensorRT Execution Provider (requires a wheel built with `onnx-tensorrt`). |
| `--trt-cache-dir PATH` | | TensorRT engine cache directory. Created automatically if missing; startup fails with a clear error when the parent path is unavailable (e.g. Google Drive not mounted on Colab). |

## USI options (`setoption`)

Declared in the `usi` response; defaults reflect the CLI flags above.

| Option | Type | Description |
| --- | --- | --- |
| `ModelPath` | filename | ONNX model path (empty = mock evaluator). |
| `Threads` / `BatchSize` / `NodeCapacity` | spin | Search resources. |
| `USI_Hash` | spin (MB) | Used to derive `NodeCapacity` when the latter is not set (approx. 512 bytes/node). `0` = ignore. |
| `UseCuda` / `UseTensorRT` | check | Execution providers (feature-gated wheel required). |
| `TrtCacheDir` | string | TensorRT engine cache directory. |
| `NetworkDelay` | spin (ms) | Communication margin subtracted from each move budget. |
| `MinimumThinkingTime` | spin (ms) | Minimum thinking time. |
| `DrawValueBlack` / `DrawValueWhite` | spin (permille) | Draw value per side (default 500; Denryu-sen 400 / 600). Converted to the search's side-to-move `draw_value`. |
| `ResignValue` | spin (permille) | Resign win-rate threshold (0 = never). |
| `ResignConsecutive` | spin | Consecutive below-threshold moves required to resign. |
| `MaxMovesToDraw` | spin | Move count for a drawn game (0 = disabled; Denryu-sen 512). |
| `USI_Ponder` | check | Enable pondering (default on). Declared so the GUI sends `go ponder`; `bestmove` carries the predicted reply (PV's 2nd move). |
| `RootDfpn` / `LeafMate` | check | Mate search toggles. |

## Example

```bash
# Manual smoke test (mock evaluator; type or pipe USI commands)
printf 'usi\nisready\nposition startpos\ngo btime 0 wtime 0 byoyomi 1000\nquit\n' \
  | maou-usi

# Start with a model and GPU (initial values; setoption can override)
maou usi --model-path model.onnx --threads 2 --batch-size 256 \
  --tensorrt --trt-cache-dir /path/to/trt-cache
```
