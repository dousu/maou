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
- Supported through milestone M4: full game loop (`usi` / `isready` /
  `setoption` / `usinewgame` / `position` / `go` with
  `btime wtime byoyomi binc winc`, `go infinite`, `go nodes`, `go movetime` /
  `stop` / `gameover` / `quit`); time strategy with soft/hard budgets and
  best-move-instability extension (byoyomi / Fischer / sudden death, network
  delay margin); streaming `info` during search; draw-value strategy
  (`DrawValueBlack` / `DrawValueWhite`); nyugyoku declaration win
  (`bestmove win`, 27-point rule); resign threshold (`ResignValue` /
  `ResignConsecutive`, off by default); `MaxMovesToDraw` (declaration check,
  budget narrowing near the limit, and an **in-search draw terminal**:
  positions past the limit are treated as draws inside the search, so a mate
  beyond the move limit correctly counts as a draw); root-dfpn + leaf-mate
  search; **pondering**
  (`USI_Ponder`, `go ponder` / `ponderhit`, and `bestmove <move> ponder
  <reply>` with the predicted reply = PV's 2nd move) — a ponder hit *continues*
  the same unbounded search under a fresh time budget, so the tree built while
  pondering carries over (the main ponder benefit); **subtree reuse across
  moves** — when the game advances along an explored line, the retained search
  tree is rerooted to the new position so its subtree warm-starts the next
  search instead of rebuilding from scratch (a ponder *miss*, or any advance
  the tree did not explore, falls back to a fresh search); and
  **`OpeningScript`** — a forced opening move sequence (e.g. the HWT
  king-shuffle time handicap): while the game path matches the script prefix
  the engine plays the next scripted move instantly without searching, and
  once the path diverges the script is disabled for the rest of the game
  (an illegal scripted move falls back to normal search); and **`go mate`**
  — dfpn mate search for the GUI's mate-search/analysis button, answering
  `checkmate <move sequence>` (shortest mate), `checkmate nomate` (only
  when no-mate is actually *proven*), or `checkmate timeout`. The last one covers
  **two different situations** the USI spec cannot distinguish — the budget or
  `stop` was reached without a conclusion, *or* a mate was proven but its move
  sequence could not be reconstructed. The second case is preceded by
  `info string checkmate timeout reason=mate-proven-but-pv-unavailable`, so a
  genuine solver regression can be told apart from a plain budget overrun. It runs on dfpn alone, so it works
  even without a model, and no `bestmove` is emitted (per the USI spec).
- For in-process self-play with the same agent, see
  [`maou selfplay`](selfplay.md).

## Engine registration in a GUI

- The `maou-usi` console script starts the engine **without arguments**
  (settings via `setoption`), for GUIs that cannot pass command-line
  arguments (e.g. Shogidokoro):
  - Linux / macOS: register `<venv>/bin/maou-usi`.
  - Windows: register `<venv>\Scripts\maou-usi.exe` (in Shogidokoro's file
    dialog, switch the filter to "all files" if needed).
- **Windows prerequisite**: the wheel links ONNX Runtime (C++) statically, so
  it needs the Microsoft Visual C++ Redistributable 2015-2022 (x64) —
  `MSVCP140.dll` ships with neither Python nor Windows. Without it the engine
  dies at startup with `ImportError: DLL load failed while importing _rust`,
  which does *not* name the missing DLL. Install it with
  `winget install Microsoft.VCRedist.2015+.x64`. CI cannot catch this: GitHub's
  Windows runners have it preinstalled. Prebuilt Windows wheels are not
  distributed — build on demand via the `Build Windows Wheel` workflow
  (manual dispatch).
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
| `--batch-size INT` | default `8` | Evaluation batch size; also the TensorRT padding size. **Use around 64 on GPU** — the measured optimum (L4 + TensorRT + ViT 19.8M fp16); batch 256 lost an A/B by 137 Elo. See [eval-batching.md](../design/position-search/eval-batching.md). |
| `--node-capacity INT` | | Node pool capacity (default 2^20 nodes). |
| `--network-delay-ms INT` | default `1000` | Communication overhead margin in milliseconds. The GUI/server measures elapsed time including transport, so the per-move budget is reduced by this amount. |
| `--min-think-ms INT` | default `100` | Minimum thinking time in milliseconds. |
| `--time-curve/--no-time-curve` | **default on** | Weight the per-move budget towards the **conversion phase** — the plies where a won position has to be turned into a win. The multiplier scales **only the discretionary share** (`remaining / horizon`), never byoyomi or the Fischer increment, so the guaranteed floor survives. Enabled after a self-play A/B scored 65% over 20 games (+108 Elo); **the interval still includes zero (t = +1.15)**, so treat it as provisional — see [verification.md §4.4.1](../design/usi-engine/verification.md). Also `setoption name TimeCurve`. |
| `--time-curve-peak-ply INT` | default `100` | Ply where the curve peaks. |
| `--time-curve-half-width-ply INT` | default `55` | Plies from the peak down to the floor weight. |
| `--time-curve-peak-permille INT` | default `2500` | Multiplier at the peak, per mille (`1000` = 1.0x). |
| `--time-curve-opening-floor-permille INT` | default `300` | Multiplier before the peak, per mille. Cutting the opening is nearly free: a neutral re-analysis put the mean winrate loss over plies 9-30 at 0.0004. |
| `--time-curve-endgame-floor-permille INT` | default `1200` | Multiplier after the peak, per mille. **Above 1000 on purpose** — the first attempt left the endgame at the flat 1000 and still lost the phase, because the same multiplier applied to a bank the midgame had already drained. |
| `--keep-alive-ms INT` | default `5000` | While answering `isready`, send a blank line every N milliseconds as a liveness signal, so the GUI does not time out when the first TensorRT engine build outlasts its `readyok` timeout. `0` disables it. A fast `isready` emits no blank line at all — silence is normal. Verified on **ShogiHome**, which ignores the blank lines harmlessly; other GUIs are unverified. |
| `--draw-value-black INT` | default `500` | Draw value for Black in permille. Repetition / max-moves draw terminals are valued at this (root side-to-move view). Denryu-sen Black 0.4 win = `400`. |
| `--draw-value-white INT` | default `500` | Draw value for White in permille (Denryu-sen White 0.6 win = `600`). |
| `--resign-value INT` | default `0` | Resign when the root win rate stays below this permille for `--resign-consecutive` moves. `0` = never resign. |
| `--resign-consecutive INT` | default `3` | Consecutive below-threshold moves required to resign (with `--resign-value > 0`). |
| `--max-moves-to-draw INT` | default `0` | Move count for a drawn game (`0` = disabled; Denryu-sen `512`). At/near the limit the engine always checks nyugyoku declaration and narrows its search budget; positions past the limit are treated as draws inside the search. |
| `--usi-ponder/--no-usi-ponder` | **default on** | Enable pondering (thinking on the opponent's turn). When on, the engine declares `USI_Ponder` and appends the predicted reply to `bestmove` so the GUI sends `go ponder`. |
| `--opening-script "MOVES"` | | Forced opening move sequence in USI notation, space-separated (e.g. `"5i5h 5a5b 5h5i 5b5a"` for the HWT king-shuffle handicap). While the game path matches this prefix the engine plays the next scripted move instantly without searching (no clock time spent, no `ponder` attached). Requires the full game path from move 1: if the engine is handed a position whose move number is already past the script (e.g. a designated position set up *after* the shuffle), the script stays disabled instead of replaying the sequence. |
| `--root-dfpn/--no-root-dfpn` | **default on** | Run dfpn mate search on the root position in parallel with MCTS. |
| `--root-dfpn-nodes INT` | default `2000000` | Node budget for the root dfpn mate search. |
| `--root-dfpn-depth INT` | default `2047` | Search depth limit for the root dfpn mate search (max 2047). |
| `--leaf-mate/--no-leaf-mate` | **default on** | Enable short mate search at MCTS leaves (async, dedicated threads). |
| `--leaf-mate-nodes INT` | default `50` | Node budget per leaf-mate df-pn call. |
| `--leaf-mate-threads INT` | default `1` | Number of dedicated leaf-mate threads. |
| `--defensive-mate/--no-defensive-mate` | **default on** | Prove whether the side to move is *being* mated (root + leaves that are in check), and drop root moves that let the opponent force mate. Without it the engine only asks whether it can mate, so a forced mate against it stays invisible until the last ply. |
| `--defensive-mate-threads INT` | default `1` | Parallelism of the defensive filter that screens root moves. Each root move is judged independently, so raising this uses spare CPU to reach a larger budget within the same wall clock (memory scales with it, ~88MB/thread). |
| `--cuda/--no-cuda` | default off | Enable CUDA Execution Provider (requires a wheel built with `onnx-cuda`). |
| `--tensorrt/--no-tensorrt` | default off | Enable TensorRT Execution Provider (requires a wheel built with `onnx-tensorrt`). |
| `--trt-cache-dir PATH` | | TensorRT engine cache directory. Created automatically if missing; startup fails with a clear error when the parent path is unavailable (e.g. Google Drive not mounted on Colab). |
| `--tensorrt/--no-tensorrt` (終了時の挙動) | | With TensorRT enabled the command flushes its output and then exits **without running destructors**: the TensorRT EP teardown corrupts the glibc heap and aborts (deterministic; see [verification.md §8.5](../design/usi-engine/verification.md)). Results are already written when this happens, and the exit code stays 0. Runs without TensorRT use the normal exit path. |

## USI options (`setoption`)

Declared in the `usi` response; defaults reflect the CLI flags above.

| Option | Type | Description |
| --- | --- | --- |
| `ModelPath` | filename | ONNX model path (empty = mock evaluator). |
| `Threads` / `BatchSize` / `NodeCapacity` | spin | Search resources. |
| `USI_Hash` | spin (MB) | Used to derive `NodeCapacity` when the latter is not set. The conversion uses the measured node footprint (node struct + edge array for the measured average branching factor ≈ 808 bytes/node), so the resulting pool stays within the megabytes you asked for. `0` = ignore. |
| `UseCuda` / `UseTensorRT` | check | Execution providers (feature-gated wheel required). |
| `TrtCacheDir` | string | TensorRT engine cache directory. |
| `NetworkDelay` | spin (ms) | Communication margin subtracted from each move budget. |
| `MinimumThinkingTime` | spin (ms) | Minimum thinking time. |
| `TimeCurve` | check | Conversion-phase-weighted time curve (**default on**). The curve **parameters** are deliberately not exposed here: `setoption` arrival order is not guaranteed, so they are fixed by the CLI flags above and this option only toggles them on. |
| `KeepAlive` | spin (ms) | Blank-line keep-alive interval while answering `isready` (default 5000; 0 = disabled). |
| `DrawValueBlack` / `DrawValueWhite` | spin (permille) | Draw value per side (default 500; Denryu-sen 400 / 600). Converted to the search's side-to-move `draw_value`. |
| `ResignValue` | spin (permille) | Resign win-rate threshold (0 = never). |
| `ResignConsecutive` | spin | Consecutive below-threshold moves required to resign. |
| `MaxMovesToDraw` | spin | Move count for a drawn game (0 = disabled; Denryu-sen 512). Also enables the in-search draw terminal past the limit. |
| `OpeningScript` | string | Forced opening move sequence in USI notation (empty = disabled). |
| `USI_Ponder` | check | Enable pondering (default on). Declared so the GUI sends `go ponder`; `bestmove` carries the predicted reply (PV's 2nd move). |
| `RootDfpn` / `LeafMate` | check | Mate search toggles. |
| `RootDfpnNodes` | spin | Node budget for the root dfpn mate search (default `2000000`). For `go` it is the search cutoff; for `go mate` it only **sizes the transposition table** — that search stops on time/`stop` as the USI spec requires. The table holds `clamp(nodes * 2, 2^18, 2^23)` entries and is written in full on allocation (≈7 ms/MB; the default is ≈352 MB), so this is a fixed cost paid per search, not just at startup. Raise it only to chase long mates; an undersized table is collected by GC rather than losing the mate, so it costs time, not correctness. |

## Example

```bash
# Manual smoke test (mock evaluator; type or pipe USI commands)
printf 'usi\nisready\nposition startpos\ngo btime 0 wtime 0 byoyomi 1000\nquit\n' \
  | maou-usi

# Start with a model and GPU (initial values; setoption can override)
maou usi --model-path model.onnx --threads 2 --batch-size 64 \
  --tensorrt --trt-cache-dir /path/to/trt-cache
```
