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
  per-move times / playouts / scores, elapsed time) can be written as
  **JSON Lines** with `--output`. A results summary is printed to stdout;
  per-game progress lines go to stderr.
- `--kifu-dir` additionally writes **one CSA game record per game**, which
  `maou analyze-game`, `maou analyze-gui` and `maou hcpe-convert` read
  directly — so an A/B match can be taken apart move by move to see *where*
  the difference came from, not just who won.
- Parallelism is "games at a time" (`--parallel`); each game additionally
  uses `--threads` search threads per move. Node pools are pre-allocated
  per agent (two per game), so keep `--node-capacity` modest when running
  many games concurrently.
- Game diversification: `--opening-random-plies N` makes the driver play
  the first N plies uniformly at random among legal moves (deterministic
  per `--seed`, mixed with the game index). `--opening-script` applies a
  forced opening to both agents instead.
- **A/B matches** (`--ab-mode`): player A plays the lever on, player B off,
  everything else identical. Colors swap every game and the pair
  (2n, 2n+1) shares one opening sequence, so the paired comparison cancels
  opening variance. Used to settle the design's open questions
  ([design §12](../design/usi-engine/index.md); GPU procedure:
  [verification.md](../design/usi-engine/verification.md)).
- **Real-clock mode** (`--clock-ms`): instead of a fixed per-move budget,
  a real clock runs and the time strategy decides each move's budget. Time
  spent is measured on the wall clock, so it requires `--parallel 1` and is
  mutually exclusive with `--playouts` / `--movetime-ms`.
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
| `--batch-size INT` | default `8` | Evaluation batch size. **Use around 64 on GPU** — the measured optimum (L4 + TensorRT + ViT 19.8M fp16); batch 256 lost an A/B by 137 Elo. See [eval-batching.md](../design/position-search/eval-batching.md). |
| `--node-capacity INT` | default `65536` | Node pool capacity per agent (two agents per game; pools are pre-allocated). |
| `--draw-value-black INT` | default `500` | Draw value for Black in permille (Denryu-sen `400`). |
| `--draw-value-white INT` | default `500` | Draw value for White in permille (Denryu-sen `600`). |
| `--resign-value INT` | default `0` | Resign when the root win rate stays below this permille for `--resign-consecutive` moves (`0` = never). |
| `--resign-consecutive INT` | default `3` | Consecutive below-threshold moves required to resign. |
| `--opening-script "MOVES"` | | Forced opening move sequence in USI notation, applied to both agents (only when `--sfen` is a move-1 position — see [`usi.md`](usi.md)). |
| `--clock-ms INT` | default `0` | Real-clock mode: initial time per side in milliseconds (`0` = off). Mutually exclusive with `--playouts` / `--movetime-ms`, and requires `--parallel 1`. |
| `--byoyomi-ms INT` | default `0` | Byoyomi in milliseconds (real-clock mode only). |
| `--inc-ms INT` | default `0` | Fischer increment per move in milliseconds (real-clock mode only). |
| `--min-think-ms INT` | | Minimum thinking time per move (engine default `100`); only meaningful in real-clock mode. |
| `--ab-mode MODE` | | A/B match instead of plain self-play: `subtree` (subtree reuse), `maxmoves` (in-search draw terminal), `budget` (same config, smaller budget for B — harness sanity check), `horizon` (time-strategy horizon; requires `--clock-ms`), `timecurve` (the time curve on — which is now the default — vs the flat allocation; requires `--clock-ms`, and **run it on GPU** — see [verification.md §4.4.1](../design/usi-engine/verification.md)), `spin` (terminal spin excluded from the budget for A; fixed `--playouts` only, rejected with `--clock-ms`), `proven` (proven children excluded from PUCT selection for A), `batch` (evaluation batch size: A = `--batch-size` / B = `--batch-size-b`; **run it under the clock** — a fixed playout budget gives both sides the same playout count, so a speed difference cannot become a strength difference), `defmate` (defensive mate search on for A / off for B: A proves whether it is *being* mated at the root and at leaves that are in check, so a forced mate against it shows up in the evaluation instead of only in the last ply). |
| `--spin-relief/--no-spin-relief` | default off | Exclude terminal spin from the playout budget, so `--playouts` counts real search volume only. A consecutive-spin limit still stops a search whose frontier is all terminals (`stop=spin_exhausted`). Fixed-budget runs only; measured to raise real playouts by only ~1-2% (see [verification.md §4.5](../design/usi-engine/verification.md)). |
| `--skip-proven/--no-skip-proven` | default off | Exclude proven children (mate / repetition / resolved subtrees) from PUCT selection, so descents open new leaves instead of backpropagating a known value (MCTS-Solver). Works under both fixed budgets and the real clock. Measured to raise real playouts per move by ~20% where proven terminals dominate; no effect where the spin comes from the depth/max-moves limit (see [verification.md §4.6](../design/usi-engine/verification.md)). |
| `--playouts-b INT` | | Player B playout budget (`--ab-mode budget`; defaults to one eighth of `--playouts`). |
| `--batch-size-b INT` | `--batch-size` × 4 | Evaluation batch size for player B (`--ab-mode batch`). |
| `--horizon INT` | | Assumed remaining moves for player A (`--ab-mode horizon`; defaults to the engine value). |
| `--horizon-b INT` | | Assumed remaining moves for player B (`--ab-mode horizon`; default `25`). |
| `--time-curve-peak-ply INT` | default `100` | Ply where the time curve peaks (`--ab-mode timecurve`). |
| `--time-curve-half-width-ply INT` | default `55` | Plies from the peak down to the floor weight. |
| `--time-curve-peak-permille INT` | default `2500` | Multiplier at the peak, per mille. Scales **only the discretionary share** (`remaining / horizon`), never byoyomi or the Fischer increment. |
| `--time-curve-opening-floor-permille INT` | default `300` | Multiplier before the peak, per mille. |
| `--time-curve-endgame-floor-permille INT` | default `1200` | Multiplier after the peak, per mille (above the flat `1000` on purpose). |
| | | All five go to **both** players; `--ab-mode timecurve` splits only the on/off flag, so the comparison isolates the curve rather than its tuning. To try different parameters, re-run the whole A/B with the new values. |
| `--alternate-colors/--no-alternate-colors` | default: on with `--ab-mode` | Swap colors every game and pair the openings (2n, 2n+1). |
| `--root-dfpn/--no-root-dfpn` | **default on** | Root-parallel dfpn mate search. |
| `--root-dfpn-nodes INT` | default `2000000` | Node budget for the root dfpn mate search. |
| `--root-dfpn-depth INT` | default `2047` | Depth limit for the root dfpn mate search. |
| `--leaf-mate/--no-leaf-mate` | **default on** | Short mate search at MCTS leaves. |
| `--leaf-mate-nodes INT` | default `50` | Node budget per leaf-mate df-pn call. |
| `--leaf-mate-threads INT` | default `1` | Dedicated leaf-mate threads. |
| `--defensive-mate/--no-defensive-mate` | **default on** | Prove whether the side to move is *being* mated (root + leaves that are in check), and drop root moves that let the opponent force mate. Without it the engine only asks whether it can mate, so a forced mate against it stays invisible until the last ply. |
| `--defensive-mate-threads INT` | default `1` | Parallelism of the defensive filter that screens root moves. Each root move is judged independently, so raising this uses spare CPU to reach a larger budget within the same wall clock (memory scales with it, ~88MB/thread). |
| `--cuda/--no-cuda` | default off | CUDA Execution Provider (requires a wheel built with `onnx-cuda`). |
| `--tensorrt/--no-tensorrt` | default off | TensorRT Execution Provider (requires a wheel built with `onnx-tensorrt`). |
| `--trt-cache-dir PATH` | | TensorRT engine cache directory. |
| `--pad-buckets/--no-pad-buckets` | default off | Round the TensorRT padding up to a power-of-two bucket instead of always padding to `--batch-size`. Fixed padding makes a 1-item root evaluation cost a full batch; bucketing removes that waste but adds one TensorRT engine build per shape and can change numeric results, so it is a measurement toggle until verified on GPU (see [eval-batching.md](../design/position-search/eval-batching.md)). |
| `--tensorrt/--no-tensorrt` (終了時の挙動) | | With TensorRT enabled the command flushes its output and then exits **without running destructors**: the TensorRT EP teardown corrupts the glibc heap and aborts (deterministic; see [verification.md §8.5](../design/usi-engine/verification.md)). Results are already written when this happens, and the exit code stays 0. Runs without TensorRT use the normal exit path. |
| `--quiet` | default off | Suppress per-game progress lines on stderr. |
| `--kifu-dir DIR` | | Write one CSA game record per game into this directory (created if missing). Feeds `analyze-game` / `analyze-gui` / `hcpe-convert` directly. One game per file. Exact milliseconds and per-move playouts stay in the `--output` JSONL — the CSA time field is whole seconds. |

## Output

- `--output` writes one JSON object per game:
  `{"game_index": 0, "sfen": "...", "moves": ["7g7f", ...], "winner":
  "black"|"white"|null, "reason": "checkmate"|"resign"|"declaration"|
  "repetition"|"perpetual_check"|"max_moves"|"illegal_move",
  "black_player": "a"|"b", "plies": N, "playouts": N,
  "terminal_backprops": N, "reused_moves": N,
  "carried_visits": N, "elapsed_ms": N, "remaining_ms": [N, N]|null}`.
  (`playouts` is the **real search volume** — playouts that evaluated a
  leaf. Descents that only hit a terminal (proven / repetition / max-moves
  / depth limit) and backpropagated without opening a new leaf are counted
  separately as `terminal_backprops`; the budget is consumed by the sum of
  the two.)
  (`reused_moves` / `carried_visits` measure how often subtree reuse
  warm-started a search and how many visits it carried over — the
  effective budget the retained tree added.)
  (`black_player` records color alternation in A/B matches; plain
  self-play always reports `"a"`. `remaining_ms` is `[black, white]` time
  left at the end and is only filled in real-clock mode.)
- `move_times_ms`, `move_playouts` and `move_scores` are per-move arrays,
  all the same length as `moves`. Moves the driver played itself
  (`--opening-random-plies`) carry `0`. `move_scores` is the engine's own
  evaluation in centipawns from the side to move (mate is clamped to
  ±30000). **`move_times_ms` is the authoritative time record** — the CSA
  file rounds it to whole seconds. Under `--parallel > 1` the wall clock is
  distorted by CPU contention between concurrent games, so take timing
  analysis from `--parallel 1` runs only.
- `--kifu-dir` writes `game_NNNN.csa` (V2.2) per game: `N+`/`N-` name the
  A/B sides, each move carries `T<seconds>` and the engine evaluation as
  `'** <score>`, and the game ends with `%TORYO` / `%KACHI` /
  `%SENNICHITE` / `%ILLEGAL_MOVE` / `%CHUDAN` / `%TIME_UP` chosen so that
  **re-reading the file reproduces the same winner**. One game per file,
  because `analyze-game` rejects multi-game CSA. When `--kifu-dir` is
  given, the record body is not duplicated into the `--output` JSONL.
- stdout prints a summary: game count, black/white/draw results, a reason
  histogram, totals for plies / playouts / summed game time, a
  `terminal spin:` line, wall-clock `throughput:` in playouts per second
  (the denominator is the whole run, so `--parallel` sweeps are
  comparable), and how much subtree reuse contributed.
- `terminal spin:` reports the backprops that never reached a leaf
  evaluation and their share of the consumed budget. Endgames dominated by
  proven terminals, repetition, or the max-moves horizon spend most of the
  budget here, so the real search volume falls far below the nominal one
  (measured: 98.1% spin with the draw horizon two plies away). The
  `throughput:` numerator counts **real playouts only** — a figure above
  the evaluator's physical ceiling means an older build that folded spin
  into `playouts`.
- With `--ab-mode` the summary additionally reports, from player A's point
  of view: `W/D/L`, the score with a Wilson 95% interval, the Elo
  conversion with its interval, the paired statistics (pairs, mean, SE,
  t value, pairs where A came out ahead) and — in real-clock mode — the
  average time left at the end for each side. Read the engagement numbers
  (carried visits, time left) before concluding "no effect": at n = 40 the
  interval only resolves differences of roughly 150 Elo, so a win rate
  alone cannot separate "no effect" from "the lever never fired".

## Example

```bash
# Mock smoke run (development only)
maou selfplay --games 2 --playouts 16 --max-moves 64 \
  --no-root-dfpn --no-leaf-mate --output records.jsonl

# Real-model self-play, 4 games at a time, varied openings
maou selfplay --model-path model.onnx --games 20 --parallel 4 \
  --playouts 800 --opening-random-plies 6 --seed 1 \
  --output selfplay.jsonl

# A/B: subtree reuse on (A) vs off (B), paired openings with color swap
maou selfplay --model-path model.onnx --games 40 --ab-mode subtree \
  --playouts 800 --opening-random-plies 8 --seed 1 --max-moves 256

# A/B on the time strategy: real clock, horizon 40 (A) vs 20 (B)
maou selfplay --model-path model.onnx --games 40 --ab-mode horizon \
  --clock-ms 32000 --inc-ms 500 --horizon 40 --horizon-b 20 \
  --resign-value 0 --opening-random-plies 8 --seed 1 --max-moves 256

# A/B that can be taken apart afterwards: keep the records and the kifu,
# then analyse the game where A lost to see *where* it went wrong
maou selfplay --model-path model.onnx --games 40 --ab-mode timecurve \
  --clock-ms 300000 --inc-ms 10000 --opening-random-plies 8 --seed 1 \
  --output ab.jsonl --kifu-dir kifu/
maou analyze-game --input-path kifu/game_0003.csa \
  --model-path model.onnx --playouts 800 --output report.json
```

The same harness is available without building the Python extension as
`cargo run --release -p maou_usi --example selfplay_ab` (a thin wrapper
around the same `maou_usi::ab` implementation, so the numbers agree).
GPU procedure for the open questions:
[verification.md](../design/usi-engine/verification.md).
