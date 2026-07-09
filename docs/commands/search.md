# `maou search`

## Overview

- Searches an arbitrary SFEN position with the Rust MCTS engine
  (`maou_search` crate) and prints the best move, evaluation score, win rate,
  principal variation (PV), candidate moves, and search statistics. The CLI
  layer forwards every option to the interface layer
  (`src/maou/interface/search.py`), which drives the use case
  `src/maou/app/search/run.py`.
- The search itself runs in Rust with the GIL released
  (`maou._rust.maou_search.search`). It includes repetition (sennichite)
  detection with perpetual-check classification, AND-OR win/loss propagation,
  and an optional root-parallel dfpn mate search (`--root-dfpn`).
- The evaluation score uses the same Ponanza-style conversion
  (`eval = 600 × logit`) as `maou evaluate` via
  `maou.app.inference.eval.Evaluation`, applied to the **searched** win rate.
  Scores are comparable in scale with `maou evaluate`, but not identical for
  the same position (search refines the raw model output).

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--sfen STRING` | ✅ | Base position in SFEN notation. |
| `--moves STRING` | | USI moves applied from the SFEN position (space-separated, like the USI `position ... moves ...` command). Intermediate positions are used as the game history for repetition detection. Illegal moves raise an error. |
| `--model-path PATH` | | ONNX model file path. When omitted, a deterministic mock evaluator is used (API verification only — move quality is meaningless). Requires a wheel built with the `onnx` cargo feature (see notes below). |
| `--threads INT` | default `1` | Number of search threads. Threads beyond 2 are not useful when GPU-bound. |
| `--batch-size INT` | default `8` | Evaluation batch size. Use around 256 on GPU. |
| `--playouts INT` | | Maximum number of playouts. |
| `--time-ms INT` | | Time limit in milliseconds. Defaults to 1000 when neither `--playouts` nor `--time-ms` is specified. |
| `--num-moves INT` | default `5` | Number of candidate moves to display. The best move is always listed first. |
| `--root-dfpn/--no-root-dfpn` | default off | Run dfpn mate search on the root position in parallel with MCTS. When a mate is proven the search stops immediately (`stop=root_proven`) and the mating sequence is returned as PV. |
| `--cuda/--no-cuda` | default `--no-cuda` | Enable the CUDA Execution Provider. Requires `--model-path` and a wheel built with `onnx-cuda`. |
| `--tensorrt/--no-tensorrt` | default `--no-tensorrt` | Enable the TensorRT Execution Provider (FP16 + engine cache). Requires `--model-path` and a wheel built with `onnx-tensorrt`. Batches are padded to `--batch-size` to keep the input shape fixed. |
| `--trt-cache-dir PATH` | | TensorRT engine cache directory (default: `trt_cache/` in the current directory). |

## Wheel build requirements (cargo features)

The default wheel is pure Rust and portable: `maou search` works with the
mock evaluator only. Real NN search requires building the extension with the
corresponding cargo feature of `maou_rust`:

```bash
uv run maturin develop --features onnx            # CPU inference
uv run maturin develop --features onnx-cuda       # + CUDA EP
uv run maturin develop --features onnx-tensorrt   # + TensorRT EP
```

Passing `--model-path` to a wheel built without `onnx` raises a
`RuntimeError` with this instruction. GPU (Colab) procedures are documented
in [docs/design/position-search/benchmarking.md](../design/position-search/benchmarking.md).

## Outputs

```
Bestmove: G*5b
Eval: 16578.56
WinRate: 1.0000
PV: G*5b
Candidates:
G*5b (visits=1, winrate=1.0000, eval=16578.56, prior=0.0187)
...
Stats: playouts=38 nps=435 elapsed_ms=87 max_depth=4 repetitions=0 proven_nodes=1 stop=root_proven
<ASCII board>
```

- `Eval` / `WinRate` are from the side to move's perspective, same convention
  as `maou evaluate` (see 評価値の解釈 in
  [docs/commands/evaluate.md](evaluate.md)). When the root result is proven
  (mate or repetition), `WinRate` is the exact value (0 / 0.5 / 1) and `Eval`
  saturates at the clipping bound (≈ ±16578).
- `Stats` fields: `playouts` (completed simulations), `nps`, `elapsed_ms`,
  `max_depth`, `repetitions` (sennichite detections), `proven_nodes` (AND-OR
  proven interior nodes), and `stop` (`playout_limit` / `time_limit` /
  `pool_exhausted` / `root_terminal` / `root_proven`).

## Example invocation

```bash
uv run maou search \
  --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
  --model-path artifacts/eval.onnx \
  --time-ms 3000 \
  --threads 2 \
  --batch-size 256 \
  --root-dfpn
```

## Implementation references

- CLI definition — `src/maou/infra/console/search_board.py`
- Interface adapter — `src/maou/interface/search.py`
- Use case (formatting, eval conversion) — `src/maou/app/search/run.py`
- Rust binding — `rust/maou_rust/src/maou_search.rs`
- Search engine design — [docs/design/position-search/](../design/position-search/index.md)
