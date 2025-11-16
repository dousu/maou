# `maou evaluate`

Use the `evaluate` subcommand when you want to score an arbitrary SFEN
position without launching the full training or benchmarking pipelines. The
flags below come directly from
`src/maou/infra/console/evaluate_board.py` and are forwarded to the interface
layer that ultimately drives `InferenceRunner`.【F:src/maou/infra/console/evaluate_board.py†L1-L51】【F:src/maou/interface/infer.py†L1-L38】

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--model-type {ONNX,TENSORRT}` | default `ONNX` | Passed verbatim to `ModelType[model_type]`. Any value outside the enum
raises a `ValueError` before inference starts, so stick to the spellings used in
`ModelType`.【F:src/maou/interface/infer.py†L10-L24】【F:src/maou/app/inference/run.py†L17-L38】 |
| `--model-path PATH` | ✅ | Absolute or relative path to the exported network. `click.Path` checks that
it already exists, so you get early feedback when pointing at the wrong file
before any GPU context is created.【F:src/maou/infra/console/evaluate_board.py†L16-L24】 |
| `--cuda/--no-cuda` | default `--no-cuda` | Turns GPU execution on or off for both ONNX Runtime and TensorRT. When
`--cuda` is omitted the inference stack runs entirely on CPU. TensorRT requires
CUDA to be enabled (see below).【F:src/maou/infra/console/evaluate_board.py†L25-L33】【F:src/maou/app/inference/onnx_inference.py†L12-L33】【F:src/maou/app/inference/tensorrt_inference.py†L64-L75】 |
| `--num-moves INT` | default `5` | Controls how many candidate moves are requested from the backend. The value
is embedded inside `InferenceRunner.InferenceOption.num_moves` and limits the
policy array that is turned into USI strings.【F:src/maou/infra/console/evaluate_board.py†L34-L42】【F:src/maou/app/inference/run.py†L40-L103】 |
| `--sfen STRING` | ✅ | Full SFEN describing the position (piece placement, side to move, hands, and
move count). The CLI enforces its presence and the board helper hands the string
to `cshogi.Board.set_sfen`, so any string accepted by cshogi works here.【F:src/maou/infra/console/evaluate_board.py†L43-L51】【F:src/maou/domain/board/shogi.py†L1-L77】 |

## Interface and runner hand-off

Once the CLI validates the options it calls `maou.interface.infer.infer`. The
interface converts the `--model-type` string into the `ModelType` enum, wraps
the rest of the flags into `InferenceRunner.InferenceOption`, and invokes
`InferenceRunner.infer`. The helper raises a descriptive error when the model
type is unknown so you can fix typos without entering the app layer.【F:src/maou/interface/infer.py†L1-L38】

`InferenceRunner` is responsible for:

1. Constructing a `Board` either from the provided SFEN or from an already
   materialized domain board and generating the float32 feature planes via
   `make_feature`.【F:src/maou/app/inference/run.py†L40-L79】
2. Dispatching to either `ONNXInference` or (optionally) `TensorRTInference`
   based on the enum. Each backend receives the normalized input tensor, the
   move limit, and the CUDA flag.【F:src/maou/app/inference/run.py†L80-L139】
3. Translating the returned policy labels into human-readable USI moves and
   turning the scalar value output into both an evaluation score and a win rate
   with the `Evaluation` helper.【F:src/maou/app/inference/run.py†L140-L180】

## Supported model types and CUDA requirements

- **ONNX** – Uses ONNX Runtime with extended graph optimizations. When
  `--cuda` is supplied the execution provider list becomes
  `["CUDAExecutionProvider", "CPUExecutionProvider"]`; otherwise it falls back
  to CPU only. Make sure the exported network names its inputs `input` and
  outputs `policy`/`value` to match the fixed fetch list in
  `ONNXInference`.【F:src/maou/app/inference/onnx_inference.py†L1-L37】
- **TensorRT** – Available when you install the optional extras
  (`poetry install -E tensorrt-infer`). The interface dynamically imports the
  TensorRT backend and surfaces a `RuntimeError` that names the missing module if
  the dependency is absent. TensorRT mode requires `--cuda`, builds an engine for
  the resident GPU, and then launches inference entirely through CUDA streams.
  Expect the command to fail fast with `ValueError("TensorRT requires CUDA.")`
  if you forget to enable the flag.【F:src/maou/app/inference/run.py†L80-L139】【F:src/maou/app/inference/tensorrt_inference.py†L1-L71】

## SFEN input expectations

`InferenceRunner` calls `Board.set_sfen`, which proxies directly to
`cshogi.Board.set_sfen`. That means your `--sfen` string must include the full
piece placement for all 9 ranks, the side to move (`b`/`w`), hands for both
players (or `-` when empty), and the move counter. Invalid strings bubble up as
exceptions before inference begins, so verify them with `cshogi` tools when in
 doubt.【F:src/maou/app/inference/run.py†L45-L79】【F:src/maou/domain/board/shogi.py†L1-L77】

## Output structure

`maou interface infer` returns a formatted string with four lines:

```
Policy: 7g7f, 2b3c, ...
Eval: +123.45
WinRate: 0.6789
+-----------------+
| ...ASCII board...|
```

- The `Policy` line lists the top `--num-moves` USI moves derived from the
  model's policy tensor. Illegal labels are logged and replaced with the string
  `"failed to convert"` so diagnostics remain visible.【F:src/maou/app/inference/run.py†L140-L169】
- `Eval` and `WinRate` come from the `Evaluation` helper, which converts the
  scalar value output into both score systems depending on the active side to
  move.【F:src/maou/app/inference/run.py†L140-L160】
- `Board` prints `Board.to_pretty_board()`, yielding the same ASCII diagram that
  `cshogi` uses. All four fields are concatenated and echoed by the CLI so the
  output is immediately usable in scripts or terminal sessions.【F:src/maou/interface/infer.py†L25-L38】【F:src/maou/app/inference/run.py†L170-L180】

## Example

```bash
poetry run maou evaluate \
  --model-type ONNX \
  --model-path artifacts/eval.onnx \
  --cuda \
  --num-moves 7 \
  --sfen "lnsgkgsnl/1r5b1/p1pppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 1"
```

The command prints the policy ranking, numerical evaluations, and an ASCII board
view that you can paste directly into bug reports or analysis logs.
