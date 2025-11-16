# `maou evaluate`

## Overview

- Scores an arbitrary SFEN position without launching the full training or
  benchmarking stacks. The CLI forwards every option from
  `src/maou/infra/console/evaluate_board.py` to the interface layer so the same
  validation logic is reused everywhere.【F:src/maou/infra/console/evaluate_board.py†L1-L51】【F:src/maou/interface/infer.py†L1-L38】
- Supports both ONNX Runtime and (optionally) TensorRT backends through the
  shared `InferenceRunner`, letting you flip between CPU-only inference and CUDA
  acceleration with a single flag.【F:src/maou/app/inference/run.py†L40-L180】【F:src/maou/app/inference/onnx_inference.py†L1-L37】【F:src/maou/app/inference/tensorrt_inference.py†L1-L75】

## CLI options

### Core flags

| Flag | Required | Description |
| --- | --- | --- |
| `--model-type {ONNX,TENSORRT}` | default `ONNX` | Passed verbatim to `ModelType[model_type]`. Any value outside the enum raises a `ValueError` before inference starts, so stick to the spellings used in `ModelType`.【F:src/maou/interface/infer.py†L10-L38】【F:src/maou/app/inference/run.py†L17-L38】 |
| `--model-path PATH` | ✅ | Absolute or relative path to the exported network. `click.Path` checks that it already exists, so you get early feedback when pointing at the wrong file before any GPU context is created.【F:src/maou/infra/console/evaluate_board.py†L16-L24】 |
| `--cuda/--no-cuda` | default `--no-cuda` | Turns GPU execution on or off for both ONNX Runtime and TensorRT. TensorRT requires `--cuda` (see guardrails below).【F:src/maou/infra/console/evaluate_board.py†L25-L33】【F:src/maou/app/inference/run.py†L80-L139】 |
| `--num-moves INT` | default `5` | Controls how many candidate moves are requested from the backend. The value is stored inside `InferenceRunner.InferenceOption.num_moves`.【F:src/maou/infra/console/evaluate_board.py†L34-L42】【F:src/maou/app/inference/run.py†L40-L103】 |
| `--sfen STRING` | ✅ | Full SFEN describing the position (piece placement, side to move, hands, move count). The string is passed to `Board.set_sfen`, so anything accepted by `cshogi` works here.【F:src/maou/infra/console/evaluate_board.py†L43-L51】【F:src/maou/domain/board/shogi.py†L1-L77】 |

## Execution flow

1. **CLI validation** – `evaluate_board.py` ensures the model path exists, the
   SFEN is present, and the CUDA flag is parsed before calling the interface
   helper.【F:src/maou/infra/console/evaluate_board.py†L1-L51】
2. **Interface hand-off** – `maou.interface.infer.infer` converts
   `--model-type` into the `ModelType` enum, wraps the rest of the options into
   `InferenceRunner.InferenceOption`, and constructs a `Board` from the SFEN
   string before invoking `InferenceRunner`.【F:src/maou/interface/infer.py†L1-L38】【F:src/maou/app/inference/run.py†L40-L79】
3. **Backend dispatch** – `InferenceRunner` sends the normalized tensor to
   either `ONNXInference` or `TensorRTInference` depending on the enum and CUDA
   flag, then receives the policy/value outputs. TensorRT is only available when
   the optional extra is installed and CUDA is active.【F:src/maou/app/inference/run.py†L80-L139】【F:src/maou/app/inference/onnx_inference.py†L1-L37】【F:src/maou/app/inference/tensorrt_inference.py†L1-L75】
4. **Result formatting** – The runner translates the top `--num-moves` labels
   into USI strings, computes evaluation/win-rate pairs, and builds the ASCII
   board representation that the CLI prints verbatim.【F:src/maou/app/inference/run.py†L140-L180】

## Validation and guardrails

- Invalid model types are rejected immediately because `ModelType[model_type]`
  raises a `KeyError`, which the interface wraps into a descriptive error
  message.【F:src/maou/interface/infer.py†L10-L38】
- `--model-path` must already exist (`click.Path(exists=True)`), so typos are
  caught before GPU contexts or TensorRT engines are created.【F:src/maou/infra/console/evaluate_board.py†L16-L24】
- TensorRT runs fail with `ValueError("TensorRT requires CUDA.")` when `--cuda`
  is omitted; ONNX runs simply stay on CPU if CUDA is off.【F:src/maou/app/inference/run.py†L80-L139】
- `cshogi.Board.set_sfen` validates the supplied SFEN string and raises when the
  placement or hand descriptors are malformed, preventing undefined board
  states.【F:src/maou/domain/board/shogi.py†L1-L77】

## Outputs and usage

- The interface returns a formatted string with `Policy`, `Eval`, `WinRate`, and
  `Board` sections. Illegal labels are logged and replaced with
  `"failed to convert"` so diagnostics stay visible.【F:src/maou/app/inference/run.py†L140-L180】
- Evaluation scores are paired with win-rate estimates based on the active side
  to move, making the output suitable for bug reports or quick comparisons.
- TensorRT and ONNX outputs share the same format, so scripts can parse the
  CLI's stdout without branching on backend type.【F:src/maou/interface/infer.py†L25-L38】

### Example invocation

```bash
poetry run maou evaluate \
  --model-type ONNX \
  --model-path artifacts/eval.onnx \
  --cuda \
  --num-moves 7 \
  --sfen "lnsgkgsnl/1r5b1/p1pppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 1"
```

## Implementation references

- CLI definition and flag parsing – `src/maou/infra/console/evaluate_board.py`.【F:src/maou/infra/console/evaluate_board.py†L1-L51】
- Interface adapter – `src/maou/interface/infer.py`.【F:src/maou/interface/infer.py†L1-L38】
- Runner and backend implementations – `src/maou/app/inference/run.py`,
  `src/maou/app/inference/onnx_inference.py`,
  `src/maou/app/inference/tensorrt_inference.py`.【F:src/maou/app/inference/run.py†L17-L180】【F:src/maou/app/inference/onnx_inference.py†L1-L37】【F:src/maou/app/inference/tensorrt_inference.py†L1-L75】
- Board utilities – `src/maou/domain/board/shogi.py`.【F:src/maou/domain/board/shogi.py†L1-L77】
