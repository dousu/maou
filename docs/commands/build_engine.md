# `maou build-engine`

## Overview

- Builds a TensorRT engine from an ONNX model and saves it to a file. This
  allows the expensive engine build to be performed once and reused across
  multiple inference runs via `maou evaluate --engine-path`.
  【F:src/maou/infra/console/build_engine.py†L1-L48】
  【F:src/maou/interface/build_engine.py†L1-L51】
- The command requires TensorRT dependencies (`uv sync --extra tensorrt-infer`).
  If the dependencies are not installed, a helpful error message is displayed.
  【F:src/maou/infra/console/app.py†L227-L240】

## CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--model-path PATH` | ✅ | Path to the ONNX model file. `click.Path(exists=True)` ensures the file exists before attempting the build.【F:src/maou/infra/console/build_engine.py†L9-L14】 |
| `--output PATH` / `-o PATH` | ✅ | Output path for the serialized TensorRT engine file. Parent directories are created automatically if they do not exist.【F:src/maou/infra/console/build_engine.py†L15-L21】 |
| `--trt-workspace-size INT` | default `256` | TensorRT workspace size in MB. Default is sufficient for this project's models. Increase for larger models or max speed. Decrease if GPU memory is limited.【F:src/maou/infra/console/build_engine.py†L22-L32】 |

## Execution flow

1. **CLI validation** -- `build_engine.py` ensures the model path exists and the
   output path is specified before calling the interface helper.
   【F:src/maou/infra/console/build_engine.py†L1-L48】
2. **Interface hand-off** -- `maou.interface.build_engine.build_engine` attempts
   to import `TensorRTInference` (raising `RuntimeError` if TensorRT is not
   installed), then calls `build_engine_from_onnx` to build the engine.
   Build time is measured with `time.perf_counter`.
   【F:src/maou/interface/build_engine.py†L1-L51】
3. **Engine serialization** -- The built engine is saved to the specified output
   path via `TensorRTInference.save_engine`, which also creates parent
   directories automatically.
   【F:src/maou/app/inference/tensorrt_inference.py†L15-L28】

## Validation and guardrails

- The ONNX model file must exist (`click.Path(exists=True)`), so missing files
  are caught before any GPU context is created.
  【F:src/maou/infra/console/build_engine.py†L9-L14】
- If TensorRT dependencies are not installed, the command is registered as a
  fallback that displays installation instructions.
  【F:src/maou/infra/console/app.py†L227-L240】

## Example invocation

```bash
# Basic engine build
maou build-engine --model-path model.onnx --output model.engine

# Custom workspace size
maou build-engine --model-path model.onnx -o model.engine --trt-workspace-size 512

# Then use the engine for inference (no rebuild needed)
maou evaluate --engine-path model.engine --cuda --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
```

## Implementation references

- CLI definition and flag parsing -- `src/maou/infra/console/build_engine.py`.
  【F:src/maou/infra/console/build_engine.py†L1-L48】
- Interface adapter -- `src/maou/interface/build_engine.py`.
  【F:src/maou/interface/build_engine.py†L1-L51】
- Engine build and serialization -- `src/maou/app/inference/tensorrt_inference.py`.
  【F:src/maou/app/inference/tensorrt_inference.py†L1-L85】
