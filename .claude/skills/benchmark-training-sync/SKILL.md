---
name: benchmark-training-sync
description: Validate that benchmark-training command stays in sync with learn-model command. Checks for missing CLI options, verifies test exclusion list accuracy, and identifies options that need to be added to benchmark-training when learn-model features are added. Use after adding new options to learn-model, before creating PRs that modify CLI commands, or when reviewing benchmark-training compatibility.
allowed-tools: Read, Grep, Glob, Bash
---

# Benchmark Training Sync Validator

Ensures that `benchmark-training` command stays in sync with `learn-model` command.
When a feature is added to `learn-model`, the corresponding option must also be available
in `benchmark-training` so users can benchmark with the same parameters they use for training.

## Background

- `learn-model` (`src/maou/infra/console/learn_model.py`) is the primary training command
- `benchmark-training` (`src/maou/infra/console/utility.py`) benchmarks training performance
- `test_cli_option_compatibility.py` enforces that learn-model options are available in benchmark-training
- Some options are legitimately excluded (output dirs, checkpointing, TensorBoard, etc.)

## Validation Steps

### 1. Extract Current CLI Options

Read both command files and extract all `@click.option` parameters:

```bash
# Learn-model options
uv run python -c "
import maou.infra.console.learn_model as lm
import maou.infra.console.utility as ut
import click

def get_opts(cmd):
    return sorted({o[2:] for p in cmd.params if isinstance(p, click.Option) for o in p.opts if o.startswith('--')})

learn_opts = get_opts(lm.learn_model)
bench_opts = get_opts(ut.benchmark_training)

learn_only = sorted(set(learn_opts) - set(bench_opts))
bench_only = sorted(set(bench_opts) - set(learn_opts))
common = sorted(set(learn_opts) & set(bench_opts))

print('=== CLI Option Sync Report ===')
print(f'Common options: {len(common)}')
print(f'learn-model only: {len(learn_only)}')
print(f'benchmark-training only: {len(bench_only)}')
print()
if learn_only:
    print('Options in learn-model but NOT in benchmark-training:')
    for o in learn_only:
        print(f'  --{o}')
print()
if bench_only:
    print('Options in benchmark-training but NOT in learn-model:')
    for o in bench_only:
        print(f'  --{o}')
"
```

### 2. Check Test Exclusion List

Read the test exclusion list in `tests/maou/infra/console/test_cli_option_compatibility.py` and verify:

- Every option in the exclusion list has a valid justification
- No option is excluded that should actually be in benchmark-training
- No option is missing from both benchmark-training AND the exclusion list

The exclusion list is the `excluded_options` set in `test_learn_model_options_available_in_benchmark_training()`.

**Legitimate exclusion categories:**
- **Output/logging**: `epoch`, `log-dir`, `model-dir`, `tensorboard-*`, `output-gcs`, `gcs-*`, `output-s3`, `s3-*`
- **Checkpointing/resume**: `resume-from`, `start-epoch`, `resume-backbone-from`, `resume-policy-head-from`, `resume-value-head-from`, `resume-reachable-head-from`, `resume-legal-moves-head-from`
- **Training control (non-benchmark)**: `freeze-backbone`, `trainable-layers`
- **Architecture-specific (non-benchmark)**: `vit-embed-dim`, `vit-num-layers`, `vit-num-heads`, `vit-mlp-ratio`, `vit-dropout`, `gradient-checkpointing`
- **Multi-stage thresholds/epochs**: `stage1-threshold`, `stage2-threshold`, `stage1-max-epochs`, `stage2-max-epochs`
- **Stage-specific batch/lr (benchmark uses shared)**: `stage1-batch-size`, `stage2-batch-size`, `stage1-learning-rate`, `stage2-learning-rate`

### 3. Run the Compatibility Test

```bash
uv run pytest tests/maou/infra/console/test_cli_option_compatibility.py -v
```

### 4. Identify Required Actions

For each option in learn-model that is NOT in benchmark-training and NOT in the exclusion list:

1. **Determine if the option is relevant for benchmarking**:
   - Training parameters (optimizer, loss, model config) -> YES, add to benchmark-training
   - Output/logging parameters -> NO, add to exclusion list with justification
   - Checkpoint/resume parameters -> NO, add to exclusion list with justification

2. **If adding to benchmark-training**:
   - Add the `@click.option` decorator to `benchmark_training()` in `src/maou/infra/console/utility.py`
   - Add the parameter to the function signature
   - Wire it through to `utility_interface.benchmark_training()`
   - Update `src/maou/interface/utility_interface.py` if needed
   - Update `docs/commands/utility_benchmark_training.md`

3. **If adding to exclusion list**:
   - Add to `excluded_options` in `test_learn_model_options_available_in_benchmark_training()`
   - Add a comment explaining why it's excluded

### 5. Verify Interface Layer Consistency

Check that `utility_interface.benchmark_training()` accepts all the parameters
that `benchmark_training()` CLI function passes to it:

```bash
uv run python -c "
import inspect
import maou.interface.utility_interface as ui
sig = inspect.signature(ui.benchmark_training)
params = sorted(sig.parameters.keys())
print('utility_interface.benchmark_training parameters:')
for p in params:
    print(f'  {p}')
"
```

### 6. Constraint: hcpe Data

`benchmark-training` does NOT need to support hcpe data format, same as `learn-model`.
Do not add hcpe-specific options.

## Report Format

```
=== Benchmark Training Sync Report ===

Status: PASS / FAIL

Common options: N
learn-model only (excluded): N
learn-model only (MISSING - action required): N
benchmark-training only: N

[If FAIL]
Missing options that need to be added to benchmark-training:
  --option-name: [description of what it does]

Missing options that need to be added to the exclusion list:
  --option-name: [justification for exclusion]

Test result: PASS / FAIL
```

## When to Use

- After adding new `@click.option` to `learn-model`
- Before creating PRs that modify `learn_model.py` or `utility.py`
- During code review of changes to CLI commands
- As part of PR preparation checks

## Files Involved

| File | Role |
|------|------|
| `src/maou/infra/console/learn_model.py` | Source of truth for training options |
| `src/maou/infra/console/utility.py` | benchmark-training command |
| `src/maou/interface/utility_interface.py` | Interface layer for benchmark |
| `tests/maou/infra/console/test_cli_option_compatibility.py` | Compatibility tests |
| `docs/commands/utility_benchmark_training.md` | CLI documentation |
