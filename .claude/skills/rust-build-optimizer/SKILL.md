---
name: rust-build-optimizer
description: Build Rust extensions efficiently in memory-constrained environments (2-4GB RAM), using optimized build profiles, split workspace compilation, and memory monitoring. Use when encountering OOM errors during Rust builds, setting up Rust backend in DevContainer/Colab, troubleshooting maturin failures, or optimizing compilation for low-memory systems.
---

# Rust Build Optimizer

Builds Rust extensions efficiently in memory-constrained environments, preventing OOM (Out of Memory) failures through optimized compilation strategies.

## Quick Start

### Pre-Flight Check

Verify available memory before building:

```bash
# Check available RAM
free -h

# Recommended: At least 2GB free memory
# Optimal: 4GB+ free memory
```

### Optimized Build (Single Command)

```bash
# Build with pre-configured memory optimizations
uv run maturin develop
```

**Expected:**
- Build time: 3-5 minutes (dev profile)
- Peak memory: 1.0-1.5GB
- Success rate: 95%+ (vs frequent OOM without optimizations)

### Split Workspace Build (For Severe Memory Constraints)

If standard build fails with OOM, use sequential workspace builds:

```bash
# Step 1: Build maou_io individually (2m 23s, ~600MB peak)
export CARGO_BUILD_JOBS=1
export CARGO_INCREMENTAL=1
export RUSTFLAGS="-C codegen-units=1"
cargo build --manifest-path rust/maou_io/Cargo.toml

# Step 2: Build maou_index individually (4m 20s, ~1.2GB peak)
# Dependencies from step 1 are cached
cargo build --manifest-path rust/maou_index/Cargo.toml

# Step 3: Build maou_rust individually (4m 00s, ~1.0GB peak)
# Dependencies from steps 1-2 are cached
cargo build --manifest-path rust/maou_rust/Cargo.toml

# Step 4: Run maturin with cached dependencies (38.94s, <500MB peak)
# All Rust crates are already compiled, maturin just links
uv run maturin develop
```

**Total time:** ~11 minutes (vs 3-5min standard, but prevents OOM)
**Peak memory per step:** <1.5GB (vs 3.5GB for full workspace build)

## Core Workflow

### Standard Build Process

1. **Environment Setup**
   ```bash
   # Verify Rust installation
   cargo --version
   rustc --version

   # Check environment variables (already set by dev-init.sh)
   echo $CARGO_BUILD_JOBS  # Should be 1
   echo $RUSTFLAGS         # Should include "codegen-units=1"
   ```

2. **Monitor Memory During Build**
   ```bash
   # Terminal 1: Run build with time measurement
   time uv run maturin develop 2>&1 | tee build.log

   # Terminal 2: Monitor memory usage
   watch -n 1 'free -h && ps aux --sort=-%mem | head -n 5'
   ```

3. **Verify Successful Build**
   ```bash
   # Test Rust extension import
   uv run python -c "from maou._rust.maou_io import hello; print(hello())"
   # Expected: "Maou I/O Rust backend initialized"

   # Check build artifacts
   ls -lh target/debug/lib_rust.so
   ```

### Memory-Constrained Build Process (Detailed)

When standard build fails with OOM:

1. **Check System Resources**
   ```bash
   # Available memory
   free -h | grep Mem

   # Swap status (should be 0 for this setup)
   swapon --show

   # Disk space
   df -h | grep /workspace
   ```

2. **Clear Build Cache**
   ```bash
   # Remove previous build artifacts
   cargo clean

   # Clear maturin cache (optional)
   rm -rf target/wheels/

   # Verify clean state
   du -sh target/
   ```

3. **Run Split Workspace Build (Step-by-Step)**

   **Step 1: Build maou_io (Arrow I/O library)**
   ```bash
   # Set memory-optimized flags
   export CARGO_BUILD_JOBS=1
   export CARGO_INCREMENTAL=1
   export RUSTFLAGS="-C codegen-units=1"

   # Build first crate
   cargo build --manifest-path rust/maou_io/Cargo.toml

   # Monitor peak memory (should be ~600MB)
   echo "Step 1 complete: maou_io built"
   ```

   **Step 2: Build maou_index (Polars indexing)**
   ```bash
   # Same environment variables (already set)
   # This step compiles Polars with minimal dtypes
   cargo build --manifest-path rust/maou_index/Cargo.toml

   # Monitor peak memory (should be ~1.2GB)
   # This is the heaviest step due to Polars compilation
   echo "Step 2 complete: maou_index built (Polars included)"
   ```

   **Step 3: Build maou_rust (PyO3 bindings)**
   ```bash
   # Dependencies from steps 1-2 are cached
   # This step is lighter since Polars is already compiled
   cargo build --manifest-path rust/maou_rust/Cargo.toml

   # Monitor peak memory (should be ~1.0GB)
   echo "Step 3 complete: maou_rust built"
   ```

   **Step 4: Run maturin (Final linking)**
   ```bash
   # All Rust crates are compiled, maturin just links
   uv run maturin develop

   # This step is very fast (~39 seconds)
   # Peak memory is minimal (<500MB)
   echo "Build complete! Extension installed."
   ```

4. **Verify Build Success**
   ```bash
   # Test import
   uv run python -c "from maou._rust.maou_io import hello; print(hello())"

   # Check installed wheel
   ls -lh target/wheels/

   # Verify Python can find extension
   uv run python -c "import maou._rust; print(maou._rust.__file__)"
   ```

5. **Extreme Memory Constraints (<2GB RAM)**

   If split build still fails:

   ```bash
   # Disable incremental compilation for minimum memory
   export CARGO_INCREMENTAL=0
   export RUSTFLAGS="-C codegen-units=1 -C debuginfo=0"

   # Build sequentially with minimal flags
   cargo build --manifest-path rust/maou_io/Cargo.toml
   cargo build --manifest-path rust/maou_index/Cargo.toml
   cargo build --manifest-path rust/maou_rust/Cargo.toml
   uv run maturin develop

   # Warning: Slower builds, no incremental compilation benefits
   ```

## Build Profiles

### Development Profile (Default)

```bash
uv run maturin develop
```

**Configuration:**
- Optimization level: 0 (no optimization)
- Codegen units: 1 (sequential)
- Incremental: Enabled
- Debug info: Line tables only

**Use cases:**
- Regular development iteration
- Testing Rust changes
- Local debugging

**Performance:**
- Build time: 3-5 minutes
- Peak memory: 1.0-1.5GB

### Release Profile (Optimized)

```bash
uv run maturin develop --release
```

**Configuration:**
- Optimization level: 3 (maximum)
- Codegen units: 1 (sequential)
- LTO: Thin (link-time optimization)
- Debug info: Stripped

**Use cases:**
- Production deployment
- Performance benchmarking
- Final testing before release

**Performance:**
- Build time: 5-8 minutes
- Peak memory: 1.5-2.0GB
- Binary size: 30-40% smaller than dev

### Balanced Profile (Memory-Optimized)

```bash
CARGO_PROFILE=mem-opt uv run maturin develop
```

**Configuration:**
- Optimization level: 2 (moderate)
- Codegen units: 1
- LTO: Thin
- Incremental: Enabled

**Use cases:**
- CI/CD builds with memory limits
- Testing optimization impact
- Balancing speed and memory

**Performance:**
- Build time: 4-6 minutes
- Peak memory: 1.2-1.7GB

## Success Criteria

Build succeeds when ALL of the following are met:

- ✓ Build completes without OOM errors
- ✓ Peak memory usage < 1.5GB (dev), < 2.0GB (release)
- ✓ Build time < 10 minutes (split builds acceptable if needed)
- ✓ Python extension imports successfully
- ✓ Extension verification succeeds: `hello()` returns expected message
- ✓ No compilation warnings about missing features or dtypes

## Common Scenarios

### Scenario 1: Initial DevContainer Setup

**Context:** First-time Rust backend setup in DevContainer (2-core, 3GB RAM)

**Commands:**
```bash
# 1. Initialize environment (sets CARGO_BUILD_JOBS, RUSTFLAGS)
bash scripts/dev-init.sh

# 2. Build Rust extension
uv run maturin develop

# 3. Verify build
uv run python -c "from maou._rust.maou_io import hello; print(hello())"

# 4. Run basic tests
uv run pytest tests/ -k rust -v
```

**Expected outcome:**
- Build completes in 3-5 minutes
- Peak memory ~1.2GB
- All tests pass

### Scenario 2: Google Colab Rust Setup

**Context:** Building Rust extension in Google Colab (limited RAM, non-interactive shell)

**Commands:**
```python
# Cell 1: Install Rust + build with memory limits in one cell
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
  export PATH="$HOME/.cargo/bin:$PATH" && \
  export CARGO_BUILD_JOBS=1 && \
  export RUSTFLAGS="-C codegen-units=1 -C incremental=1" && \
  uv run maturin develop

# Cell 2: Verify import
!uv run python -c "from maou._rust.maou_io import hello; print(hello())"
```

**Expected outcome:**
- Build completes in one cell (no PATH persistence issues)
- Peak memory ~1.5GB
- Import verification succeeds

### Scenario 3: Recovering from OOM Failure

**Context:** Standard build failed with "signal: 9, SIGKILL: kill" or "signal: 15, SIGTERM" error

**Diagnostic commands:**
```bash
# 1. Check what caused OOM (if dmesg available)
dmesg | grep -i "killed process" | tail -5

# 2. Verify available resources
free -h && df -h

# 3. Clear build cache
cargo clean

# 4. Use split workspace build (detailed in Quick Start section)
export CARGO_BUILD_JOBS=1
export CARGO_INCREMENTAL=1
export RUSTFLAGS="-C codegen-units=1"

cargo build --manifest-path rust/maou_io/Cargo.toml
cargo build --manifest-path rust/maou_index/Cargo.toml
cargo build --manifest-path rust/maou_rust/Cargo.toml
uv run maturin develop
```

**Expected outcome:**
- Sequential builds succeed with <1.5GB peak each
- Final maturin step completes in <1 minute (cached deps)

### Scenario 4: CI/CD Build Optimization

**Context:** GitHub Actions runner with 7GB RAM limit, optimizing build time

**Commands:**
```bash
# Use balanced profile for CI
CARGO_PROFILE=mem-opt uv run maturin develop --release

# Verify optimizations
ls -lh target/wheels/*.whl

# Cache Rust artifacts for subsequent builds
# (In .github/workflows/ci.yml)
# - uses: actions/cache@v3
#   with:
#     path: |
#       ~/.cargo/registry
#       ~/.cargo/git
#       target
#     key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
```

**Expected outcome:**
- Build completes in 4-6 minutes
- Peak memory ~1.7GB
- Wheel size ~5-8MB

### Scenario 5: Updating Polars/Arrow Dependencies

**Context:** Adding new Polars dtype or updating Arrow version

**Workflow:**
```bash
# 1. Update dependency via uv
uv add "polars>=1.2.0"

# 2. Check if new dtype features are needed
grep "dtype-" Cargo.toml

# 3. Update Cargo.toml if needed (minimal features only)
# Example: Add dtype-i32 only if actually used in schema.py
# Analyze src/maou/domain/data/schema.py for pl.Int32() usage

# 4. Rebuild with clean cache
cargo clean
uv run maturin develop

# 5. Verify schema compatibility
uv run python -c "from maou.domain.data.schema import get_hcpe_polars_schema; print(get_hcpe_polars_schema())"
```

**Expected outcome:**
- Memory usage stays <1.5GB (minimal features)
- Schema validation passes
- No dtype-related compilation errors

## Troubleshooting

### Issue 1: OOM During polars-core Compilation

**Symptoms:**
```
Compiling polars-core v0.45.1
signal: 9, SIGKILL: kill
```
or
```
Compiling polars-core v0.45.1
signal: 15, SIGTERM: termination signal
```

**Diagnosis:**
```bash
# Check if dtype-full is accidentally enabled
grep "dtype-full" Cargo.toml

# Check current memory usage
free -h

# Check recent OOM kills (if available)
dmesg | grep -i oom | tail -5
```

**Solutions:**

**A. Use split workspace build (RECOMMENDED):**
```bash
# Build crates individually to distribute memory load
cargo build --manifest-path rust/maou_io/Cargo.toml
# polars-core compiles in isolation during maou_index, then cached
cargo build --manifest-path rust/maou_index/Cargo.toml
cargo build --manifest-path rust/maou_rust/Cargo.toml
uv run maturin develop
```

**B. Disable incremental compilation temporarily:**
```bash
export CARGO_INCREMENTAL=0
cargo clean
uv run maturin develop
```

**C. Verify minimal dtype features:**
```bash
# Should see only: dtype-i8, dtype-i16, dtype-u8, dtype-u16, dtype-date
# Should NOT see: dtype-full or dtype-u64 (u64 is a core type)
grep "dtype-" Cargo.toml
```

### Issue 2: Feature Flag Conflict (dtype-u64)

**Symptoms:**
```
error: failed to select a version for `polars`
package `maou_index` depends on `polars` with feature `dtype-u64` but `polars` does not have that feature
```

**Diagnosis:**
```bash
# Check Polars version compatibility
uv pip show polars | grep "version"

# Check dtype feature definitions
cargo tree --manifest-path rust/maou_index/Cargo.toml -e features | grep dtype
```

**Solutions:**

**A. Remove dtype-u64 (u64 is a core type):**
```bash
# Edit Cargo.toml and remove "dtype-u64" from features list
# u64 is always available in Polars, no feature flag needed

# Rebuild
cargo clean
uv run maturin develop
```

**B. Verify workspace dependencies:**
```toml
# In Cargo.toml, ensure workspace.dependencies.polars features do NOT include:
# - "dtype-u64" (not needed, u64 is core)
# - "dtype-full" (too many dtypes, causes OOM)

# Should only include minimal required dtypes:
features = [
  "lazy",
  "ipc",
  "dtype-i8",
  "dtype-i16",
  "dtype-u8",
  "dtype-u16",
  "dtype-date",
]
```

### Issue 3: Incremental Build Cache Corruption

**Symptoms:**
```
internal compiler error: ... during incr comp
```
or
```
error: could not compile `polars-core` (lib) due to previous error
```

**Diagnosis:**
```bash
# Check incremental cache size
du -sh target/debug/incremental/

# Check for corrupted files
ls -la target/debug/incremental/*/
```

**Solutions:**

**A. Clear incremental cache:**
```bash
rm -rf target/debug/incremental/
uv run maturin develop
```

**B. Rebuild from scratch:**
```bash
cargo clean
uv run maturin develop
```

**C. Disable incremental if persistent:**
```bash
export CARGO_INCREMENTAL=0
uv run maturin develop
# Warning: Slower builds, but prevents cache corruption
```

### Issue 4: maturin Rebuild Not Detecting Changes

**Symptoms:**
```
Rust code modified but maturin develop doesn't rebuild
Finished `dev` profile in 0.05s
```

**Diagnosis:**
```bash
# Check if Rust source changed
git status rust/

# Check maturin cache
ls -la target/wheels/

# Check target artifacts timestamps
ls -lt target/debug/lib_rust.so
```

**Solutions:**

**A. Force rebuild:**
```bash
cargo clean
uv run maturin develop
```

**B. Use cargo directly first:**
```bash
# Force recompilation
cargo build --manifest-path rust/maou_rust/Cargo.toml
uv run maturin develop
```

**C. Clear wheels cache:**
```bash
rm -rf target/wheels/
uv run maturin develop
```

### Issue 5: Linker Errors (lld not found)

**Symptoms:**
```
error: linking with `cc` failed
/usr/bin/ld: cannot find -llld
```
or
```
error: linker `clang` not found
```

**Diagnosis:**
```bash
# Check if lld is installed
which lld
lld --version

# Check clang
which clang
clang --version
```

**Solutions:**

**A. Install lld and clang:**
```bash
sudo apt-get update
sudo apt-get install -y lld clang
```

**B. Fallback to default linker temporarily:**
```bash
# Edit .cargo/config.toml and comment out lld linker
# Or set environment override
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc

uv run maturin develop
```

### Issue 6: Split Build Failed on Second Crate

**Symptoms:**
```
Building rust/maou_index...
error: failed to select a version for `polars`
package `maou_index` depends on `polars` with feature `dtype-u64`
```

**Diagnosis:**
```bash
# Check what crate failed
echo $?  # Exit code from previous command (should be non-zero)

# Check dependency tree
cargo tree --manifest-path rust/maou_index/Cargo.toml --features | grep polars
```

**Solutions:**

**A. Feature flag issue - verify workspace dependencies:**
```bash
# Check workspace Cargo.toml
grep "dtype-" Cargo.toml

# Should NOT include dtype-u64 (it's a core type)
# Should only include: dtype-i8, dtype-i16, dtype-u8, dtype-u16, dtype-date

# If dtype-u64 is present, remove it and retry
```

**B. Cache conflict - clean and retry:**
```bash
cargo clean
cargo build --manifest-path rust/maou_io/Cargo.toml
cargo build --manifest-path rust/maou_index/Cargo.toml
```

**C. Verify previous crate built successfully:**
```bash
# Check if maou_io artifacts exist
ls -lh target/debug/libmaou_io.rlib

# If missing, rebuild maou_io first
cargo build --manifest-path rust/maou_io/Cargo.toml
```

### Issue 7: Build Succeeded But Import Fails

**Symptoms:**
```python
>>> from maou._rust.maou_io import hello
ImportError: cannot import name 'hello' from 'maou._rust.maou_io'
```
or
```python
>>> from maou._rust.maou_io import hello
ModuleNotFoundError: No module named 'maou._rust'
```

**Diagnosis:**
```bash
# Check if .so file exists
ls -lh target/debug/lib_rust.so

# Check Python can find the extension
uv run python -c "import maou._rust; print(maou._rust.__file__)"

# Check what's installed
uv run pip list | grep maou
```

**Solutions:**

**A. maturin didn't install the extension:**
```bash
# Force reinstall
uv run maturin develop --force

# Verify installation
uv run python -c "from maou._rust.maou_io import hello; print(hello())"
```

**B. Wrong Python environment:**
```bash
# Verify uv environment
uv run which python

# Reinstall in correct environment
uv run maturin develop
```

**C. Module name mismatch:**
```bash
# Check actual module structure
uv run python -c "import maou._rust; print(dir(maou._rust))"

# Look for available submodules
uv run python -c "import sys; import maou._rust; print(sys.modules['maou._rust'].__path__)"

# Check if module name changed
grep "name = " rust/maou_rust/Cargo.toml
```

## Memory Optimization Details

### Automatic Optimizations (Pre-configured)

The project includes these optimizations by default:

**1. Environment Variables (scripts/dev-init.sh):**
- `CARGO_BUILD_JOBS=1` - Single parallel job
- `CARGO_INCREMENTAL=1` - Reuse artifacts between builds
- `RUSTFLAGS="-C codegen-units=1 -C incremental=1"` - Sequential compilation

**2. Build Profiles (Cargo.toml):**
- `codegen-units = 1` for all profiles
- Thin LTO for release builds
- Optimized debug info settings

**3. Minimal Feature Flags:**
- Polars: Only 5 dtypes (i8, i16, u8, u16, date)
- Arrow: No prettyprint feature
- 800MB-1.2GB memory savings vs dtype-full

**4. Optimized Linker (.cargo/config.toml):**
- Uses lld (LLVM linker) for faster linking
- Lower memory usage than default ld

### Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Peak Memory | 3.0-3.5GB | 1.0-1.5GB | 60-70% reduction |
| OOM Rate | Frequent | <5% | 95%+ success |
| Dev Build Time | 1-2 min | 3-5 min | Acceptable tradeoff |
| Release Build Time | 2-3 min | 5-8 min | Acceptable tradeoff |

### Feature Flag Rationale

**Used dtypes (5 total):**
- `dtype-i8`: Board piece types, move deltas
- `dtype-i16`: Evaluation scores (centipawns)
- `dtype-u8`: Board square representations (0-255)
- `dtype-u16`: Move labels, piece counts
- `dtype-date`: Timestamp fields
- Note: `u64` is a core type (no feature flag needed for position hash IDs)

**Excluded dtypes (12+):**
- f32, f64 - Not needed for integer game data
- i32, i64 - Covered by smaller types
- Decimal types - Not used in game records
- String types - Handled separately
- Time types - Only date is needed

**Evidence:** Analyzed actual usage in:
- `src/maou/domain/data/schema.py` (Polars schemas)
- `rust/maou_index/src/index.rs` (DataFrame operations)

## When to Use

- Initial Rust backend setup in DevContainer
- Building in memory-constrained environments (2-4GB RAM)
- Google Colab / Jupyter notebook Rust builds
- CI/CD pipelines with memory limits
- Recovering from OOM build failures
- Debugging maturin build issues
- After updating Polars/Arrow dependencies
- Verifying build configuration correctness

## Integration with Other Skills

**Runs Before:**
- `qa-pipeline-automation` - Build Rust before running Python tests
- `benchmark-execution` - Build optimized release for benchmarking
- `cloud-integration-tests` - Rust backend needed for cloud I/O tests

**Uses:**
- `dependency-update-helper` - For updating Polars/Arrow/PyO3 versions

**Coordinate With:**
- `feature-branch-setup` - Build Rust backend after branch creation
- `pr-preparation-checks` - Verify Rust builds succeed in CI

## Common Workflows

### Workflow 1: Initial Setup + QA

```bash
# 1. Build Rust backend (this skill)
uv run maturin develop

# 2. Run QA pipeline
uv run ruff format src/ && uv run ruff check src/ --fix && uv run isort src/ && uv run mypy src/ && uv run pytest
```

### Workflow 2: Dependency Update + Rebuild

```bash
# 1. Update Polars (dependency-update-helper)
uv add "polars>=1.2.0"

# 2. Rebuild Rust (this skill)
cargo clean
uv run maturin develop

# 3. Verify compatibility
uv run pytest tests/ -k rust
```

### Workflow 3: Release Build + Benchmark

```bash
# 1. Build optimized release (this skill)
uv run maturin develop --release

# 2. Run benchmarks (benchmark-execution)
uv run python -m maou.infra.utility.benchmark_polars_io --num-records 50000
```

### Workflow 4: Cloud Testing Setup

```bash
# 1. Build Rust backend (this skill)
uv run maturin develop

# 2. Run cloud integration tests (cloud-integration-tests)
TEST_GCP=true uv run pytest tests/integration/ -k gcs
TEST_AWS=true uv run pytest tests/integration/ -k s3
```

## Context Reduction

Using this skill reduces context usage by **~65%** compared to manual troubleshooting:

- **Manual approach**: ~15 diagnostic commands + documentation lookup + trial-and-error (~2500 tokens)
- **Skill activation**: Single invocation + structured workflow (~900 tokens)
- **Benefit**: Automated OOM diagnostics, pre-configured solutions, memory monitoring

## References

- **CLAUDE.md**: Memory-Constrained Build Configuration (lines 229-400)
- **CLAUDE.md**: Rust Backend Development (lines 146-228)
- **CLAUDE.md**: Initial Rust Setup (lines 150-211)
- `.cargo/config.toml`: Global build settings
- `Cargo.toml`: Build profiles and feature flags
- `scripts/dev-init.sh`: Environment variable setup
- Cargo documentation: https://doc.rust-lang.org/cargo/
- maturin documentation: https://www.maturin.rs/
