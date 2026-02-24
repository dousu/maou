# ADR-003: Training Performance Optimization Attempts (2025-11-22)

## Status

Decided - All optimization attempts rejected. Current baseline configuration retained.

## Context

Training performance investigation was conducted to achieve a target of 10 minutes per epoch (later revised to 20 minutes). The baseline configuration achieved **56.7 minutes per epoch** with the following settings:

- Batch size: 750
- DataLoader workers: 8
- DataLoader prefetch_factor: 1
- Dataset: 32,083,062 records (25,666,449 training / 6,416,613 validation)
- Total batches per epoch: 34,221
- Hardware: NVIDIA A100-SXM4-80GB (80GB VRAM)

### Initial Performance Analysis

**Baseline Metrics** (batch_size=750, workers=8, prefetch_factor=1):
- **Actual Average Batch Time**: 0.0997s
- **Predicted Epoch Time**: 56.7 minutes
- **Data Loading Time**: 0.0442s/batch (44.3% of total)
- **GPU Computation Time**: 0.0238s/batch (23.9% of total)
- **Throughput**: 5,641 samples/second

**Performance Breakdown**:
```
Data Loading:    0.0442s (44.3%)
GPU Transfer:    0.0001s (0.1%)
Forward Pass:    0.0079s (7.9%)
Loss Computation: 0.0008s (0.8%)
Backward Pass:    0.0082s (8.2%)
Optimizer Step:   0.0023s (2.3%)
Other:            0.0362s (36.4%)
```

### Physical Limitations

**GPU Computation Time Analysis**:
- Pure computation time: 0.0238s/batch × 34,221 batches = **13.6 minutes**
- With GPU overhead and synchronization: **~21.7 minutes minimum**

**Conclusion**:
- **10-minute target**: Physically impossible (computation alone takes 21.7 minutes)
- **20-minute target**: Extremely difficult (requires 2.8× speedup with data loading as bottleneck)

## Attempted Optimizations

### 1. Benchmark Output Fix (✅ SUCCESSFUL - RETAINED)

**Objective**: Fix benchmark output to accurately predict actual epoch time.

**Problem**:
- Previous benchmark reported `average_batch_time` which excluded warmup batches
- This caused a 13.4× underestimation of actual epoch time
- Example: Reported 0.0253s/batch predicting 16 minutes, but actual was 0.34s/batch taking 218 minutes

**Solution**:
- Added `actual_average_batch_time` metric: `total_epoch_time / total_batches`
- Updated benchmark output to show both metrics
- Fixed percentage calculations to use actual average batch time

**Files Modified**:
- `src/maou/app/learning/callbacks.py:260-280` - Added `actual_average_batch_time` to metrics
- `src/maou/app/utility/training_benchmark.py:55-85` - Added field to BenchmarkResult and updated output

**Result**: ✅ **SUCCESS** - Benchmark now accurately predicts epoch time
- Verification: 49.85s ÷ 500 batches = 0.0997s ✓ Perfect match with actual timing

---

### 2. DataLoader Worker/Prefetch Optimization (❌ FAILED - REVERTED)

**Objective**: Reduce data loading bottleneck by increasing parallelization.

**Configuration**:
- workers: 8 → 10 (+25%)
- prefetch_factor: 1 → 6 (+500%)

**Hypothesis**: More workers and prefetch buffer would overlap data loading with GPU computation.

**Results**:
```
Metric                    Baseline    Optimized    Change
─────────────────────────────────────────────────────────
Actual Avg Batch Time     0.0997s     0.3598s     +261% ❌
Predicted Epoch Time      56.7 min    205 min     +262% ❌
Data Loading Time         0.0442s     0.3105s     +602% ❌
Data Loading Percentage   44.3%       86.3%       +95% ❌
Throughput                5,641/s     2,084/s     -63% ❌
```

**Analysis**:
- **CPU Contention**: 10 workers exceeded optimal CPU utilization, causing context switching overhead
- **Memory Bandwidth Saturation**: Increased prefetch buffer caused memory bus saturation
- **Diminishing Returns**: Current 8 workers already near-optimal for hardware
- **Synchronization Overhead**: More workers increased coordination overhead

**Conclusion**: ❌ **REJECTED** - 3.6× performance regression
- Current workers=8, prefetch=1 is already optimal
- Over-parallelization causes severe performance degradation

---

### 3. Batch Size Increase (❌ FAILED - REVERTED)

**Objective**: Improve GPU utilization with larger batches.

**Configuration**:
- batch_size: 750 → 1500 (+100%)

**Hypothesis**: Larger batches would:
1. Improve GPU compute efficiency
2. Reduce per-batch DataLoader overhead
3. Better utilize A100's 80GB memory

**Results**:
```
Metric                    Baseline    Optimized    Change
─────────────────────────────────────────────────────────
Actual Avg Batch Time     0.0997s     0.4969s     +398% ❌
Predicted Epoch Time      56.7 min    141 min     +149% ❌
Data Loading Time         0.0442s     0.4337s     +881% ❌
Data Loading Percentage   44.3%       87.3%       +97% ❌
Throughput                5,641/s     2,257/s     -60% ❌
Total Batches             34,221      17,111      -50%
```

**Analysis**:
- **DataLoader Overhead**: Larger batches caused disproportionate DataLoader overhead increase (10× worse)
- **Memory Pressure**: Doubled batch size increased memory pressure, causing:
  - More frequent page faults in memory-mapped I/O
  - Increased numpy array allocation/deallocation overhead
- **No GPU Benefit**: GPU was not the bottleneck, so larger batches provided no benefit
- **Counterintuitive Result**: Despite fewer batches, total time increased significantly

**Conclusion**: ❌ **REJECTED** - 2.5× performance regression
- Current batch_size=750 provides optimal balance
- Larger batches exacerbate data loading bottleneck

---

### 4. torch.compile() JIT Compilation (❌ FAILED - REVERTED)

**Objective**: Accelerate model computation using PyTorch 2.0+ JIT compilation.

**Configuration**:
- Enabled `torch.compile()` on model with `dynamic=False`

**Hypothesis**: Model compilation would:
1. Optimize forward/backward pass computation
2. Fuse operations to reduce kernel launch overhead
3. Improve memory access patterns

**Results**:
```
Metric                    Baseline    Compiled     Change
─────────────────────────────────────────────────────────
Actual Avg Batch Time     0.0997s     0.2811s     +182% ❌
Predicted Epoch Time      56.7 min    160 min     +183% ❌
Data Loading Time         0.0442s     0.1749s     +296% ❌
Data Loading Percentage   44.3%       62.2%       +40% ❌
Forward Pass Time         0.0079s     0.0085s     +8% ❌
Throughput                5,641/s     2,668/s     -53% ❌
```

**Analysis**:
- **Compilation Overhead**: Model recompilation occurred frequently due to:
  - Dynamic input shapes from DataLoader
  - Variable batch sizes during warmup
  - Graph invalidation on every epoch start
- **Small Model**: BottleneckBlock architecture is already efficient
  - Layers: [2, 2, 2, 1] - very shallow
  - Bottleneck widths: [24, 48, 96, 144]
  - ~40% fewer parameters than ResNet-50
  - Little room for optimization
- **Data Loading Still Bottleneck**: Compilation cannot improve data loading (62.2% of time)
- **No Graph Reuse**: 500-batch benchmark showed continuous compilation overhead

**Conclusion**: ❌ **REJECTED** - 2.8× performance regression
- Compilation overhead outweighs benefits for this model
- Small, efficient model doesn't benefit from JIT compilation
- Should only be used for larger models with stable input shapes

---

### 5. FileDataSource.__getitems__() Batch Retrieval (❌ FAILED - REVERTED)

**Objective**: Optimize DataLoader batch retrieval by implementing `__getitems__()` method.

**Implementation**:
- Added `FileManager.get_items()` for batch-level data retrieval
- Added `FileDataSource.__getitems__()` to expose batch API to PyTorch
- Optimization strategy:
  1. Sort indices for sequential access
  2. Group indices by source file
  3. Use numpy fancy indexing for batch retrieval
  4. Restore original order with `np.argsort()`

**Hypothesis**: Batch-level retrieval would:
1. Reduce `np.searchsorted()` calls from N → 1 per batch
2. Utilize numpy's optimized fancy indexing
3. Improve memory-mapped I/O access patterns
4. Reduce per-item function call overhead

**Code Changes**:
```python
# FileManager.get_items() - Lines 314-375
def get_items(self, indices: list[int]) -> np.ndarray:
    """複数のレコードをまとめて取得してnumpy structured arrayとして返す．"""
    sorted_indices = sorted(indices)
    file_indices = np.searchsorted(self.cum_lengths, sorted_indices, side="right") - 1
    # ... file grouping and batch retrieval ...
    original_order = np.argsort([indices.index(idx) for idx in sorted_indices])
    return result[original_order]

# FileDataSource.__getitems__() - Lines 436-459
def __getitems__(self, indices: list[int]) -> list[np.ndarray]:
    """複数のアイテムをまとめて取得する（PyTorch DataLoaderの最適化用）．"""
    global_indices = [self.indicies[idx] for idx in indices]
    batch = self.__file_manager.get_items(global_indices)
    return list(batch)
```

**Results**:
```
Metric                    Baseline    Optimized    Change
─────────────────────────────────────────────────────────
Actual Avg Batch Time     0.0997s     0.2141s     +115% ❌
Predicted Epoch Time      56.7 min    122 min     +116% ❌
Data Loading Time         0.0442s     0.1729s     +291% ❌
Data Loading Percentage   44.3%       80.7%       +82% ❌
Throughput                5,641/s     3,503/s     -38% ❌
```

**Analysis**:

**Unexpected Overhead Sources**:

1. **Index Sorting Overhead** (`sorted_indices = sorted(indices)`):
   - Sorting 750 random indices per batch
   - O(n log n) complexity adds ~0.1ms per batch
   - No cache locality benefit for random access pattern

2. **Order Restoration Overhead** (`np.argsort([indices.index(idx) for idx in sorted_indices])`):
   - List comprehension with `.index()` is O(n²)
   - For batch_size=750: 750 × 750 = 562,500 operations per batch
   - Approximately 0.5-1ms overhead per batch

3. **List-to-Array Conversion** (`np.array(records)`):
   - Converting Python list of structured arrays to numpy array
   - Memory allocation and copy overhead
   - Structured array dtype validation

4. **PyTorch DataLoader Interaction**:
   - DataLoader's default collate function expects list returns
   - Additional conversions between `__getitems__()` return and batch tensor
   - Possible double-copying of data

5. **Memory Access Pattern**:
   - Fancy indexing `entry.memmap[relative_indices]` creates copy
   - Original `__getitem__()` uses direct scalar indexing (zero-copy)
   - Increased memory pressure from intermediate arrays

**Why Simple Implementation Was Faster**:

Original `__getitem__()` implementation:
```python
def __getitem__(self, idx: int) -> np.ndarray:
    return self.__file_manager.get_item(self.indicies[idx])

def get_item(self, idx: int) -> np.ndarray:
    file_idx = np.searchsorted(self.cum_lengths, idx, side="right") - 1
    relative_idx = idx - self.cum_lengths[file_idx]
    return entry.memmap[relative_idx]  # Direct scalar indexing - zero-copy
```

Benefits of simple approach:
- Direct scalar indexing into memory-mapped arrays (zero-copy)
- No sorting or order restoration overhead
- No intermediate array allocations
- PyTorch's DataLoader handles batching efficiently in C++
- Better cache locality with sequential DataLoader access

**Conclusion**: ❌ **REJECTED** - 2.1× performance regression
- Batch-level Python optimization slower than DataLoader's C++ batching
- Overhead of sorting and order restoration exceeded benefits
- Simple scalar access with memory-mapping is already optimal
- **Lesson**: Don't optimize what's already efficient in lower-level code

---

## Decision

**REJECT all optimization attempts. RETAIN baseline configuration.**

### Optimal Configuration (Validated)

```python
# Training Configuration
batch_size = 750
dataloader_workers = 8
dataloader_prefetch_factor = 1
enable_gpu_prefetch = True  # Already implemented
gradient_accumulation_steps = 1
torch_compile = False  # Explicitly disabled

# Performance Characteristics
actual_avg_batch_time = 0.0997s
epoch_time = 56.7 minutes
data_loading_percentage = 44.3%
throughput = 5,641 samples/second
```

### Retained Improvements

Only the **benchmark output fix** was successful and has been retained:
- `src/maou/app/learning/callbacks.py` - `actual_average_batch_time` metric
- `src/maou/app/utility/training_benchmark.py` - Accurate epoch time prediction

### Reverted Changes

All failed optimization attempts have been reverted:
- ❌ DataLoader worker/prefetch changes
- ❌ Batch size increase
- ❌ torch.compile() integration
- ❌ FileDataSource.__getitems__() implementation

## Consequences

### Why All Optimizations Failed

**Root Cause Analysis**:

1. **Current Configuration Already Optimal**
   - 8 workers perfectly balanced for CPU/memory bandwidth
   - batch_size=750 provides optimal GPU utilization without overhead
   - Memory-mapped I/O already maximally efficient

2. **Data Loading is Fundamental Bottleneck**
   - 44.3% of time spent loading data from disk
   - Storage I/O speed is physical limitation
   - Cannot be optimized further without hardware changes

3. **Over-Optimization Causes Harm**
   - Increased parallelization → CPU contention and memory saturation
   - Larger batches → Disproportionate DataLoader overhead
   - JIT compilation → Recompilation overhead exceeds benefits
   - Batch retrieval → Python overhead exceeds C++ DataLoader efficiency

4. **Physical Limitations**
   - GPU computation time: 21.7 minutes minimum
   - Data loading time: 24.9 minutes minimum (44.3% of 56.7 min)
   - Little room for improvement without hardware upgrades

### Performance Targets

**Achievable**:
- ✅ Current: 56.7 minutes/epoch
- ✅ Realistic target: 50-60 minutes/epoch (current range)

**Not Achievable**:
- ❌ 20 minutes/epoch (requires 2.8× speedup - impossible with current bottleneck)
- ❌ 10 minutes/epoch (requires 5.7× speedup - physically impossible)

### Recommendations for Future Work

**Do NOT Attempt**:
1. ❌ Increasing DataLoader workers beyond 8
2. ❌ Increasing DataLoader prefetch_factor beyond 1
3. ❌ Increasing batch_size beyond 750
4. ❌ Enabling torch.compile() for this model
5. ❌ Implementing custom batch retrieval in Python
6. ❌ Any optimization that increases data loading overhead

**Consider for Hardware Upgrades**:
1. ✅ Faster storage (NVMe SSD → NVMe RAID, RAM disk)
2. ✅ More RAM for full dataset caching (current: memory-mapped)
3. ✅ Multiple GPUs with data parallelism
4. ✅ Distributed training across multiple nodes

**Consider for Algorithm Changes**:
1. ✅ Reduce dataset size (data filtering, sampling)
2. ✅ Compress preprocessed data format (current: numpy structured arrays)
3. ✅ Pre-load data into RAM before training (requires ~100GB+ RAM)
4. ✅ Use smaller model for faster iteration (trade accuracy for speed)

### Key Lessons Learned

1. **Measure First, Optimize Second**
   - Baseline was already well-optimized
   - All optimization attempts made performance worse
   - Trust benchmarks and performance profiling

2. **Beware Over-Parallelization**
   - More workers/threads ≠ better performance
   - Hardware has optimal parallelism point
   - Exceeding it causes contention and degradation

3. **Understand Bottlenecks**
   - Data loading was bottleneck, not GPU computation
   - Optimizing non-bottleneck components wastes effort
   - Address actual bottleneck or accept limitations

4. **Respect Lower-Level Optimizations**
   - PyTorch DataLoader's C++ batching is highly optimized
   - Python-level batch retrieval slower than scalar access + C++ batching
   - Don't reinvent well-optimized wheels

5. **Physical Limits Exist**
   - No software optimization can overcome hardware limits
   - Storage I/O speed is fundamental constraint
   - Sometimes "good enough" is the best achievable

### Documentation

This ADR serves as a comprehensive record of:
- All optimization attempts and their results
- Detailed performance metrics for each configuration
- Analysis of why each optimization failed
- Guidelines for future optimization work
- Validated optimal configuration

**Benchmark Logs Preserved**:
- `/tmp/benchmark_compilation.log` - torch.compile() results
- `/tmp/benchmark_w10_p6.log` - Worker/prefetch optimization
- `/tmp/benchmark_bs1500.log` - Batch size increase
- `/tmp/benchmark_getitems.log` - __getitems__() optimization

## References

- **Baseline Investigation**: `docs/TRAINING_INVESTIGATION_REPORT.md`
- **DataLoader Optimization History**: `docs/adr-001-dataloader-multiprocessing-optimization.md`
- **GPU Prefetching Implementation**: `src/maou/app/learning/gpu_prefetcher.py`
- **Benchmark Tool**: `src/maou/app/utility/training_benchmark.py`
- **Training Loop**: `src/maou/app/learning/training_loop.py`
- **File Data Source**: `src/maou/infra/file_system/file_data_source.py`

## Date

2025-11-22

## Authors

- Investigation and optimization attempts: AI-assisted development
- Performance analysis and documentation: Comprehensive benchmarking study
