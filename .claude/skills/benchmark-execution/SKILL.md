---
name: benchmark-execution
description: Execute performance benchmarks for DataLoader configurations, training performance analysis, GPU utilization monitoring, and optimization validation. Use when analyzing performance bottlenecks, finding optimal training settings, validating speed improvements, or testing array bundling efficiency.
---

# Benchmark Execution

Performance analysis and optimization for Maou training workflows.

## Benchmark Types

### 1. DataLoader Benchmarking

Find optimal DataLoader configuration:

```bash
poetry run maou utility benchmark-dataloader \
  --input-dir /path/to/processed \
  --gpu cuda:0 \
  --batch-size 256
```

**What it measures**:
- Data loading throughput (samples/sec)
- GPU memory utilization
- CPU/GPU synchronization overhead
- Data prefetching efficiency
- Worker process performance
- Memory mapping effectiveness

**Output metrics**:
- Load time per batch
- Total throughput
- GPU utilization percentage
- Memory consumption
- Worker efficiency

### 2. Training Performance Analysis

Analyze end-to-end training performance:

```bash
poetry run maou utility benchmark-training \
  --input-dir /path/to/processed \
  --gpu cuda:0 \
  --batch-size 256
```

**What it measures**:
- Forward pass timing
- Backward pass timing
- Loss computation performance
- Gradient computation overhead
- Parameter update timing
- Overall training throughput

**Output metrics**:
- Samples processed per second
- GPU compute utilization
- Memory bandwidth utilization
- Training stability
- Iteration timing breakdown

## Performance Targets

### Expected Performance (CUDA GPU)

**DataLoader**:
- Throughput: >1000 samples/sec
- GPU utilization: >90%
- Memory efficiency: Minimal idle VRAM
- Worker overhead: <10%

**Training**:
- Forward+Backward: >100 samples/sec
- GPU utilization: >85%
- Mixed precision speedup: 1.5-2x
- Memory reduction: ~50% with AMP

### Performance Indicators

**Good Performance**:
- ✓ Sustained high throughput
- ✓ Stable GPU memory usage
- ✓ Consistent batch processing times
- ✓ High GPU utilization (>85%)

**Poor Performance**:
- ✗ GPU utilization <70%
- ✗ Frequent memory allocations
- ✗ Variable batch times (data loading bottleneck)
- ✗ Excessive CPU overhead

## Benchmarking Large Datasets

### Use Sample Ratio for Efficient Testing

Test on subset of data for quick iteration:

```bash
poetry run maou utility benchmark-training \
  --input-dir /path/to/processed \
  --sample-ratio 0.1 \
  --gpu cuda:0 \
  --batch-size 256
```

Runs on 10% of dataset - sufficient for performance analysis.

**Benefits**:
- Faster iteration cycles
- Reduced I/O overhead
- Quick configuration testing
- Same performance characteristics

## Cloud-Based Benchmarking

### S3 Performance Testing

```bash
poetry run maou utility benchmark-training \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --gpu cuda:0 \
  --batch-size 256
```

**Additional metrics**:
- Download throughput (MB/s)
- Cache hit rate
- Network overhead
- Parallel download efficiency

### GCS Performance Testing

```bash
poetry run maou utility benchmark-training \
  --input-gcs \
  --input-bucket-name my-bucket \
  --input-local-cache-dir ./cache \
  --gpu cuda:0 \
  --batch-size 256
```

### Array Bundling Performance Impact

Compare bundled vs non-bundled performance:

```bash
# Without bundling
poetry run maou utility benchmark-training \
  --input-s3 \
  --input-bucket-name my-bucket \
  --gpu cuda:0 \
  --batch-size 256

# With bundling
poetry run maou utility benchmark-training \
  --input-s3 \
  --input-bucket-name my-bucket \
  --input-enable-bundling \
  --input-bundle-size-gb 1.0 \
  --gpu cuda:0 \
  --batch-size 256
```

**Expected improvements with bundling**:
- 3-5x faster data loading
- Reduced file system overhead
- Better cache locality
- Lower network request count

## Mixed Precision Training Analysis

The project uses Automatic Mixed Precision (AMP) for CUDA:

**Benefits**:
- 1.5-2x training speedup
- ~50% GPU memory reduction
- Maintains FP32 accuracy
- Enabled automatically for CUDA

**Verify AMP is working**:

```bash
# Enable debug logging
export MAOU_LOG_LEVEL=DEBUG

# Run benchmark
poetry run maou utility benchmark-training \
  --gpu cuda:0 \
  --batch-size 256

# Look for: "Using automatic mixed precision (AMP)" in logs
```

## Optimization Workflow

### Recommended Approach

1. **Baseline Benchmark**
   ```bash
   poetry run maou utility benchmark-dataloader --input-dir ./data --gpu cuda:0
   ```

2. **Identify Bottlenecks**
   - Low GPU utilization → Data loading bottleneck
   - High memory usage → Reduce batch size
   - Slow iteration → Check model complexity

3. **Optimize Configuration**
   - Adjust `--num-workers`
   - Tune `--batch-size`
   - Enable `--pin-memory`
   - Test `--persistent-workers`

4. **Validate Improvements**
   ```bash
   poetry run maou utility benchmark-training --input-dir ./data --gpu cuda:0
   ```

5. **Production Testing**
   ```bash
   poetry run maou learn-model --input-dir ./data --gpu cuda:0 --epoch 1
   ```

## Batch Size Optimization

### Finding Optimal Batch Size

Test multiple batch sizes:

```bash
# Small batch
poetry run maou utility benchmark-training --batch-size 64 --gpu cuda:0

# Medium batch
poetry run maou utility benchmark-training --batch-size 256 --gpu cuda:0

# Large batch
poetry run maou utility benchmark-training --batch-size 512 --gpu cuda:0
```

**Trade-offs**:
- Larger batches: Better GPU utilization, more memory
- Smaller batches: Less memory, potential gradient noise

**Optimal selection**:
- Maximize batch size without OOM
- Ensure GPU utilization >85%
- Monitor training stability

## Worker Process Optimization

### Test Different Worker Counts

```bash
# No workers (main process only)
poetry run maou utility benchmark-dataloader --num-workers 0

# Few workers
poetry run maou utility benchmark-dataloader --num-workers 4

# Many workers
poetry run maou utility benchmark-dataloader --num-workers 16
```

**Guidelines**:
- CPU count: Typically 2x CPU cores
- Memory: Ensure sufficient RAM per worker
- I/O: More workers for cloud storage
- Overhead: Diminishing returns after ~16 workers

## Interpreting Results

### DataLoader Metrics

```
DataLoader Benchmark Results:
=============================
Throughput: 1250 samples/sec
GPU Utilization: 92%
CPU Usage: 45%
Memory: 8.2 GB / 16 GB
Workers: 8 active
Cache Hit Rate: 78%
```

**Analysis**:
- ✓ High throughput (>1000)
- ✓ Excellent GPU utilization (>90%)
- ✓ Reasonable CPU usage
- ✓ Good cache hit rate

### Training Metrics

```
Training Benchmark Results:
==========================
Forward Pass: 12.5 ms/batch
Backward Pass: 18.2 ms/batch
Total: 30.7 ms/batch (820 samples/sec)
GPU Memory: 10.5 GB / 16 GB
Mixed Precision: Enabled
Gradient Norm: 2.34
```

**Analysis**:
- Backward pass slower than forward (expected)
- Good throughput for training
- Memory usage reasonable
- AMP working correctly

## Common Performance Issues

### Issue: Low GPU Utilization (<70%)

**Diagnosis**: Data loading bottleneck

**Solutions**:
- Increase `--num-workers`
- Enable array bundling
- Use local caching for cloud data
- Optimize data preprocessing

### Issue: Out of Memory (OOM)

**Diagnosis**: Batch size too large

**Solutions**:
- Reduce `--batch-size`
- Enable gradient checkpointing
- Use mixed precision (AMP)
- Clear cache between runs

### Issue: Variable Batch Times

**Diagnosis**: Inconsistent data loading

**Solutions**:
- Use `--persistent-workers`
- Enable array bundling
- Pre-download cloud data
- Check disk I/O performance

### Issue: Slow Training Throughput

**Diagnosis**: Model or hardware limitations

**Solutions**:
- Verify AMP is enabled
- Check model architecture efficiency
- Profile with PyTorch profiler
- Consider model optimization

## Profiling with PyTorch Profiler

For detailed performance analysis:

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    # Training code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Benchmark Reporting

### Generate Performance Report

Create a summary after benchmarking:

```markdown
# Performance Benchmark Report

## Configuration
- GPU: NVIDIA A100 40GB
- Batch Size: 256
- Workers: 16
- Dataset: 100K samples

## Results
- DataLoader: 1450 samples/sec
- Training: 820 samples/sec
- GPU Utilization: 94%
- Memory: 12.8 GB / 40 GB

## Optimizations Applied
- Array bundling enabled (1.0 GB chunks)
- Mixed precision training (AMP)
- Persistent workers
- Local caching

## Improvements vs Baseline
- 3.2x faster data loading
- 2.1x faster training throughput
- 45% memory reduction
```

## When to Benchmark

- Before production training runs
- After optimization changes
- When changing hardware
- After dependency updates
- During performance regression investigation
- When validating cloud configurations

## Integration with Other Skills

**Combine with**:
- `cloud-integration-tests` - Test cloud performance
- `qa-pipeline-automation` - Ensure code quality before benchmarking
- `architecture-validator` - Verify structural efficiency

## References

- **CLAUDE.md**: Performance optimization (lines 294-309)
- **AGENTS.md**: Benchmarking workflows (lines 176-190)
- PyTorch profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Mixed precision training: https://pytorch.org/docs/stable/amp.html
