# Performance Optimization Guide

## Neural Network Architecture

### BottleneckBlock Implementation
The project uses optimized BottleneckBlock architecture (1x1→3x3→1x1 convolution):

**Shogi-optimized configuration:**
- Layers: [2, 2, 2, 1] - Wide and shallow
- Bottleneck widths: [24, 48, 96, 144]
- ~40% fewer parameters than ResNet-50

### Mixed Precision Training
Automatic mixed precision (AMP) enabled for CUDA:
- 1.5-2x faster training on GPU
- ~50% GPU memory reduction
- Maintains FP32 accuracy

## Performance Optimization

### Recommended Workflow
1. **DataLoader Benchmarking**: Find optimal settings
2. **Training Performance Analysis**: Identify bottlenecks
3. **Apply Optimizations**: Use recommended settings

### Sample Ratio for Large Datasets
Use `--sample-ratio` for efficient benchmarking:
```bash
poetry run maou utility benchmark-training \
  --input-s3 \
  --sample-ratio 0.1 \
  --gpu cuda:0
```

### GPU Prefetching (Auto-Enabled)
Automatic GPU prefetching overlaps data loading with computation. **Enabled by default** on CUDA devices.

**Performance**: -93.6% data loading time, +53.2% training throughput (2,202 → 3,374 samples/sec)

```python
# Default: enable_gpu_prefetch=True, gpu_prefetch_buffer_size=3
# To disable: enable_gpu_prefetch=False (not recommended)
```

### Gradient Accumulation
Simulate larger batch sizes without increasing GPU memory. Effective batch size = `batch_size × gradient_accumulation_steps`.

```python
training_loop = TrainingLoop(
    gradient_accumulation_steps=4,  # 256 × 4 = 1024 effective batch
)
# Memory usage: Same as batch_size=256
# Training time: Increases proportionally with steps
```

## Benchmarking Commands

### DataLoader Benchmarking
```bash
poetry run maou utility benchmark-dataloader \
  --input-path /path/to/processed \
  --gpu cuda:0 \
  --batch-size 256
```

### Training Performance
```bash
poetry run maou utility benchmark-training \
  --input-path /path/to/processed \
  --gpu cuda:0 \
  --batch-size 256
```

### Polars + Rust I/O Performance
```bash
poetry run python -m maou.infra.utility.benchmark_polars_io \
  --num-records 50000 \
  --output-dir /tmp/benchmark
```
