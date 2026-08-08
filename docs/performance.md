# Performance Optimization Guide

## Neural Network Architecture

### BottleneckBlock Implementation
The project uses optimized BottleneckBlock architecture (1x1→3x3→1x1 convolution):

**Shogi-optimized configuration:**
- Layers: [2, 2, 2, 2]，strides [1, 2, 2, 2]
- Stage 出力チャンネル: [64, 128, 256, 512] (`BottleneckBlock`)

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
uv run maou utility benchmark-training \
  --input-s3 \
  --sample-ratio 0.1 \
  --gpu cuda:0
```

### GPU Prefetching (削除済み)

GPU プリフェッチ (`DataPrefetcher` / `gpu_prefetcher.py`) は削除された．
H2D 転送は DataLoader の `--pin-memory` + `--prefetch-factor` と，
`TrainingLoop._iterate_cuda_overlap` の CUDA ストリーム
オーバーラップに一本化されている．

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
uv run maou utility benchmark-dataloader \
  --stage3-data-path /path/to/processed \
  --gpu cuda:0 \
  --batch-size 256
```

### Training Performance
```bash
uv run maou utility benchmark-training \
  --stage3-data-path /path/to/processed \
  --gpu cuda:0 \
  --batch-size 256
```

### Polars + Rust I/O Performance
```bash
uv run python -m maou.infra.utility.benchmark_polars_io \
  --num-records 50000 \
  --output-dir /tmp/benchmark
```
