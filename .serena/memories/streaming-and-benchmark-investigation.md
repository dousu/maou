# Maou Streaming & Benchmarking Code Investigation Report

## Executive Summary

This document provides a detailed investigation of the Maou project's code for:
1. **SingleEpochBenchmark** - Timing measurement infrastructure (data loading, GPU transfer, forward pass, loss, backward, optimizer step)
2. **StreamingFileSource / StreamingKifDataset** - File-level streaming data loading pattern
3. **DataLoaderFactory.create_streaming_dataloaders()** - Streaming DataLoader creation with spawn context
4. **Benchmark-Training CLI/Config** - warmup_batches handling in benchmark-training command
5. **Learn-Model Streaming Integration** - How learn-model uses streaming sources

---

## 1. SingleEpochBenchmark Class

**File:** `src/maou/app/utility/training_benchmark.py`

### Class Signature

```python
class SingleEpochBenchmark:
    """Single epoch benchmark for training performance measurement."""
    
    def __init__(
        self,
        *,
        model: Network,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn_policy: torch.nn.Module,
        loss_fn_value: torch.nn.Module,
        policy_loss_ratio: float,
        value_loss_ratio: float,
        enable_resource_monitoring: bool = False,
    ):
```

### Key Methods

#### `benchmark_epoch()`
- **Purpose:** Run single epoch benchmark with detailed timing breakdown
- **Signature:**
  ```python
  def benchmark_epoch(
      self,
      dataloader: DataLoader,
      *,
      warmup_batches: int = 5,
      max_batches: Optional[int] = None,
      enable_profiling: bool = False,
  ) -> BenchmarkResult:
  ```
- **Warmup Handling:** First `warmup_batches` (default: 5) are excluded from timing statistics
- **Returns:** `BenchmarkResult` with detailed timing metrics

#### `benchmark_validation()`
- **Purpose:** Run validation/inference-only benchmark
- **Signature:** Similar to `benchmark_epoch` but no warmup, inference mode
- **Returns:** Validation-specific `BenchmarkResult`

### Timing Measurement Flow

**Architecture:** Uses `TimingCallback` to instrument the training loop at specific points:

1. **Data Loading Time**: Measured between previous batch end and current batch start
   - Formula: `batch_start_time - previous_batch_end_time`
   
2. **GPU Transfer Time**: DataLoader → GPU transfer
   - Measured: `on_data_transfer_start()` → `on_data_transfer_end()`
   
3. **Forward Pass Time**: Model forward computation
   - Measured: `on_forward_pass_start()` → `on_forward_pass_end()`
   - Pure forward = total_forward_time - loss_computation_time
   
4. **Loss Computation Time**: Loss function evaluation
   - Measured: `on_loss_computation_start()` → `on_loss_computation_end()`
   
5. **Backward Pass Time**: Gradient computation
   - Measured: `on_backward_pass_start()` → `on_backward_pass_end()`
   
6. **Optimizer Step Time**: Parameter update
   - Measured: `on_optimizer_step_start()` → `on_optimizer_step_end()`

**Key Features:**
- Warmup exclusion: Only batches `>= warmup_batches` contribute to timing stats
- GradScaler support for mixed precision (GPU only)
- Resource monitoring callback integration (CPU, Memory, GPU)
- Detailed breakdown in `BenchmarkResult` dataclass

### BenchmarkResult Structure

```python
@dataclass(frozen=True)
class BenchmarkResult:
    # Overall Timing
    total_epoch_time: float              # Wall-clock epoch duration
    average_batch_time: float            # Avg per-batch processing (excl. data loading)
    actual_average_batch_time: float     # Avg per-batch including all overhead
    total_batches: int                   # Processed batch count
    
    # Warmup Information
    warmup_time: float                   # Time spent in warmup phase
    warmup_batches: int                  # Number of warmup batches
    measured_time: float                 # Time from end of warmup to epoch end
    measured_batches: int                # Batches after warmup
    
    # Detailed Timing Breakdown (per batch, averaged)
    data_loading_time: float             # I/O + DataLoader overhead
    gpu_transfer_time: float             # CPU→GPU transfer
    forward_pass_time: float             # Pure forward computation
    loss_computation_time: float         # Loss evaluation
    backward_pass_time: float            # Gradient computation
    optimizer_step_time: float           # Parameter update
    
    # Performance Metrics
    samples_per_second: float            # Throughput metric
    batches_per_second: float            # Batch throughput
    
    # Loss
    final_loss: float                    # Final batch loss
    average_loss: float                  # Average over measured batches
    
    # Optional Resource Monitoring
    resource_usage: Optional[ResourceUsage] = None
```

### TrainingBenchmarkConfig

```python
@dataclass(frozen=True)
class TrainingBenchmarkConfig:
    datasource: LearningDataSource.DataSourceSpliter
    gpu: Optional[str] = None
    compilation: bool = False
    detect_anomaly: bool = False
    batch_size: int = 256
    dataloader_workers: int = 4
    pin_memory: Optional[bool] = None
    prefetch_factor: int = 2
    cache_transforms: Optional[bool] = None
    gce_parameter: float = 0.1
    policy_loss_ratio: float = 1.0
    value_loss_ratio: float = 1.0
    learning_ratio: float = 0.01
    momentum: float = 0.9
    optimizer_name: str = "adamw"
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    lr_scheduler_name: Optional[str] = None
    warmup_batches: int = 5              # ← Warmup batches config
    max_batches: Optional[int] = None    # ← Max batches to process
    enable_profiling: bool = False
    test_ratio: float = 0.2
    run_validation: bool = False
    sample_ratio: Optional[float] = None
    enable_resource_monitoring: bool = False
    model_architecture: BackboneArchitecture = "resnet"
```

---

## 2. Streaming Data Loading Infrastructure

### 2.1 StreamingFileSource Class

**File:** `src/maou/infra/file_system/streaming_file_source.py`

**Purpose:** Infrastructure layer that handles file-level streaming with memory-efficient pattern.

### Class Signature

```python
class StreamingFileSource:
    """ファイル単位のストリーミングデータソース．
    
    全ファイルを一度にメモリにロードする FileManager とは異なり，
    1ファイルずつ読み込み → 消費 → 解放のストリーミングパターンを使用する．
    """
    
    def __init__(
        self,
        file_paths: list[Path],
        array_type: Literal[
            "hcpe", "preprocessing", "stage1", "stage2"
        ],
    ) -> None:
```

### Key Properties & Methods

#### Lazy Row Count Scanning

```python
@property
def total_rows(self) -> int:
    """全ファイルの合計行数(初回アクセス時にスキャン実行)."""
    # Lazy initialization: scans only on first access
    self._ensure_row_counts()
    return self._total_rows

@property
def row_counts(self) -> list[int]:
    """各ファイルの行数リスト(初回アクセス時にスキャン実行)."""
    self._ensure_row_counts()
    return list(self._row_counts)
```

**Optimization:** Deferred row count scanning to avoid blocking initialization

#### File Iteration Methods

```python
def iter_files_columnar(
    self,
) -> Generator[ColumnarBatch, None, None]:
    """ファイル単位で ColumnarBatch をyieldする．
    
    1. featherファイルをPolars DataFrameとして読み込み
    2. Polars DataFrame → ColumnarBatch に変換
    3. ColumnarBatch をyield
    4. DataFrame参照を即座に切る(GC対象にする)
    """
    for fp in self._file_paths:
        df = self._loader(fp)          # Rust FFI call
        batch = self._converter(df)    # Convert to SOA format
        del df                         # Explicit GC hint
        yield batch

def iter_files_columnar_subset(
    self,
    file_paths: list[Path],
) -> Generator[ColumnarBatch, None, None]:
    """指定されたファイルパスのみを読み込む（worker分割用）．
    
    DEBUG レベルでタイミング情報を出力:
    - ファイル読込時間 (t_load)
    - 変換時間 (t_conv)
    """
    for i, fp in enumerate(file_paths):
        logger.log(log_level, "Loading file %d/%d: %s", i+1, n, fp.name)
        t0 = time.perf_counter()
        df = self._loader(fp)
        t_load = time.perf_counter() - t0
        logger.log(log_level, "File loaded: %d rows in %.2fs, converting...", len(df), t_load)
        
        t1 = time.perf_counter()
        batch = self._converter(df)
        t_conv = time.perf_counter() - t1
        logger.debug("Conversion complete: %.2fs (file %d/%d)", t_conv, i+1, n)
        
        del df
        yield batch
```

#### File Format Detection (Arrow IPC)

```python
def _is_arrow_ipc_file_format(file_path: Path) -> bool:
    """Arrow IPC File形式 (File vs Stream) を判定する．"""
    _ARROW_FILE_MAGIC = b"ARROW1\x00\x00"  # First 8 bytes
    with open(file_path, "rb") as f:
        header = f.read(8)
    return header == _ARROW_FILE_MAGIC

def _scan_row_count(file_path: Path) -> int:
    """featherファイルの行数のみを取得する．
    
    - Arrow IPC File形式: メタデータのみ読み（高速）
    - Arrow IPC Stream形式: 全データ読み（警告ログ出力）
    """
    if _is_arrow_ipc_file_format(file_path):
        lf = pl.scan_ipc(file_path)
        return lf.select(pl.len()).collect().item()
    else:
        logger.warning(
            "File %s is Arrow IPC Stream format. "
            "Reading full data for row count (consider converting to File format).",
            file_path,
        )
        df = pl.read_ipc_stream(file_path)
        row_count = df.height
        del df
        return row_count
```

### 2.2 StreamingDataSource Protocol

**File:** `src/maou/app/learning/streaming_dataset.py`

**Purpose:** Application layer abstraction to avoid direct infra dependency.

```python
@runtime_checkable
class StreamingDataSource(Protocol):
    """ストリーミングデータソースのプロトコル．"""
    
    @property
    def file_paths(self) -> list[Path]:
        """ファイルパスのリスト(worker分割用)."""
        ...
    
    @property
    def total_rows(self) -> int:
        """全ファイルの合計行数."""
        ...
    
    @property
    def row_counts(self) -> list[int]:
        """各ファイルの行数リスト."""
        ...
    
    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        """ファイル単位で ColumnarBatch をyieldする."""
        ...
    
    def iter_files_columnar_subset(
        self,
        file_paths: list[Path],
    ) -> Generator[ColumnarBatch, None, None]:
        """指定されたファイルパスのみを読み込み ColumnarBatch をyieldする."""
        ...
```

### 2.3 Streaming Datasets (IterableDataset)

**Key Classes:**
1. **StreamingKifDataset** - Stage 3 (Policy + Value) training
2. **StreamingStage1Dataset** - Stage 1 (Reachable Squares)
3. **StreamingStage2Dataset** - Stage 2 (Legal Moves) with file concatenation

#### Common Pattern for All Three

```python
class StreamingKifDataset(IterableDataset):
    """IterableDataset版のKifDataset(バッチ単位yield)．"""
    
    def __init__(
        self,
        *,
        streaming_source: StreamingDataSource,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self._source = streaming_source
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
    
    def set_epoch(self, epoch: int) -> None:
        """エポック番号を設定（シード管理用）．"""
        self._epoch = epoch
    
    def __iter__(self) -> Iterator[tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """バッチ単位でTensorをyield．
        
        persistent_workers対応:
        - worker_info.seed を使用（エポックごとに変更）
        - または (seed + epoch) で異なるシード生成
        
        Worker分割:
        - _resolve_worker_files() でファイルをラウンドロビン分配
        - 各workerが担当ファイルのみ読み込み
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if worker_info is not None:
            epoch_seed = worker_info.seed  # persistent_workers対応
        else:
            epoch_seed = (self._seed or 0) + self._epoch
        
        rng = np.random.default_rng(epoch_seed)
        
        try:
            # Worker分割 + ファイル順シャッフル
            worker_files = _resolve_worker_files(
                self._source,
                shuffle=self._shuffle,
                epoch_seed=epoch_seed,
            )
            
            if not worker_files:
                logger.debug("Worker %d: no files assigned", worker_id)
                return
            
            total_batches = 0
            file_count = 0
            for file_idx, columnar_batch in enumerate(
                self._source.iter_files_columnar_subset(worker_files)
            ):
                file_count += 1
                if file_idx == 0:
                    _log_worker_memory(worker_id, "after_first_file", level=logging.DEBUG)
                
                for batch in _yield_kif_batches(
                    columnar_batch,
                    batch_size=self._batch_size,
                    shuffle=self._shuffle,
                    rng=rng,
                ):
                    total_batches += 1
                    if total_batches == 1:
                        logger.debug(
                            "Worker %d: first batch produced (pid=%d)",
                            worker_id,
                            os.getpid(),
                        )
                    yield batch
            
            logger.debug(
                "Worker %d: iteration complete (%d batches from %d files)",
                worker_id,
                total_batches,
                file_count,
            )
        except Exception as exc:
            logger.error(
                "Worker %d crashed during iteration (pid=%d): %s",
                worker_id,
                os.getpid(),
                exc,
                exc_info=True,
            )
            raise
    
    def __len__(self) -> int:
        """バッチ数を返す(tqdmプログレスバー用)."""
        return _compute_total_batches(
            self._source.row_counts,
            self._batch_size,
        )
```

#### Stage 2 Specific Feature: File Concatenation

```python
_FILES_PER_CONCAT: int = 10
"""ファイル結合のグループサイズ．
Stage 2 の小ファイル(~100K行)をこの個数まとめて
ColumnarBatch.concatenate() で結合し，ファイルロード
間隔を広げてI/Oストールを軽減する．
"""

# In StreamingStage2Dataset.__iter__():
buffer: list[ColumnarBatch] = []
for file_idx, columnar_batch in enumerate(self._source.iter_files_columnar_subset(worker_files)):
    buffer.append(columnar_batch)
    
    if len(buffer) >= _FILES_PER_CONCAT:
        merged = ColumnarBatch.concatenate(buffer)
        buffer.clear()
        for batch in _yield_stage2_batches(merged, ...):
            yield batch

# 残りのバッファを処理
if buffer:
    merged = ColumnarBatch.concatenate(buffer)
    buffer.clear()
    for batch in _yield_stage2_batches(merged, ...):
        yield batch
```

#### Helper: Worker File Resolution

```python
def _resolve_worker_files(
    source: StreamingDataSource,
    shuffle: bool,
    epoch_seed: int,
) -> list[Path]:
    """Workerファイル分割 + ファイル順シャッフルを行う共通関数．
    
    1. オプショナルにファイル順をシャッフル (epoch_seed でランダマイズ)
    2. Worker数でラウンドロビン分配 (i % n_workers == worker_id)
    """
    file_paths = source.file_paths
    
    # エポックごとのファイル順シャッフル
    if shuffle:
        file_rng = np.random.default_rng(epoch_seed + 1_000_000)
        file_indices = file_rng.permutation(len(file_paths))
        file_paths = [file_paths[i] for i in file_indices]
    
    # Worker分割
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        n_workers = worker_info.num_workers
        worker_id = worker_info.id
        if len(file_paths) < n_workers:
            logger.warning(
                "Number of files (%d) < num_workers (%d). Some workers will be idle.",
                len(file_paths),
                n_workers,
            )
        # ラウンドロビン分配
        file_paths = [
            fp
            for i, fp in enumerate(file_paths)
            if i % n_workers == worker_id
        ]
    
    return file_paths
```

#### Batch Yield Helpers

```python
def _yield_kif_batches(
    columnar_batch: ColumnarBatch,
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Generator[tuple[tuple[torch.Tensor, torch.Tensor], ...], None, None]:
    """ColumnarBatchからKifDataset互換のバッチTensorをyieldする．"""
    n = len(columnar_batch)
    if n == 0:
        return
    
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    
    for start in range(0, n, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = columnar_batch.slice(batch_indices)
        
        # .clone() でPyTorch-nativeストレージに変換
        board_tensor = torch.from_numpy(batch.board_positions).clone()
        pieces_tensor = torch.from_numpy(batch.pieces_in_hand).clone()
        move_label_tensor = torch.from_numpy(batch.move_label).clone()
        result_value_tensor = torch.from_numpy(batch.result_value).float().unsqueeze(1)
        legal_move_mask_tensor = torch.ones_like(move_label_tensor)
        
        yield (
            (board_tensor, pieces_tensor),
            (move_label_tensor, result_value_tensor, legal_move_mask_tensor),
        )
```

---

## 3. DataLoaderFactory.create_streaming_dataloaders()

**File:** `src/maou/app/learning/setup.py`

### Method Signature

```python
@classmethod
def create_streaming_dataloaders(
    cls,
    train_dataset: IterableDataset,
    val_dataset: IterableDataset,
    dataloader_workers: int,
    pin_memory: bool,
    prefetch_factor: int = 2,
    n_train_files: int = 0,
    n_val_files: int = 0,
    file_paths: list[Path] | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Streaming用DataLoader作成．"""
```

### Key Design Decisions

#### 1. spawn Context for Multiprocessing

```python
"""ストリーミングワーカーは Rust FFI (Polars/Arrow) を呼び出すため，
multiprocessing_context="spawn" を使用する．

fork/forkserver では jemalloc の内部状態が子プロセスに継承され，
segfault する可能性がある．

spawn は os.exec() で完全に新しいプロセスを生成するため安全．
"""
mp_context: str | None = "spawn" if train_workers > 0 else None
mp_context_val: str | None = "spawn" if val_workers > 0 else None
```

#### 2. Memory-Based Worker Limiting

```python
# ワーカー数の動的制限: メモリとファイル数
memory_limit = _estimate_max_workers_by_memory(
    pin_memory=pin_memory,
    logger=cls.logger,
    file_paths=file_paths,
)

train_workers = cls._clamp_workers(
    dataloader_workers,        # 要求されたworker数
    n_train_files,             # ファイル数上限
    "training",
    cls.logger,
    memory_limit=memory_limit, # メモリ上限
)

val_workers = cls._clamp_workers(
    dataloader_workers,
    n_val_files,
    "validation",
    cls.logger,
    memory_limit=memory_limit,
)
```

**Helper Function: _clamp_workers()**

```python
@staticmethod
def _clamp_workers(
    requested_workers: int,
    n_files: int,
    label: str,
    logger: logging.Logger,
    *,
    memory_limit: int | None = None,
) -> int:
    """ワーカー数をファイル数およびメモリ制約で制限する．
    
    ストリーミングモードでは各ワーカーが1つ以上のファイルを担当するため，
    ファイル数を超えるワーカーは不要かつ有害(アイドルワーカーがリソース消費)．
    """
    if n_files <= 0:
        return 0
    if requested_workers <= 0:
        return 0
    
    effective = min(requested_workers, n_files)  # ファイル数で上限
    if memory_limit is not None and memory_limit > 0:
        effective = min(effective, memory_limit)  # メモリで上限
    
    if effective < requested_workers:
        logger.info(
            "Clamped %s workers from %d to %d "
            "(file_count=%d, memory_limit=%s)",
            label,
            requested_workers,
            effective,
            n_files,
            memory_limit,
        )
    return effective
```

#### 3. Memory Estimation

```python
def _estimate_max_workers_by_memory(
    pin_memory: bool,
    logger: logging.Logger,
    file_paths: list[Path] | None = None,
) -> int:
    """システムの利用可能メモリからワーカー数の上限を推定する．
    
    利用可能メモリの50%をDataLoaderワーカーに割当．
    ファイルサイズベースの動的推定:
    - 圧縮済みファイルの平均サイズ
    - 展開倍率: LZ4 = 4.0倍
    - 安全マージン: 1.5倍
    """
    available_mb = psutil.virtual_memory().available / (1024 * 1024)
    worker_budget_mb = available_mb * 0.5
    per_worker_mb = _estimate_per_worker_mb(file_paths, logger)
    if pin_memory:
        per_worker_mb += 50.0
    max_workers = max(1, int(worker_budget_mb / per_worker_mb))
    return max_workers
```

#### 4. /dev/shm Verification

```python
def _check_shm_size(
    num_workers: int,
    batch_size: int | None,
    prefetch_factor: int,
    logger: logging.Logger,
) -> None:
    """Linux環境で /dev/shm の空き容量を確認し不足時に警告する．
    
    DataLoaderのワーカー間通信は共有メモリ(/dev/shm)を使用する．
    Docker等で /dev/shm サイズが制限されている場合，警告を発出．
    """
    _BYTES_PER_SAMPLE_KB = 154  # Stage 3の1バッチあたりの概算
    effective_batch_size = batch_size if batch_size is not None else 1024
    threshold_mb = (
        effective_batch_size
        * _BYTES_PER_SAMPLE_KB
        * num_workers
        * prefetch_factor
        / 1024
    )
    
    if shm_available_mb < threshold_mb:
        logger.warning(
            "/dev/shm available space (%.0fMB) is below "
            "estimated requirement (%.0fMB) for %d workers "
            "with prefetch_factor=%d. "
            "Consider increasing /dev/shm size "
            "(e.g. docker run --shm-size=8g) or "
            "reducing --dataloader-workers.",
            shm_available_mb,
            threshold_mb,
            num_workers,
            prefetch_factor,
        )
```

#### 5. DataLoader Construction with batch_size=None

```python
# batch_size=None (自動バッチングOFF)
# StreamingDatasetがバッチ単位でTensorをyieldするため
training_loader = DataLoader(
    train_dataset,
    batch_size=None,           # ← 自動バッチング OFF
    shuffle=False,             # ← StreamingDataset内でシャッフル
    num_workers=train_workers,
    pin_memory=pin_memory,
    persistent_workers=train_workers > 0,
    prefetch_factor=prefetch_factor if train_workers > 0 else None,
    timeout=_STREAMING_TIMEOUT,
    worker_init_fn=train_worker_init_fn,
    multiprocessing_context=mp_context,  # spawn コンテキスト
)
```

**Important:** `batch_size=None` tells PyTorch DataLoader to:
- Do NOT apply automatic batching
- Yield items directly from the IterableDataset
- This matches WebDataset/Mosaic StreamingDataset patterns

#### 6. Timeout Configuration

```python
# spawn + persistent_workers での通信をカバーする有限タイムアウト
_STREAMING_TIMEOUT = 300 if train_workers > 0 else 0  # 5分

training_loader = DataLoader(
    ...,
    timeout=_STREAMING_TIMEOUT,
    ...
)
```

### Full Implementation

```python
@classmethod
def create_streaming_dataloaders(
    cls,
    train_dataset: IterableDataset,
    val_dataset: IterableDataset,
    dataloader_workers: int,
    pin_memory: bool,
    prefetch_factor: int = 2,
    n_train_files: int = 0,
    n_val_files: int = 0,
    file_paths: list[Path] | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Streaming用DataLoader作成．"""
    
    # 1. Estimate memory limit based on available system resources
    memory_limit = _estimate_max_workers_by_memory(
        pin_memory=pin_memory,
        logger=cls.logger,
        file_paths=file_paths,
    )
    
    # 2. Clamp worker counts (file count + memory limits)
    train_workers = cls._clamp_workers(
        dataloader_workers,
        n_train_files,
        "training",
        cls.logger,
        memory_limit=memory_limit,
    )
    val_workers = cls._clamp_workers(
        dataloader_workers,
        n_val_files,
        "validation",
        cls.logger,
        memory_limit=memory_limit,
    )
    
    # 3. Check /dev/shm capacity
    _check_shm_size(
        num_workers=max(train_workers, val_workers),
        batch_size=None,
        prefetch_factor=prefetch_factor,
        logger=cls.logger,
    )
    
    # 4. Setup worker init functions
    train_worker_init_fn = (
        default_worker_init_fn if train_workers > 0 else None
    )
    val_worker_init_fn = (
        default_worker_init_fn if val_workers > 0 else None
    )
    
    # 5. Setup multiprocessing context (spawn for Rust FFI safety)
    mp_context: str | None = "spawn" if train_workers > 0 else None
    mp_context_val: str | None = "spawn" if val_workers > 0 else None
    
    # 6. Setup timeouts
    _STREAMING_TIMEOUT = 300 if train_workers > 0 else 0
    _STREAMING_TIMEOUT_VAL = 300 if val_workers > 0 else 0
    
    # 7. Create training DataLoader
    training_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=train_workers,
        pin_memory=pin_memory,
        persistent_workers=train_workers > 0,
        prefetch_factor=prefetch_factor if train_workers > 0 else None,
        timeout=_STREAMING_TIMEOUT,
        worker_init_fn=train_worker_init_fn,
        multiprocessing_context=mp_context,
    )
    
    # 8. Create validation DataLoader
    validation_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
        persistent_workers=val_workers > 0,
        prefetch_factor=prefetch_factor if val_workers > 0 else None,
        timeout=_STREAMING_TIMEOUT_VAL,
        worker_init_fn=val_worker_init_fn,
        multiprocessing_context=mp_context_val,
    )
    
    # 9. Log batch counts
    if hasattr(train_dataset, "__len__"):
        cls.logger.info(
            "Streaming Training: %d batches",
            len(train_dataset),
        )
    if hasattr(val_dataset, "__len__"):
        cls.logger.info(
            "Streaming Validation: %d batches",
            len(val_dataset),
        )
    
    return training_loader, validation_loader
```

---

## 4. Benchmark-Training Related CLI & Config

**File:** `src/maou/infra/console/utility.py` (CLI command)

### CLI Command: benchmark-training

Located in: `/workspaces/maou/src/maou/infra/console/utility.py`

**Key Parameters Related to Warmup:**

```python
@click.option(
    "--warmup-batches",
    type=int,
    default=5,
    help="Number of warmup batches to exclude from timing statistics (default: 5).",
    show_default=True,
)
@click.option(
    "--max-batches",
    type=int,
    default=100,
    help="Maximum number of batches to process during benchmark (default: 100).",
    show_default=True,
)
```

### Interface Layer: benchmark_training()

**File:** `src/maou/interface/utility_interface.py`

```python
def benchmark_training(
    datasource: LearningDataSource.DataSourceSpliter,
    *,
    warmup_batches: Optional[int] = None,  # ← Warmup config
    max_batches: Optional[int] = None,      # ← Max batches config
    enable_profiling: Optional[bool] = None,
    run_validation: Optional[bool] = None,
    sample_ratio: Optional[float] = None,
    enable_resource_monitoring: Optional[bool] = None,
    # ... other params
) -> str:
    """Benchmark single epoch training performance with detailed timing analysis."""
    
    # Validation & defaults
    if warmup_batches is None:
        warmup_batches = 5
    elif warmup_batches < 0:
        raise ValueError(
            f"warmup_batches must be non-negative, got {warmup_batches}"
        )
    
    if max_batches is None:
        max_batches = 100
    elif max_batches <= 0:
        raise ValueError(
            f"max_batches must be positive, got {max_batches}"
        )
    
    if enable_profiling is None:
        enable_profiling = False
    
    if run_validation is None:
        run_validation = False
    
    if enable_resource_monitoring is None:
        enable_resource_monitoring = False
    
    # Create config
    config = TrainingBenchmarkConfig(
        datasource=datasource,
        gpu=gpu,
        compilation=compilation,
        test_ratio=test_ratio,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        cache_transforms=cache_transforms_enabled,
        gce_parameter=gce_parameter,
        policy_loss_ratio=policy_loss_ratio,
        value_loss_ratio=value_loss_ratio,
        learning_ratio=learning_ratio,
        momentum=momentum,
        lr_scheduler_name=lr_scheduler_key,
        optimizer_name=optimizer_key,
        optimizer_beta1=optimizer_beta1,
        optimizer_beta2=optimizer_beta2,
        optimizer_eps=optimizer_eps,
        warmup_batches=warmup_batches,        # ← Passed to config
        max_batches=max_batches,              # ← Passed to config
        enable_profiling=enable_profiling,
        run_validation=run_validation,
        sample_ratio=sample_ratio,
        enable_resource_monitoring=enable_resource_monitoring,
        detect_anomaly=detect_anomaly,
        model_architecture=model_architecture,
    )
    
    use_case = TrainingBenchmarkUseCase()
    return use_case.execute(config)
```

---

## 5. Learn-Model Streaming Integration

**File:** `src/maou/infra/console/learn_model.py` (CLI) and `src/maou/interface/learn.py` (interface)

### CLI Option

```python
@click.option(
    "--no-streaming",
    is_flag=True,
    default=False,
    help="Disable streaming mode for file input (use map-style dataset instead).",
)
```

**Default Behavior:** Streaming is ENABLED unless `--no-streaming` is specified.

### Interface Function Signatures

```python
# From src/maou/interface/learn.py

def run_stage3(
    datasource: DataSource,
    *,
    streaming: bool = False,
    streaming_train_source: Optional[StreamingDataSource] = None,
    streaming_val_source: Optional[StreamingDataSource] = None,
    ...
) -> LearnResult:
    """Stage 3 training (Policy + Value)."""

def _run_stage1_streaming(
    streaming_source: StreamingDataSource,
    # ... training params
) -> LearnResult:
    """Stage 1 streaming-specific implementation."""

def _run_stage2_streaming(
    streaming_source: StreamingDataSource,
    # ... training params
) -> LearnResult:
    """Stage 2 streaming-specific implementation."""

def run_multi_stage(
    *,
    streaming: bool = False,
    stage1_streaming_source: Optional[StreamingDataSource] = None,
    stage2_streaming_source: Optional[StreamingDataSource] = None,
    stage3_streaming_train_source: Optional[StreamingDataSource] = None,
    stage3_streaming_val_source: Optional[StreamingDataSource] = None,
    # ... other params
) -> LearnResult:
    """Multi-stage training with streaming support."""
```

### Streaming DataLoader Creation in learn.py

```python
# When streaming is enabled:
dataloaders = DataLoaderFactory.create_streaming_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    dataloader_workers=dataloader_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
    n_train_files=len(streaming_source.file_paths),
    n_val_files=len(streaming_val_source.file_paths),
    file_paths=streaming_source.file_paths,
)
```

### Learn-Model CLI Processing Flow

```python
# From src/maou/infra/console/learn_model.py

# 1. Collect file paths for each stage
stage1_data_config = StageDataConfig(...)  # Lazy config
stage2_data_config = StageDataConfig(...)
stage3_data_config = StageDataConfig(...)

# 2. Flag: Check --no-streaming
use_streaming = not no_streaming  # Default: True

# 3. If streaming enabled, create StreamingFileSource objects
if use_streaming:
    # For each stage:
    s1_streaming_source = StreamingFileSource(
        file_paths=stage1_file_paths,
        array_type="stage1",
    )
    
    # Similar for Stage 2 and Stage 3
    s2_streaming_source = StreamingFileSource(
        file_paths=stage2_file_paths,
        array_type="stage2",
    )
    
    s3_streaming_train_source = StreamingFileSource(
        file_paths=stage3_train_file_paths,
        array_type="stage3",
    )
    s3_streaming_val_source = StreamingFileSource(
        file_paths=stage3_val_file_paths,
        array_type="stage3",
    )

# 4. Call learn interface with streaming sources
result = learn.run_multi_stage(
    streaming=use_streaming,
    stage1_streaming_source=s1_streaming_source if use_streaming else None,
    stage2_streaming_source=s2_streaming_source if use_streaming else None,
    stage3_streaming_train_source=s3_streaming_train_source if use_streaming else None,
    stage3_streaming_val_source=s3_streaming_val_source if use_streaming else None,
    # ... other params
)
```

---

## Summary Table

| Component | Key Responsibility | Key Files |
|-----------|-------------------|-----------|
| **SingleEpochBenchmark** | Measure timing for data loading, GPU transfer, forward, loss, backward, optimizer step. Exclude warmup batches. | `src/maou/app/utility/training_benchmark.py` |
| **TimingCallback** | Instrument training loop with `on_*_start/end` callbacks. Record measurements only after warmup. | `src/maou/app/learning/callbacks.py` |
| **StreamingFileSource** | Infra: Read feather files one at a time, convert to ColumnarBatch, explicitly release memory. Lazy row count scanning. | `src/maou/infra/file_system/streaming_file_source.py` |
| **StreamingDataSource** | Protocol: Abstract interface for app layer. Avoids direct infra dependency. | `src/maou/app/learning/streaming_dataset.py` |
| **StreamingKifDataset** | App: IterableDataset for Stage 3. Worker file distribution + intra-file shuffling + batch yielding. | `src/maou/app/learning/streaming_dataset.py` |
| **DataLoaderFactory.create_streaming_dataloaders()** | Create DataLoaders with `batch_size=None`, spawn context (Rust safety), dynamic worker limiting (files + memory). | `src/maou/app/learning/setup.py` |
| **benchmark-training CLI** | Expose warmup_batches, max_batches, enable_profiling, run_validation parameters. | `src/maou/infra/console/utility.py` |
| **benchmark_training() interface** | Validate and pass warmup_batches to TrainingBenchmarkConfig. | `src/maou/interface/utility_interface.py` |
| **learn-model CLI** | `--no-streaming` flag to disable streaming (default: enabled). Create StreamingFileSource objects for each stage. | `src/maou/infra/console/learn_model.py` |
| **learn interface** | Route to streaming or map-style dataset implementation based on `streaming` flag. | `src/maou/interface/learn.py` |

---

## Key Implementation Details to Remember

### Timing Measurement
- **Warmup**: First N batches (default 5) excluded from statistics
- **Measured Time**: Wall-clock time from end of warmup to epoch end
- **Per-Batch Breakdown**: Averaged over (total_batches - warmup_batches) measured batches
- **Formula**: `actual_average_batch_time = measured_time / measured_batches`

### Streaming Architecture
1. **Infrastructure (infra/)**: StreamingFileSource reads feather files with Rust FFI
2. **Protocol (app/)**: StreamingDataSource protocol abstracts storage
3. **Application (app/)**: StreamingKifDataset/Stage1/Stage2 implement PyTorch IterableDataset
4. **Interface (interface/)**: Learn interface switches between streaming and map-style modes
5. **Console (infra/console/)**: CLI provides `--no-streaming` flag

### Worker Safety
- **Multiprocessing Context**: `spawn` (NOT fork/forkserver) to safely call Rust/jemalloc code
- **File Distribution**: Round-robin per worker (worker_id % n_workers)
- **Memory Limiting**: Dynamic estimation based on file sizes + compression ratio
- **Timeouts**: 300 seconds (5 minutes) for spawn context startup + large file reads

### Performance Optimizations
- **Lazy Row Counting**: Only scan on `total_rows` property access
- **File Format Detection**: Arrow IPC File (fast metadata read) vs Stream (requires full read)
- **Stage 2 File Concatenation**: Group small files (10 at a time) before batching to reduce I/O
- **Worker Memory Logging**: Debug logs show RSS memory after first file load

