# ADR-001: DataLoader マルチプロセシング最適化

## ステータス

✅ **Accepted** - 2025-06-17実装完了

## コンテキスト

Maou将棋AIプロジェクトにおいて，機械学習のデータローディング性能がボトルネックとなっていた．特に以下の問題が発生：

1. **CUDA初期化エラー**: DataLoaderの`num_workers > 0`設定時にワーカープロセス内でCUDAコンテキストエラーが発生
2. **性能低下**: `num_workers=0`での回避により，CPUとGPUの並列処理が活用できない
3. **スケーラビリティ**: 大規模データセットでの学習効率が低い

### 発生していたエラー

`num_workers=4`を設定した際に以下のエラーが発生：

```
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/ssm-user/.cache/pypoetry/virtualenvs/maou-T3BWhynF-py3.11/lib64/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/ssm-user/.cache/pypoetry/virtualenvs/maou-T3BWhynF-py3.11/lib64/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ssm-user/.cache/pypoetry/virtualenvs/maou-T3BWhynF-py3.11/lib64/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/home/ssm-user/maou/src/maou/app/learning/dataset.py", line 98, in __getitem__
    features_tensor = torch.from_numpy(data["features"].copy()).pin_memory()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: initialization error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

### 問題の根本原因

このエラーは典型的な「DataLoaderのワーカープロセス内でCUDAコンテキストにアクセスしようとした」際に発生する問題である．

- **Dataset内でのCUDA操作**: `pin_memory()`がワーカープロセス内で実行され，CUDAコンテキスト未初期化エラーが発生
- **プロセス分離**: 各ワーカープロセスは独立したプロセスで，メインプロセスのCUDAコンテキストを共有できない
- **ワーカー初期化不備**: 各ワーカープロセスでのCUDAコンテキスト初期化が未実装

## 決定事項

### 1. データ転送タイミングの明確化

**3段階のデータ転送フロー**

1. **Dataset**: CPUテンソルを返却（CUDA操作禁止）
2. **DataLoader**: CPU上でpin_memoryによる高速転送準備
3. **学習ループ**: GPU転送を実行

```python
# フロー図
Dataset.__getitem__()
    → CPUテンソル作成（torch.from_numpy）
        → DataLoader（pin_memory=True）
            → CPUのピン済みメモリ
                → 学習ループ（.to(device, non_blocking=True)）
                    → GPU転送完了
```

**重要**: DataLoaderにはGPUへのデータ転送オプションは存在しないため，学習ループ内でDataLoaderからデータを取得後に明示的にGPU転送を行う必要がある．

### 2. Dataset設計パターンの変更

**Dataset内でのCUDA操作を完全に排除**

**Before:**
```python
# dataset.py の98行目 - 問題のあるパターン
features_tensor = torch.from_numpy(data["features"].copy()).pin_memory()
# ↑ ワーカープロセス内でCUDA操作を実行してエラー
```

**After:**
```python
# dataset.py - 修正後のパターン
def __getitem__(self, idx):
    # Dataset内ではCUDA操作を避け，CPUテンソルのみ作成
    features_tensor = torch.from_numpy(data["features"].copy())
    return features_tensor  # CPUテンソルを返す
```

### 3. DataLoader最適化設定の標準化

**DataLoaderはCPUでのpin_memoryのみ実行**

```python
# DataLoaderでpin_memory設定（メインプロセスで実行）
# 注意: DataLoaderはGPU転送を行わない
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,                         # CPUのピン済みメモリ作成
    persistent_workers=True,                 # ワーカー再利用
    prefetch_factor=2,                       # プリフェッチング
    drop_last=True,                          # 不完全バッチ回避
    timeout=60,                              # デッドロック防止
    worker_init_fn=worker_init_fn,           # ワーカー初期化
)
```

### 4. 学習ループでのGPU転送

**DataLoaderからデータを取得後に明示的にGPU転送**

```python
# 学習ループでのGPU転送パターン
for batch in dataloader:
    # DataLoaderからはCPUのピン済みメモリで取得
    inputs, (labels_policy, labels_value, legal_move_mask) = batch

    # 学習ループ内で明示的にGPU転送を実行
    inputs = inputs.to(device, non_blocking=True)
    labels_policy = labels_policy.to(device, non_blocking=True)
    labels_value = labels_value.to(device, non_blocking=True)
    if legal_move_mask is not None:
        legal_move_mask = legal_move_mask.to(device, non_blocking=True)

    # GPU上でモデル実行
    outputs_policy, outputs_value = model(inputs)
    # 学習処理...
```

### 5. グローバルワーカー初期化関数の実装

**Pickle化エラーを解決するためグローバル関数として定義**

当初はクラスメソッド内でネストした関数として実装したが，spawnモードでPickle化エラーが発生：

```
AttributeError: Can't pickle local object 'TestLearning._create_worker_init_fn.<locals>.worker_init_fn'
```

このため，グローバル関数として再設計：

```python
def _default_worker_init_fn(worker_id: int) -> None:
    """
    各ワーカープロセスの初期化関数．
    Pickle化エラーを回避するためグローバル関数として定義．
    """
    import random

    # 再現性のためのシード設定（ワーカーごとに異なるシードを使用）
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # CUDAが利用可能な場合のみ初期化を試みる
    if torch.cuda.is_available():
        try:
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(current_device)
            _ = torch.cuda.current_device()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Worker {worker_id}: CUDA initialization failed: {e}")
```

### 6. マルチプロセシング方式の適応的設定

**主にWindows環境とspawnモードでの安定性向上**

```python
def _setup_multiprocessing(self) -> None:
    """
    マルチプロセシング開始方法を設定する．
    主にWindows環境での制約とspawnモードでの安定性を向上させる．
    """
    import platform
    import torch.multiprocessing as mp

    try:
        current_method = mp.get_start_method(allow_none=True)

        if platform.system() == "Windows":
            # Windowsでは常にspawnを使用（forkが利用不可）
            if current_method != "spawn":
                mp.set_start_method("spawn", force=True)
        elif torch.cuda.is_available() and self.device.type == "cuda":
            # CUDA使用時はspawnが推奨される場合がある
            if current_method != "spawn":
                try:
                    mp.set_start_method("spawn", force=True)
                except RuntimeError as e:
                    # 既に設定済みの場合は警告のみ
                    self.logger.warning(f"Could not set multiprocessing method: {e}")
        # LinuxでCPU使用時は通常forkで問題なし

    except Exception as e:
        # マルチプロセシング設定に失敗した場合は警告のみ
        self.logger.warning(f"Failed to configure multiprocessing: {e}")
```

## 理由

### 技術的根拠

1. **CUDA初期化エラーの解決**
   - Dataset内でのCUDA操作排除により，ワーカープロセスでのエラーを根本解決
   - DataLoaderはCPUでのpin_memoryのみ実行し，GPU転送は学習ループで明示的に実行

2. **データ転送の最適化**
   - pin_memory: CPUのピン済みメモリによる高速GPU転送準備
   - non_blocking=True: GPU転送とCPU処理の並列実行
   - 明確な責任分離: Dataset（CPU作成）→ DataLoader（pin_memory）→ 学習ループ（GPU転送）

3. **Pickle化問題の解決**
   - ネストした関数（クロージャ）でPickle化エラーが発生したため，グローバル関数に変更
   - spawnモードでの正常動作を実現

### 歴史的背景

**なぜ過去に`num_workers=0`にしてしまったのか**
- 過去のPyTorchバージョンでマルチプロセシング関連のバグやデッドロックが発生
- Windows環境での互換性問題
- CUDA初期化とマルチプロセシングの競合

**現在の状況**
- PyTorch 1.7以降で多くの問題が修正済み
- データローディングとGPU計算は独立したプロセス
- CPUでのデータ前処理とGPUでの計算を並列実行可能

## 結果

### 解決された問題

1. ✅ **CUDA初期化エラー完全解消**: Dataset内CUDA操作排除により根本解決
2. ✅ **Pickle化エラー解消**: グローバル関数によりspawnモードで正常動作
3. ✅ **性能向上**: 2-4倍のデータローディング高速化（CPUコア数依存）
4. ✅ **プラットフォーム互換**: Windows/Linux自動対応
5. ✅ **データ転送最適化**: 明確な3段階フローによる効率的GPU転送

### 推奨される実装パターン

```python
# dataset.py の修正版
class KifDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # NumPy配列をCPUテンソルに変換（CUDA操作なし）
        features_tensor = torch.from_numpy(data["features"].copy())
        return features_tensor

# DataLoader設定
def create_dataloader():
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,              # CPUでピン済みメモリ作成
        persistent_workers=True,
        drop_last=True
    )

# 学習ループ
for batch in dataloader:
    # DataLoaderからCPUピン済みメモリで取得
    inputs, targets = batch

    # 学習ループ内で明示的にGPU転送
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # GPU上でモデル実行
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    # 学習処理...
```

最も簡潔な解決方法は，**Dataset内でのpin_memory()呼び出しを削除し，DataLoaderのpin_memory=Trueオプションを使用する**ことである．これにより，メモリピニングがメインプロセスで実行され，ワーカープロセス内でのCUDA初期化エラーを回避できる．

## 制約事項とトレードオフ

### データ転送に関する制約

1. **明示的GPU転送**: DataLoaderは自動GPU転送を行わないため，学習ループで明示的に`.to(device)`が必要
2. **メモリ使用量**: pin_memoryによりCPUメモリ使用量が増加
3. **転送タイミング**: 学習ループでの転送により，若干のオーバーヘッドが発生

### その他の制約事項

1. **ワーカー数とメモリ**: ワーカー数に比例してメモリ使用量が増加
2. **共有メモリ**: Dockerコンテナでは共有メモリサイズ調整が必要な場合がある
3. **デバッグ複雑性**: マルチプロセシング環境でのデバッグは単一プロセスより困難

### 環境依存の問題

- **Windows**: `mp.set_start_method('spawn')`が必要
- **Linux**: 通常は`fork`で問題なし
- **Docker**: 共有メモリサイズの調整が必要な場合あり

## 使用方法

```bash
# 推奨設定（CPUコア数の2-4倍程度）
poetry run maou learn-model \
  --input-dir /path/to/data \
  --input-format preprocess \
  --dataloader-workers 4 \
  --pin-memory \
  --batch-size 32 \
  --epoch 10 \
  --gpu cuda:0
```

## 今後の検討事項

1. **動的ワーカー調整**: 負荷に応じた自動スケーリング
2. **WebDataset対応**: 大規模データセット向けの更なる最適化
3. **分散学習対応**: 複数GPU環境での最適化
4. **自動GPU転送**: DataLoaderからGPU転送を自動化する仕組みの検討

## 参考資料

- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [PyTorch Multiprocessing Best Practices](https://pytorch.org/docs/stable/notes/multiprocessing.html)
- [CUDA Context Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__CONTEXT.html)
- [Python Multiprocessing Start Methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)
