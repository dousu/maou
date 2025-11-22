"""GPU prefetcher for overlapping data loading with computation.

このモジュールは，データローディングとGPU計算を並列化するための
GPU prefetchラッパーを提供する．

主な機能:
- バックグラウンドスレッドでのデータローディング
- CUDA streamを使用した非同期GPU転送
- 複数バッチのバッファリング
- データローディングボトルネックの解消

使用例:
    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(dataset, batch_size=256)
    >>> prefetcher = DataPrefetcher(loader, device='cuda:0')
    >>> for batch in prefetcher:
    ...     # GPU上で既に利用可能なバッチを処理
    ...     output = model(batch)
"""

import logging
import queue
import threading
from typing import Any, Dict, Iterator, Optional, Union

import torch
from torch.utils.data import DataLoader


def calculate_recommended_buffer_size(batch_size: int) -> int:
    """バッチサイズに基づいて推奨バッファサイズを計算する．

    大きなバッチサイズではより多くのバッファが必要．
    小さなバッチサイズでは少ないバッファで十分．

    Args:
        batch_size: バッチサイズ

    Returns:
        推奨バッファサイズ（バッチ数）
    """
    if batch_size <= 128:
        return 3
    elif batch_size <= 256:
        return 5
    elif batch_size <= 512:
        return 8
    elif batch_size <= 1024:
        return 12
    else:
        return 16


class DataPrefetcher:
    """DataLoaderのGPU prefetchラッパー．

    バックグラウンドスレッドでデータをロードし，CUDA streamを使用して
    非同期でGPUに転送する．複数バッチをバッファリングすることで，
    データローディングの待ち時間を隠蔽し，GPU稼働率を向上させる．

    Attributes:
        loader: ラップするDataLoader
        device: GPU device (例: 'cuda:0')
        buffer_size: バッファに保持するバッチ数
        pin_memory_override: DataLoaderのpin_memory設定を上書き
    """

    def __init__(
        self,
        loader: DataLoader,
        device: Union[str, torch.device],
        buffer_size: int = 3,
        pin_memory_override: Optional[bool] = None,
    ) -> None:
        """DataPrefetcherを初期化する．

        Args:
            loader: ラップするDataLoader
            device: データを転送するGPU device
            buffer_size: バッファに保持するバッチ数（デフォルト: 3）
            pin_memory_override: Trueの場合，DataLoaderのpin_memoryを強制的に有効化
        """
        self.loader = loader
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.logger = logging.getLogger(__name__)

        # CUDA streamを作成（非同期転送用）
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream(device=self.device)
        else:
            self.stream = None
            self.logger.warning(
                "DataPrefetcher is optimized for CUDA devices. "
                f"Current device: {self.device}"
            )

        # pin_memoryの設定を確認・上書き
        if pin_memory_override is not None and pin_memory_override:
            if hasattr(loader, "pin_memory"):
                if not loader.pin_memory:
                    self.logger.info(
                        "Overriding DataLoader pin_memory to True for better "
                        "GPU transfer performance"
                    )
                    loader.pin_memory = True
            else:
                self.logger.warning(
                    "DataLoader does not have pin_memory attribute"
                )

        # バッファリング用のキュー
        self.queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.exception: Optional[Exception] = None

        self.logger.debug(
            f"DataPrefetcher initialized: device={device}, "
            f"buffer_size={buffer_size}"
        )

    def _transfer_to_device(
        self, batch: Any, non_blocking: bool = True
    ) -> Any:
        """バッチデータをGPUに転送する．

        テンソルを再帰的に探索し，全てのテンソルをGPUに転送する．
        辞書，リスト，タプルなどのネストした構造にも対応．

        Args:
            batch: 転送するバッチデータ
            non_blocking: 非同期転送を使用するかどうか

        Returns:
            GPUに転送されたバッチデータ
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(
                self.device, non_blocking=non_blocking
            )
        elif isinstance(batch, dict):
            return {
                k: self._transfer_to_device(v, non_blocking)
                for k, v in batch.items()
            }
        elif isinstance(batch, list):
            return [
                self._transfer_to_device(item, non_blocking)
                for item in batch
            ]
        elif isinstance(batch, tuple):
            return tuple(
                self._transfer_to_device(item, non_blocking)
                for item in batch
            )
        else:
            # テンソルでない場合はそのまま返す
            return batch

    def _loader_thread(self) -> None:
        """バックグラウンドスレッドでDataLoaderからバッチを読み込む．

        DataLoaderからバッチを読み込み，GPUに転送してキューに追加する．
        CUDA streamを使用して非同期転送を行う．
        """
        try:
            for batch in self.loader:
                if self.stop_event.is_set():
                    break

                # CUDA streamを使用して非同期転送
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        gpu_batch = self._transfer_to_device(
                            batch, non_blocking=True
                        )
                else:
                    # CPU deviceの場合は通常の転送
                    gpu_batch = self._transfer_to_device(
                        batch, non_blocking=False
                    )

                # キューに追加（キューが満杯の場合は待機）
                self.queue.put(gpu_batch)

        except Exception as e:
            self.logger.error(f"Error in loader thread: {e}")
            self.exception = e
        finally:
            # 終了シグナルをキューに追加
            self.queue.put(None)

    def __iter__(self) -> Iterator[Any]:
        """イテレータを返す．

        バックグラウンドスレッドを開始し，prefetchを実行する．

        Yields:
            GPUに転送済みのバッチデータ

        Raises:
            Exception: ローダースレッドで例外が発生した場合
        """
        # 前回のスレッドが残っている場合は停止
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()

        # 初期化
        self.stop_event.clear()
        self.exception = None
        self.queue = queue.Queue(maxsize=self.buffer_size)

        # バックグラウンドスレッドを開始
        self.thread = threading.Thread(
            target=self._loader_thread,
            daemon=True,
        )
        self.thread.start()

        # キューからバッチを取得して返す
        while True:
            # ローダースレッドで例外が発生していないか確認
            if self.exception is not None:
                raise self.exception

            # キューからバッチを取得（タイムアウト付き）
            try:
                batch = self.queue.get(timeout=120)
            except queue.Empty:
                # タイムアウト時は例外をチェック
                if self.exception is not None:
                    raise self.exception
                raise TimeoutError(
                    "Timeout waiting for batch from DataPrefetcher"
                )

            # 終了シグナル（None）を受け取ったら終了
            if batch is None:
                break

            # CUDA streamの同期を待つ
            if self.stream is not None:
                self.stream.synchronize()

            yield batch

    def __len__(self) -> int:
        """DataLoaderの長さを返す．

        Returns:
            バッチ数
        """
        return len(self.loader)

    def shutdown(self) -> None:
        """バックグラウンドスレッドを停止する．

        通常はwith文を使用するか，明示的に呼び出す必要がある．
        """
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            # キューをクリア
            try:
                while not self.queue.empty():
                    self.queue.get_nowait()
            except queue.Empty:
                pass
            self.thread.join(timeout=5)

    def __enter__(self) -> "DataPrefetcher":
        """コンテキストマネージャのエントリ．

        Returns:
            self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """コンテキストマネージャの終了時にスレッドを停止する．"""
        self.shutdown()


def create_prefetched_loader(
    loader: DataLoader,
    device: Union[str, torch.device],
    buffer_size: int = 3,
    pin_memory_override: bool = True,
) -> DataPrefetcher:
    """DataPrefetcherを作成するヘルパー関数．

    Args:
        loader: ラップするDataLoader
        device: データを転送するGPU device
        buffer_size: バッファに保持するバッチ数（デフォルト: 3）
        pin_memory_override: pin_memoryを強制的に有効化するかどうか

    Returns:
        DataPrefetcherインスタンス
    """
    return DataPrefetcher(
        loader=loader,
        device=device,
        buffer_size=buffer_size,
        pin_memory_override=pin_memory_override,
    )
