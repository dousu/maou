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
from typing import Any, ClassVar, Iterator, Optional, Union

import torch
from torch.utils.data import DataLoader


def calculate_recommended_buffer_size(
    batch_size: int,
    num_workers: int = 0,
) -> int:
    """バッチサイズとワーカー数に基づいて推奨バッファサイズを計算する．

    小〜中バッチではバッファを増やして転送レイテンシを隠蔽し，
    大バッチではバッチあたりのメモリコストが大きく，かつGPU計算時間が
    十分に長いため，少ないバッファで転送を隠蔽できる．

    ワーカー数が多い場合はワーカー+pin_memory+バッファの合計メモリが
    増大するため，大バッチ時にバッファサイズを抑制する．
    小バッチはバッファあたりのメモリコストが低いため抑制不要．

    Args:
        batch_size: バッチサイズ
        num_workers: DataLoaderのワーカー数(0で抑制なし)

    Returns:
        推奨バッファサイズ（バッチ数，最小2）
    """
    if batch_size <= 128:
        base = 3
    elif batch_size <= 256:
        base = 5
    elif batch_size <= 512:
        base = 8
    elif batch_size <= 1024:
        base = 12
    elif batch_size <= 2048:
        base = 8
    else:
        base = 4

    # ワーカー数による抑制: 大バッチ(≥512)時のみ適用
    # 小バッチはバッファあたりのメモリコストが低いため抑制不要
    if batch_size >= 512:
        if num_workers >= 8:
            return max(2, base // 2)
        elif num_workers >= 4:
            return max(2, base * 2 // 3)

    return base


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

    #: 通常バッチのタイムアウト(秒)
    DEFAULT_TIMEOUT: ClassVar[float] = 120.0
    #: 初回バッチのタイムアウト(秒):
    #: spawnワーカーの初期化+ファイル読込を考慮．
    #: ワーカーキャップ(合計8)により初期化時間が短縮されたため300sから180sに変更．
    FIRST_BATCH_TIMEOUT: ClassVar[float] = 180.0

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
        self.stream: torch.cuda.Stream | None
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream(device=self.device)
        else:
            self.stream = None
            self.logger.warning(
                "DataPrefetcher is optimized for CUDA devices. "
                f"Current device: {self.device}"
            )

        # pin_memoryの設定を確認・上書き
        if (
            pin_memory_override is not None
            and pin_memory_override
        ):
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
        self.queue: queue.Queue = queue.Queue(
            maxsize=buffer_size
        )
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
        初回バッチにはワーカー初期化やファイル読込を考慮した
        長めのタイムアウトを適用する．

        Yields:
            GPUに転送済みのバッチデータ

        Raises:
            Exception: ローダースレッドで例外が発生した場合
            TimeoutError: バッチ取得がタイムアウトした場合
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

        is_first_batch = True

        # キューからバッチを取得して返す
        while True:
            # ローダースレッドで例外が発生していないか確認
            if self.exception is not None:
                raise self.exception

            # 初回バッチには長めのタイムアウトを適用
            timeout = (
                self.FIRST_BATCH_TIMEOUT
                if is_first_batch
                else self.DEFAULT_TIMEOUT
            )

            # キューからバッチを取得（タイムアウト付き）
            try:
                batch = self.queue.get(timeout=timeout)
            except queue.Empty:
                # タイムアウト時は診断情報を出力
                self._log_timeout_diagnostics(is_first_batch)
                if self.exception is not None:
                    raise self.exception
                raise TimeoutError(
                    f"Timeout ({timeout:.0f}s) waiting for "
                    f"{'first ' if is_first_batch else ''}"
                    f"batch from DataPrefetcher. "
                    f"This may indicate slow data loading "
                    f"in spawn workers rather than a deadlock. "
                    f"Try --dataloader-workers 0 to test "
                    f"without spawn."
                )

            # 終了シグナル（None）を受け取ったら終了
            if batch is None:
                # 残留アイテムのドレイン（複数のNoneセンチネル対策）
                while True:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break
                # ローダースレッドで例外が発生していた場合は伝播
                if self.exception is not None:
                    raise self.exception
                break

            # CUDA streamの同期を待つ
            if self.stream is not None:
                self.stream.synchronize()

            is_first_batch = False
            yield batch

    def _log_timeout_diagnostics(
        self, is_first_batch: bool
    ) -> None:
        """タイムアウト発生時の診断情報をログ出力する．

        メモリ使用量，ワーカー状態，GPUメモリ等を出力し，
        タイムアウトの原因特定を支援する．

        Args:
            is_first_batch: 初回バッチでのタイムアウトかどうか
        """
        diag_lines = [
            "=== DataPrefetcher Timeout Diagnostics ===",
            f"First batch: {is_first_batch}",
            f"Buffer size: {self.buffer_size}",
            f"Queue size: {self.queue.qsize()}/{self.buffer_size}",
            f"Loader thread alive: "
            f"{self.thread.is_alive() if self.thread else 'N/A'}",
            f"Stop event set: {self.stop_event.is_set()}",
            f"Exception in loader: {self.exception}",
        ]

        # システムメモリ情報
        try:
            import psutil

            vm = psutil.virtual_memory()
            diag_lines.append(
                f"System memory: "
                f"total={vm.total // (1024**2)}MB, "
                f"available={vm.available // (1024**2)}MB, "
                f"percent={vm.percent}%"
            )
        except ImportError:
            try:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)
                rss_mb = usage.ru_maxrss / 1024
                diag_lines.append(f"Peak RSS: {rss_mb:.0f}MB")
            except Exception:
                pass

        # GPUメモリ情報
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (
                    1024**2
                )
                reserved = torch.cuda.memory_reserved() / (
                    1024**2
                )
                diag_lines.append(
                    f"GPU memory: "
                    f"allocated={allocated:.0f}MB, "
                    f"reserved={reserved:.0f}MB"
                )
            except Exception:
                pass

        # DataLoader情報
        if hasattr(self.loader, "num_workers"):
            diag_lines.append(
                f"DataLoader workers: {self.loader.num_workers}"
            )
        if hasattr(self.loader, "pin_memory"):
            diag_lines.append(
                f"DataLoader pin_memory: "
                f"{self.loader.pin_memory}"
            )

        for line in diag_lines:
            self.logger.error(line)

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
    pin_memory_override: Optional[bool] = None,
) -> DataPrefetcher:
    """DataPrefetcherを作成するヘルパー関数．

    Args:
        loader: ラップするDataLoader
        device: データを転送するGPU device
        buffer_size: バッファに保持するバッチ数（デフォルト: 3）
        pin_memory_override: pin_memoryの上書き設定(Noneで上書きなし)

    Returns:
        DataPrefetcherインスタンス
    """
    return DataPrefetcher(
        loader=loader,
        device=device,
        buffer_size=buffer_size,
        pin_memory_override=pin_memory_override,
    )
