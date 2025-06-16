import queue
import threading
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from maou.app.learning.dataset import KifDataset


class PrefetchDataset(Dataset):
    """
    Background threadでデータ変換を先行実行するDataset
    CPU→GPU転送のオーバーラップを実現
    """

    def __init__(
        self,
        base_dataset: KifDataset,
        prefetch_factor: int = 2,
        max_workers: int = 1,
    ) -> None:
        self.base_dataset = base_dataset
        self.prefetch_factor = prefetch_factor
        self.max_workers = max_workers

        self._cache: Dict[int, Any] = {}
        self._queue: queue.Queue[Tuple[int, Any]] = queue.Queue(maxsize=prefetch_factor)
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()

        # prefetch workerを開始
        for _ in range(max_workers):
            worker = threading.Thread(target=self._prefetch_worker, daemon=True)
            worker.start()
            self._workers.append(worker)

    def _prefetch_worker(self) -> None:
        """Background threadでデータを先行ロード"""
        idx = 0
        while not self._stop_event.is_set():
            try:
                if idx < len(self.base_dataset) and idx not in self._cache:
                    data = self.base_dataset[idx]
                    self._queue.put((idx, data), timeout=1.0)
                    idx += 1
                else:
                    threading.Event().wait(0.01)  # 短い待機
            except queue.Full:
                pass
            except Exception:
                break

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # キャッシュから取得を試行
        if idx in self._cache:
            return self._cache.pop(idx)

        # Queueから取得を試行
        try:
            while True:
                cached_idx, data = self._queue.get_nowait()
                if cached_idx == idx:
                    return data
                else:
                    self._cache[cached_idx] = data
        except queue.Empty:
            pass

        # フォールバック: 直接ロード
        return self.base_dataset[idx]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __del__(self) -> None:
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=1.0)
