import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from cshogi import HuffmanCodedPosAndEval  # type: ignore
from torch.utils.data import Dataset


class KifDataset(Dataset):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        paths: list[Path],
        transform: Callable[
            [np.ndarray, int, int, int],
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        ],
    ):
        # 各局面を棋譜の区別なくフラットにいれておく
        hcpes = np.concatenate(
            [np.fromfile(path, dtype=HuffmanCodedPosAndEval) for path in paths]
        )
        self.data: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = []
        for i, path in enumerate(paths):
            if i % 100000 == 0:
                self.logger.info(f"進捗: {i / len(paths) * 100}%")
            try:
                hcpes = np.fromfile(path, dtype=HuffmanCodedPosAndEval)
                self.data.extend(
                    [
                        transform(
                            hcpe["hcp"],
                            hcpe["bestMove16"],
                            hcpe["gameResult"],
                            hcpe["eval"],
                        )
                        for hcpe in hcpes
                    ]
                )
            except Exception:
                self.logger.error(f"path: {path}")
        self.logger.info(f"{len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.data[idx]
