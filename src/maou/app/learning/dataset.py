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
            [np.ndarray, int, int],
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        ],
    ):
        self.hcps: list[np.ndarray]
        self.transform = transform

        # 各局面を棋譜の区別なくフラットにいれておく
        self.hcpes = np.concatenate(
            [np.fromfile(path, dtype=HuffmanCodedPosAndEval) for path in paths]
        )
        self.logger.info(f"{self.__len__()} samples")

    def __len__(self) -> int:
        return len(self.hcpes)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.transform(
            self.hcpes[idx]["hcp"],
            self.hcpes[idx]["bestMove16"],
            self.hcpes[idx]["gameResult"],
        )
