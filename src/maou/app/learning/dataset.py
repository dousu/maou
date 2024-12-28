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
        self.data: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]
        for path in paths:
            try:
                hcpes = np.fromfile(path, dtype=HuffmanCodedPosAndEval)
                for hcpe in hcpes:
                    self.data.append(
                        transform(
                            hcpe["hcp"],
                            hcpe["bestMove16"],
                            hcpe["gameResult"],
                            hcpe["eval"],
                        )
                    )
            except Exception as e:
                self.logger.error(f"path: {path}")
                raise e
        self.logger.info(f"{len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.data[idx]
