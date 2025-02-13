import abc
import logging
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from maou.app.pre_process.transform import Transform


class DataSource:
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class KifDataset(Dataset):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        datasource: DataSource,
        transform: Optional[Transform] = None,
        pin_memory: bool,
        device: torch.device,
    ):
        self.__datasource = datasource
        self.transform: Optional[Transform] = transform
        self.device = device
        self.pin_memory = pin_memory
        self.logger.info(f"{len(self.__datasource)} samples")

    def __len__(self) -> int:
        return len(self.__datasource)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.transform is not None:
            # 最初にtransformしないパターン
            features, move_label, result_value = self.transform(
                hcp=self.__datasource[idx]["hcp"],
                move16=self.__datasource[idx]["bestMove16"],
                game_result=self.__datasource[idx]["gameResult"],
                eval=self.__datasource[idx]["eval"],
            )
            return (
                torch.from_numpy(features).to(self.device),
                (
                    torch.tensor(
                        move_label, dtype=torch.long, pin_memory=self.pin_memory
                    )
                    # one_hotのtargetsにいれるのでLongTEnsorに変換しておく
                    .long()
                    .to(self.device),
                    torch.tensor(
                        result_value, dtype=torch.float32, pin_memory=self.pin_memory
                    )
                    .reshape((1))
                    .to(self.device),
                ),
            )
        else:
            # 前処理済みのデータを使うパターン
            return (
                torch.from_numpy(self.__datasource[idx]["features"].copy()).to(
                    self.device
                ),
                (
                    torch.tensor(self.__datasource[idx]["moveLabel"])
                    # one_hotのtargetsにいれるのでLongTEnsorに変換しておく
                    .long()
                    .to(self.device),
                    torch.tensor(self.__datasource[idx]["resultValue"])
                    .reshape((1))
                    .to(self.device),
                ),
            )
