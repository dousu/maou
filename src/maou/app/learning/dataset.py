import abc
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from cshogi import HuffmanCodedPosAndEval  # type: ignore
from torch.utils.data import Dataset

from maou.app.pre_process.transform import Transform


class DataSource(Sequence, metaclass=abc.ABCMeta):
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
        paths: list[Path],
        transform: Optional[Transform] = None,
        pin_memory: bool,
        device: torch.device,
        datasource: Optional[DataSource] = None,
    ):
        self.__datasource = datasource
        self.transform: Optional[Transform] = transform
        self.device = device
        self.pin_memory = pin_memory
        # TODO: ここをデータソースがBQでもファイルでもいいように抽象化する
        # 各局面を棋譜の区別なくフラットにいれておく
        self.hcpes = np.concatenate(
            [np.fromfile(path, dtype=HuffmanCodedPosAndEval) for path in paths]
        )
        self.logger.info(f"hcpes shape: {self.hcpes.shape}")
        self.logger.info(f"hcpes dtype: {self.hcpes.dtype}")
        self.logger.info(f"{len(self.hcpes)} samples")

        # デバッグ用のコード
        # self.paths: list[Path] = []
        # for path in paths:
        #     hcpes = np.fromfile(path, dtype=HuffmanCodedPosAndEval)
        #     self.paths.extend([path for i in range(len(hcpes))])

        # 最初にtransformするパターン
        # これにするとなぜかプログラムが落ちてしまうのでデバッグ用途で残しておく
        # たぶんメモリが原因
        # # 各局面を棋譜の区別なくフラットにいれておく
        # hcpes = np.concatenate(
        #     [np.fromfile(path, dtype=HuffmanCodedPosAndEval) for path in paths]
        # )
        # self.transformed_data: list[
        #     tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        # ] = []
        # for i, path in enumerate(paths):
        #     if i % 100000 == 0:
        #         self.logger.info(f"進捗: {i / len(paths) * 100}%")
        #     try:
        #         hcpes = np.fromfile(path, dtype=HuffmanCodedPosAndEval)
        #         self.transformed_data.extend(
        #             [
        #                 transform(
        #                     hcpe["hcp"],
        #                     hcpe["bestMove16"],
        #                     hcpe["gameResult"],
        #                     hcpe["eval"],
        #                 )
        #                 for hcpe in hcpes
        #             ]
        #         )
        #     except Exception:
        #         self.logger.error(f"path: {path}")
        # self.logger.info(f"{len(self.transformed_data)} samples")

    def __len__(self) -> int:
        # 最初にtransformしないパターン
        return len(self.hcpes)

        # 最初にtransformするパターン
        # return len(self.transformed_data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.transform is not None:
            # 最初にtransformしないパターン
            features, move_label, result_value = self.transform(
                hcp=self.hcpes[idx]["hcp"],
                move16=self.hcpes[idx]["bestMove16"],
                game_result=self.hcpes[idx]["gameResult"],
                eval=self.hcpes[idx]["eval"],
            )
            return (
                torch.from_numpy(features).to(self.device),
                (
                    torch.tensor(
                        move_label, dtype=torch.long, pin_memory=self.pin_memory
                    ).to(self.device),
                    torch.tensor(
                        result_value, dtype=torch.float32, pin_memory=self.pin_memory
                    )
                    .reshape((1))
                    .to(self.device),
                ),
            )

            # デバッグ用のコード
            # try:
            #     return self.transform(
            #         hcp=self.hcpes[idx]["hcp"],
            #         move16=self.hcpes[idx]["bestMove16"],
            #         game_result=self.hcpes[idx]["gameResult"],
            #         eval=self.hcpes[idx]["eval"],
            #     )
            # except Exception as e:
            #     self.logger.error(f"error: {self.paths[idx]}")
            #     raise e

            # 最初にtransformするパターン
            # return self.transformed_data[idx]
        else:
            # 前処理済みのデータを使うパターン
            return (
                torch.from_numpy(self.hcpes[idx]["features"]).to(self.device),
                (
                    torch.tensor(self.hcpes[idx]["moveLabel"]).to(self.device),
                    torch.tensor(self.hcpes[idx]["resultValue"])
                    .reshape((1))
                    .to(self.device),
                ),
            )
