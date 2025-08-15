import abc
import logging
from collections.abc import Sized
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from maou.app.pre_process.transform import Transform


class DataSource:
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> np.ndarray:
        """
        指定されたインデックスのレコードをnumpy structured arrayとして返す

        Returns:
            np.ndarray: structured arrayの単一レコード（0次元配列）
        """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class KifDataset(Dataset, Sized):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        datasource: DataSource,
        transform: Optional[Transform] = None,
    ):
        self.__datasource = datasource
        self.transform: Optional[Transform] = transform
        self.logger.info(f"{len(self.__datasource)} samples")

    def __len__(self) -> int:
        return len(self.__datasource)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if self.transform is not None:
            # transformを使用するパターン（GPU転送の最適化）
            data = self.__datasource[
                idx
            ]  # numpy structured array (0次元)
            (
                features,
                move_label,
                result_value,
                legal_move_mask,
            ) = self.transform(
                hcp=data["hcp"],
                move16=data["bestMove16"].item(),
                game_result=data["gameResult"].item(),
                eval=data["eval"].item(),
            )

            # torch.from_numpy()を使用してゼロコピー変換
            # Dataset内ではCUDA操作を避け、DataLoaderのpin_memory機能を活用
            features_tensor = torch.from_numpy(features).to(
                torch.float32
            )
            legal_move_mask_tensor = torch.from_numpy(
                legal_move_mask
            ).to(torch.float32)
            move_label_tensor = torch.tensor(
                move_label, dtype=torch.long
            )
            result_value_tensor = torch.tensor(
                result_value, dtype=torch.float32
            ).reshape((1))

            # DataLoaderのpin_memory機能と競合を避けるため、Dataset内ではCPUテンソルを返す
            # GPU転送はDataLoaderが自動的に処理する
            return (
                features_tensor,
                (
                    move_label_tensor,
                    result_value_tensor,
                    legal_move_mask_tensor,
                ),
            )
        else:
            # 前処理済みのデータを使うパターン（structured arrayから直接アクセス）
            data = self.__datasource[
                idx
            ]  # numpy structured array (0次元)

            # torch.from_numpy()を使用してゼロコピー変換（read-onlyの場合はcopy()で回避）
            # Dataset内ではCUDA操作を避け、DataLoaderのpin_memory機能を活用
            features_tensor = torch.from_numpy(
                data["features"].copy()
            ).to(torch.float32)
            legal_move_mask_tensor = torch.from_numpy(
                data["legalMoveMask"].copy()
            ).to(torch.float32)
            move_label_tensor = torch.tensor(
                data["moveLabel"].item(), dtype=torch.long
            )
            result_value_tensor = torch.tensor(
                data["resultValue"].item(), dtype=torch.float32
            ).reshape((1))

            # DataLoaderのpin_memory機能と競合を避けるため、Dataset内ではCPUテンソルを返す
            # GPU転送はDataLoaderが自動的に処理する
            return (
                features_tensor,
                (
                    move_label_tensor,
                    result_value_tensor,
                    legal_move_mask_tensor,
                ),
            )
