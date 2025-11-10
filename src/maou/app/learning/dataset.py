import abc
import logging
from dataclasses import dataclass
from collections.abc import Sized
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from maou.app.pre_process.label import MOVE_LABELS_NUM
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


@dataclass(frozen=True)
class _CachedSample:
    features: tuple[torch.Tensor, torch.Tensor]
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def byte_size(self) -> int:
        total = 0
        for tensor in (*self.features, *self.targets):
            total += tensor.element_size() * tensor.nelement()
        return total


class KifDataset(Dataset, Sized):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        datasource: DataSource,
        transform: Optional[Transform] = None,
        cache_transforms: bool = False,
    ):
        self.__datasource = datasource
        self.transform: Optional[Transform] = transform
        self.logger.info(f"{len(self.__datasource)} samples")
        self._cached_samples: Optional[list[_CachedSample]] = None
        self._cache_transforms_enabled: bool = False

        should_cache = cache_transforms and self.transform is not None
        if cache_transforms and self.transform is None:
            self.logger.warning(
                "cache_transforms is enabled but transform is None; skipping cache"
            )

        if should_cache:
            cached_samples: list[_CachedSample] = []
            total_bytes = 0
            for idx in range(len(self.__datasource)):
                cached_sample = self._create_cached_sample(
                    data=self.__datasource[idx]
                )
                cached_samples.append(cached_sample)
                total_bytes += cached_sample.byte_size

            self._cached_samples = cached_samples
            self._cache_transforms_enabled = True
            memory_mib = total_bytes / (1024**2)
            self.logger.info(
                "Cached %d transformed samples (%.2f MiB)",
                len(cached_samples),
                memory_mib,
            )

    def __len__(self) -> int:
        return len(self.__datasource)

    @property
    def cache_transforms_enabled(self) -> bool:
        """Return whether transformed samples are cached in memory."""

        return self._cache_transforms_enabled

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if self._cached_samples is not None:
            cached_sample = self._cached_samples[idx]
            return cached_sample.features, cached_sample.targets

        if self.transform is not None:
            # transformを使用するパターン（GPU転送の最適化）
            sample = self._create_cached_sample(
                data=self.__datasource[idx]
            )
            return sample.features, sample.targets
        else:
            # 前処理済みのデータを使うパターン（structured arrayから直接アクセス）
            data = self.__datasource[
                idx
            ]  # numpy structured array (0次元)

            # torch.from_numpy()を使用してゼロコピー変換（read-onlyの場合はcopy()で回避）
            # Dataset内ではCUDA操作を避け、DataLoaderのpin_memory機能を活用
            if data.dtype.names is None:
                raise ValueError("Preprocessed record lacks named fields")

            if "boardIdPositions" not in data.dtype.names:
                raise ValueError("Preprocessed record lacks boardIdPositions")

            if "piecesInHand" not in data.dtype.names:
                raise ValueError("Preprocessed record lacks piecesInHand")

            board_array = np.asarray(
                data["boardIdPositions"], dtype=np.uint8
            )
            board_tensor = torch.from_numpy(board_array.copy()).to(torch.long)
            pieces_in_hand_array = np.asarray(
                data["piecesInHand"], dtype=np.uint8
            )
            pieces_in_hand_tensor = torch.from_numpy(
                pieces_in_hand_array.copy()
            ).to(torch.float32)
            move_label_tensor = torch.from_numpy(
                data["moveLabel"].copy()
            ).to(torch.float32)
            result_value_tensor = torch.tensor(
                data["resultValue"].item(), dtype=torch.float32
            ).reshape((1))

            legal_move_mask_tensor = torch.ones_like(move_label_tensor)

            # DataLoaderのpin_memory機能と競合を避けるため、Dataset内ではCPUテンソルを返す
            # GPU転送はDataLoaderが自動的に処理する
            return (
                (board_tensor, pieces_in_hand_tensor),
                (
                    move_label_tensor,
                    result_value_tensor,
                    legal_move_mask_tensor,
                ),
            )
    def _create_cached_sample(self, *, data: np.ndarray) -> _CachedSample:
        if self.transform is None:
            raise ValueError("Transform is required to cache samples")

        (
            board_id_positions,
            pieces_in_hand,
            move_label,
            result_value,
            legal_move_mask,
        ) = self.transform(
            hcp=data["hcp"],
            move16=data["bestMove16"].item(),
            game_result=data["gameResult"].item(),
            eval=data["eval"].item(),
        )

        board_tensor = torch.from_numpy(board_id_positions.copy()).to(
            torch.long
        )
        pieces_in_hand_tensor = torch.from_numpy(
            pieces_in_hand.copy()
        ).to(torch.float32)
        legal_move_mask_tensor = torch.from_numpy(legal_move_mask).to(
            torch.float32
        )
        move_label_tensor = torch.nn.functional.one_hot(
            torch.tensor(move_label),
            num_classes=MOVE_LABELS_NUM,
        ).to(torch.float32)
        result_value_tensor = torch.tensor(
            result_value, dtype=torch.float32
        ).reshape((1))

        return _CachedSample(
            features=(board_tensor, pieces_in_hand_tensor),
            targets=(
                move_label_tensor,
                result_value_tensor,
                legal_move_mask_tensor,
            ),
        )
