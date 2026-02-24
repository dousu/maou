import abc
import logging
from collections.abc import Sized
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import torch
from numpy.typing import DTypeLike
from torch.utils.data import Dataset

from maou.app.pre_process.transform import Transform
from maou.domain.move.label import MOVE_LABELS_NUM


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
        self._cached_samples: Optional[list[_CachedSample]] = (
            None
        )
        self._cache_transforms_enabled: bool = False

        should_cache = (
            cache_transforms and self.transform is not None
        )
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
                raise ValueError(
                    "Preprocessed record lacks named fields"
                )

            if "boardIdPositions" not in data.dtype.names:
                raise ValueError(
                    "Preprocessed record lacks boardIdPositions"
                )

            if "piecesInHand" not in data.dtype.names:
                raise ValueError(
                    "Preprocessed record lacks piecesInHand"
                )

            board_tensor = self._structured_field_to_tensor(
                data,
                field_name="boardIdPositions",
                expected_dtype=np.uint8,
            )
            pieces_in_hand_tensor = (
                self._structured_field_to_tensor(
                    data,
                    field_name="piecesInHand",
                    expected_dtype=np.uint8,
                )
            )
            move_label_tensor = (
                self._structured_field_to_tensor(
                    data,
                    field_name="moveLabel",
                    expected_dtype=(np.float16, np.float32),
                )
            )
            result_value_tensor = torch.tensor(
                data["resultValue"].item(), dtype=torch.float32
            ).reshape((1))

            legal_move_mask_tensor = torch.ones_like(
                move_label_tensor
            )

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

    def _create_cached_sample(
        self, *, data: np.ndarray
    ) -> _CachedSample:
        if self.transform is None:
            raise ValueError(
                "Transform is required to cache samples"
            )

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

        board_tensor = self._numpy_to_tensor(
            board_id_positions,
            field_name="boardIdPositions",
            expected_dtype=np.uint8,
        )
        pieces_in_hand_tensor = self._numpy_to_tensor(
            pieces_in_hand,
            field_name="piecesInHand",
            expected_dtype=np.uint8,
        )
        legal_move_mask_tensor = self._numpy_to_tensor(
            legal_move_mask,
            field_name="legalMoveMask",
            expected_dtype=np.uint8,
        )
        move_label_tensor = torch.nn.functional.one_hot(
            torch.tensor(move_label),
            num_classes=MOVE_LABELS_NUM,
        )
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

    @staticmethod
    def _structured_field_to_tensor(
        record: np.ndarray,
        *,
        field_name: str,
        expected_dtype: DTypeLike | tuple[DTypeLike, ...],
    ) -> torch.Tensor:
        try:
            field = record[field_name]
        except (
            ValueError
        ) as exc:  # pragma: no cover - numpy raises ValueError
            msg = f"Preprocessed record lacks field `{field_name}`"
            raise ValueError(msg) from exc

        return KifDataset._numpy_to_tensor(
            field,
            field_name=field_name,
            expected_dtype=expected_dtype,
        )

    @staticmethod
    def _numpy_to_tensor(
        array: np.ndarray,
        *,
        field_name: str,
        expected_dtype: DTypeLike | tuple[DTypeLike, ...],
    ) -> torch.Tensor:
        np_array = np.asarray(array)
        expected_dtypes = (
            tuple(
                np.dtype(cast(DTypeLike, dtype))
                for dtype in expected_dtype
            )
            if isinstance(expected_dtype, tuple)
            else (np.dtype(cast(DTypeLike, expected_dtype)),)
        )
        if np_array.dtype not in expected_dtypes:
            expected_desc = (
                expected_dtypes[0].name
                if len(expected_dtypes) == 1
                else " or ".join(
                    dtype.name for dtype in expected_dtypes
                )
            )
            msg = (
                f"Field `{field_name}` must have dtype {expected_desc}, "
                f"got {np_array.dtype}"
            )
            raise TypeError(msg)
        if not np_array.flags.c_contiguous:
            msg = (
                f"Field `{field_name}` must be C-contiguous to enable zero-copy "
                "conversion"
            )
            raise ValueError(msg)
        if not np_array.flags.writeable:
            msg = (
                f"Field `{field_name}` was loaded as read-only. "
                "Ensure preprocessing files are opened via copy-on-write "
                "memory mapping so tensors can share storage."
            )
            raise ValueError(msg)
        return torch.from_numpy(np_array)


class Stage1Dataset(Dataset, Sized):
    """Dataset for Stage 1 (reachable squares) training.

    This dataset is used for the first stage of multi-stage training，
    where the model learns which board squares pieces can move to.
    The target is a 9×9 binary map indicating reachable squares.
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        datasource: DataSource,
    ):
        """Initialize Stage 1 dataset.

        Args:
            datasource: Data source providing Stage 1 training data
                with schema defined by get_stage1_dtype()
        """
        self.__datasource = datasource
        self.logger.info(
            f"Stage 1 Dataset: {len(self.__datasource)} samples"
        )

    def __len__(self) -> int:
        return len(self.__datasource)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],  # features
        torch.Tensor,  # target
    ]:
        """Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features，target):
                - features: (board_tensor，pieces_in_hand_tensor)
                    - board_tensor: (9，9) uint8 tensor
                    - pieces_in_hand_tensor: (14，) uint8 tensor
                - target: (81，) float32 tensor of binary labels
        """
        data = self.__datasource[idx]

        board_tensor = KifDataset._structured_field_to_tensor(
            data,
            field_name="boardIdPositions",
            expected_dtype=np.uint8,
        )
        pieces_in_hand_tensor = (
            KifDataset._structured_field_to_tensor(
                data,
                field_name="piecesInHand",
                expected_dtype=np.uint8,
            )
        )
        reachable_squares_tensor = (
            KifDataset._structured_field_to_tensor(
                data,
                field_name="reachableSquares",
                expected_dtype=np.uint8,
            )
            .flatten()
            .float()
        )  # (9，9) -> (81，) and convert to float for BCE

        return (
            (board_tensor, pieces_in_hand_tensor),
            reachable_squares_tensor,
        )


class Stage2Dataset(Dataset, Sized):
    """Dataset for Stage 2 (legal moves) training.

    This dataset is used for the second stage of multi-stage training，
    where the model learns which moves are legal in a given position.
    The target is a MOVE_LABELS_NUM-dimensional binary vector indicating legal moves.
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        datasource: DataSource,
    ):
        """Initialize Stage 2 dataset.

        Args:
            datasource: Data source providing Stage 2 training data
                with schema defined by get_stage2_dtype()
        """
        self.__datasource = datasource
        self.logger.info(
            f"Stage 2 Dataset: {len(self.__datasource)} samples"
        )

    def __len__(self) -> int:
        return len(self.__datasource)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],  # features
        torch.Tensor,  # target
    ]:
        """Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features，target):
                - features: (board_tensor，pieces_in_hand_tensor)
                    - board_tensor: (9，9) uint8 tensor
                    - pieces_in_hand_tensor: (14，) uint8 tensor
                - target: (MOVE_LABELS_NUM，) float32 tensor of binary labels
        """
        data = self.__datasource[idx]

        board_tensor = KifDataset._structured_field_to_tensor(
            data,
            field_name="boardIdPositions",
            expected_dtype=np.uint8,
        )
        pieces_in_hand_tensor = (
            KifDataset._structured_field_to_tensor(
                data,
                field_name="piecesInHand",
                expected_dtype=np.uint8,
            )
        )
        legal_moves_tensor = (
            KifDataset._structured_field_to_tensor(
                data,
                field_name="legalMovesLabel",
                expected_dtype=np.uint8,
            ).float()
        )  # Convert to float for BCE loss

        return (
            (board_tensor, pieces_in_hand_tensor),
            legal_moves_tensor,
        )
