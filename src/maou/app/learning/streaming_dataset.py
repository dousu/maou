"""IterableDatasetベースのストリーミングデータセットモジュール．

ファイル単位ストリーミング + SOA(ColumnarBatch) + バッチ単位yieldを組み合わせ，
メモリ使用量を大幅に削減する．

DataLoaderは ``batch_size=None`` (自動バッチングOFF)で使用する．
これはWebDataset，Mosaic StreamingDataset等と同様のPyTorch推奨パターンである．
"""

from __future__ import annotations

import logging
import math
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import IterableDataset

from maou.domain.data.columnar_batch import ColumnarBatch

logger = logging.getLogger(__name__)


# ============================================================================
# StreamingDataSource Protocol
# ============================================================================


@runtime_checkable
class StreamingDataSource(Protocol):
    """ストリーミングデータソースのプロトコル．

    ``StreamingFileSource`` (infra層)がこのプロトコルを実装する．
    app層のStreamingDatasetはこのプロトコルに依存し，
    infra層への直接依存を回避する．
    """

    @property
    def file_paths(self) -> list[Path]:
        """ファイルパスのリスト(worker分割用)."""
        ...

    @property
    def total_rows(self) -> int:
        """全ファイルの合計行数."""
        ...

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        """ファイル単位で ``ColumnarBatch`` をyieldする."""
        ...


# ============================================================================
# Streaming Datasets
# ============================================================================


class StreamingKifDataset(IterableDataset):
    """IterableDataset版のKifDataset(バッチ単位yield)．

    Stage 3 (Policy + Value) 学習用．
    ファイル単位でストリーミング読み込みし，ファイル内シャッフル後，
    バッチサイズ分のTensorをyieldする．

    DataLoaderは ``batch_size=None`` (自動バッチングOFF)で使用する．
    """

    def __init__(
        self,
        *,
        streaming_source: StreamingDataSource,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """ストリーミングデータセットを初期化する．

        Args:
            streaming_source: ストリーミングデータソース
            batch_size: yieldするバッチサイズ
            shuffle: ファイル内レコードをシャッフルするか
            seed: シャッフル用の基本シード
        """
        super().__init__()
        self._source = streaming_source
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """エポック番号を設定し，シャッフルのシード管理に使用する．

        エポックごとに異なるシャッフル順序を保証するため，
        TrainingLoopからエポック開始時に呼び出す．

        Args:
            epoch: 現在のエポック番号(0-based)
        """
        self._epoch = epoch

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ]:
        """バッチ単位でTensorをyieldするイテレータ．

        persistent_workers=True対応のため，worker_info.seedを使用して
        エポックごとに異なるRNGを生成する．

        Yields:
            ((board_tensor, pieces_tensor), (move_label_tensor, result_value_tensor, legal_move_mask_tensor))
        """
        # persistent_workers対応: worker_info.seedはエポックごとに変わる
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            epoch_seed = worker_info.seed
        else:
            epoch_seed = (self._seed or 0) + self._epoch

        rng = np.random.default_rng(epoch_seed)

        for (
            columnar_batch
        ) in self._source.iter_files_columnar():
            yield from _yield_kif_batches(
                columnar_batch,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                rng=rng,
            )

    def __len__(self) -> int:
        """バッチ数を返す(tqdmプログレスバー用)."""
        return math.ceil(
            self._source.total_rows / self._batch_size
        )


class StreamingStage1Dataset(IterableDataset):
    """IterableDataset版のStage1Dataset(バッチ単位yield)．

    Stage 1 (Reachable Squares) 学習用．
    出力形式: (features, reachable_squares_target)．
    """

    def __init__(
        self,
        *,
        streaming_source: StreamingDataSource,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """ストリーミングStage1データセットを初期化する．

        Args:
            streaming_source: ストリーミングデータソース
            batch_size: yieldするバッチサイズ
            shuffle: ファイル内レコードをシャッフルするか
            seed: シャッフル用の基本シード
        """
        super().__init__()
        self._source = streaming_source
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """エポック番号を設定(シード管理用)．

        Args:
            epoch: 現在のエポック番号(0-based)
        """
        self._epoch = epoch

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
        ]
    ]:
        """バッチ単位でStage1用Tensorをyield．

        Yields:
            ((board_tensor, pieces_tensor), reachable_squares_tensor)
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            epoch_seed = worker_info.seed
        else:
            epoch_seed = (self._seed or 0) + self._epoch

        rng = np.random.default_rng(epoch_seed)

        for (
            columnar_batch
        ) in self._source.iter_files_columnar():
            yield from _yield_stage1_batches(
                columnar_batch,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                rng=rng,
            )

    def __len__(self) -> int:
        """バッチ数を返す(tqdmプログレスバー用)."""
        return math.ceil(
            self._source.total_rows / self._batch_size
        )


class StreamingStage2Dataset(IterableDataset):
    """IterableDataset版のStage2Dataset(バッチ単位yield)．

    Stage 2 (Legal Moves) 学習用．
    出力形式: (features, legal_moves_target)．
    """

    def __init__(
        self,
        *,
        streaming_source: StreamingDataSource,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """ストリーミングStage2データセットを初期化する．

        Args:
            streaming_source: ストリーミングデータソース
            batch_size: yieldするバッチサイズ
            shuffle: ファイル内レコードをシャッフルするか
            seed: シャッフル用の基本シード
        """
        super().__init__()
        self._source = streaming_source
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """エポック番号を設定(シード管理用)．

        Args:
            epoch: 現在のエポック番号(0-based)
        """
        self._epoch = epoch

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
        ]
    ]:
        """バッチ単位でStage2用Tensorをyield．

        Yields:
            ((board_tensor, pieces_tensor), legal_moves_tensor)
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            epoch_seed = worker_info.seed
        else:
            epoch_seed = (self._seed or 0) + self._epoch

        rng = np.random.default_rng(epoch_seed)

        for (
            columnar_batch
        ) in self._source.iter_files_columnar():
            yield from _yield_stage2_batches(
                columnar_batch,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                rng=rng,
            )

    def __len__(self) -> int:
        """バッチ数を返す(tqdmプログレスバー用)."""
        return math.ceil(
            self._source.total_rows / self._batch_size
        )


# ============================================================================
# Batch yield helpers
# ============================================================================


def _yield_kif_batches(
    columnar_batch: ColumnarBatch,
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Generator[
    tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    None,
    None,
]:
    """ColumnarBatchからKifDataset互換のバッチTensorをyieldする．

    Args:
        columnar_batch: 変換元のColumnarBatch
        batch_size: バッチサイズ
        shuffle: インデックスをシャッフルするか
        rng: 乱数生成器

    Yields:
        ((board_tensor, pieces_tensor), (move_label_tensor, result_value_tensor, legal_move_mask_tensor))
    """
    n = len(columnar_batch)
    if n == 0:
        return

    assert columnar_batch.move_label is not None
    assert columnar_batch.result_value is not None

    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = columnar_batch.slice(batch_indices)

        board_tensor = torch.from_numpy(
            batch.board_positions.copy()
        )
        pieces_tensor = torch.from_numpy(
            batch.pieces_in_hand.copy()
        )

        assert batch.move_label is not None
        assert batch.result_value is not None

        move_label_tensor = torch.from_numpy(
            batch.move_label.copy()
        )
        result_value_tensor = (
            torch.from_numpy(batch.result_value.copy())
            .float()
            .unsqueeze(1)
        )  # (N,) → (N, 1)
        legal_move_mask_tensor = torch.ones_like(
            move_label_tensor
        )

        yield (
            (board_tensor, pieces_tensor),
            (
                move_label_tensor,
                result_value_tensor,
                legal_move_mask_tensor,
            ),
        )


def _yield_stage1_batches(
    columnar_batch: ColumnarBatch,
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Generator[
    tuple[
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    None,
    None,
]:
    """ColumnarBatchからStage1Dataset互換のバッチTensorをyieldする．

    Args:
        columnar_batch: 変換元のColumnarBatch
        batch_size: バッチサイズ
        shuffle: インデックスをシャッフルするか
        rng: 乱数生成器

    Yields:
        ((board_tensor, pieces_tensor), reachable_squares_tensor)
    """
    n = len(columnar_batch)
    if n == 0:
        return

    assert columnar_batch.reachable_squares is not None

    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = columnar_batch.slice(batch_indices)

        board_tensor = torch.from_numpy(
            batch.board_positions.copy()
        )
        pieces_tensor = torch.from_numpy(
            batch.pieces_in_hand.copy()
        )

        assert batch.reachable_squares is not None
        # (N, 9, 9) → (N, 81) and convert to float for BCE
        reachable_tensor = (
            torch.from_numpy(batch.reachable_squares.copy())
            .flatten(start_dim=1)
            .float()
        )

        yield (
            (board_tensor, pieces_tensor),
            reachable_tensor,
        )


def _yield_stage2_batches(
    columnar_batch: ColumnarBatch,
    *,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Generator[
    tuple[
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    None,
    None,
]:
    """ColumnarBatchからStage2Dataset互換のバッチTensorをyieldする．

    Args:
        columnar_batch: 変換元のColumnarBatch
        batch_size: バッチサイズ
        shuffle: インデックスをシャッフルするか
        rng: 乱数生成器

    Yields:
        ((board_tensor, pieces_tensor), legal_moves_tensor)
    """
    n = len(columnar_batch)
    if n == 0:
        return

    assert columnar_batch.legal_moves_label is not None

    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = columnar_batch.slice(batch_indices)

        board_tensor = torch.from_numpy(
            batch.board_positions.copy()
        )
        pieces_tensor = torch.from_numpy(
            batch.pieces_in_hand.copy()
        )

        assert batch.legal_moves_label is not None
        legal_moves_tensor = torch.from_numpy(
            batch.legal_moves_label.copy()
        ).float()

        yield (
            (board_tensor, pieces_tensor),
            legal_moves_tensor,
        )
