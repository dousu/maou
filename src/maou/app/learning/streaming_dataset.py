"""IterableDatasetベースのストリーミングデータセットモジュール．

ファイル単位ストリーミング + SOA(ColumnarBatch) + バッチ単位yieldを組み合わせ，
メモリ使用量を大幅に削減する．

DataLoaderは ``batch_size=None`` (自動バッチングOFF)で使用する．
これはWebDataset，Mosaic StreamingDataset等と同様のPyTorch推奨パターンである．
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import IterableDataset

from maou.app.learning.setup import _log_worker_memory
from maou.domain.data.columnar_batch import ColumnarBatch

logger = logging.getLogger(__name__)


# ============================================================================
# Worker file resolution helper
# ============================================================================


def _resolve_worker_files(
    source: StreamingDataSource,
    shuffle: bool,
    epoch_seed: int,
) -> list[Path]:
    """workerファイル分割 + ファイル順シャッフルを行う共通関数．

    マルチワーカー環境でファイルをラウンドロビン方式で各workerに分配し，
    エポックごとにファイル順をシャッフルすることでworker間の分布偏りを防ぐ．

    Args:
        source: ストリーミングデータソース
        shuffle: ファイル順をシャッフルするか
        epoch_seed: エポックシード(シャッフル用)

    Returns:
        このworkerが担当するファイルパスのリスト
    """
    file_paths = source.file_paths

    # エポックごとのファイル順シャッフル
    if shuffle:
        file_rng = np.random.default_rng(epoch_seed + 1_000_000)
        file_indices = file_rng.permutation(len(file_paths))
        file_paths = [file_paths[i] for i in file_indices]

    # worker分割
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        n_workers = worker_info.num_workers
        worker_id = worker_info.id
        if len(file_paths) < n_workers:
            logger.warning(
                "Number of files (%d) < num_workers (%d). "
                "Some workers will be idle.",
                len(file_paths),
                n_workers,
            )
        # ラウンドロビン分配
        file_paths = [
            fp
            for i, fp in enumerate(file_paths)
            if i % n_workers == worker_id
        ]

    return file_paths


def _compute_total_batches(
    row_counts: list[int],
    batch_size: int,
) -> int:
    """ファイルごとの ceil(rows / batch_size) の合計を返す．

    ストリーミングデータセットはファイル単位でバッチを生成するため，
    各ファイルの端数バッチを考慮した正確なバッチ数を計算する．

    Args:
        row_counts: 各ファイルの行数リスト
        batch_size: バッチサイズ

    Returns:
        合計バッチ数
    """
    return sum(math.ceil(n / batch_size) for n in row_counts)


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

    @property
    def row_counts(self) -> list[int]:
        """各ファイルの行数リスト."""
        ...

    def iter_files_columnar(
        self,
    ) -> Generator[ColumnarBatch, None, None]:
        """ファイル単位で ``ColumnarBatch`` をyieldする."""
        ...

    def iter_files_columnar_subset(
        self,
        file_paths: list[Path],
    ) -> Generator[ColumnarBatch, None, None]:
        """指定されたファイルパスのみを読み込み ``ColumnarBatch`` をyieldする."""
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

        workerファイル分割により，マルチワーカー環境では各workerが
        担当ファイルのみを読み込む．

        Yields:
            ((board_tensor, pieces_tensor), (move_label_tensor, result_value_tensor, legal_move_mask_tensor))
        """
        # persistent_workers対応: worker_info.seedはエポックごとに変わる
        worker_info = torch.utils.data.get_worker_info()
        worker_id = (
            worker_info.id if worker_info is not None else 0
        )
        if worker_info is not None:
            epoch_seed = worker_info.seed
        else:
            epoch_seed = (self._seed or 0) + self._epoch

        rng = np.random.default_rng(epoch_seed)

        try:
            # workerファイル分割 + ファイル順シャッフル
            worker_files = _resolve_worker_files(
                self._source,
                shuffle=self._shuffle,
                epoch_seed=epoch_seed,
            )

            if not worker_files:
                logger.debug(
                    "Worker %d: no files assigned", worker_id
                )
                return

            total_batches = 0
            file_count = 0
            for file_idx, columnar_batch in enumerate(
                self._source.iter_files_columnar_subset(
                    worker_files
                )
            ):
                file_count += 1
                if file_idx == 0:
                    _log_worker_memory(
                        worker_id,
                        "after_first_file",
                        level=logging.DEBUG,
                    )
                for batch in _yield_kif_batches(
                    columnar_batch,
                    batch_size=self._batch_size,
                    shuffle=self._shuffle,
                    rng=rng,
                ):
                    total_batches += 1
                    if total_batches == 1:
                        logger.info(
                            "Worker %d: first batch produced"
                            " (pid=%d)",
                            worker_id,
                            os.getpid(),
                        )
                    yield batch
            logger.debug(
                "Worker %d: iteration complete"
                " (%d batches from %d files)",
                worker_id,
                total_batches,
                file_count,
            )
        except Exception as exc:
            logger.error(
                "Worker %d crashed during iteration"
                " (pid=%d): %s",
                worker_id,
                os.getpid(),
                exc,
                exc_info=True,
            )
            raise

    def __len__(self) -> int:
        """バッチ数を返す(tqdmプログレスバー用)."""
        return _compute_total_batches(
            self._source.row_counts,
            self._batch_size,
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

        workerファイル分割により，マルチワーカー環境では各workerが
        担当ファイルのみを読み込む．

        Yields:
            ((board_tensor, pieces_tensor), reachable_squares_tensor)
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = (
            worker_info.id if worker_info is not None else 0
        )
        if worker_info is not None:
            epoch_seed = worker_info.seed
        else:
            epoch_seed = (self._seed or 0) + self._epoch

        rng = np.random.default_rng(epoch_seed)

        try:
            # workerファイル分割 + ファイル順シャッフル
            worker_files = _resolve_worker_files(
                self._source,
                shuffle=self._shuffle,
                epoch_seed=epoch_seed,
            )

            if not worker_files:
                logger.debug(
                    "Worker %d: no files assigned", worker_id
                )
                return

            total_batches = 0
            file_count = 0
            for file_idx, columnar_batch in enumerate(
                self._source.iter_files_columnar_subset(
                    worker_files
                )
            ):
                file_count += 1
                if file_idx == 0:
                    _log_worker_memory(
                        worker_id,
                        "after_first_file",
                        level=logging.DEBUG,
                    )
                for batch in _yield_stage1_batches(
                    columnar_batch,
                    batch_size=self._batch_size,
                    shuffle=self._shuffle,
                    rng=rng,
                ):
                    total_batches += 1
                    if total_batches == 1:
                        logger.info(
                            "Worker %d: first batch produced"
                            " (pid=%d)",
                            worker_id,
                            os.getpid(),
                        )
                    yield batch
            logger.debug(
                "Worker %d: iteration complete"
                " (%d batches from %d files)",
                worker_id,
                total_batches,
                file_count,
            )
        except Exception as exc:
            logger.error(
                "Worker %d crashed during iteration"
                " (pid=%d): %s",
                worker_id,
                os.getpid(),
                exc,
                exc_info=True,
            )
            raise

    def __len__(self) -> int:
        """バッチ数を返す(tqdmプログレスバー用)."""
        return _compute_total_batches(
            self._source.row_counts,
            self._batch_size,
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

        workerファイル分割により，マルチワーカー環境では各workerが
        担当ファイルのみを読み込む．

        Yields:
            ((board_tensor, pieces_tensor), legal_moves_tensor)
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = (
            worker_info.id if worker_info is not None else 0
        )
        if worker_info is not None:
            epoch_seed = worker_info.seed
        else:
            epoch_seed = (self._seed or 0) + self._epoch

        rng = np.random.default_rng(epoch_seed)

        try:
            # workerファイル分割 + ファイル順シャッフル
            worker_files = _resolve_worker_files(
                self._source,
                shuffle=self._shuffle,
                epoch_seed=epoch_seed,
            )

            if not worker_files:
                logger.debug(
                    "Worker %d: no files assigned", worker_id
                )
                return

            total_batches = 0
            file_count = 0
            for file_idx, columnar_batch in enumerate(
                self._source.iter_files_columnar_subset(
                    worker_files
                )
            ):
                file_count += 1
                if file_idx == 0:
                    _log_worker_memory(
                        worker_id,
                        "after_first_file",
                        level=logging.DEBUG,
                    )
                for batch in _yield_stage2_batches(
                    columnar_batch,
                    batch_size=self._batch_size,
                    shuffle=self._shuffle,
                    rng=rng,
                ):
                    total_batches += 1
                    if total_batches == 1:
                        logger.info(
                            "Worker %d: first batch produced"
                            " (pid=%d)",
                            worker_id,
                            os.getpid(),
                        )
                    yield batch
            logger.debug(
                "Worker %d: iteration complete"
                " (%d batches from %d files)",
                worker_id,
                total_batches,
                file_count,
            )
        except Exception as exc:
            logger.error(
                "Worker %d crashed during iteration"
                " (pid=%d): %s",
                worker_id,
                os.getpid(),
                exc,
                exc_info=True,
            )
            raise

    def __len__(self) -> int:
        """バッチ数を返す(tqdmプログレスバー用)."""
        return _compute_total_batches(
            self._source.row_counts,
            self._batch_size,
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

        board_tensor = torch.from_numpy(batch.board_positions)
        pieces_tensor = torch.from_numpy(batch.pieces_in_hand)

        assert batch.move_label is not None
        assert batch.result_value is not None

        move_label_tensor = torch.from_numpy(batch.move_label)
        result_value_tensor = (
            torch.from_numpy(batch.result_value)
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

        board_tensor = torch.from_numpy(batch.board_positions)
        pieces_tensor = torch.from_numpy(batch.pieces_in_hand)

        assert batch.reachable_squares is not None
        # (N, 9, 9) → (N, 81) and convert to float for BCE
        reachable_tensor = (
            torch.from_numpy(batch.reachable_squares)
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

        board_tensor = torch.from_numpy(batch.board_positions)
        pieces_tensor = torch.from_numpy(batch.pieces_in_hand)

        assert batch.legal_moves_label is not None
        legal_moves_tensor = torch.from_numpy(
            batch.legal_moves_label
        ).float()

        yield (
            (board_tensor, pieces_tensor),
            legal_moves_tensor,
        )
