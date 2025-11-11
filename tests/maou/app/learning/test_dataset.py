"""Tests for dataset handling in the learning module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from maou.app.learning.dataset import DataSource, KifDataset


class _ArrayDataSource(DataSource):
    """Minimal ``DataSource`` backed by a numpy structured array."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


def test_preprocessed_batches_provide_legal_move_masks() -> None:
    """Collating preprocessed records should yield a tensor mask, not ``None``."""

    dtype = np.dtype(
        [
            ("boardIdPositions", np.uint8, (9, 9)),
            ("piecesInHand", np.uint8, (14,)),
            ("moveLabel", np.float32, (5,)),
            ("resultValue", np.float32),
        ]
    )
    data = np.array(
        [
            (
                np.eye(9, dtype=np.uint8),
                np.zeros(14, dtype=np.uint8),
                np.array([1, 0, 0, 0, 0], dtype=np.float32),
                np.float32(1.0),
            ),
            (
                np.fliplr(np.eye(9, dtype=np.uint8)),
                np.zeros(14, dtype=np.uint8),
                np.array([0, 1, 0, 0, 0], dtype=np.float32),
                np.float32(-1.0),
            ),
        ],
        dtype=dtype,
    )

    dataset = KifDataset(datasource=_ArrayDataSource(data), transform=None)

    loader = DataLoader(dataset, batch_size=2)
    (boards, pieces), (_, _, legal_move_mask) = next(iter(loader))

    assert isinstance(boards, torch.Tensor)
    assert boards.dtype == torch.uint8
    assert boards.shape == (2, 9, 9)
    assert isinstance(pieces, torch.Tensor)
    assert pieces.dtype == torch.uint8
    assert pieces.shape == (2, 14)
    assert isinstance(legal_move_mask, torch.Tensor)
    assert torch.all(legal_move_mask == 1)


def test_dataset_accepts_float16_move_labels() -> None:
    """Structured arrays with float16 policy labels remain loadable."""

    dtype = np.dtype(
        [
            ("boardIdPositions", np.uint8, (2, 2)),
            ("piecesInHand", np.uint8, (4,)),
            ("moveLabel", np.float16, (3,)),
            ("resultValue", np.float32),
        ]
    )
    data = np.array(
        [
            (
                np.ones((2, 2), dtype=np.uint8),
                np.zeros(4, dtype=np.uint8),
                np.array([0.5, 0.25, 0.25], dtype=np.float16),
                np.float32(0.0),
            )
        ],
        dtype=dtype,
    )

    dataset = KifDataset(datasource=_ArrayDataSource(data), transform=None)

    (_, _), (policy, _, _) = dataset[0]

    assert policy.dtype == torch.float16


def test_dataset_requires_board_identifiers() -> None:
    """Datasets missing board ID grids should raise a helpful error."""

    dtype = np.dtype(
        [
            ("features", np.uint8, (4, 9, 9)),
            ("moveLabel", np.float16, (5,)),
            ("resultValue", np.float16),
        ]
    )
    data = np.zeros(1, dtype=dtype)

    dataset = KifDataset(datasource=_ArrayDataSource(data), transform=None)

    with pytest.raises(ValueError):
        dataset[0]

