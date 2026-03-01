"""Tests for dataset handling in the learning module."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from maou.app.common.data_io_service import DataIOService
from maou.app.learning.dataset import DataSource, KifDataset
from maou.domain.data.array_io import save_preprocessing_df
from maou.domain.data.schema import (
    convert_numpy_to_preprocessing_df,
    create_empty_preprocessing_array,
)


class _ArrayDataSource(DataSource):
    """Minimal ``DataSource`` backed by a numpy structured array."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


def test_preprocessed_batches_provide_legal_move_masks() -> (
    None
):
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

    dataset = KifDataset(
        datasource=_ArrayDataSource(data), transform=None
    )

    loader = DataLoader(dataset, batch_size=2)
    (boards, pieces), (_, _, legal_move_mask) = next(
        iter(loader)
    )

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

    dataset = KifDataset(
        datasource=_ArrayDataSource(data), transform=None
    )

    (_, _), (policy, _, _) = dataset[0]

    assert policy.dtype == torch.float16


def test_dataset_returns_move_win_rate_when_present() -> None:
    """moveWinRate field in structured array is returned as 4th target element."""

    dtype = np.dtype(
        [
            ("boardIdPositions", np.uint8, (2, 2)),
            ("piecesInHand", np.uint8, (4,)),
            ("moveLabel", np.float16, (3,)),
            ("resultValue", np.float32),
            ("moveWinRate", np.float32, (3,)),
        ]
    )
    data = np.array(
        [
            (
                np.ones((2, 2), dtype=np.uint8),
                np.zeros(4, dtype=np.uint8),
                np.array([0.5, 0.25, 0.25], dtype=np.float16),
                np.float32(0.0),
                np.array([0.8, 0.6, 0.1], dtype=np.float32),
            )
        ],
        dtype=dtype,
    )

    dataset = KifDataset(
        datasource=_ArrayDataSource(data), transform=None
    )

    (_, _), (_, _, _, move_win_rate) = dataset[0]

    assert move_win_rate is not None
    assert move_win_rate.dtype == torch.float32
    assert move_win_rate.shape == (3,)
    assert torch.allclose(
        move_win_rate,
        torch.tensor([0.8, 0.6, 0.1], dtype=torch.float32),
    )


def test_dataset_returns_3_element_tuple_when_no_win_rate() -> (
    None
):
    """Target tuple has 3 elements when moveWinRate field is absent."""

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

    dataset = KifDataset(
        datasource=_ArrayDataSource(data), transform=None
    )

    (_, _), targets = dataset[0]

    assert len(targets) == 3


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

    dataset = KifDataset(
        datasource=_ArrayDataSource(data), transform=None
    )

    with pytest.raises(ValueError):
        dataset[0]


def test_numpy_to_tensor_requires_writeable_buffer() -> None:
    """Read-only buffers should surface actionable guidance."""

    dtype = np.dtype(
        [
            ("boardIdPositions", np.uint8, (2, 2)),
            ("piecesInHand", np.uint8, (2,)),
            ("moveLabel", np.float32, (1,)),
            ("resultValue", np.float32),
        ]
    )
    data = np.zeros(1, dtype=dtype)
    data.setflags(write=False)

    dataset = KifDataset(
        datasource=_ArrayDataSource(data), transform=None
    )

    with pytest.raises(ValueError, match="read-only"):
        dataset[0]


@pytest.mark.skip(reason="Needs update for DataFrame-based I/O")
def test_numpy_to_tensor_preserves_memmap_zero_copy(
    tmp_path: pathlib.Path,
) -> None:
    """Copy-on-write preprocessing memmaps stay writeable for tensors."""

    prep_array = create_empty_preprocessing_array(1)
    prep_array["boardIdPositions"] = np.ones(
        prep_array["boardIdPositions"].shape,
        dtype=np.uint8,
    )

    file_path = tmp_path / "zero_copy.feather"
    # Convert numpy array to Polars DataFrame and save as .feather
    df = convert_numpy_to_preprocessing_df(prep_array)
    save_preprocessing_df(df, file_path)

    loaded_array = DataIOService.load_array(
        file_path,
        array_type="preprocessing",
        bit_pack=False,
    )

    assert isinstance(loaded_array, np.memmap)
    assert loaded_array.flags.writeable

    record = loaded_array[0]
    board_np = record["boardIdPositions"]
    tensor = KifDataset._numpy_to_tensor(
        board_np,
        field_name="boardIdPositions",
        expected_dtype=np.uint8,
    )

    ptr = tensor.data_ptr()
    board_np[0, 0] = 9

    assert tensor.data_ptr() == ptr
    assert int(tensor[0, 0]) == 9
