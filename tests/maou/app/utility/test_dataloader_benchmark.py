from __future__ import annotations

import numpy as np
import torch

from maou.app.learning.dataset import DataSource
from maou.app.utility.dataloader_benchmark import (
    BenchmarkConfig,
    DataLoaderBenchmark,
)


class _DummyDataSource(DataSource):
    def __getitem__(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        return 0


def test_move_inputs_to_device_handles_nested_structures() -> (
    None
):
    config = BenchmarkConfig(
        datasource=_DummyDataSource(),
        batch_size=1,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    benchmark = DataLoaderBenchmark(config)

    board = torch.zeros((2, 2), dtype=torch.float32)
    pieces = torch.ones((2, 2), dtype=torch.float32)
    additional = torch.full((2, 2), 2.0, dtype=torch.float32)

    inputs = [board, (pieces, [additional])]

    moved = benchmark._move_inputs_to_device(inputs)

    assert isinstance(moved, list)
    assert torch.equal(moved[0], board)

    nested = moved[1]
    assert isinstance(nested, tuple)
    assert torch.equal(nested[0], pieces)

    deeper = nested[1]
    assert isinstance(deeper, list)
    assert torch.equal(deeper[0], additional)
