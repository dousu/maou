from __future__ import annotations

import torch

from maou.app.learning.callbacks import ModelInputs
from maou.app.learning.network import Network
from maou.app.learning.training_loop import TrainingLoop


def test_resolve_batch_size_supports_list_inputs() -> None:
    board = torch.zeros((4, 9, 9), dtype=torch.long)
    pieces = torch.zeros((4, 14), dtype=torch.float32)
    inputs: ModelInputs = [board, pieces]

    batch_size = TrainingLoop._resolve_batch_size(inputs)

    assert batch_size == 4


def test_network_separate_inputs_accepts_list() -> None:
    board = torch.zeros((2, 9, 9), dtype=torch.long)
    pieces = torch.zeros((2, 14), dtype=torch.float32)

    separated_board, separated_pieces = (
        Network._separate_inputs([board, pieces])
    )

    assert separated_board is board
    assert separated_pieces is pieces
