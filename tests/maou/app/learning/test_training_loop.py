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


def test_training_loop_stores_logical_batch_size() -> None:
    """logical_batch_sizeがTrainingLoopに正しく保存されることを確認する．"""
    model = torch.nn.Linear(10, 2)
    loop = TrainingLoop(
        model=model,
        device=torch.device("cpu"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn_policy=torch.nn.CrossEntropyLoss(),
        loss_fn_value=torch.nn.MSELoss(),
        policy_loss_ratio=1.0,
        value_loss_ratio=1.0,
        logical_batch_size=4096,
    )

    assert loop.logical_batch_size == 4096


def test_training_loop_logical_batch_size_defaults_to_none() -> (
    None
):
    """logical_batch_size未指定時はNoneとなることを確認する．"""
    model = torch.nn.Linear(10, 2)
    loop = TrainingLoop(
        model=model,
        device=torch.device("cpu"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn_policy=torch.nn.CrossEntropyLoss(),
        loss_fn_value=torch.nn.MSELoss(),
        policy_loss_ratio=1.0,
        value_loss_ratio=1.0,
    )

    assert loop.logical_batch_size is None
