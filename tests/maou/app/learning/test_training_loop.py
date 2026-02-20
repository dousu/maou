from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from maou.app.learning.callbacks import (
    ModelInputs,
    TrainingContext,
)
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


def test_nan_loss_does_not_call_scaler_update() -> None:
    """NaN損失検出時にscaler.update()が呼ばれないことを確認する．"""
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    mock_scaler = MagicMock()

    loop = TrainingLoop(
        model=model,
        device=torch.device("cpu"),
        optimizer=optimizer,
        loss_fn_policy=torch.nn.CrossEntropyLoss(),
        loss_fn_value=torch.nn.MSELoss(),
        policy_loss_ratio=1.0,
        value_loss_ratio=1.0,
    )
    loop.scaler = mock_scaler

    # NaN損失を含むcontextを作成
    context = TrainingContext(
        batch_idx=0,  # finitude_check_intervalの倍数でチェック発生
        epoch_idx=0,
        inputs=torch.zeros(2, 10),
        labels_policy=torch.zeros(2, dtype=torch.long),
        labels_value=torch.zeros(2),
        legal_move_mask=None,
        batch_size=2,
    )

    # モデル出力とNaN損失を設定
    context.outputs_policy = torch.zeros(2, 2)
    context.outputs_value = torch.zeros(2)
    context.loss = torch.tensor(float("nan"))

    # _compute_policy_lossとモデルをモックして
    # _train_batch_mixed_precisionのNaN検出パスを直接テスト
    with (
        patch.object(
            loop,
            "_compute_policy_loss",
            return_value=torch.tensor(float("nan")),
        ),
        patch.object(
            loop.model,
            "forward",
            return_value=(torch.zeros(2, 2), torch.zeros(2)),
        ),
    ):
        loop._train_batch_mixed_precision(
            context, is_accumulation_step=False
        )

    # scaler.update()はNaN検出時に呼ばれてはいけない
    mock_scaler.update.assert_not_called()
