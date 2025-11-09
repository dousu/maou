"""Tests for the shogi ResNet based Network."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from maou.app.learning.network import (
    BACKBONE_ARCHITECTURES,
    DEFAULT_BOARD_VOCAB_SIZE,
    HeadlessNetwork,
    Network,
)
from maou.app.pre_process.label import MOVE_LABELS_NUM


def test_network_outputs_have_expected_shapes() -> None:
    network = Network()
    batch_size = 2
    boards = torch.randint(
        0, DEFAULT_BOARD_VOCAB_SIZE, (batch_size, 9, 9), dtype=torch.long
    )
    pieces = torch.randint(0, 3, (batch_size, 14), dtype=torch.float32)

    policy, value = network((boards, pieces))

    assert policy.shape == (batch_size, MOVE_LABELS_NUM)
    assert value.shape == (batch_size, 1)


def test_backbone_feature_dimension_matches_channels() -> None:
    network = Network()
    boards = torch.randint(0, DEFAULT_BOARD_VOCAB_SIZE, (3, 9, 9))

    features = network.forward_features(boards)

    assert features.shape == (3, network.embedding_dim)


def test_network_allows_custom_head_configuration() -> None:
    network = Network(
        policy_hidden_dim=128, value_hidden_dim=64
    )
    boards = torch.randint(0, DEFAULT_BOARD_VOCAB_SIZE, (1, 9, 9))
    pieces = torch.zeros((1, 14), dtype=torch.float32)

    policy, value = network((boards, pieces))

    assert policy.shape == (1, MOVE_LABELS_NUM)
    assert value.shape == (1, 1)


def test_hand_projection_matches_board_embedding_channels() -> None:
    network = Network()

    assert network._hand_projection.out_features == network.input_channels


def test_forward_features_performs_early_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = Network()
    boards = torch.randint(0, DEFAULT_BOARD_VOCAB_SIZE, (2, 9, 9))
    hands = torch.rand((2, network._hand_projection.in_features))

    captured_inputs: dict[str, Tensor] = {}

    def fake_forward_features(self: HeadlessNetwork, inputs: Tensor) -> Tensor:
        captured_inputs["inputs"] = inputs
        return torch.zeros(
            (inputs.shape[0], network.embedding_dim),
            dtype=inputs.dtype,
            device=inputs.device,
        )

    monkeypatch.setattr(HeadlessNetwork, "forward_features", fake_forward_features)

    features = network.forward_features((boards, hands))

    expected_embedded = network._prepare_inputs(boards)
    projected = network._hand_projection(
        hands.to(dtype=expected_embedded.dtype, device=expected_embedded.device)
    ).view(expected_embedded.shape[0], network.input_channels, 1, 1)

    assert torch.allclose(
        captured_inputs["inputs"], expected_embedded + projected
    )
    assert torch.equal(
        features,
        torch.zeros(
            (boards.shape[0], network.embedding_dim),
            dtype=expected_embedded.dtype,
            device=expected_embedded.device,
        ),
    )


@pytest.mark.parametrize("architecture", BACKBONE_ARCHITECTURES)
def test_headless_network_embeds_board_ids(architecture: str) -> None:
    model = HeadlessNetwork(architecture=architecture)
    inputs = torch.randint(0, model.board_vocab_size, (2, 9, 9))

    embedded = model._prepare_inputs(inputs)

    assert embedded.shape == (2, model.input_channels, 9, 9)
    assert embedded.dtype == model.embedding.weight.dtype
