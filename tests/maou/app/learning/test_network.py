"""Tests for the shogi ResNet based Network."""

from __future__ import annotations

import pytest
import torch

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
        0,
        DEFAULT_BOARD_VOCAB_SIZE,
        (batch_size, 9, 9),
        dtype=torch.long,
    )
    pieces = torch.randint(
        0, 3, (batch_size, 14), dtype=torch.float32
    )

    policy, value = network((boards, pieces))

    assert policy.shape == (batch_size, MOVE_LABELS_NUM)
    assert value.shape == (batch_size, 1)


def test_backbone_feature_dimension_matches_channels() -> None:
    network = Network()
    boards = torch.randint(
        0, DEFAULT_BOARD_VOCAB_SIZE, (3, 9, 9)
    )

    features = network.forward_features(boards)

    assert features.shape == (3, network.embedding_dim)


def test_network_allows_custom_head_configuration() -> None:
    network = Network(
        policy_hidden_dim=128, value_hidden_dim=64
    )
    boards = torch.randint(
        0, DEFAULT_BOARD_VOCAB_SIZE, (1, 9, 9)
    )
    pieces = torch.zeros((1, 14), dtype=torch.float32)

    policy, value = network((boards, pieces))

    assert policy.shape == (1, MOVE_LABELS_NUM)
    assert value.shape == (1, 1)


def test_hand_projection_matches_hand_projection_dim() -> None:
    network = Network()

    assert (
        network._hand_projection.out_features
        == network._hand_projection_dim
    )


def test_forward_features_concatenates_hand_and_board() -> None:
    """Test that hand features are concatenated with board features."""
    network = Network()
    boards = torch.randint(
        0, DEFAULT_BOARD_VOCAB_SIZE, (2, 9, 9)
    )
    hands = torch.rand(
        (2, network._hand_projection.in_features)
    )

    # Get intermediate representations
    embedded_board = network._prepare_inputs(boards)
    projected = network._hand_projection(
        hands.to(
            dtype=embedded_board.dtype,
            device=embedded_board.device,
        )
    )

    # Verify that forward_features produces valid output
    features = network.forward_features((boards, hands))
    assert features.shape == (2, network.embedding_dim)

    # Verify concatenation dimensions by checking backbone input would be correct
    expected_channels = (
        network._embedding_channels
        + network._hand_projection_dim
    )
    assert (
        embedded_board.shape[1] + projected.shape[1]
        == expected_channels
    )


@pytest.mark.parametrize("architecture", BACKBONE_ARCHITECTURES)
def test_headless_network_embeds_board_ids(
    architecture: str,
) -> None:
    model = HeadlessNetwork(architecture=architecture)
    inputs = torch.randint(0, model.board_vocab_size, (2, 9, 9))

    embedded = model._prepare_inputs(inputs)

    assert embedded.shape == (2, model.input_channels, 9, 9)
    assert embedded.dtype == model.embedding.weight.dtype
