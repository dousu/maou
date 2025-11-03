"""Tests for the shogi ResNet based Network."""

from __future__ import annotations

import torch

from maou.app.learning.network import Network
from maou.app.pre_process.label import MOVE_LABELS_NUM
from maou.domain.board.shogi import FEATURES_NUM


def test_network_outputs_have_expected_shapes() -> None:
    network = Network()
    batch_size = 2
    inputs = torch.randn(batch_size, FEATURES_NUM, 9, 9)

    policy, value = network(inputs)

    assert policy.shape == (batch_size, MOVE_LABELS_NUM)
    assert value.shape == (batch_size, 1)


def test_backbone_feature_dimension_matches_channels() -> None:
    network = Network()
    inputs = torch.randn(3, FEATURES_NUM, 9, 9)

    features = network.forward_features(inputs)

    assert features.shape == (3, network.embedding_dim)


def test_network_allows_custom_head_configuration() -> None:
    network = Network(
        policy_hidden_dim=128, value_hidden_dim=64
    )
    inputs = torch.randn(1, FEATURES_NUM, 9, 9)

    policy, value = network(inputs)

    assert policy.shape == (1, MOVE_LABELS_NUM)
    assert value.shape == (1, 1)
