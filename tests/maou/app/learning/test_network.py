"""Tests for the shogi ResNet based Network."""

from __future__ import annotations

import pytest
import torch

from maou.app.learning.network import (
    BACKBONE_ARCHITECTURES,
    DEFAULT_BOARD_VOCAB_SIZE,
    DEFAULT_HAND_PROJECTION_DIM,
    PIECES_IN_HAND_VECTOR_SIZE,
    HeadlessNetwork,
    Network,
)
from maou.domain.move.label import MOVE_LABELS_NUM


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

    assert network._hand_projection is not None
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
    assert network._hand_projection is not None
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


class TestHeadlessNetworkHandProjection:
    """HeadlessNetwork の hand projection 統合テスト．"""

    def test_hand_projection_created_when_dim_positive(
        self,
    ) -> None:
        """hand_projection_dim > 0 のとき _hand_projection が作成される．"""
        model = HeadlessNetwork(
            hand_projection_dim=32,
            out_channels=(16, 32, 64, 64),
        )
        assert model._hand_projection is not None
        assert (
            model._hand_projection.in_features
            == PIECES_IN_HAND_VECTOR_SIZE
        )
        assert model._hand_projection.out_features == 32

    def test_hand_projection_none_when_dim_zero(self) -> None:
        """hand_projection_dim=0 のとき _hand_projection が None になる．"""
        model = HeadlessNetwork(
            hand_projection_dim=0,
            out_channels=(16, 32, 64, 64),
        )
        assert model._hand_projection is None

    def test_forward_features_with_board_and_hand_tuple(
        self,
    ) -> None:
        """(board_tensor, hand_tensor) タプルで正しい出力形状を得られる．"""
        model = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=16,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        board = torch.randint(0, 32, (2, 9, 9))
        hand = torch.randn(2, PIECES_IN_HAND_VECTOR_SIZE)

        features = model.forward_features((board, hand))

        assert isinstance(features, torch.Tensor)
        assert features.shape == (2, model.embedding_dim)

    def test_forward_features_with_hand_none_uses_zero_padding(
        self,
    ) -> None:
        """hand_tensor が None のときゼロパディングで動作する．"""
        model = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=16,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        board = torch.randint(0, 32, (2, 9, 9))

        features = model.forward_features((board, None))

        assert isinstance(features, torch.Tensor)
        assert features.shape == (2, model.embedding_dim)

    def test_forward_features_single_tensor_backward_compat(
        self,
    ) -> None:
        """hand_projection_dim=0 で単一テンソルを渡す後方互換性．"""
        model = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        board = torch.randint(0, 32, (2, 9, 9))

        features = model.forward_features(board)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (2, model.embedding_dim)

    def test_separate_inputs_tensor(self) -> None:
        """_separate_inputs: torch.Tensor → (tensor, None)．"""
        tensor = torch.zeros(2, 9, 9)
        board, hand = HeadlessNetwork._separate_inputs(tensor)
        assert board is tensor
        assert hand is None

    def test_separate_inputs_tuple(self) -> None:
        """_separate_inputs: (board, hand) → (board, hand)．"""
        board_in = torch.zeros(2, 9, 9)
        hand_in = torch.zeros(2, 38)
        board, hand = HeadlessNetwork._separate_inputs(
            (board_in, hand_in)
        )
        assert board is board_in
        assert hand is hand_in


class TestNetworkForwardFeaturesRegression:
    """Network.forward_features のリファクタリング前後で同一出力を検証する回帰テスト．"""

    def test_network_forward_features_produces_deterministic_output(
        self,
    ) -> None:
        """同一の入力・重みで forward_features が一貫した出力を返す．"""
        torch.manual_seed(42)
        network = Network(
            hand_projection_dim=DEFAULT_HAND_PROJECTION_DIM,
            out_channels=(16, 32, 64, 64),
        )
        network.eval()

        boards = torch.randint(
            0, DEFAULT_BOARD_VOCAB_SIZE, (2, 9, 9)
        )
        hands = torch.randn(2, PIECES_IN_HAND_VECTOR_SIZE)

        with torch.no_grad():
            features1 = network.forward_features(
                (boards, hands)
            )
            features2 = network.forward_features(
                (boards, hands)
            )

        assert torch.allclose(features1, features2)

    def test_network_delegates_to_headless(self) -> None:
        """Network.forward_features が super() に委譲して同じ結果を返す．"""
        torch.manual_seed(42)
        network = Network(
            hand_projection_dim=DEFAULT_HAND_PROJECTION_DIM,
            out_channels=(16, 32, 64, 64),
        )
        network.eval()

        boards = torch.randint(
            0, DEFAULT_BOARD_VOCAB_SIZE, (2, 9, 9)
        )
        hands = torch.randn(2, PIECES_IN_HAND_VECTOR_SIZE)

        with torch.no_grad():
            via_network = network.forward_features(
                (boards, hands)
            )
            via_headless = HeadlessNetwork.forward_features(
                network, (boards, hands)
            )

        assert torch.allclose(via_network, via_headless)


class TestValidateInputsTracingGuard:
    """_validate_inputs のトレーシングガード動作を検証する．"""

    def test_validate_inputs_skips_validation_during_tracing(
        self,
    ) -> None:
        """トレーシング中はサイズ不一致でもエラーにならず分類のみ行う．"""
        from unittest.mock import patch

        model = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        # board_size は (9, 9) なので (5, 5) は不一致
        wrong_size_input = torch.randint(0, 32, (1, 5, 5))

        with patch(
            "maou.app.learning.network.is_tracing",
            return_value=True,
        ):
            input_type, tensor = model._validate_inputs(
                wrong_size_input
            )

        assert input_type == "board"
        assert tensor is wrong_size_input

    def test_validate_inputs_raises_during_eager(self) -> None:
        """通常実行時は従来通り ValueError を送出する．"""
        model = HeadlessNetwork(
            board_vocab_size=32,
            embedding_dim=64,
            hand_projection_dim=0,
            architecture="resnet",
            out_channels=(16, 32, 64, 64),
        )
        wrong_size_input = torch.randint(0, 32, (1, 5, 5))

        with pytest.raises(ValueError, match="board size"):
            model._validate_inputs(wrong_size_input)


def test_embed_inputs_matches_backbone_input() -> None:
    """embed_inputs はバックボーンへ渡す直前のテンソルを返す．

    段階学習の truncated model がこの前処理を private メソッド
    呼び出しで書き直していたため公開した．
    """
    net = Network(hand_projection_dim=4)
    board = torch.zeros((2, 9, 9), dtype=torch.long)
    hand = torch.zeros((2, 14), dtype=torch.float32)

    combined = net.embed_inputs((board, hand))

    assert combined.shape == (
        2,
        net.backbone_input_channels,
        *net.board_size,
    )
    # forward_features はこの出力をそのままバックボーンへ渡す．
    expected = net.backbone.forward_features(combined)
    assert torch.equal(
        net.forward_features((board, hand)), expected
    )


def test_backbone_input_channels_is_derived() -> None:
    """backbone_input_channels は盤面埋め込み次元と持ち駒射影の和．"""
    from maou.app.learning.network import BOARD_EMBEDDING_DIM

    net = Network(hand_projection_dim=4)

    assert (
        net.backbone_input_channels == BOARD_EMBEDDING_DIM + 4
    )
    assert net.hand_projection_dim == 4
    assert net.board_size == (9, 9)


def test_custom_pooling_is_still_applied_for_resnet() -> None:
    """pooling 引数は ResNet バックボーンへ渡っても生きている．

    プーリングの持ち主を移す変更で，この拡張ポイントが黙って
    死ぬのを防ぐ．
    """
    calls: list[torch.Tensor] = []

    class RecordingPool(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls.append(x)
            return torch.nn.functional.adaptive_avg_pool2d(
                x, (1, 1)
            )

    pool = RecordingPool()
    net = Network(architecture="resnet", pooling=pool)
    board = torch.zeros((2, 9, 9), dtype=torch.long)
    hand = torch.zeros((2, 14), dtype=torch.float32)

    net.forward_features((board, hand))

    assert len(calls) == 1
    assert net.pool is pool
