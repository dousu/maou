"""Tests for multi-stage training neural network heads."""

import torch

from maou.app.learning.network import (
    LegalMovesHead,
    ReachableSquaresHead,
)
from maou.app.pre_process.label import MOVE_LABELS_NUM


class TestReachableSquaresHead:
    """Test ReachableSquaresHead for Stage 1 training."""

    def test_initialization_without_hidden_dim(self) -> None:
        """Test initialization without hidden layer."""
        input_dim = 256
        head = ReachableSquaresHead(input_dim=input_dim)

        assert head.board_size == (9, 9)
        # Without hidden_dim, should have single linear layer
        assert isinstance(head.head, torch.nn.Sequential)

    def test_initialization_with_hidden_dim(self) -> None:
        """Test initialization with hidden layer."""
        input_dim = 256
        hidden_dim = 128
        head = ReachableSquaresHead(
            input_dim=input_dim, hidden_dim=hidden_dim
        )

        assert head.board_size == (9, 9)
        # With hidden_dim, should have 3 layers (Linear -> GELU -> Linear)
        assert isinstance(head.head, torch.nn.Sequential)
        assert len(head.head) == 3

    def test_initialization_custom_board_size(self) -> None:
        """Test initialization with custom board size."""
        input_dim = 256
        board_size = (7, 7)
        head = ReachableSquaresHead(
            input_dim=input_dim, board_size=board_size
        )

        assert head.board_size == board_size

    def test_forward_output_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        input_dim = 256
        batch_size = 4
        head = ReachableSquaresHead(input_dim=input_dim)

        features = torch.randn(batch_size, input_dim)
        output = head(features)

        # Output should be (batch, 81) for 9x9 board
        expected_output_dim = 9 * 9
        assert output.shape == (batch_size, expected_output_dim)

    def test_forward_output_shape_with_hidden(self) -> None:
        """Test forward pass with hidden layer produces correct output shape."""
        input_dim = 256
        hidden_dim = 128
        batch_size = 4
        head = ReachableSquaresHead(
            input_dim=input_dim, hidden_dim=hidden_dim
        )

        features = torch.randn(batch_size, input_dim)
        output = head(features)

        # Output should be (batch, 81)
        expected_output_dim = 9 * 9
        assert output.shape == (batch_size, expected_output_dim)

    def test_forward_returns_logits(self) -> None:
        """Test that forward pass returns logits (not probabilities)."""
        input_dim = 256
        batch_size = 4
        head = ReachableSquaresHead(input_dim=input_dim)

        features = torch.randn(batch_size, input_dim)
        output = head(features)

        # Logits can be any real number (not bounded to [0, 1])
        assert output.dtype == torch.float32
        # At least some values should be outside [0, 1] range
        assert (output < 0).any() or (output > 1).any()

    def test_forward_custom_board_size(self) -> None:
        """Test forward pass with custom board size."""
        input_dim = 256
        batch_size = 4
        board_size = (7, 7)
        head = ReachableSquaresHead(
            input_dim=input_dim, board_size=board_size
        )

        features = torch.randn(batch_size, input_dim)
        output = head(features)

        # Output should match custom board size
        expected_output_dim = 7 * 7
        assert output.shape == (batch_size, expected_output_dim)

    def test_backward_pass(self) -> None:
        """Test that gradients can be computed."""
        input_dim = 256
        batch_size = 4
        head = ReachableSquaresHead(input_dim=input_dim)

        features = torch.randn(
            batch_size, input_dim, requires_grad=True
        )
        output = head(features)

        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Gradients should be computed
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_device_transfer(self) -> None:
        """Test head can be moved to different devices."""
        input_dim = 256
        batch_size = 4
        head = ReachableSquaresHead(input_dim=input_dim)

        # CPU
        features_cpu = torch.randn(batch_size, input_dim)
        output_cpu = head(features_cpu)
        assert output_cpu.device.type == "cpu"

        # CUDA if available
        if torch.cuda.is_available():
            head_cuda = head.cuda()
            features_cuda = features_cpu.cuda()
            output_cuda = head_cuda(features_cuda)
            assert output_cuda.device.type == "cuda"


class TestLegalMovesHead:
    """Test LegalMovesHead for Stage 2 training."""

    def test_initialization_without_hidden_dim(self) -> None:
        """Test initialization without hidden layer."""
        input_dim = 256
        head = LegalMovesHead(input_dim=input_dim)

        # Without hidden_dim, should have single linear layer
        assert isinstance(head.head, torch.nn.Sequential)

    def test_initialization_with_hidden_dim(self) -> None:
        """Test initialization with hidden layer."""
        input_dim = 256
        hidden_dim = 512
        head = LegalMovesHead(
            input_dim=input_dim, hidden_dim=hidden_dim
        )

        # With hidden_dim, should have 3 layers (Linear -> GELU -> Linear)
        assert isinstance(head.head, torch.nn.Sequential)
        assert len(head.head) == 3

    def test_initialization_custom_num_move_labels(
        self,
    ) -> None:
        """Test initialization with custom number of move labels."""
        input_dim = 256
        num_move_labels = 1000
        head = LegalMovesHead(
            input_dim=input_dim, num_move_labels=num_move_labels
        )

        batch_size = 4
        features = torch.randn(batch_size, input_dim)
        output = head(features)

        assert output.shape == (batch_size, num_move_labels)

    def test_forward_output_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        input_dim = 256
        batch_size = 4
        head = LegalMovesHead(input_dim=input_dim)

        features = torch.randn(batch_size, input_dim)
        output = head(features)

        # Output should be (batch, MOVE_LABELS_NUM)
        assert output.shape == (batch_size, MOVE_LABELS_NUM)

    def test_forward_output_shape_with_hidden(self) -> None:
        """Test forward pass with hidden layer produces correct output shape."""
        input_dim = 256
        hidden_dim = 512
        batch_size = 4
        head = LegalMovesHead(
            input_dim=input_dim, hidden_dim=hidden_dim
        )

        features = torch.randn(batch_size, input_dim)
        output = head(features)

        # Output should be (batch, MOVE_LABELS_NUM)
        assert output.shape == (batch_size, MOVE_LABELS_NUM)

    def test_forward_returns_logits(self) -> None:
        """Test that forward pass returns logits (not probabilities)."""
        input_dim = 256
        batch_size = 4
        head = LegalMovesHead(input_dim=input_dim)

        features = torch.randn(batch_size, input_dim)
        output = head(features)

        # Logits can be any real number (not bounded to [0, 1])
        assert output.dtype == torch.float32
        # At least some values should be outside [0, 1] range
        assert (output < 0).any() or (output > 1).any()

    def test_forward_large_output_dimension(self) -> None:
        """Test forward pass handles large output dimension (2187)."""
        input_dim = 256
        batch_size = 8
        head = LegalMovesHead(input_dim=input_dim)

        features = torch.randn(batch_size, input_dim)
        output = head(features)

        # Should handle large output dimension without issues
        assert output.shape == (batch_size, MOVE_LABELS_NUM)
        assert torch.isfinite(output).all()

    def test_backward_pass(self) -> None:
        """Test that gradients can be computed."""
        input_dim = 256
        batch_size = 4
        head = LegalMovesHead(input_dim=input_dim)

        features = torch.randn(
            batch_size, input_dim, requires_grad=True
        )
        output = head(features)

        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Gradients should be computed
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_device_transfer(self) -> None:
        """Test head can be moved to different devices."""
        input_dim = 256
        batch_size = 4
        head = LegalMovesHead(input_dim=input_dim)

        # CPU
        features_cpu = torch.randn(batch_size, input_dim)
        output_cpu = head(features_cpu)
        assert output_cpu.device.type == "cpu"

        # CUDA if available
        if torch.cuda.is_available():
            head_cuda = head.cuda()
            features_cuda = features_cpu.cuda()
            output_cuda = head_cuda(features_cuda)
            assert output_cuda.device.type == "cuda"

    def test_parameter_count(self) -> None:
        """Test parameter count is reasonable."""
        input_dim = 256
        head = LegalMovesHead(input_dim=input_dim)

        num_params = sum(p.numel() for p in head.parameters())

        # Without hidden layer: input_dim * MOVE_LABELS_NUM + MOVE_LABELS_NUM
        expected_params = (
            input_dim * MOVE_LABELS_NUM + MOVE_LABELS_NUM
        )

        assert num_params == expected_params

    def test_parameter_count_with_hidden(self) -> None:
        """Test parameter count with hidden layer."""
        input_dim = 256
        hidden_dim = 512
        head = LegalMovesHead(
            input_dim=input_dim, hidden_dim=hidden_dim
        )

        num_params = sum(p.numel() for p in head.parameters())

        # With hidden layer:
        # (input_dim * hidden_dim + hidden_dim) +
        # (hidden_dim * MOVE_LABELS_NUM + MOVE_LABELS_NUM)
        expected_params = (
            input_dim * hidden_dim
            + hidden_dim
            + hidden_dim * MOVE_LABELS_NUM
            + MOVE_LABELS_NUM
        )

        assert num_params == expected_params
