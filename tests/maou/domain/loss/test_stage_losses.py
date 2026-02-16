"""Tests for Stage 1 and Stage 2 loss functions."""

import numpy as np
import torch

from maou.domain.loss.loss_fn import (
    LegalMovesLoss,
    ReachableSquaresLoss,
)
from maou.domain.move.label import MOVE_LABELS_NUM


class TestReachableSquaresLoss:
    """Test ReachableSquaresLoss (Stage 1 loss function)."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        loss_fn = ReachableSquaresLoss()

        assert loss_fn.pos_weight.item() == 1.0
        assert loss_fn.reduction == "mean"

    def test_initialization_custom_pos_weight(self) -> None:
        """Test initialization with custom pos_weight."""
        loss_fn = ReachableSquaresLoss(pos_weight=2.0)

        assert loss_fn.pos_weight.item() == 2.0

    def test_initialization_custom_reduction(self) -> None:
        """Test initialization with custom reduction mode."""
        loss_fn = ReachableSquaresLoss(reduction="sum")

        assert loss_fn.reduction == "sum"

    def test_forward_shape(self) -> None:
        """Test forward pass returns correct shape."""
        loss_fn = ReachableSquaresLoss()

        batch_size = 4
        num_squares = 81  # 9x9 flattened

        logits = torch.randn(batch_size, num_squares)
        targets = torch.randint(
            0, 2, (batch_size, num_squares)
        ).float()

        loss = loss_fn(logits, targets)

        # With reduction='mean', output should be scalar
        assert loss.shape == torch.Size([])
        assert loss.ndim == 0

    def test_forward_reduction_none(self) -> None:
        """Test forward pass with reduction='none'."""
        loss_fn = ReachableSquaresLoss(reduction="none")

        batch_size = 4
        num_squares = 81

        logits = torch.randn(batch_size, num_squares)
        targets = torch.randint(
            0, 2, (batch_size, num_squares)
        ).float()

        loss = loss_fn(logits, targets)

        # With reduction='none', output shape should match input
        assert loss.shape == (batch_size, num_squares)

    def test_forward_perfect_prediction(self) -> None:
        """Test loss is low for perfect predictions."""
        loss_fn = ReachableSquaresLoss()

        batch_size = 4
        num_squares = 81

        # Perfect predictions: high logits for 1, low logits for 0
        targets = torch.randint(
            0, 2, (batch_size, num_squares)
        ).float()
        logits = torch.where(
            targets == 1,
            torch.tensor(10.0),
            torch.tensor(-10.0),
        )

        loss = loss_fn(logits, targets)

        # Loss should be very small for perfect predictions
        assert loss.item() < 0.01

    def test_forward_worst_prediction(self) -> None:
        """Test loss is high for worst predictions."""
        loss_fn = ReachableSquaresLoss()

        batch_size = 4
        num_squares = 81

        # Worst predictions: opposite of targets
        targets = torch.randint(
            0, 2, (batch_size, num_squares)
        ).float()
        logits = torch.where(
            targets == 1,
            torch.tensor(-10.0),
            torch.tensor(10.0),
        )

        loss = loss_fn(logits, targets)

        # Loss should be high for worst predictions
        assert loss.item() > 5.0

    def test_forward_pos_weight_effect(self) -> None:
        """Test that pos_weight affects loss for positive class."""
        loss_fn_balanced = ReachableSquaresLoss(pos_weight=1.0)
        loss_fn_weighted = ReachableSquaresLoss(pos_weight=2.0)

        batch_size = 4
        num_squares = 81

        # Create targets with some positives
        targets = torch.zeros(batch_size, num_squares)
        targets[:, :10] = 1.0  # First 10 squares are reachable

        # Imperfect predictions
        logits = torch.randn(batch_size, num_squares)

        loss_balanced = loss_fn_balanced(logits, targets)
        loss_weighted = loss_fn_weighted(logits, targets)

        # Weighted loss should be different due to pos_weight
        assert loss_balanced.item() != loss_weighted.item()

    def test_device_transfer(self) -> None:
        """Test loss function works on different devices."""
        loss_fn = ReachableSquaresLoss()

        batch_size = 4
        num_squares = 81

        # CPU tensors
        logits_cpu = torch.randn(batch_size, num_squares)
        targets_cpu = torch.randint(
            0, 2, (batch_size, num_squares)
        ).float()

        loss_cpu = loss_fn(logits_cpu, targets_cpu)

        assert loss_cpu.device.type == "cpu"

        # Test with CUDA if available
        if torch.cuda.is_available():
            logits_cuda = logits_cpu.cuda()
            targets_cuda = targets_cpu.cuda()

            loss_cuda = loss_fn(logits_cuda, targets_cuda)

            assert loss_cuda.device.type == "cuda"


class TestLegalMovesLoss:
    """Test LegalMovesLoss (Stage 2 loss function)."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        loss_fn = LegalMovesLoss()

        assert loss_fn.pos_weight.item() == 1.0
        assert loss_fn.reduction == "mean"

    def test_initialization_custom_pos_weight(self) -> None:
        """Test initialization with custom pos_weight."""
        loss_fn = LegalMovesLoss(pos_weight=3.0)

        assert loss_fn.pos_weight.item() == 3.0

    def test_initialization_custom_reduction(self) -> None:
        """Test initialization with custom reduction mode."""
        loss_fn = LegalMovesLoss(reduction="sum")

        assert loss_fn.reduction == "sum"

    def test_forward_shape(self) -> None:
        """Test forward pass returns correct shape."""
        loss_fn = LegalMovesLoss()

        batch_size = 4
        num_moves = MOVE_LABELS_NUM  # 1496

        logits = torch.randn(batch_size, num_moves)
        targets = torch.randint(
            0, 2, (batch_size, num_moves)
        ).float()

        loss = loss_fn(logits, targets)

        # With reduction='mean', output should be scalar
        assert loss.shape == torch.Size([])
        assert loss.ndim == 0

    def test_forward_reduction_none(self) -> None:
        """Test forward pass with reduction='none'."""
        loss_fn = LegalMovesLoss(reduction="none")

        batch_size = 4
        num_moves = MOVE_LABELS_NUM

        logits = torch.randn(batch_size, num_moves)
        targets = torch.randint(
            0, 2, (batch_size, num_moves)
        ).float()

        loss = loss_fn(logits, targets)

        # With reduction='none', output shape should match input
        assert loss.shape == (batch_size, num_moves)

    def test_forward_perfect_prediction(self) -> None:
        """Test loss is low for perfect predictions."""
        loss_fn = LegalMovesLoss()

        batch_size = 4
        num_moves = MOVE_LABELS_NUM

        # Perfect predictions
        targets = torch.zeros(batch_size, num_moves)
        # Set some moves as legal
        targets[:, :50] = 1.0

        logits = torch.where(
            targets == 1,
            torch.tensor(10.0),
            torch.tensor(-10.0),
        )

        loss = loss_fn(logits, targets)

        # Loss should be very small for perfect predictions
        assert loss.item() < 0.01

    def test_forward_worst_prediction(self) -> None:
        """Test loss is high for worst predictions."""
        loss_fn = LegalMovesLoss()

        batch_size = 4
        num_moves = MOVE_LABELS_NUM

        # Worst predictions
        targets = torch.zeros(batch_size, num_moves)
        targets[:, :50] = 1.0

        logits = torch.where(
            targets == 1,
            torch.tensor(-10.0),
            torch.tensor(10.0),
        )

        loss = loss_fn(logits, targets)

        # Loss should be high for worst predictions
        assert loss.item() > 5.0

    def test_forward_multi_label_nature(self) -> None:
        """Test that loss correctly handles multi-label classification."""
        loss_fn = LegalMovesLoss()

        batch_size = 2
        num_moves = MOVE_LABELS_NUM

        # Multi-label targets: multiple legal moves per position
        targets = torch.zeros(batch_size, num_moves)
        targets[0, :30] = 1.0  # Position 1 has 30 legal moves
        targets[1, :50] = 1.0  # Position 2 has 50 legal moves

        # Predictions that are moderately good
        logits = torch.randn(batch_size, num_moves)

        loss = loss_fn(logits, targets)

        # Loss should be finite and reasonable
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_forward_sparse_labels(self) -> None:
        """Test loss with sparse legal moves (realistic scenario)."""
        loss_fn = LegalMovesLoss()

        batch_size = 4
        num_moves = MOVE_LABELS_NUM

        # Sparse labels: only a few legal moves (realistic for shogi)
        targets = torch.zeros(batch_size, num_moves)
        for i in range(batch_size):
            # Each position has 20-40 legal moves
            num_legal = np.random.randint(20, 40)
            legal_indices = np.random.choice(
                num_moves, num_legal, replace=False
            )
            targets[i, legal_indices] = 1.0

        logits = torch.randn(batch_size, num_moves)

        loss = loss_fn(logits, targets)

        # Loss should be finite
        assert torch.isfinite(loss)

    def test_device_transfer(self) -> None:
        """Test loss function works on different devices."""
        loss_fn = LegalMovesLoss()

        batch_size = 4
        num_moves = MOVE_LABELS_NUM

        # CPU tensors
        logits_cpu = torch.randn(batch_size, num_moves)
        targets_cpu = torch.randint(
            0, 2, (batch_size, num_moves)
        ).float()

        loss_cpu = loss_fn(logits_cpu, targets_cpu)

        assert loss_cpu.device.type == "cpu"

        # Test with CUDA if available
        if torch.cuda.is_available():
            logits_cuda = logits_cpu.cuda()
            targets_cuda = targets_cpu.cuda()

            loss_cuda = loss_fn(logits_cuda, targets_cuda)

            assert loss_cuda.device.type == "cuda"

    def test_backward_pass(self) -> None:
        """Test that gradients can be computed."""
        loss_fn = LegalMovesLoss()

        batch_size = 4
        num_moves = MOVE_LABELS_NUM

        logits = torch.randn(
            batch_size, num_moves, requires_grad=True
        )
        targets = torch.randint(
            0, 2, (batch_size, num_moves)
        ).float()

        loss = loss_fn(logits, targets)
        loss.backward()

        # Gradients should be computed
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
