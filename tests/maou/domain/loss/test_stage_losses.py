"""Tests for Stage 1 and Stage 2 loss functions."""

import logging
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch

from maou.domain.loss.loss_fn import (
    AsymmetricLoss,
    LegalMovesLoss,
    ReachableSquaresLoss,
)
from maou.domain.move.label import MOVE_LABELS_NUM


@contextmanager
def warnings_captured_by_logging() -> Generator[
    list[logging.LogRecord], None, None
]:
    """Context manager to capture logging warnings."""
    records: list[logging.LogRecord] = []

    class Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Handler()
    logger = logging.getLogger("maou.domain.loss.loss_fn")
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


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


class TestAsymmetricLoss:
    """Test AsymmetricLoss for multi-label classification."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        loss_fn = AsymmetricLoss()

        assert loss_fn.gamma_pos == 0.0
        assert loss_fn.gamma_neg == 2.0
        assert loss_fn.clip == 0.02
        assert loss_fn.reduction == "mean"

    def test_initialization_custom(self) -> None:
        """Test initialization with custom parameters."""
        loss_fn = AsymmetricLoss(
            gamma_pos=1.0,
            gamma_neg=4.0,
            clip=0.05,
            reduction="sum",
        )

        assert loss_fn.gamma_pos == 1.0
        assert loss_fn.gamma_neg == 4.0
        assert loss_fn.clip == 0.05
        assert loss_fn.reduction == "sum"

    def test_invalid_gamma_pos_raises(self) -> None:
        """Test that negative gamma_pos raises ValueError."""
        try:
            AsymmetricLoss(gamma_pos=-1.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "gamma_pos" in str(e)

    def test_invalid_gamma_neg_raises(self) -> None:
        """Test that negative gamma_neg raises ValueError."""
        try:
            AsymmetricLoss(gamma_neg=-1.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "gamma_neg" in str(e)

    def test_invalid_clip_raises(self) -> None:
        """Test that negative clip raises ValueError."""
        try:
            AsymmetricLoss(clip=-0.01)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "clip" in str(e)

    def test_forward_shape(self) -> None:
        """Test forward pass returns correct shape."""
        loss_fn = AsymmetricLoss()

        batch_size = 4
        num_labels = 1496

        logits = torch.randn(batch_size, num_labels)
        targets = torch.randint(
            0, 2, (batch_size, num_labels)
        ).float()

        loss = loss_fn(logits, targets)

        assert loss.shape == torch.Size([])
        assert loss.ndim == 0

    def test_forward_reduction_none(self) -> None:
        """Test forward pass with reduction='none'."""
        loss_fn = AsymmetricLoss(reduction="none")

        batch_size = 4
        num_labels = 100

        logits = torch.randn(batch_size, num_labels)
        targets = torch.randint(
            0, 2, (batch_size, num_labels)
        ).float()

        loss = loss_fn(logits, targets)

        assert loss.shape == (batch_size, num_labels)

    def test_gamma_neg_zero_clip_zero_equals_bce(self) -> None:
        """Test that gamma_pos=0, gamma_neg=0, clip=0 equals standard BCE."""
        asl = AsymmetricLoss(
            gamma_pos=0.0, gamma_neg=0.0, clip=0.0
        )
        bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

        torch.manual_seed(42)
        batch_size = 8
        num_labels = 100

        logits = torch.randn(batch_size, num_labels)
        targets = torch.randint(
            0, 2, (batch_size, num_labels)
        ).float()

        asl_loss = asl(logits, targets)
        bce_loss = bce(logits, targets)

        torch.testing.assert_close(
            asl_loss, bce_loss, atol=1e-5, rtol=1e-5
        )

    def test_clip_effect(self) -> None:
        """Test that clip > 0 changes loss compared to clip=0."""
        asl_no_clip = AsymmetricLoss(
            gamma_pos=0.0, gamma_neg=2.0, clip=0.0
        )
        asl_with_clip = AsymmetricLoss(
            gamma_pos=0.0, gamma_neg=2.0, clip=0.05
        )

        torch.manual_seed(42)
        batch_size = 8
        num_labels = 100

        logits = torch.randn(batch_size, num_labels)
        targets = torch.randint(
            0, 2, (batch_size, num_labels)
        ).float()

        loss_no_clip = asl_no_clip(logits, targets)
        loss_with_clip = asl_with_clip(logits, targets)

        # Clip should change the loss value
        assert loss_no_clip.item() != loss_with_clip.item()

    def test_asymmetric_weighting(self) -> None:
        """Test that gamma_pos != gamma_neg produces asymmetric loss."""
        # When gamma_pos=0 (no down-weighting of positives) and gamma_neg=2
        # (strong down-weighting of easy negatives), the loss should differ from
        # symmetric case (gamma_pos=gamma_neg=2)
        asl_asymmetric = AsymmetricLoss(
            gamma_pos=0.0, gamma_neg=2.0, clip=0.0
        )
        asl_symmetric = AsymmetricLoss(
            gamma_pos=2.0, gamma_neg=2.0, clip=0.0
        )

        torch.manual_seed(42)
        batch_size = 8
        num_labels = 100

        logits = torch.randn(batch_size, num_labels)
        targets = torch.randint(
            0, 2, (batch_size, num_labels)
        ).float()

        loss_asymmetric = asl_asymmetric(logits, targets)
        loss_symmetric = asl_symmetric(logits, targets)

        assert loss_asymmetric.item() != loss_symmetric.item()

    def test_fp32_cast(self) -> None:
        """Test that FP16 inputs are correctly cast to FP32."""
        loss_fn = AsymmetricLoss()

        batch_size = 4
        num_labels = 100

        # Create FP16 tensors
        logits = torch.randn(batch_size, num_labels).half()
        targets = torch.randint(
            0, 2, (batch_size, num_labels)
        ).half()

        loss = loss_fn(logits, targets)

        # Output should be FP32 due to internal cast
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)

    def test_backward_pass(self) -> None:
        """Test that gradients can be computed."""
        loss_fn = AsymmetricLoss()

        batch_size = 4
        num_labels = 100

        logits = torch.randn(
            batch_size, num_labels, requires_grad=True
        )
        targets = torch.randint(
            0, 2, (batch_size, num_labels)
        ).float()

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_perfect_prediction(self) -> None:
        """Test loss is low for perfect predictions."""
        loss_fn = AsymmetricLoss()

        batch_size = 4
        num_labels = 100

        targets = torch.zeros(batch_size, num_labels)
        targets[:, :20] = 1.0

        logits = torch.where(
            targets == 1,
            torch.tensor(10.0),
            torch.tensor(-10.0),
        )

        loss = loss_fn(logits, targets)

        assert loss.item() < 0.01

    def test_worst_prediction(self) -> None:
        """Test loss is high for worst predictions."""
        loss_fn = AsymmetricLoss()

        batch_size = 4
        num_labels = 100

        targets = torch.zeros(batch_size, num_labels)
        targets[:, :20] = 1.0

        logits = torch.where(
            targets == 1,
            torch.tensor(-10.0),
            torch.tensor(10.0),
        )

        loss = loss_fn(logits, targets)

        assert loss.item() > 1.0

    def test_clip_suppresses_easy_negatives_completely(
        self,
    ) -> None:
        """Test that clip suppresses easy negatives via focusing weight.

        When clip > 0 and gamma_neg > 0, easy negatives (where the model
        is very confident that the label is negative, i.e., probs_pos is very
        small) should have their loss contribution suppressed to near zero.
        This is because the focusing weight uses clipped probabilities.
        """
        loss_fn = AsymmetricLoss(
            gamma_pos=0.0,
            gamma_neg=4.0,
            clip=0.05,
            reduction="none",
        )

        batch_size = 1
        num_labels = 10

        # All targets are negative (0)
        targets = torch.zeros(batch_size, num_labels)

        # Model is very confident these are negatives (logits << 0)
        # This means probs_pos is very small (easy negatives)
        logits = torch.full((batch_size, num_labels), -10.0)

        loss = loss_fn(logits, targets)

        # Easy negatives should have very small loss due to
        # clip + focusing weight suppression
        assert loss.max().item() < 0.01


class TestLegalMovesLoss:
    """Test LegalMovesLoss (Stage 2 loss function)."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        loss_fn = LegalMovesLoss()

        assert loss_fn._pos_weight.item() == 1.0
        assert loss_fn._reduction == "mean"
        assert loss_fn._use_asl is False

    def test_initialization_custom_pos_weight(self) -> None:
        """Test initialization with custom pos_weight."""
        loss_fn = LegalMovesLoss(pos_weight=3.0)

        assert loss_fn._pos_weight.item() == 3.0

    def test_initialization_custom_reduction(self) -> None:
        """Test initialization with custom reduction mode."""
        loss_fn = LegalMovesLoss(reduction="sum")

        assert loss_fn._reduction == "sum"

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

    def test_asl_mode_initialization(self) -> None:
        """Test that ASL mode is activated when gamma_neg > 0."""
        loss_fn = LegalMovesLoss(gamma_neg=2.0, clip=0.02)

        assert loss_fn._use_asl is True
        assert isinstance(loss_fn._loss_fn, AsymmetricLoss)

    def test_default_backward_compatible(self) -> None:
        """Test that default LegalMovesLoss matches standard BCE behavior."""
        legal_loss = LegalMovesLoss()
        bce_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0]), reduction="mean"
        )

        torch.manual_seed(42)
        batch_size = 4
        num_moves = MOVE_LABELS_NUM

        logits = torch.randn(batch_size, num_moves)
        targets = torch.randint(
            0, 2, (batch_size, num_moves)
        ).float()

        legal_result = legal_loss(logits, targets)
        bce_result = bce_loss(logits, targets)

        torch.testing.assert_close(
            legal_result, bce_result, atol=1e-6, rtol=1e-6
        )

    def test_asl_integration(self) -> None:
        """Test LegalMovesLoss with ASL mode produces valid loss."""
        loss_fn = LegalMovesLoss(
            gamma_pos=0.0, gamma_neg=2.0, clip=0.02
        )

        batch_size = 4
        num_moves = MOVE_LABELS_NUM

        targets = torch.zeros(batch_size, num_moves)
        for i in range(batch_size):
            num_legal = np.random.randint(20, 40)
            legal_indices = np.random.choice(
                num_moves, num_legal, replace=False
            )
            targets[i, legal_indices] = 1.0

        logits = torch.randn(batch_size, num_moves)

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_pos_weight_warning_with_asl(self) -> None:
        """Test that a warning is logged when pos_weight != 1.0 with ASL."""
        with warnings_captured_by_logging() as log_records:
            LegalMovesLoss(
                pos_weight=5.0, gamma_neg=2.0, clip=0.02
            )

        # Check that at least one warning about pos_weight was logged
        warning_messages = [
            r.message
            for r in log_records
            if r.levelno == logging.WARNING
        ]
        assert any(
            "pos_weight" in msg for msg in warning_messages
        )
