import pytest
import torch

from maou.app.learning.callbacks import (
    TrainingContext,
    ValidationCallback,
    ValidationMetrics,
)


def _create_context(
    *,
    outputs_policy: torch.Tensor,
    policy_target_distribution: torch.Tensor,
    labels_value: torch.Tensor,
    outputs_value: torch.Tensor,
    loss: float,
) -> TrainingContext:
    batch_size = int(policy_target_distribution.size(0))
    return TrainingContext(
        batch_idx=0,
        epoch_idx=0,
        inputs=torch.zeros(
            (batch_size, 1), dtype=torch.float32
        ),
        labels_policy=policy_target_distribution,
        labels_value=labels_value,
        legal_move_mask=None,
        outputs_policy=outputs_policy,
        outputs_value=outputs_value,
        loss=torch.tensor(loss, dtype=torch.float32),
        batch_size=batch_size,
        policy_target_distribution=policy_target_distribution,
    )


def test_validation_callback_collects_policy_and_value_metrics() -> (
    None
):
    callback = ValidationCallback()

    policy_targets = torch.tensor(
        [[0.0, 1.0, 0.0]], dtype=torch.float32
    )
    outputs_policy = torch.tensor(
        [[0.1, 2.0, 0.0]], dtype=torch.float32
    )
    labels_value = torch.tensor([1.0], dtype=torch.float32)
    outputs_value = torch.tensor([4.0], dtype=torch.float32)

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.5,
    )

    callback.on_batch_end(context)

    metrics = callback.get_average_metrics()

    log_probs = torch.nn.functional.log_softmax(
        outputs_policy, dim=1
    )
    expected_cross_entropy = (
        -torch.sum(policy_targets * log_probs, dim=1)
        .mean()
        .item()
    )
    probabilities = torch.sigmoid(outputs_value)
    expected_brier = (
        torch.square(probabilities - labels_value).mean().item()
    )

    assert isinstance(metrics, ValidationMetrics)
    assert metrics.policy_cross_entropy == pytest.approx(
        expected_cross_entropy
    )
    assert metrics.value_brier_score == pytest.approx(
        expected_brier
    )
    assert metrics.policy_top5_accuracy == pytest.approx(1.0)
    assert metrics.value_high_confidence_rate == pytest.approx(
        1.0
    )


def test_validation_callback_computes_policy_top5_accuracy() -> (
    None
):
    callback = ValidationCallback()

    policy_targets = torch.tensor(
        [
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.0, 0.6, 0.0, 0.5, 0.0, 0.4],
        ],
        dtype=torch.float32,
    )
    outputs_policy = torch.tensor(
        [
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        ],
        dtype=torch.float32,
    )
    labels_value = torch.tensor([1.0, 0.0], dtype=torch.float32)
    outputs_value = torch.tensor(
        [0.0, 0.0], dtype=torch.float32
    )

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.0,
    )

    callback.on_batch_end(context)

    metrics = callback.get_average_metrics()

    expected_accuracy = pytest.approx((1.0 + (1.0 / 3.0)) / 2.0)

    assert metrics.policy_top5_accuracy == expected_accuracy


def test_validation_callback_limits_prediction_topk_to_target_span() -> (
    None
):
    callback = ValidationCallback()

    policy_targets = torch.tensor(
        [[0.6, 0.5, 0.4, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    outputs_policy = torch.tensor(
        [[0.1, 0.2, 0.3, 5.0, 4.0, 3.0]],
        dtype=torch.float32,
    )
    labels_value = torch.tensor([0.0], dtype=torch.float32)
    outputs_value = torch.tensor([0.0], dtype=torch.float32)

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.0,
    )

    callback.on_batch_end(context)

    metrics = callback.get_average_metrics()

    assert metrics.policy_top5_accuracy == pytest.approx(0.0)


def test_validation_callback_handles_absent_high_confidence_targets() -> (
    None
):
    callback = ValidationCallback()

    policy_targets = torch.tensor(
        [[1.0, 0.0]], dtype=torch.float32
    )
    outputs_policy = torch.tensor(
        [[1.0, 0.0]], dtype=torch.float32
    )
    labels_value = torch.tensor([0.2], dtype=torch.float32)
    outputs_value = torch.tensor([-2.0], dtype=torch.float32)

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.3,
    )

    callback.on_batch_end(context)

    metrics = callback.get_average_metrics()

    assert metrics.value_high_confidence_rate == pytest.approx(
        0.0
    )


def test_validation_callback_measures_value_precision() -> None:
    callback = ValidationCallback()

    policy_targets = torch.tensor(
        [[1.0], [1.0]], dtype=torch.float32
    )
    outputs_policy = torch.tensor(
        [[0.0], [0.0]], dtype=torch.float32
    )
    labels_value = torch.tensor([0.1, 0.9], dtype=torch.float32)
    outputs_value = torch.tensor(
        [4.0, 4.0], dtype=torch.float32
    )

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.7,
    )

    callback.on_batch_end(context)

    metrics = callback.get_average_metrics()

    assert metrics.value_high_confidence_rate == pytest.approx(
        0.5
    )


def test_validation_callback_computes_policy_f1_score_all_correct() -> (
    None
):
    """Test F1 score when all predictions match labels (perfect score)."""
    callback = ValidationCallback()

    # 3 classes, predict top-2, labels are {0, 1}
    policy_targets = torch.tensor(
        [[0.5, 0.5, 0.0]],
        dtype=torch.float32,
    )
    outputs_policy = torch.tensor(
        [[2.0, 1.5, 0.1]],  # Top-2: [0, 1]
        dtype=torch.float32,
    )
    labels_value = torch.tensor([0.0], dtype=torch.float32)
    outputs_value = torch.tensor([0.0], dtype=torch.float32)

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.0,
    )

    callback.on_batch_end(context)
    metrics = callback.get_average_metrics()

    # TP=2 (both predictions correct), FP=1 (prediction {2}), FN=0
    # Top-3 predictions because only 3 classes
    # Precision=2/3≈0.667, Recall=2/2=1.0, F1=2*(0.667*1.0)/(0.667+1.0)=0.8
    expected_f1 = 2.0 * (2.0 / 3.0 * 1.0) / (2.0 / 3.0 + 1.0)
    assert metrics.policy_f1_score == pytest.approx(expected_f1)


def test_validation_callback_computes_policy_f1_score_all_wrong() -> (
    None
):
    """Test F1 score when no predictions match labels (worst score)."""
    callback = ValidationCallback()

    # 5 classes, predict top-2, labels are {3, 4}
    policy_targets = torch.tensor(
        [[0.0, 0.0, 0.0, 0.5, 0.5]],
        dtype=torch.float32,
    )
    outputs_policy = torch.tensor(
        [[2.0, 1.5, 0.0, 0.0, 0.0]],  # Top-5: [0, 1, 2, 3, 4]
        dtype=torch.float32,
    )
    labels_value = torch.tensor([0.0], dtype=torch.float32)
    outputs_value = torch.tensor([0.0], dtype=torch.float32)

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.0,
    )

    callback.on_batch_end(context)
    metrics = callback.get_average_metrics()

    # TP=2 (predictions {3,4} match labels), FP=3 (predictions {0,1,2}), FN=0
    # Precision=2/5=0.4, Recall=2/2=1.0
    # F1 = 2 * (0.4 * 1.0) / (0.4 + 1.0) = 0.8 / 1.4
    expected_f1 = 2.0 * (2.0 / 5.0 * 1.0) / (2.0 / 5.0 + 1.0)
    assert metrics.policy_f1_score == pytest.approx(expected_f1)


def test_validation_callback_computes_policy_f1_score_partial_overlap() -> (
    None
):
    """Test F1 score with partial overlap between predictions and labels."""
    callback = ValidationCallback()

    # 6 classes, predict top-5, labels are {0, 2, 4}
    policy_targets = torch.tensor(
        [[0.4, 0.0, 0.3, 0.0, 0.3, 0.0]],
        dtype=torch.float32,
    )
    outputs_policy = torch.tensor(
        [
            [2.0, 1.8, 1.5, 0.5, 0.3, 0.1]
        ],  # Top-5: [0, 1, 2, 3, 4]
        dtype=torch.float32,
    )
    labels_value = torch.tensor([0.0], dtype=torch.float32)
    outputs_value = torch.tensor([0.0], dtype=torch.float32)

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.0,
    )

    callback.on_batch_end(context)
    metrics = callback.get_average_metrics()

    # TP=3 (predictions {0,2,4} match labels), FP=2 (predictions {1,3}), FN=0
    # Precision=3/5=0.6, Recall=3/3=1.0
    # F1 = 2 * (0.6 * 1.0) / (0.6 + 1.0) = 1.2 / 1.6 = 0.75
    assert metrics.policy_f1_score == pytest.approx(0.75)


def test_validation_callback_computes_policy_f1_across_multiple_batches() -> (
    None
):
    """Test F1 score accumulation and aggregation across multiple batches."""
    callback = ValidationCallback()

    # Batch 1: Perfect predictions
    policy_targets_1 = torch.tensor(
        [[0.5, 0.5, 0.0]],
        dtype=torch.float32,
    )
    outputs_policy_1 = torch.tensor(
        [
            [2.0, 1.5, 0.1]
        ],  # Top-5: [0, 1, 2] (but only 3 classes)
        dtype=torch.float32,
    )

    context_1 = _create_context(
        outputs_policy=outputs_policy_1,
        policy_target_distribution=policy_targets_1,
        labels_value=torch.tensor([0.0], dtype=torch.float32),
        outputs_value=torch.tensor([0.0], dtype=torch.float32),
        loss=0.0,
    )
    callback.on_batch_end(context_1)

    # Batch 2: No overlap
    policy_targets_2 = torch.tensor(
        [[0.0, 0.0, 0.0, 0.5, 0.5]],
        dtype=torch.float32,
    )
    outputs_policy_2 = torch.tensor(
        [[2.0, 1.5, 1.0, 0.0, 0.0]],  # Top-5: [0, 1, 2, 3, 4]
        dtype=torch.float32,
    )

    context_2 = _create_context(
        outputs_policy=outputs_policy_2,
        policy_target_distribution=policy_targets_2,
        labels_value=torch.tensor([0.0], dtype=torch.float32),
        outputs_value=torch.tensor([0.0], dtype=torch.float32),
        loss=0.0,
    )
    callback.on_batch_end(context_2)

    metrics = callback.get_average_metrics()

    # Batch 1: TP=2, FP=1, FN=0 (top-3 because only 3 classes)
    # Batch 2: TP=2, FP=3, FN=0
    # Total: TP=4, FP=4, FN=0
    # Precision=4/8=0.5, Recall=4/4=1.0, F1=2*(0.5*1.0)/(0.5+1.0)=2/3
    expected_f1 = 2.0 * (4.0 / 8.0 * 1.0) / (4.0 / 8.0 + 1.0)
    assert metrics.policy_f1_score == pytest.approx(expected_f1)


def test_validation_callback_f1_handles_varying_label_counts() -> (
    None
):
    """Test F1 score with samples having different numbers of labels."""
    callback = ValidationCallback()

    # Sample 1: 1 label
    # Sample 2: 3 labels
    policy_targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # 1 label
            [0.33, 0.33, 0.34, 0.0],  # 3 labels
        ],
        dtype=torch.float32,
    )
    outputs_policy = torch.tensor(
        [
            [2.0, 1.0, 0.5, 0.0],  # Top-4: [0, 1, 2, 3]
            [2.0, 1.5, 0.5, 0.0],  # Top-4: [0, 1, 2, 3]
        ],
        dtype=torch.float32,
    )
    labels_value = torch.tensor([0.0, 0.0], dtype=torch.float32)
    outputs_value = torch.tensor(
        [0.0, 0.0], dtype=torch.float32
    )

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.0,
    )

    callback.on_batch_end(context)
    metrics = callback.get_average_metrics()

    # Sample 1: TP=1, FP=3, FN=0
    # Sample 2: TP=3, FP=1, FN=0
    # Total: TP=4, FP=4, FN=0
    # Precision=4/8=0.5, Recall=4/4=1.0
    # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2/3
    expected_f1 = 2.0 * (4.0 / 8.0 * 1.0) / (4.0 / 8.0 + 1.0)
    assert metrics.policy_f1_score == pytest.approx(expected_f1)


def test_validation_callback_f1_handles_zero_labels() -> None:
    """Test F1 score when sample has no positive labels (edge case)."""
    callback = ValidationCallback()

    policy_targets = torch.tensor(
        [[0.0, 0.0, 0.0]],  # No positive labels
        dtype=torch.float32,
    )
    outputs_policy = torch.tensor(
        [[2.0, 1.0, 0.5]],  # Still makes predictions
        dtype=torch.float32,
    )
    labels_value = torch.tensor([0.0], dtype=torch.float32)
    outputs_value = torch.tensor([0.0], dtype=torch.float32)

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.0,
    )

    callback.on_batch_end(context)
    metrics = callback.get_average_metrics()

    # TP=0 (no labels to match), FP=3 (all predictions), FN=0 (no labels)
    # Precision=0/3=0.0, Recall=0/0=0.0 (undefined, set to 0)
    # F1=0.0
    assert metrics.policy_f1_score == pytest.approx(0.0)


class TestValidationCallbackWinRateMetrics:
    """Tests for move_win_rate-based validation metrics."""

    def _create_context_with_win_rate(
        self,
        *,
        outputs_policy: torch.Tensor,
        policy_target_distribution: torch.Tensor,
        labels_value: torch.Tensor,
        outputs_value: torch.Tensor,
        move_win_rate: torch.Tensor | None,
        loss: float = 0.0,
    ) -> TrainingContext:
        batch_size = int(policy_target_distribution.size(0))
        return TrainingContext(
            batch_idx=0,
            epoch_idx=0,
            inputs=torch.zeros(
                (batch_size, 1), dtype=torch.float32
            ),
            labels_policy=policy_target_distribution,
            labels_value=labels_value,
            legal_move_mask=None,
            outputs_policy=outputs_policy,
            outputs_value=outputs_value,
            loss=torch.tensor(loss, dtype=torch.float32),
            batch_size=batch_size,
            policy_target_distribution=policy_target_distribution,
            move_win_rate=move_win_rate,
        )

    def test_top1_win_rate(self) -> None:
        """policy_top1_win_rate returns average win rate of top-1 predicted move."""
        callback = ValidationCallback()

        # 2 samples, 4 moves each
        outputs_policy = torch.tensor(
            [
                [3.0, 1.0, 0.0, 0.0],  # argmax=0
                [0.0, 0.0, 5.0, 1.0],  # argmax=2
            ],
            dtype=torch.float32,
        )
        move_win_rate = torch.tensor(
            [
                [
                    0.8,
                    0.2,
                    0.1,
                    0.0,
                ],  # win rate of move 0 = 0.8
                [
                    0.1,
                    0.3,
                    0.9,
                    0.5,
                ],  # win rate of move 2 = 0.9
            ],
            dtype=torch.float32,
        )
        policy_targets = torch.tensor(
            [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]],
            dtype=torch.float32,
        )

        ctx = self._create_context_with_win_rate(
            outputs_policy=outputs_policy,
            policy_target_distribution=policy_targets,
            labels_value=torch.zeros(2),
            outputs_value=torch.zeros(2),
            move_win_rate=move_win_rate,
        )
        callback.on_batch_end(ctx)
        metrics = callback.get_average_metrics()

        # top-1 win rates: 0.8 and 0.9, average = 0.85
        assert metrics.policy_top1_win_rate == pytest.approx(
            0.85
        )

    def test_expected_win_rate(self) -> None:
        """policy_expected_win_rate computes softmax-weighted win rate."""
        callback = ValidationCallback()

        # Single sample for simplicity
        outputs_policy = torch.tensor(
            [[0.0, 0.0]],  # equal logits → softmax = [0.5, 0.5]
            dtype=torch.float32,
        )
        move_win_rate = torch.tensor(
            [[0.6, 0.4]],
            dtype=torch.float32,
        )
        policy_targets = torch.tensor(
            [[0.5, 0.5]],
            dtype=torch.float32,
        )

        ctx = self._create_context_with_win_rate(
            outputs_policy=outputs_policy,
            policy_target_distribution=policy_targets,
            labels_value=torch.zeros(1),
            outputs_value=torch.zeros(1),
            move_win_rate=move_win_rate,
        )
        callback.on_batch_end(ctx)
        metrics = callback.get_average_metrics()

        # softmax([0, 0]) = [0.5, 0.5]
        # expected = 0.5*0.6 + 0.5*0.4 = 0.5
        assert (
            metrics.policy_expected_win_rate
            == pytest.approx(0.5)
        )

    def test_move_label_ce(self) -> None:
        """policy_move_label_ce computes CE against moveLabel."""
        callback = ValidationCallback()

        policy_targets = torch.tensor(
            [[0.7, 0.3, 0.0]],
            dtype=torch.float32,
        )
        outputs_policy = torch.tensor(
            [[2.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        move_win_rate = torch.tensor(
            [[0.5, 0.5, 0.0]],
            dtype=torch.float32,
        )

        ctx = self._create_context_with_win_rate(
            outputs_policy=outputs_policy,
            policy_target_distribution=policy_targets,
            labels_value=torch.zeros(1),
            outputs_value=torch.zeros(1),
            move_win_rate=move_win_rate,
        )
        callback.on_batch_end(ctx)
        metrics = callback.get_average_metrics()

        # CE = -sum(normalized_targets * log_softmax(logits))
        log_probs = torch.nn.functional.log_softmax(
            outputs_policy, dim=1
        )
        target_sum = policy_targets.sum(dim=1, keepdim=True)
        normalized = policy_targets / target_sum
        expected_ce = (
            -torch.sum(normalized * log_probs, dim=1)
            .mean()
            .item()
        )
        assert metrics.policy_move_label_ce == pytest.approx(
            expected_ce
        )

    def test_metrics_none_without_move_win_rate(self) -> None:
        """Win rate metrics are None when move_win_rate is absent."""
        callback = ValidationCallback()

        ctx = self._create_context_with_win_rate(
            outputs_policy=torch.tensor(
                [[1.0, 0.0]], dtype=torch.float32
            ),
            policy_target_distribution=torch.tensor(
                [[1.0, 0.0]], dtype=torch.float32
            ),
            labels_value=torch.zeros(1),
            outputs_value=torch.zeros(1),
            move_win_rate=None,
        )
        callback.on_batch_end(ctx)
        metrics = callback.get_average_metrics()

        assert metrics.policy_top1_win_rate is None
        assert metrics.policy_move_label_ce is None
        assert metrics.policy_expected_win_rate is None

    def test_metrics_accumulate_across_batches(self) -> None:
        """Win rate metrics accumulate correctly across batches."""
        callback = ValidationCallback()

        # Batch 1: top-1 win rate = 1.0
        ctx1 = self._create_context_with_win_rate(
            outputs_policy=torch.tensor(
                [[5.0, 0.0]], dtype=torch.float32
            ),
            policy_target_distribution=torch.tensor(
                [[1.0, 0.0]], dtype=torch.float32
            ),
            labels_value=torch.zeros(1),
            outputs_value=torch.zeros(1),
            move_win_rate=torch.tensor(
                [[1.0, 0.0]], dtype=torch.float32
            ),
        )
        callback.on_batch_end(ctx1)

        # Batch 2: top-1 win rate = 0.6
        ctx2 = self._create_context_with_win_rate(
            outputs_policy=torch.tensor(
                [[5.0, 0.0]], dtype=torch.float32
            ),
            policy_target_distribution=torch.tensor(
                [[1.0, 0.0]], dtype=torch.float32
            ),
            labels_value=torch.zeros(1),
            outputs_value=torch.zeros(1),
            move_win_rate=torch.tensor(
                [[0.6, 0.4]], dtype=torch.float32
            ),
        )
        callback.on_batch_end(ctx2)

        metrics = callback.get_average_metrics()

        # Average top-1 win rate: (1.0 + 0.6) / 2 = 0.8
        assert metrics.policy_top1_win_rate == pytest.approx(
            0.8
        )

    def test_reset_clears_win_rate_metrics(self) -> None:
        """reset() clears accumulated win rate metrics."""
        callback = ValidationCallback()

        ctx = self._create_context_with_win_rate(
            outputs_policy=torch.tensor(
                [[5.0, 0.0]], dtype=torch.float32
            ),
            policy_target_distribution=torch.tensor(
                [[1.0, 0.0]], dtype=torch.float32
            ),
            labels_value=torch.zeros(1),
            outputs_value=torch.zeros(1),
            move_win_rate=torch.tensor(
                [[0.9, 0.1]], dtype=torch.float32
            ),
        )
        callback.on_batch_end(ctx)
        callback.reset()

        metrics = callback.get_average_metrics()

        assert metrics.policy_top1_win_rate is None
        assert metrics.policy_move_label_ce is None
        assert metrics.policy_expected_win_rate is None
