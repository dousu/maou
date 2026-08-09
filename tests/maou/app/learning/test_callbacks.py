import pytest
import torch

from maou.app.learning.callbacks import (
    LRSchedulerStepCallback,
    TrainingContext,
    ValidationCallback,
    ValidationMetrics,
)
from maou.app.learning.policy_targets import (
    normalize_policy_targets,
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
        legal_move_mask: torch.Tensor | None = None,
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
            legal_move_mask=legal_move_mask,
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
        """policy_move_label_ce computes CE against moveLabel with legal_move_mask."""
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
        # legal_move_mask必須: maskなしではpolicyラベルと確率空間が不整合になる
        legal_move_mask = torch.tensor(
            [[1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )

        ctx = self._create_context_with_win_rate(
            outputs_policy=outputs_policy,
            policy_target_distribution=policy_targets,
            labels_value=torch.zeros(1),
            outputs_value=torch.zeros(1),
            move_win_rate=move_win_rate,
            legal_move_mask=legal_move_mask,
        )
        callback.on_batch_end(ctx)
        metrics = callback.get_average_metrics()

        # CE = -sum(normalize(targets * mask) * log_softmax(logits))
        log_probs = torch.nn.functional.log_softmax(
            outputs_policy, dim=1
        )
        masked = policy_targets * legal_move_mask
        target_sum = masked.sum(dim=1, keepdim=True)
        normalized = masked / target_sum
        expected_ce = (
            -torch.sum(normalized * log_probs, dim=1)
            .mean()
            .item()
        )
        assert metrics.policy_move_label_ce == pytest.approx(
            expected_ce
        )

    def test_move_label_ce_computed_without_legal_move_mask(
        self,
    ) -> None:
        """legal_move_mask が None でも policy_move_label_ce を出す．

        以前は ``legal_move_mask is not None`` でゲートして
        いたため，全データ経路がマスクの供給をやめた時点で
        この指標が黙って消えてしまう．マスク無しでも
        ``normalize_policy_targets`` は素の正規化を行い，
        ``policy_log_probs`` と同じ確率空間になる．
        """
        callback = ValidationCallback()

        outputs_policy = torch.tensor(
            [[1.0, 0.0]], dtype=torch.float32
        )
        policy_targets = torch.tensor(
            [[0.8, 0.2]], dtype=torch.float32
        )
        move_win_rate = torch.tensor(
            [[0.5, 0.5]],
            dtype=torch.float32,
        )
        ctx = self._create_context_with_win_rate(
            outputs_policy=outputs_policy,
            policy_target_distribution=policy_targets,
            labels_value=torch.zeros(1),
            outputs_value=torch.zeros(1),
            move_win_rate=move_win_rate,
            legal_move_mask=None,
        )
        callback.on_batch_end(ctx)
        metrics = callback.get_average_metrics()

        log_probs = torch.nn.functional.log_softmax(
            outputs_policy, dim=1
        )
        normalized = policy_targets / policy_targets.sum(
            dim=1, keepdim=True
        )
        expected_ce = (
            -torch.sum(normalized * log_probs, dim=1)
            .mean()
            .item()
        )
        assert metrics.policy_move_label_ce is not None
        assert metrics.policy_move_label_ce == pytest.approx(
            expected_ce
        )

    def test_move_label_ce_matches_all_ones_mask(
        self,
    ) -> None:
        """全 1 マスクを渡した場合と None の場合で値が一致する．

        データ経路からマスクを外した変更が指標を動かして
        いないことを固定する (全 1 マスクは no-op なので，
        マスク有無で結果が変わってはいけない)．
        """
        outputs_policy = torch.tensor(
            [[2.0, 1.0, 0.0]], dtype=torch.float32
        )
        policy_targets = torch.tensor(
            [[0.7, 0.3, 0.0]], dtype=torch.float32
        )
        move_win_rate = torch.tensor(
            [[0.5, 0.5, 0.0]], dtype=torch.float32
        )

        values = []
        for mask in (
            None,
            torch.ones(1, 3, dtype=torch.float32),
        ):
            callback = ValidationCallback()
            ctx = self._create_context_with_win_rate(
                outputs_policy=outputs_policy,
                policy_target_distribution=policy_targets,
                labels_value=torch.zeros(1),
                outputs_value=torch.zeros(1),
                move_win_rate=move_win_rate,
                legal_move_mask=mask,
            )
            callback.on_batch_end(ctx)
            ce = callback.get_average_metrics().policy_move_label_ce
            assert ce is not None
            values.append(ce)

        assert values[0] == pytest.approx(values[1])

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


def _create_minimal_context(
    *, optimizer_stepped: bool
) -> TrainingContext:
    """LRSchedulerStepCallback 用の最小 TrainingContext を生成する．"""
    return TrainingContext(
        batch_idx=0,
        epoch_idx=0,
        inputs=torch.zeros((1, 1), dtype=torch.float32),
        labels_policy=torch.zeros((1,), dtype=torch.long),
        labels_value=torch.zeros((1,), dtype=torch.float32),
        legal_move_mask=None,
        optimizer_stepped=optimizer_stepped,
    )


class TestLRSchedulerStepCallback:
    """optimizer step のスキップに追随して scheduler を制御する挙動の検証．"""

    def _make_scheduler(
        self,
    ) -> torch.optim.lr_scheduler.StepLR:
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.5
        )

    def test_steps_scheduler_when_optimizer_stepped(
        self,
    ) -> None:
        scheduler = self._make_scheduler()
        callback = LRSchedulerStepCallback(scheduler)
        start = scheduler.last_epoch
        # 実際に optimizer.step() が走った状況を再現する．これを省くと
        # PyTorch 自身が「optimizer.step() 前に scheduler.step()」warning を
        # 出すため，本番と同じ順序で呼ぶ．
        scheduler.optimizer.step()

        callback.on_optimizer_step_end(
            _create_minimal_context(optimizer_stepped=True)
        )

        assert scheduler.last_epoch == start + 1

    def test_skips_scheduler_when_optimizer_not_stepped(
        self,
    ) -> None:
        # GradScaler が optimizer.step() をスキップした場合を模擬する．
        scheduler = self._make_scheduler()
        callback = LRSchedulerStepCallback(scheduler)
        start = scheduler.last_epoch

        callback.on_optimizer_step_end(
            _create_minimal_context(optimizer_stepped=False)
        )

        assert scheduler.last_epoch == start


class TestValidationCallbackSharedComputation:
    """同一 logits に対する log_softmax / topk の共有に関する回帰テスト．

    ``on_batch_end`` は 1 バッチにつき ``log_softmax`` を 1 回，
    ``topk`` を 1 回だけ計算し，CE 2 種・expected_win_rate・
    top5 精度・F1 の各指標へ使い回す．共有によって値が変わって
    いないことを固定する．
    """

    @staticmethod
    def _context_with_win_rate() -> TrainingContext:
        policy_targets = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        outputs_policy = torch.tensor(
            [
                [0.1, 2.0, 0.0, -1.0, 0.3, 0.2],
                [1.5, 0.2, 1.4, 0.1, -0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        legal_move_mask = torch.ones(
            (2, 6), dtype=torch.float32
        )
        move_win_rate = torch.tensor(
            [
                [0.1, 0.9, 0.2, 0.3, 0.4, 0.5],
                [0.7, 0.2, 0.6, 0.1, 0.3, 0.4],
            ],
            dtype=torch.float32,
        )
        return TrainingContext(
            batch_idx=0,
            epoch_idx=0,
            inputs=torch.zeros((2, 1), dtype=torch.float32),
            labels_policy=policy_targets,
            labels_value=torch.tensor(
                [1.0, 0.0], dtype=torch.float32
            ),
            legal_move_mask=legal_move_mask,
            outputs_policy=outputs_policy,
            outputs_value=torch.tensor(
                [4.0, -4.0], dtype=torch.float32
            ),
            loss=torch.tensor(0.5, dtype=torch.float32),
            batch_size=2,
            policy_target_distribution=policy_targets,
            move_win_rate=move_win_rate,
        )

    def test_expected_win_rate_matches_independent_softmax(
        self,
    ) -> None:
        """使い回した log_probs でも独立計算と同じ値になる．"""
        context = self._context_with_win_rate()
        callback = ValidationCallback()

        callback.on_batch_end(context)
        metrics = callback.get_average_metrics()

        assert context.outputs_policy is not None
        assert context.move_win_rate is not None
        probs = torch.softmax(context.outputs_policy, dim=1)
        expected = (
            (probs * context.move_win_rate)
            .sum(dim=1)
            .mean()
            .item()
        )

        assert metrics.policy_expected_win_rate is not None
        assert (
            metrics.policy_expected_win_rate
            == pytest.approx(expected)
        )

    def test_move_label_ce_matches_independent_softmax(
        self,
    ) -> None:
        """2 つ目の CE も同じ log_probs を使うが値は変わらない．"""
        context = self._context_with_win_rate()
        callback = ValidationCallback()

        callback.on_batch_end(context)
        metrics = callback.get_average_metrics()

        assert context.outputs_policy is not None
        log_probs = torch.log_softmax(
            context.outputs_policy, dim=1
        )
        targets = normalize_policy_targets(
            context.labels_policy, context.legal_move_mask
        )
        expected = (
            -torch.sum(targets * log_probs, dim=1).mean().item()
        )

        assert metrics.policy_move_label_ce is not None
        assert metrics.policy_move_label_ce == pytest.approx(
            expected
        )

    def test_precomputed_top_indices_match_internal_topk(
        self,
    ) -> None:
        """topk を外から渡しても内部計算と同じ結果になる．"""
        context = self._context_with_win_rate()
        callback = ValidationCallback()
        logits = context.outputs_policy
        targets = context.policy_target_distribution
        assert logits is not None
        assert targets is not None
        top_indices = torch.topk(
            logits.detach(), k=5, dim=1
        ).indices

        internal_ratio, internal_count = (
            callback._compute_policy_top5_accuracy_stats_gpu(
                logits=logits, targets=targets
            )
        )
        shared_ratio, shared_count = (
            callback._compute_policy_top5_accuracy_stats_gpu(
                logits=logits,
                targets=targets,
                prediction_top_indices=top_indices,
            )
        )
        assert internal_count == shared_count
        assert torch.equal(internal_ratio, shared_ratio)

        internal_f1 = (
            callback._compute_policy_f1_components_gpu(
                logits=logits, targets=targets
            )
        )
        shared_f1 = callback._compute_policy_f1_components_gpu(
            logits=logits,
            targets=targets,
            prediction_top_indices=top_indices,
        )
        for internal_t, shared_t in zip(
            internal_f1, shared_f1, strict=True
        ):
            assert torch.equal(internal_t, shared_t)

    def test_log_softmax_and_topk_run_once_per_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """同じ logits に対する log_softmax / topk を再計算しない．

        以前は log_softmax が 3 回 (CE 2 種 + expected_win_rate)，
        topk が 2 回 (top5 精度 + F1) 走っていた．
        """
        context = self._context_with_win_rate()
        callback = ValidationCallback()

        log_softmax_calls = 0
        topk_calls = 0
        original_log_softmax = torch.log_softmax
        original_topk = torch.topk

        def counting_log_softmax(
            *args: object, **kwargs: object
        ):  # type: ignore[no-untyped-def]
            nonlocal log_softmax_calls
            log_softmax_calls += 1
            return original_log_softmax(*args, **kwargs)  # type: ignore[arg-type]

        def counting_topk(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            nonlocal topk_calls
            topk_calls += 1
            return original_topk(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            torch, "log_softmax", counting_log_softmax
        )
        monkeypatch.setattr(torch, "topk", counting_topk)

        callback.on_batch_end(context)

        assert log_softmax_calls == 1
        # targets 側の topk は logits と別物なので 1 回残る．
        assert topk_calls == 2
