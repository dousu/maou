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
        inputs=torch.zeros((batch_size, 1), dtype=torch.float32),
        labels_policy=policy_target_distribution,
        labels_value=labels_value,
        legal_move_mask=None,
        outputs_policy=outputs_policy,
        outputs_value=outputs_value,
        loss=torch.tensor(loss, dtype=torch.float32),
        batch_size=batch_size,
        policy_target_distribution=policy_target_distribution,
    )


def test_validation_callback_collects_policy_and_value_metrics() -> None:
    callback = ValidationCallback()

    policy_targets = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    outputs_policy = torch.tensor([[0.1, 2.0, 0.0]], dtype=torch.float32)
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

    log_probs = torch.nn.functional.log_softmax(outputs_policy, dim=1)
    expected_cross_entropy = (
        -torch.sum(policy_targets * log_probs, dim=1).mean().item()
    )
    probabilities = torch.sigmoid(outputs_value)
    expected_brier = torch.square(probabilities - labels_value).mean().item()

    assert isinstance(metrics, ValidationMetrics)
    assert metrics.policy_cross_entropy == pytest.approx(
        expected_cross_entropy
    )
    assert metrics.value_brier_score == pytest.approx(expected_brier)
    assert metrics.policy_top1_accuracy == pytest.approx(1.0)
    assert metrics.value_high_confidence_rate == pytest.approx(1.0)


def test_validation_callback_handles_absent_high_confidence_targets() -> None:
    callback = ValidationCallback()

    policy_targets = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    outputs_policy = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
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

    assert metrics.value_high_confidence_rate == pytest.approx(0.0)


def test_validation_callback_measures_value_precision() -> None:
    callback = ValidationCallback()

    policy_targets = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    outputs_policy = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    labels_value = torch.tensor([0.1, 0.9], dtype=torch.float32)
    outputs_value = torch.tensor([4.0, 4.0], dtype=torch.float32)

    context = _create_context(
        outputs_policy=outputs_policy,
        policy_target_distribution=policy_targets,
        labels_value=labels_value,
        outputs_value=outputs_value,
        loss=0.7,
    )

    callback.on_batch_end(context)

    metrics = callback.get_average_metrics()

    assert metrics.value_high_confidence_rate == pytest.approx(0.5)
