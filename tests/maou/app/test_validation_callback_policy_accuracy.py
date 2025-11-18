"""Tests for learning callbacks."""

from __future__ import annotations

import pytest
import torch

from maou.app.learning.callbacks import ValidationCallback

pytest.importorskip("torch")


def _reference_policy_top5_stats(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[float, int]:
    """Reference implementation mirroring the pre-optimization logic."""

    if logits.ndim != 2 or targets.ndim != 2:
        raise ValueError("Tensors logits and targets must be 2-dimensional.")

    batch_size = int(targets.size(0))
    if batch_size == 0:
        return 0.0, 0

    topk_pred = min(5, int(logits.size(1)))
    if topk_pred == 0:
        return 0.0, 0

    prediction_top_indices = torch.topk(logits, k=topk_pred, dim=1).indices

    ratio_sum = 0.0
    sample_count = 0

    for sample_idx in range(batch_size):
        target_values = targets[sample_idx]
        positive_indices = torch.nonzero(target_values > 0, as_tuple=False).view(-1)

        if int(positive_indices.numel()) == 0:
            ratio = 0.0
        else:
            positive_values = target_values[positive_indices]
            label_topk = min(5, int(positive_values.numel()))
            top_label_rel_indices = torch.topk(positive_values, k=label_topk).indices
            label_top_indices = positive_indices[top_label_rel_indices]
            current_topk = min(label_topk, topk_pred)
            predicted_indices_set = set(
                prediction_top_indices[sample_idx, :current_topk].tolist()
            )
            match_count = sum(
                1 for idx in label_top_indices.tolist() if idx in predicted_indices_set
            )
            ratio = float(match_count) / float(label_topk)

        ratio_sum += ratio
        sample_count += 1

    return ratio_sum, sample_count


def test_policy_accuracy_top5_overlap() -> None:
    """ValidationCallback should compute label overlap within prediction Top-5."""

    callback = ValidationCallback()
    predictions = torch.tensor(
        [
            [0.9, 0.1, 0.3, 0.2, 0.5, 0.4],
            [0.2, 0.7, 0.4, 0.1, 0.8, 0.6],
        ]
    )
    labels = torch.tensor(
        [
            [0.6, 0.2, 0.0, 0.1, 0.0, 0.3],
            [0.0, 0.5, 0.4, 0.0, 0.3, 0.2],
        ]
    )

    # First sample: label positives {0, 1, 3, 5}; prediction Top-5 overlap = 0.5.
    # Second sample: label positives {1, 2, 4, 5} (Top-5 truncated to 4 entries),
    # prediction Top-5 contains {1, 2, 4, 5}, overlap = 1.0.
    expected_accuracy = (0.5 + 1.0) / 2

    accuracy = callback._policy_accuracy(predictions, labels)

    assert accuracy == pytest.approx(expected_accuracy)


def test_policy_accuracy_includes_empty_label_samples() -> None:
    """Samples without positive labels should contribute zero to the average."""

    callback = ValidationCallback()
    predictions = torch.tensor(
        [
            [0.9, 0.1, 0.3, 0.2, 0.5, 0.4],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        ]
    )
    labels = torch.tensor(
        [
            [0.6, 0.2, 0.0, 0.1, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    expected_accuracy = (0.5 + 0.0) / 2

    accuracy = callback._policy_accuracy(predictions, labels)

    assert accuracy == pytest.approx(expected_accuracy)


def test_policy_accuracy_handles_no_positive_labels() -> None:
    """Accuracy should be zero when no labels have positive probability."""

    callback = ValidationCallback()
    predictions = torch.rand((1, 3))
    labels = torch.zeros((1, 3))

    accuracy = callback._policy_accuracy(predictions, labels)

    assert accuracy == 0.0


@pytest.mark.parametrize(
    "logits,targets",
    [
        (
            torch.tensor(
                [
                    [0.9, 0.1, 0.3, 0.2, 0.5, 0.4],
                    [0.2, 0.7, 0.4, 0.1, 0.8, 0.6],
                ]
            ),
            torch.tensor(
                [
                    [0.6, 0.2, 0.0, 0.1, 0.0, 0.3],
                    [0.0, 0.5, 0.4, 0.0, 0.3, 0.2],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1],
                    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                ]
            ),
            torch.tensor(
                [
                    [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                ]
            ),
        ),
        (
            torch.tensor(
                [
                    [0.9, 0.8, 0.7],
                    [0.6, 0.5, 0.4],
                    [0.3, 0.2, 0.1],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.9, 0.8, 0.7],
                ]
            ),
        ),
    ],
)
def test_policy_top5_stats_matches_reference(
    logits: torch.Tensor, targets: torch.Tensor
) -> None:
    """Vectorised implementation should match the historical Python logic."""

    callback = ValidationCallback()

    ratio_sum, sample_count = callback._compute_policy_top5_accuracy_stats(
        logits=logits, targets=targets
    )
    ref_ratio_sum, ref_sample_count = _reference_policy_top5_stats(logits, targets)

    assert ratio_sum == pytest.approx(ref_ratio_sum)
    assert sample_count == ref_sample_count
