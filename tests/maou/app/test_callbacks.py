"""Tests for learning callbacks."""

from __future__ import annotations

import pytest

from maou.app.learning.callbacks import ValidationCallback

torch = pytest.importorskip("torch")


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

    # First sample: label positives at indices {0, 1, 3, 5} with Top-5 identical
    # Prediction Top-5 contains indices {0, 1, 2, 4, 5}, overlap = 3/4.
    # Second sample: label positives {1, 2, 4, 5} (Top-5 truncated to 4 entries),
    # prediction Top-5 contains {1, 2, 4, 5}, overlap = 1.0.
    expected_accuracy = (0.75 + 1.0) / 2

    accuracy = callback._policy_accuracy(predictions, labels)

    assert accuracy == expected_accuracy


def test_policy_accuracy_handles_no_positive_labels() -> None:
    """Accuracy should be zero when no labels have positive probability."""

    callback = ValidationCallback()
    predictions = torch.rand((1, 3))
    labels = torch.zeros((1, 3))

    accuracy = callback._policy_accuracy(predictions, labels)

    assert accuracy == 0.0
