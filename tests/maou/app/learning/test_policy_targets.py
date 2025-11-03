"""Tests for policy target normalization utilities."""

from __future__ import annotations

import torch

from maou.app.learning.policy_targets import normalize_policy_targets


def test_normalize_policy_targets_without_mask() -> None:
    labels = torch.tensor(
        [[0.25, 0.75, 1.0], [2.0, 1.0, 1.0]],
        dtype=torch.float32,
    )

    normalized = normalize_policy_targets(labels, legal_move_mask=None)

    expected = torch.tensor(
        [[0.125, 0.375, 0.5], [0.5, 0.25, 0.25]],
        dtype=torch.float32,
    )
    assert torch.allclose(normalized, expected)


def test_normalize_policy_targets_with_mask_and_dtype() -> None:
    labels = torch.tensor(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )

    normalized = normalize_policy_targets(
        labels,
        mask,
        dtype=torch.float64,
    )

    expected = torch.tensor(
        [[0.25, 0.0, 0.75], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    assert normalized.dtype == torch.float64
    assert torch.allclose(normalized, expected)
    assert torch.count_nonzero(normalized[1]).item() == 0
